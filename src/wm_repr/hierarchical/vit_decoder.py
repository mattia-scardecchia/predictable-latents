# Decoder-forward portions are adapted from the RAE/Hugging Face ViT decoder.
# The following notices apply to those upstream portions.
# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# RAE source: https://github.com/bytetriper/RAE/tree/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041
# RAE additions: Copyright (c) 2025 Boyang Zheng, MIT License.
# Modified: added past-frame conditioning and cross-attention adapters between decoder layers.

"""Reconstruct frames from target latents and past images.

The pretrained RAE ViT decoder uses zero-initialized cross-attention adapters
between transformer layers.
"""

import logging
from contextlib import nullcontext
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_repr.wm.model import RotaryType, rope_mix, rope_nd


# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------


class ViTCrossAttentionAdapter(nn.Module):
    """
    Zero-initialized cross-attention adapter for conditioning ViT decoder layers.

    Inserted after each transformer layer (on patch tokens only, CLS excluded).
    Due to zero-init of output projection, initially acts as identity.
    Learns to incorporate context information during training.

    Uses 2D RoPE for position encoding on both queries and keys, enabling
    position-aware cross-attention between decoder features and context tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
        )
        assert self.head_dim % 4 == 0, (
            f"head_dim {self.head_dim} (hidden_dim/num_heads) must be "
            f"divisible by 4 for 2D RoPE"
        )

        # Pre-norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Query from decoder features, K/V from context
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(context_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Zero-init output projection for identity behavior at start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        query_shape: Tuple[int, int],
        context_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Apply cross-attention to incorporate context.

        Args:
            x: Patch token features ``(B, N, hidden_dim)`` — CLS excluded.
            context: Context tokens ``(B, N_ctx, context_dim)``.
            query_shape: Spatial grid shape of query tokens ``(H_q, W_q)``.
            context_shape: Spatial grid shape of context tokens ``(H_ctx, W_ctx)``.

        Returns:
            Modulated features ``(B, N, hidden_dim)`` — same shape as input.
        """
        B, N, _ = x.shape
        H_q, W_q = query_shape
        H_ctx, W_ctx = context_shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Project queries from decoder features
        q = self.q_proj(x)  # (B, N, hidden_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)

        # Project context to K/V
        kv = self.kv_proj(context)  # (B, N_ctx, hidden_dim * 2)
        k, v = kv.chunk(2, dim=-1)
        N_ctx = context.shape[1]
        k = k.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        # Both: (B, num_heads, N_ctx, head_dim)

        # 2D RoPE for queries at decoder token grid resolution
        q = q.view(B, self.num_heads, H_q, W_q, self.head_dim)
        rope_q = rope_nd(
            (H_q, W_q),
            self.head_dim,
            rotary_type=RotaryType.PIXEL,
            dtype=q.dtype,
            device=q.device,
        )
        rope_q = rope_q.view(H_q, W_q, 2, 2, -1)  # (H, W, num_axes, cos/sin, half)
        cos_q, sin_q = rope_q.unbind(-2)
        cos_q = cos_q.reshape(H_q, W_q, -1)
        sin_q = sin_q.reshape(H_q, W_q, -1)
        q = rope_mix(q, cos_q, sin_q)
        q = q.reshape(B, self.num_heads, N, self.head_dim)

        # 2D RoPE for keys at context grid resolution
        k = k.view(B, self.num_heads, H_ctx, W_ctx, self.head_dim)
        rope_k = rope_nd(
            (H_ctx, W_ctx),
            self.head_dim,
            rotary_type=RotaryType.PIXEL,
            dtype=k.dtype,
            device=k.device,
        )
        rope_k = rope_k.view(H_ctx, W_ctx, 2, 2, -1)
        cos_k, sin_k = rope_k.unbind(-2)
        cos_k = cos_k.reshape(H_ctx, W_ctx, -1)
        sin_k = sin_k.reshape(H_ctx, W_ctx, -1)
        k = rope_mix(k, cos_k, sin_k)
        k = k.reshape(B, self.num_heads, N_ctx, self.head_dim)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.hidden_dim)

        # Project and residual
        out = self.dropout(self.out_proj(out))
        return residual + out


class ViTContextEncoder(nn.Module):
    """
    Encodes s_past into context tokens for cross-attention.

    Sources of context (fused per-position along feature dimension):
    1. Patchified s_past (raw pixel info): patch_size × patch_size patches → linear projection
    2. Encoder features (semantic info): frozen RAE encoder output

    Both share the same spatial grid (determined by encoder patch size).
    """

    def __init__(
        self,
        rae_encoder: nn.Module,
        encoder_mean: torch.Tensor,
        encoder_std: torch.Tensor,
        encoder_input_size: int,
        context_dim: int = 512,
        patch_size: int = 14,
        latent_channels: int = 768,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.patch_size = patch_size
        self.encoder_input_size = encoder_input_size

        # ImageNet normalization for encoder input
        self.register_buffer("encoder_mean", encoder_mean.clone())
        self.register_buffer("encoder_std", encoder_std.clone())

        # Patch projection: flatten patch_size × patch_size × 3 patches → context_dim
        patch_dim = patch_size * patch_size * 3
        self.patch_proj = nn.Linear(patch_dim, context_dim)

        # Encoder features and fusion of the two sources
        self.encoder = deepcopy(rae_encoder)
        self.encoder_proj = nn.Linear(latent_channels, context_dim)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fusion_proj = nn.Linear(2 * context_dim, context_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches matching encoder spatial grid.

        Args:
            x: Image ``(B, 3, H, W)``.

        Returns:
            Patches ``(B, num_patches, patch_dim)``.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, (
            f"Image size ({H}, {W}) must be divisible by patch_size {p}"
        )

        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x

    def _extract_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract semantic features from frozen RAE encoder.

        Args:
            x: Image ``(B, 3, H, W)`` in ``[0, 1]``, already resized to
               ``encoder_input_size``.

        Returns:
            Encoder features ``(B, N, encoder_hidden_size)``.
        """
        # ImageNet normalization
        x = (x - self.encoder_mean) / self.encoder_std

        # ViT encoder forward → (B, N, hidden_size)
        return self.encoder(x)

    def forward(self, s_past: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Encode s_past into context tokens.

        Args:
            s_past: Past frame ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            context: Context tokens ``(B, N, context_dim)``.
            context_shape: Spatial grid shape ``(H_ctx, W_ctx)`` for RoPE.
        """
        # Resize s_past to encoder input size for consistent patchification
        _, _, h, w = s_past.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            s_past_resized = F.interpolate(
                s_past,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        else:
            s_past_resized = s_past

        p = self.patch_size
        context_h = self.encoder_input_size // p
        context_w = self.encoder_input_size // p

        # Source 1: Patchified pixels
        patches = self._patchify(s_past_resized)  # (B, N, patch_dim)
        patch_tokens = self.patch_proj(patches)  # (B, N, context_dim)

        # Source 2: Frozen encoder features
        with torch.no_grad() if self._is_encoder_frozen() else nullcontext():
            enc_features = self._extract_encoder_features(s_past_resized)
        enc_tokens = self.encoder_proj(enc_features)  # (B, N, context_dim)

        # Fuse sources
        context = self.fusion_proj(torch.cat([patch_tokens, enc_tokens], dim=-1))

        return context, (context_h, context_w)

    def _is_encoder_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.parameters())


# -----------------------------------------------------------------------------
# Main Decoder
# -----------------------------------------------------------------------------


class PastConditionedViTDecoder(nn.Module):
    """
    ViT decoder with cross-attention conditioning from s_past.

    Wraps pretrained RAE ViT decoder and injects zero-initialized
    ViTCrossAttentionAdapters after each transformer layer (on patch tokens
    only; CLS token passes through unchanged). Initial behavior is identical
    to vanilla RAE decoding; conditioning effect grows during training.

    Supports unconditioned forward pass (skip context encoder + adapters),
    which reduces to a vanilla RAE decode of z_t.

    Args:
        rae: Pretrained RAE model.
        context_dim: Dimension for context tokens.
        num_heads: Number of attention heads in adapters.
        dropout: Dropout rate.
        adapter_interval: Insert adapter every K-th layer (1 = all layers).
    """

    def __init__(
        self,
        rae,
        context_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        adapter_interval: int = 1,
        pretrained_decoder: bool = False,
    ):
        super().__init__()
        self.supports_unconditioned = True
        self.context_dim = context_dim

        # The zero-init adapter strategy assumes the ViT decoder is pretrained
        # so that initial output is a reasonable reconstruction. Without a
        # pretrained decoder the adapters start from zero on top of random noise.
        if not pretrained_decoder:
            logging.warning(
                "PastConditionedViTDecoder: no pretrained RAE decoder provided. "
                "The zero-init adapter strategy will start from random "
                "reconstructions instead of high-quality ones. Set "
                "pretrained_decoder_path in the RAE config to fix this."
            )

        # Read decoder hidden size from the actual decoder
        decoder_hidden_size = rae.decoder.config.decoder_hidden_size

        # Validate adapter dimensions
        assert decoder_hidden_size % num_heads == 0, (
            f"decoder_hidden_size {decoder_hidden_size} not divisible "
            f"by num_heads {num_heads}"
        )
        head_dim = decoder_hidden_size // num_heads
        assert head_dim % 4 == 0, (
            f"head_dim {head_dim} (decoder_hidden_size/num_heads) must be "
            f"divisible by 4 for 2D RoPE"
        )

        # --- Context encoder ---
        self.context_encoder = ViTContextEncoder(
            rae_encoder=rae.encoder,
            encoder_mean=rae.encoder_mean,
            encoder_std=rae.encoder_std,
            encoder_input_size=rae.encoder_input_size,
            context_dim=context_dim,
            patch_size=rae.encoder_patch_size,
            latent_channels=rae.latent_dim,
        )

        # --- ViT decoder (copy to avoid modifying original) ---
        self.vit_decoder = deepcopy(rae.decoder)

        # --- Normalization stats for input/output transforms ---
        self._do_normalization: bool = rae.do_normalization
        self._eps: float = getattr(rae, "eps", 1e-5)

        self.register_buffer("encoder_mean", rae.encoder_mean.clone())
        self.register_buffer("encoder_std", rae.encoder_std.clone())
        if self._do_normalization:
            self.register_buffer(
                "latent_mean",
                rae.latent_mean.clone() if rae.latent_mean is not None else None,
            )
            self.register_buffer(
                "latent_var",
                rae.latent_var.clone() if rae.latent_var is not None else None,
            )

        # --- Spatial grid shape for RoPE (from encoder patch layout) ---
        grid_side = rae.encoder_input_size // rae.encoder_patch_size
        self._query_shape = (grid_side, grid_side)

        # --- Cross-attention adapters ---
        num_layers = len(self.vit_decoder.decoder_layers)
        self.adapters = nn.ModuleList()
        for i in range(num_layers):
            if i % adapter_interval == 0:
                self.adapters.append(
                    ViTCrossAttentionAdapter(
                        hidden_dim=decoder_hidden_size,
                        context_dim=context_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                self.adapters.append(None)

    def forward(
        self,
        s_past: torch.Tensor,
        z_past: torch.Tensor,
        z_t: torch.Tensor,
        unconditioned: bool = False,
    ) -> torch.Tensor:
        """
        Decode z_t with conditioning from s_past.

        Args:
            s_past: Past frame ``(B, 3, H, W)`` in ``[0, 1]``.
            z_past: Past embedding ``(B, C, h, w)``; not used.
            z_t: Target embedding ``(B, C, h, w)``.
            unconditioned: If True, skip context encoder and adapters.

        Returns:
            Reconstructed frame ``(B, 3, H_out, W_out)`` in approx ``[0, 1]``.
        """
        # --- Encode context ---
        if not unconditioned:
            context, context_shape = self.context_encoder(s_past)

        # --- Prepare decoder input: denormalize + reshape to tokens ---
        z = z_t
        if self._do_normalization:
            mean = self.latent_mean if self.latent_mean is not None else 0
            var = self.latent_var if self.latent_var is not None else 1
            z = z * torch.sqrt(var + self._eps) + mean

        # (B, C, h, w) → (B, N, C) token sequence
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)

        # --- ViT decoder forward with adapter injection ---
        # Replicate GeneralDecoder.forward() to interleave adapters
        dec = self.vit_decoder

        # Embed tokens: Linear(encoder_hidden → decoder_hidden)
        x = dec.decoder_embed(z)

        # Interpolate if num tokens differs from expected
        x = dec.interpolate_latent(x)

        # Prepend trainable CLS token
        cls_token = dec.trainable_cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings
        x = x + dec.decoder_pos_embed

        # Process through decoder layers with adapters
        for i, layer_module in enumerate(dec.decoder_layers):
            # Transformer layer (self-attention + FFN)
            if getattr(dec, "gradient_checkpointing", False) and self.training:
                layer_outputs = dec._gradient_checkpointing_func(
                    layer_module.__call__,
                    x,
                    None,   # head_mask
                    False,  # output_attentions
                )
            else:
                x_out = layer_module(x, head_mask=None, output_attentions=False)
                layer_outputs = x_out
            x = layer_outputs[0]

            # Apply cross-attention adapter to patch tokens (skip CLS at index 0)
            if not unconditioned and self.adapters[i] is not None:
                cls, patches = x[:, :1], x[:, 1:]
                patches = self.adapters[i](
                    patches, context, self._query_shape, context_shape
                )
                x = torch.cat([cls, patches], dim=1)

        # Final norm + prediction head
        x = dec.decoder_norm(x)
        logits = dec.decoder_pred(x)

        # Remove CLS token
        logits = logits[:, 1:]

        # Unpatchify → (B, 3, H_out, W_out) in encoder-normalized space
        x_rec = dec.unpatchify(logits)

        # Reverse ImageNet normalization → approximately [0, 1]
        x_rec = x_rec * self.encoder_std + self.encoder_mean

        return x_rec
