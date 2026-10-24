# Decoder-forward portions are adapted from Hugging Face Diffusers.
# Source: https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/autoencoders/vae.py
# The following notice applies to those upstream portions.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# Modified: added past-frame context and cross-attention adapters between decoder stages.

"""Reconstruct frames from target latents and past images.

The pretrained convolutional VAE decoder uses zero-initialized cross-attention
adapters between decoder stages.
"""

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from wm_repr.wm.model import RotaryType, rope_mix, rope_nd


# -----------------------------------------------------------------------------
# Validation Helper
# -----------------------------------------------------------------------------


def _validate_conv_decoder_config(
    vae: AutoencoderKL,
    context_dim: int,
    num_heads: int,
    num_groups: int,
    patch_size: int,
) -> int:
    """
    Validate PastConditionedConvDecoder config. Returns VAE downsampling factor.

    Checks:
    - VAE type is AutoencoderKL
    - patch_size matches VAE downsampling factor
    - context_dim divisible by num_heads
    - head_dim (context_dim // num_heads) divisible by 4 for 2D RoPE
    - All VAE channel counts divisible by num_groups and num_heads

    Args:
        vae: VAE model to validate
        context_dim: Context dimension for cross-attention
        num_heads: Number of attention heads
        num_groups: GroupNorm groups
        patch_size: Patch size for context encoder

    Returns:
        VAE downsampling factor (typically 8 for SD VAE)

    Raises:
        TypeError: If vae is not AutoencoderKL
        ValueError: If any config validation fails
    """
    # VAE type
    if not isinstance(vae, AutoencoderKL):
        raise TypeError(
            f"Expected AutoencoderKL from diffusers.models.autoencoders.autoencoder_kl, "
            f"got {type(vae).__name__}"
        )

    # Downsampling factor
    num_downsamplers = len(vae.config.block_out_channels) - 1
    downsampling_factor = 2**num_downsamplers
    if patch_size != downsampling_factor:
        raise ValueError(
            f"patch_size={patch_size} must equal VAE downsampling factor={downsampling_factor}"
        )

    # Context dim / heads compatibility
    if context_dim % num_heads != 0:
        raise ValueError(
            f"context_dim={context_dim} must be divisible by num_heads={num_heads}"
        )

    head_dim = context_dim // num_heads
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} (context_dim/num_heads) must be divisible by 4 for 2D RoPE"
        )

    # VAE channel compatibility with adapters
    for i, ch in enumerate(vae.config.block_out_channels):
        if ch % num_heads != 0:
            raise ValueError(
                f"VAE block_out_channels[{i}]={ch} not divisible by num_heads={num_heads}"
            )
        adapter_head_dim = ch // num_heads
        if adapter_head_dim % 4 != 0:
            raise ValueError(
                f"VAE block_out_channels[{i}]={ch} / num_heads={num_heads} = {adapter_head_dim}, "
                f"which is not divisible by 4 (required for 2D RoPE)"
            )
        if ch % num_groups != 0:
            raise ValueError(
                f"VAE block_out_channels[{i}]={ch} not divisible by num_groups={num_groups}"
            )

    return downsampling_factor


# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------


class CrossAttentionAdapter(nn.Module):
    """
    Zero-initialized cross-attention adapter for conditioning VAE decoder.

    Inserted after each decoder stage. Due to zero-init of output projection,
    initially acts as identity (returns input unchanged). Learns to incorporate
    context information during training.

    Uses 2D RoPE for position encoding on both queries and keys, enabling
    position-aware cross-attention between decoder features and context tokens.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, (
            f"channels {channels} not divisible by num_heads {num_heads}"
        )

        # Pre-norm
        self.norm = nn.GroupNorm(num_groups, channels)

        # Query from spatial features, K/V from context
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(context_dim, channels * 2)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        # Zero-init output projection for identity behavior at start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Apply cross-attention to incorporate context.

        Args:
            x: Spatial features (B, C, H, W)
            context: Context tokens (B, N, context_dim)
            context_shape: Spatial shape of context grid (ctx_H, ctx_W) for RoPE

        Returns:
            Modulated features (B, C, H, W)
        """
        B, C, H, W = x.shape
        ctx_H, ctx_W = context_shape
        residual = x

        # Pre-norm and flatten
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project queries
        q = self.q_proj(x)  # (B, H*W, C)
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # Project context to K/V
        kv = self.kv_proj(context)  # (B, N, C*2)
        k, v = kv.chunk(2, dim=-1)
        N = context.shape[1]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE to queries and keys at their respective resolutions
        # Both use PIXEL coordinates (linspace -1 to +1) for approximate alignment

        # RoPE for queries at decoder resolution (H, W)
        q = q.view(B, self.num_heads, H, W, self.head_dim)
        rope_q = rope_nd(
            (H, W),
            self.head_dim,
            rotary_type=RotaryType.PIXEL,
            dtype=q.dtype,
            device=q.device,
        )
        rope_q = rope_q.view(H, W, 2, 2, -1)  # (H, W, num_axes, cos/sin, half)
        cos_q, sin_q = rope_q.unbind(-2)
        cos_q, sin_q = cos_q.reshape(H, W, -1), sin_q.reshape(H, W, -1)
        q = rope_mix(q, cos_q, sin_q)
        q = q.reshape(B, self.num_heads, H * W, self.head_dim)

        # RoPE for keys at context resolution (ctx_H, ctx_W)
        k = k.view(B, self.num_heads, ctx_H, ctx_W, self.head_dim)
        rope_k = rope_nd(
            (ctx_H, ctx_W),
            self.head_dim,
            rotary_type=RotaryType.PIXEL,
            dtype=k.dtype,
            device=k.device,
        )
        rope_k = rope_k.view(ctx_H, ctx_W, 2, 2, -1)
        cos_k, sin_k = rope_k.unbind(-2)
        cos_k, sin_k = cos_k.reshape(ctx_H, ctx_W, -1), sin_k.reshape(ctx_H, ctx_W, -1)
        k = rope_mix(k, cos_k, sin_k)
        k = k.reshape(B, self.num_heads, N, self.head_dim)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, H*W, head_dim)
        out = out.transpose(1, 2).reshape(B, H * W, C)

        # Project and dropout
        out = self.dropout(self.out_proj(out))

        # Reshape and residual
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return residual + out


class ContextEncoder(nn.Module):
    """
    Encodes s_past into context tokens for cross-attention.

    Sources of context (fused per-position along feature dimension):
    1. Patchified s_past (raw pixel info): 8x8 patches → linear projection
    2. Bottleneck features (semantic info): frozen VAE encoder mid_block output

    Both share the same spatial grid (patch_size matches VAE downsampling factor).
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        context_dim: int = 512,
        patch_size: int = 8,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.patch_size = patch_size

        # Patch projection: flatten 8x8x3 patches → context_dim
        patch_dim = patch_size * patch_size * 3
        self.patch_proj = nn.Linear(patch_dim, context_dim)

        # Bottleneck projection and fusion of the two sources
        self.encoder = deepcopy(vae.encoder)
        bottleneck_channels = vae.config.block_out_channels[-1]
        self.bottleneck_proj = nn.Linear(bottleneck_channels, context_dim)
        self.fusion_proj = nn.Linear(2 * context_dim, context_dim)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.

        Args:
            x: Image tensor (B, 3, H, W)

        Returns:
            Patches (B, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, (
            f"Image size ({H}, {W}) must be divisible by patch_size {p}"
        )

        # Reshape to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)  # (B, num_patches, patch_dim)
        return x

    def _extract_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract bottleneck features from VAE encoder (mid_block output).

        Args:
            x: Image tensor (B, 3, H, W) in [0, 1] range

        Returns:
            Bottleneck features (B, bottleneck_channels, h, w)
        """
        # Normalize to [-1, 1]
        x = 2.0 * x - 1.0

        # Run through encoder up to mid_block (skip conv_norm_out, conv_act, conv_out)
        h = self.encoder.conv_in(x)
        for down_block in self.encoder.down_blocks:
            h = down_block(h)  # Let down_block handle its own components
        h = self.encoder.mid_block(h)

        return h

    def forward(self, s_past: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Encode s_past into context tokens.

        Args:
            s_past: Past frame image (B, 3, H, W) in [0, 1] range

        Returns:
            context: Context tokens (B, N, context_dim) where N = (H/p) * (W/p)
            context_shape: Spatial grid shape (H/p, W/p) for RoPE
        """
        B, C, H, W = s_past.shape
        p = self.patch_size
        context_h, context_w = H // p, W // p

        # Patchify and project
        patches = self._patchify(s_past)  # (B, num_patches, patch_dim)
        patch_tokens = self.patch_proj(patches)  # (B, num_patches, context_dim)

        # Extract and project bottleneck
        bottleneck = self._extract_bottleneck(s_past)  # (B, bottleneck_ch, h, w)
        bottleneck_tokens = bottleneck.flatten(2).transpose(1, 2)  # (B, h*w, ch)
        bottleneck_tokens = self.bottleneck_proj(
            bottleneck_tokens
        )  # (B, h*w, context_dim)

        # Fuse per-position along feature dimension
        fused = torch.cat([patch_tokens, bottleneck_tokens], dim=-1)
        context = self.fusion_proj(fused)  # (B, N, context_dim)

        return context, (context_h, context_w)


class PastConditionedConvDecoder(nn.Module):
    """
    VAE decoder with cross-attention conditioning from s_past.

    Wraps pretrained SD VAE decoder and injects zero-initialized
    CrossAttentionAdapters after the middle block and after each upsampling
    block. Initial behavior is identical to vanilla VAE decoding; conditioning
    effect grows during training.

    Supports unconditioned forward pass (skip context encoder + adapters),
    which reduces to a vanilla VAE decode of z_t.

    Args:
        vae: Pretrained SD VAE (AutoencoderKL)
        context_dim: Dimension for context tokens
        num_heads: Number of attention heads in adapters
        num_groups: GroupNorm groups
        dropout: Dropout rate
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        context_dim: int = 512,
        num_heads: int = 8,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.supports_unconditioned = True
        self.context_dim = context_dim

        # Validate configuration
        patch_size = 8  # Fixed to match VAE downsampling
        _validate_conv_decoder_config(
            vae=vae,
            context_dim=context_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            patch_size=patch_size,
        )

        # Context encoder
        self.context_encoder = ContextEncoder(
            vae=vae,
            context_dim=context_dim,
            patch_size=patch_size,
        )

        # VAE decoder (copy to avoid modifying original)
        self.vae_decoder = deepcopy(vae.decoder)
        self.post_quant_conv = (
            deepcopy(vae.post_quant_conv) if vae.post_quant_conv is not None else None
        )

        # Get channel configuration from VAE
        block_out_channels = list(vae.config.block_out_channels)
        reversed_channels = list(reversed(block_out_channels))

        # Mid adapter (operates at bottleneck resolution)
        self.mid_adapter = CrossAttentionAdapter(
            channels=reversed_channels[0],
            context_dim=context_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
        )

        # One adapter after each up block
        self.up_adapters = nn.ModuleList(
            CrossAttentionAdapter(
                channels=reversed_channels[i],
                context_dim=context_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout=dropout,
            )
            for i in range(len(self.vae_decoder.up_blocks))
        )

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
            s_past: Past frame (B, 3, H, W) in [0, 1] range
            z_past: Past embedding (B, C, h, w); not used
            z_t: Target embedding (B, C, h, w)
            unconditioned: If True, skip context encoder and adapters (vanilla VAE decode)

        Returns:
            Reconstructed target frame (B, 3, H, W) in [0, 1] range
        """
        if not unconditioned:
            context, context_shape = self.context_encoder(s_past)

        # Post-quant conv if present
        z = z_t
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        # VAE decoder forward with adapters
        sample = self.vae_decoder.conv_in(z)

        # Get upscale dtype
        upscale_dtype = next(iter(self.vae_decoder.up_blocks.parameters())).dtype

        # Mid block + adapter
        sample = self.vae_decoder.mid_block(sample)
        if not unconditioned:
            sample = self.mid_adapter(sample, context, context_shape)

        sample = sample.to(upscale_dtype)

        # Up blocks + adapters
        for i, up_block in enumerate(self.vae_decoder.up_blocks):
            sample = up_block(sample)
            if not unconditioned:
                sample = self.up_adapters[i](sample, context, context_shape)

        # Output projection
        sample = self.vae_decoder.conv_norm_out(sample)
        sample = self.vae_decoder.conv_act(sample)
        sample = self.vae_decoder.conv_out(sample)

        # VAE outputs [-1, 1], convert to [0, 1]
        sample = (sample + 1.0) / 2.0

        return sample
