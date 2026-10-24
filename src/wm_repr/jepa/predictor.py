from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from wm_repr.wm.model import RotaryType, apply_rope_nd


class ActionEncoder(nn.Module):
    """
    Action tokens with an MLP projection and final LayerNorm.

    Interface::

        forward(actions) -> (tokens, cls, key_padding_mask)
    """

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # MLP projection: action_dim -> d_model -> d_model
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embed = nn.Embedding(max_len, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        seq: torch.Tensor = actions["seq"]
        mask: Optional[torch.Tensor] = actions["mask"]

        B, L, _ = seq.shape
        if L > self.max_len:
            raise ValueError(
                f"Action sequence length {L} > max_len {self.max_len} "
                "in ActionEncoder"
            )

        # MLP projection
        x = self.action_proj(seq)  # (B, L, D)

        # Positional embeddings
        pos_ids = torch.arange(L, device=seq.device)
        x = x + self.pos_embed(pos_ids)[None, :, :]  # (B, L, D)

        # Prepend CLS
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, D)

        # Key-padding mask: True = PAD
        encoder_kpm = None
        token_pad_mask = None
        if mask is not None:
            valid_mask = mask > 0 if mask.dtype != torch.bool else mask
            token_pad_mask = ~valid_mask  # (B, L)
            encoder_kpm = torch.cat(
                [
                    torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
                    token_pad_mask,
                ],
                dim=1,
            )  # (B, L+1)

        # Transformer encoder
        h = x
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=encoder_kpm)

        # Final norm
        h = self.final_norm(h)

        tokens_out = h[:, 1:]  # (B, L, D)
        cls_out = h[:, 0]  # (B, D)

        key_padding_mask_tokens = token_pad_mask if mask is not None else None
        return tokens_out, cls_out, key_padding_mask_tokens


class ConvPredictorBlock(nn.Module):
    """
    Hybrid conv + attention block operating in (B, D, H, W) image format.

    Sub-layers (all pre-norm with residual connections):
      1. Depthwise separable conv  — local spatial + channel mixing
      2. Self-attention             — global spatial mixing (2D RoPE)
      3. Cross-attention            — action conditioning (cross_attn mode only)
      4. FFN via 1×1 convolutions   — channel mixing

    Conditioning modes:
      - cross_attn: sub-layer 3 present
      - none: unconditioned spatial processing
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conditioning: str,
        dropout: float = 0.0,
        mlp_mult: float = 4.0,
        conv_kernel_size: int = 3,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        assert conditioning in ("cross_attn", "none")
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.conditioning = conditioning

        # 1. Depthwise separable conv (always unconditioned)
        self.conv_norm = nn.GroupNorm(norm_num_groups, d_model)
        self.dwconv = nn.Conv2d(
            d_model, d_model, conv_kernel_size,
            padding=conv_kernel_size // 2, groups=d_model,
        )
        self.pwconv = nn.Conv2d(d_model, d_model, 1)

        # 2. Self-attention (custom QKV projections for RoPE)
        self.attn_norm = nn.GroupNorm(norm_num_groups, d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_out_proj = nn.Linear(d_model, d_model)

        # 3. Action conditioning
        if conditioning == "cross_attn":
            self.cross_norm = nn.GroupNorm(norm_num_groups, d_model)
            self.cross_q_proj = nn.Linear(d_model, d_model)
            self.cross_kv_proj = nn.Linear(d_model, 2 * d_model)
            self.cross_out_proj = nn.Linear(d_model, d_model)

        # 4. FFN (1×1 convolutions)
        self.ffn_norm = nn.GroupNorm(norm_num_groups, d_model)
        inner_dim = int(mlp_mult * d_model)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, d_model, 1),
            nn.Dropout(dropout),
        )

    # -- helpers --------------------------------------------------------------

    def _self_attention(
        self, h: torch.Tensor, H: int, W: int,
    ) -> torch.Tensor:
        """h: (B, D, H, W) → (B, D, H, W)"""
        B, D = h.shape[:2]
        N = H * W

        # Flatten to (B, N, D) and project
        h = h.flatten(2).transpose(1, 2)            # (B, N, D)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)

        # Multi-head reshape: (B, heads, N, head_dim)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Reshape for 2-D positions: (B, heads, H, W, head_dim)
        q = q.view(B, self.n_heads, H, W, self.head_dim)
        k = k.view(B, self.n_heads, H, W, self.head_dim)
        q, k = apply_rope_nd(q, k, (H, W), RotaryType.PIXEL)
        q = q.reshape(B, self.n_heads, N, self.head_dim)
        k = k.reshape(B, self.n_heads, N, self.head_dim)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, D)     # (B, N, D)
        h = self.attn_out_proj(h)
        return h.transpose(1, 2).reshape(B, D, H, W)

    def _cross_attention(
        self,
        h: torch.Tensor,           # (B, D, H, W)
        action_tokens: torch.Tensor,  # (B, L, D)
        action_kpm: Optional[torch.Tensor],  # (B, L) bool, True = PAD
        action_cls: Optional[torch.Tensor] = None,  # (B, D)
    ) -> torch.Tensor:
        B, D, H, W = h.shape
        N = H * W

        # Prepend CLS to K/V sequence so spatial queries can attend to
        # both per-timestep action tokens and the global summary.
        if action_cls is not None:
            cls_kv = action_cls.unsqueeze(1)  # (B, 1, D)
            action_tokens = torch.cat([cls_kv, action_tokens], dim=1)  # (B, L+1, D)
            if action_kpm is not None:
                cls_valid = torch.zeros(
                    B, 1, dtype=torch.bool, device=h.device,
                )
                action_kpm = torch.cat([cls_valid, action_kpm], dim=1)  # (B, L+1)

        L = action_tokens.shape[1]

        h_flat = h.flatten(2).transpose(1, 2)       # (B, N, D)
        q = self.cross_q_proj(h_flat)
        kv = self.cross_kv_proj(action_tokens)       # (B, L, 2D)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if action_kpm is not None:
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(action_kpm[:, None, None, :], float("-inf"))

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.cross_out_proj(h)
        return h.transpose(1, 2).reshape(B, D, H, W)

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                               # (B, D, H, W)
        action_tokens: Optional[torch.Tensor] = None,   # (B, L, D)
        action_cls: Optional[torch.Tensor] = None,      # (B, D)
        action_kpm: Optional[torch.Tensor] = None,      # (B, L) bool
    ) -> torch.Tensor:
        B, D, H, W = x.shape

        # 1. Depthwise separable conv
        h = self.conv_norm(x)
        h = self.pwconv(F.silu(self.dwconv(h)))
        x = x + h

        # 2. Self-attention
        h = self.attn_norm(x)
        x = x + self._self_attention(h, H, W)

        # 3. Cross-attention (cross_attn mode only)
        if self.conditioning == "cross_attn":
            h = self.cross_norm(x)
            x = x + self._cross_attention(h, action_tokens, action_kpm, action_cls)

        # 4. FFN
        h = self.ffn_norm(x)
        x = x + self.ffn(h)

        return x


class InverseDynamicsHead(nn.Module):
    """Predict six standardized motion components and a gripper logit from a latent pair."""

    def __init__(self, latent_channels: int, cfg):
        super().__init__()
        self.latent_channels = latent_channels
        self.in_proj = nn.Conv2d(2 * latent_channels, cfg.d_model, 1)
        self.blocks = nn.ModuleList([
            ConvPredictorBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                conditioning="none",
                dropout=cfg.dropout,
                mlp_mult=cfg.mlp_mult,
                conv_kernel_size=cfg.conv_kernel_size,
                norm_num_groups=cfg.norm_num_groups,
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.GroupNorm(cfg.norm_num_groups, cfg.d_model)
        self.readout = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 7),
        )

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if current.ndim != 4 or current.shape != future.shape:
            raise ValueError("Inverse dynamics requires two identically shaped BCHW latents")
        if current.shape[1] != self.latent_channels:
            raise ValueError("Latent channel count does not match the inverse head")
        x = self.in_proj(torch.cat((current, future), dim=1))
        for block in self.blocks:
            x = block(x)
        return self.readout(self.final_norm(x).mean(dim=(-2, -1)))


class ConvPredictor(nn.Module):
    """
    Hybrid conv + attention predictor operating in (B, D, H, W) image format.

    Interface:
        forward(z: (B, C, H, W), actions: dict) → z_pred: (B, C, H, W)

    Each block combines depthwise separable convolution, GroupNorm, self-attention
    with 2-D rotary position embeddings, and cross-attention to the action tokens.
    """

    def __init__(
        self,
        latent_channels: int,
        action_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.0,
        max_actions: int = 64,
        mlp_mult: float = 4.0,
        action_encoder_layers: int = 2,
        conv_kernel_size: int = 3,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        # Input / output projections
        self.in_proj = nn.Conv2d(latent_channels, d_model, 1)
        self.out_proj = nn.Conv2d(d_model, latent_channels, 1)

        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            d_model=d_model,
            num_layers=action_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_actions,
        )

        # Main body
        self.blocks = nn.ModuleList(
            [
                ConvPredictorBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    conditioning="cross_attn",
                    dropout=dropout,
                    mlp_mult=mlp_mult,
                    conv_kernel_size=conv_kernel_size,
                    norm_num_groups=norm_num_groups,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.GroupNorm(norm_num_groups, d_model)

    def forward(
        self,
        z: torch.Tensor,  # (B, C, H, W)
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 1. Encode actions
        action_tokens, action_cls, action_kpm = self.action_encoder(actions)

        # 2. Project to d_model
        x = self.in_proj(z)  # (B, D, H, W)

        # 3. Blocks
        for blk in self.blocks:
            x = blk(
                x,
                action_tokens=action_tokens,
                action_cls=action_cls,
                action_kpm=action_kpm,
            )

        # 4. Post-process
        x = self.final_norm(x)
        return self.out_proj(x)


Predictor = ConvPredictor
