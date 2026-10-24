"""WorldGym transformer with action dropout and DDT decoding."""

import einops
from torch import nn
from torch.nn import functional as F

from wm_repr.third_party.worldgym import model as upstream

AttentionType = upstream.AttentionType
RotaryType = upstream.RotaryType
rope_nd = upstream.rope_nd
rope_mix = upstream.rope_mix
apply_rope_nd = upstream.apply_rope_nd


def _spatial_condition(block, tokens, condition):
    """Apply a decoder block's modulation separately at each spatial position."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
        block.adaLN_modulation(condition).chunk(6, dim=-1)
    )
    tokens = tokens + block.attn(block.norm1(tokens) * (1 + scale_a) + shift_a) * gate_a
    return tokens + block.ffwd(block.norm2(tokens) * (1 + scale_f) + shift_f) * gate_f


class DecoderBlock(upstream.Block):
    def forward(self, x, c):
        return _spatial_condition(self.t_block, _spatial_condition(self.s_block, x, c), c)


class DecoderFinalLayer(upstream.FinalLayer):
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class DiT(upstream.DiT):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        action_dim: int = 0,
        max_frames: int = 16,
        rope_config: dict[AttentionType, RotaryType] | None = None,
        cond_dropout: float = 0.1,
        decoder_dim: int | None = None,
        decoder_depth: int = 0,
        decoder_heads: int | None = None,
    ):
        nn.Module.__init__(self)
        self.in_channels, self.patch_size = in_channels, patch_size
        self.action_dim, self.max_frames = action_dim, max_frames
        self.action_dropout_prob = cond_dropout
        self.decoder_dim = decoder_dim if decoder_dim and decoder_depth > 0 else None
        self._encoder_depth = num_layers - decoder_depth if self.decoder_dim else num_layers
        if self._encoder_depth <= 0:
            raise ValueError("decoder_depth must be less than num_layers")

        # Registration order determines checkpoint keys and seeded initialization.
        self.timestep_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_embedder = nn.Linear(action_dim, dim)
        if self.decoder_dim:
            self.s_proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
            self.x_proj = nn.Conv2d(in_channels, self.decoder_dim, patch_size, stride=patch_size)
            self.s_projector = nn.Linear(dim, self.decoder_dim)
            self.blocks = nn.ModuleList(
                [upstream.Block(dim, num_heads, rope_config) for _ in range(self._encoder_depth)]
                + [DecoderBlock(self.decoder_dim, decoder_heads or num_heads, rope_config)
                   for _ in range(decoder_depth)]
            )
            self.final_layer = DecoderFinalLayer(self.decoder_dim, patch_size, in_channels)
        else:
            self.x_proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
            self.blocks = nn.ModuleList(
                [upstream.Block(dim, num_heads, rope_config) for _ in range(num_layers)]
            )
            self.final_layer = upstream.FinalLayer(dim, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        if self.decoder_dim is None:
            return super().initialize_weights()
        # DDT has two patch projections and a projection between encoder and decoder.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for projection in (self.s_proj, self.x_proj):
            nn.init.xavier_uniform_(projection.weight.flatten(1))
            nn.init.zeros_(projection.bias)
        for layer in (self.timestep_mlp[0], self.timestep_mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        zero_layers = [self.final_layer.adaLN_modulation[-1], self.final_layer.linear]
        for block in self.blocks:
            zero_layers.extend((block.s_block.adaLN_modulation[-1], block.t_block.adaLN_modulation[-1]))
        for layer in zero_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, action):
        frames = x.shape[1]
        # Upstream owns action masking, timestep embedding, and the null-action convention.
        condition = super().get_cond(t, action)
        if self.decoder_dim is None:
            tokens = self.patchify(x)
            for block in self.blocks:
                tokens = block(tokens, condition)
            tokens = self.final_layer(tokens, condition)
        else:
            tokens = self.s_proj(einops.rearrange(x, "b t h w c -> (b t) c h w"))
            encoded = einops.rearrange(tokens, "(b t) d h w -> b t h w d", t=frames)
            for block in self.blocks[:self._encoder_depth]:
                encoded = block(encoded, condition)
            # DDT adds the timestep without the action before its width projection.
            time = self.timestep_mlp(self.timestep_embedding(t.reshape(-1)))
            time = time.reshape(*t.shape, -1)[:, :, None, None, :]
            condition = self.s_projector(F.silu(time + encoded))
            tokens = self.patchify(x)
            for block in self.blocks[self._encoder_depth:]:
                tokens = block(tokens, condition)
            tokens = self.final_layer(tokens, condition)
        pixels = self.unpatchify(einops.rearrange(tokens, "b t h w d -> (b t) h w d"))
        return einops.rearrange(pixels, "(b t) h w c -> b t h w c", t=frames)
