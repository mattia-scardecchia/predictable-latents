# Adapted from RAE (https://github.com/bytetriper/RAE), Copyright (c) 2025 Boyang Zheng, MIT License.
"""DINOv2 and the pretrained RAE decoder (derived from bytetriper/RAE)."""
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoImageProcessor
from .encoders.dinov2 import Dinov2withNorm
from .decoders.decoder import GeneralDecoder

ARCHS = {"dinov2": Dinov2withNorm, "Dinov2withNorm": Dinov2withNorm}


class RAE(nn.Module):
    """Container for the RAE encoder, decoder and normalization statistics.

    EncoderFromRAE and PastConditionedViTDecoder take their components from it.
    """

    def __init__(
        self,
        encoder_cls: str,
        encoder_config_path: str,
        encoder_input_size: int,
        encoder_params: dict,
        decoder_config_path: str,
        encoder_patch_size: int,
        decoder_patch_size: int,
        eps: float = 1e-5,
        normalization_stat_path: Optional[str] = None,
        pretrained_decoder_path: Optional[str] = None,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: nn.Module = encoder_cls(**encoder_params)
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = encoder_patch_size
        self.latent_dim = self.encoder.hidden_size
        self.base_patches = (
            self.encoder_input_size // self.encoder_patch_size
        ) ** 2  # number of patches of the latent
        assert self.encoder.patch_size == encoder_patch_size, (
            f"encoder patch size from config ({encoder_patch_size}) does not match the patch size of the {self.encoder.__class__.__name__} encoder, which is {self.encoder.patch_size}"
        )
        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        )
        # decoder (patch size may differ from encoder, e.g. encoder=14, decoder=16)
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = (
            self.encoder.hidden_size
        )  # NOTE: this is a modelling choice!
        decoder_config.patch_size = decoder_patch_size
        num_patches_per_side = self.encoder_input_size // self.encoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * num_patches_per_side)
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(
                    f"Missing keys when loading pretrained decoder: {keys.missing_keys}"
                )
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

