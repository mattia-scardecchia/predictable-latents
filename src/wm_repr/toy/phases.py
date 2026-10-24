import logging
import os
from omegaconf import DictConfig
import torch

from .data import data_loaders
from .models import (
    ContrastiveVAE,
    DynamicsModel,
    Identity,
    Probe,
    ReadoutDecoder,
)
from .trainer import GenericTrainer


def create_contrastive_vae(cfg: DictConfig, device: torch.device) -> ContrastiveVAE:
    """Create ContrastiveVAE model from config"""
    D = cfg.data.state_dim
    hybrid_cfg = cfg.model.contrastive_vae
    return ContrastiveVAE(
        D,
        hybrid_cfg.z_dim,
        cfg.data.num_actions,
        hybrid_cfg.enc_widths,
        hybrid_cfg.proj_widths,
        hybrid_cfg.dec_widths,
        temperature=hybrid_cfg.temperature,
        beta=hybrid_cfg.beta,
        temporal_weight=hybrid_cfg.temporal_weight,
        activation=hybrid_cfg.activation,
    ).to(device)


def create_dynamics(
    cfg: DictConfig, device: torch.device, z_space: str
) -> DynamicsModel:
    """Create Dynamics model from config"""
    z_dim = cfg.model.repr.z_dim
    dynamics_cfg = cfg.model.dynamics
    return DynamicsModel(
        z_dim,
        cfg.data.num_actions,
        dynamics_cfg.dyn_widths,
        repr_method=z_space,
        activation=dynamics_cfg.activation,
    ).to(device)


def create_probe(cfg: DictConfig, device: torch.device, repr_method: str) -> Probe:
    """Create Probe model from config"""
    z_dim = cfg.model.repr.z_dim
    probe_cfg = cfg.model.probe
    return Probe(
        z_dim,
        cfg.data.signal_dim,
        repr_method=repr_method,
        widths=probe_cfg.probe_widths,
        activation=probe_cfg.activation,
    ).to(device)


def create_readout_decoder(
    cfg: DictConfig, device: torch.device, repr_method: str
) -> ReadoutDecoder:
    """Create ReadoutDecoder model from config"""
    z_dim = cfg.model.repr.z_dim
    readout_cfg = cfg.model.readout
    return ReadoutDecoder(
        z_dim,
        cfg.data.state_dim,
        repr_method=repr_method,
        widths=readout_cfg.dec_widths,
        activation=readout_cfg.activation,
    ).to(device)


def load_encoder(
    cfg: DictConfig,
    device: torch.device,
    repr_method: str,
    freeze: bool,
    ckpt_path: str,
):
    """Load pre-trained encoder and return model + encoding function"""
    if repr_method == "identity":
        return Identity()  # not learned: no phase-1 checkpoint

    model = create_contrastive_vae(cfg, device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    )
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model


def train_phase1_contrastive(cfg: DictConfig, device: torch.device, model):
    """Contrastive training using GenericTrainer.

    Reads single-transition batches and uses the optimizer settings in cfg.train.contrastive.
    """
    train_loader, val_loader = data_loaders(cfg, cfg.train.contrastive.batch_size)
    trainer = GenericTrainer(cfg, device)

    return trainer.train(
        model,
        train_loader,
        val_loader,
        cfg.train.steps_phase1,
    )


def train_phase2_dynamics(
    cfg: DictConfig, device: torch.device, repr_method: str, ckpt_dir: str
):
    """Dynamics training using GenericTrainer"""
    train_loader, val_loader = data_loaders(cfg, cfg.train.dynamics.batch_size)
    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{repr_method}.pt",
    )
    logging.info(f"Loading encoder from {ckpt_path} for dynamics training")
    encoder_model = load_encoder(
        cfg,
        device,
        repr_method,
        freeze=True,
        ckpt_path=ckpt_path,
    )
    encoder_fn = encoder_model.get_encoder_fn()

    model = create_dynamics(cfg, device, repr_method)
    model.set_encoder(encoder_fn)
    trainer = GenericTrainer(cfg, device)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg.train.steps_phase2
    )

    out_path = os.path.join(cfg.train.ckpt_dir, "dynamics.pt")
    return out_path, final_metrics


def train_probes(
    cfg: DictConfig, device: torch.device, repr_method: str, ckpt_dir: str
):
    """Probe training using GenericTrainer"""
    train_loader, val_loader = data_loaders(cfg, cfg.train.probe.batch_size)

    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{repr_method}.pt",
    )
    logging.info(f"Loading encoder from {ckpt_path} for probe training")
    encoder_model = load_encoder(
        cfg,
        device,
        repr_method,
        freeze=True,
        ckpt_path=ckpt_path,
    )
    encoder_fn = encoder_model.get_encoder_fn()

    model = create_probe(cfg, device, repr_method)
    model.set_encoder(encoder_fn)
    trainer = GenericTrainer(cfg, device)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg.train.steps_probe
    )

    out_path = os.path.join(cfg.train.ckpt_dir, "probe.pt")
    return out_path, final_metrics


def train_readout_decoder(
    cfg: DictConfig, device: torch.device, repr_method: str, ckpt_dir: str
):
    """Readout decoder training using GenericTrainer.

    Same shape as train_probes -- frozen encoder, detached latents -- with the observation as the
    target instead of the signal.
    """
    train_loader, val_loader = data_loaders(cfg, cfg.train.readout_decoder.batch_size)

    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{repr_method}.pt",
    )
    logging.info(f"Loading encoder from {ckpt_path} for readout decoder training")
    encoder_model = load_encoder(
        cfg,
        device,
        repr_method,
        freeze=True,
        ckpt_path=ckpt_path,
    )
    encoder_fn = encoder_model.get_encoder_fn()

    model = create_readout_decoder(cfg, device, repr_method)
    model.set_encoder(encoder_fn)
    trainer = GenericTrainer(cfg, device)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg.train.steps_readout
    )

    out_path = os.path.join(cfg.train.ckpt_dir, "readout_decoder.pt")
    return out_path, final_metrics
