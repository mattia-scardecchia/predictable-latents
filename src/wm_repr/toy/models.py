from typing import Dict, Any, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from .utils import make_mlp


def _unpack_transition(
    batch: Dict[str, torch.Tensor], device: torch.device, num_actions: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to prepare batch data. Works with data from TransitionDataset."""
    observation = batch["observation"].to(device)
    B, _ = observation.shape
    action_onehot = (
        torch.nn.functional.one_hot(
            batch["action"].reshape((B,)).long(), num_classes=num_actions
        )
        .float()
        .to(device)
    )
    observation_next = batch["observation_next"].to(device)
    return observation, observation_next, action_onehot


class TrainableModel(nn.Module, ABC):
    """Base class for models that can be trained with the generic trainer"""

    @abstractmethod
    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one training step
        Returns: (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one validation step
        Returns: (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        """Return optimizer configuration for this model"""
        pass

    def save_checkpoint(self, ckpt_dir: str, cfg: Any) -> Dict[str, Any]:
        """
        Save the state dict and config to ckpt_dir/{id}.pt and return the path
        Returns: {"paths": list}
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        filename = f"{self.id}.pt"
        path = os.path.join(ckpt_dir, filename)
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, path)

        return {"paths": [path]}

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the model identifier for logging and file naming"""
        pass


def beta_vae_loss(x_hat, x, mu, logvar, beta: float):
    """Beta-VAE loss function.

    `recon` is mean-reduced (divides by x_dim), `kl` is summed over latent dims and
    then averaged over the batch, i.e. the true per-sample KL. Keeping the KL a *sum*
    makes the effective regularization strength (beta * x_dim in textbook beta-VAE
    terms) independent of z_dim, so z_dim can be swept without silently rescaling beta.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + beta * kl, {"recon": recon.item(), "kl": kl.item()}


class VAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        enc_widths,
        dec_widths,
        activation: str = "relu",
    ):
        super().__init__()
        # The trunk ends on an activation, so the mu/logvar heads are the only Linear after its
        # last hidden layer. A trunk ending on a bare Linear would stack two Linears there: an
        # over-parameterized factorization of one map, which is expressively a no-op but not an
        # optimization no-op.
        self.enc = make_mlp([x_dim] + list(enc_widths), activation, out_act=activation)
        hid = enc_widths[-1]
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)
        self.dec = make_mlp([z_dim] + list(dec_widths) + [x_dim], activation)

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def decode(self, z):
        return self.dec(z)

    def get_encoder_fn(self):
        """Return a function that encodes input states to latent representations"""

        def encode_fn(x):
            mu, logvar, z = self.encode(x)
            return mu

        return encode_fn

    def get_decoder_fn(self):
        """Return a function that maps latent representations back to observations"""
        return self.decode


class Identity(nn.Module):
    """The "identity" method: the entire observation, with no representation training."""

    def forward(self, x):
        return x

    def get_encoder_fn(self):
        return self.forward


class ContrastiveVAE(TrainableModel):
    """Temporal contrastive and beta-VAE objectives sharing one encoder.

        loss = w * contrastive + (1 - w) * (recon + beta * kl)

    with w = temporal_weight, so w=1 is exactly the contrastive objective on this architecture and
    w=0 is exactly the VAE's; sweeping w traces the frontier between them.

    The encoder has a shared trunk and separate mu/logvar heads. Normalization is
    applied only to the contrastive queries and keys, leaving the latent distribution
    available for KL regularization. The key network `g` is an MLP.

    Reconstruction and KL apply to the query latent alone -- dec(z) against the query observation,
    KL on the (mu, logvar) that produced it. The next observation is never encoded by the VAE, so
    the reconstruction term costs no forward pass beyond the ones the contrastive term already
    needs. Training samples z from the posterior; only mu goes downstream.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        num_actions: int,
        enc_widths,
        proj_widths,
        dec_widths,
        temperature: float,
        beta: float,
        temporal_weight: float,
        activation: str,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.temperature = temperature
        self.beta = beta
        self.temporal_weight = temporal_weight
        # Layer-normalized queries and keys have squared norm approximately z_dim;
        # dividing by z_dim keeps the logit scale comparable across latent sizes.
        self.logit_denom = temperature * z_dim
        self.vae = VAE(x_dim, z_dim, enc_widths, dec_widths, activation)
        self.g = make_mlp(
            [x_dim + num_actions] + list(proj_widths) + [z_dim], activation
        )

    def _normalize(self, v):
        """Normalize logits' inputs to zero mean and unit variance across coordinates."""
        return F.layer_norm(v, (self.z_dim,))

    def contrastive_vae_loss(self, obs, obs_next, action, device):
        obs_query = obs
        mu, logvar, z = self.vae.encode(obs_query)
        keys = self.g(torch.cat([obs_next, action], dim=-1))  # B, D
        diff = self._normalize(z)[:, None, :] - self._normalize(keys)[None, :, :]
        logits = -torch.sum(diff**2, dim=-1) / self.logit_denom  # B, B
        target = torch.arange(obs.size(0), device=device)

        contrastive = F.cross_entropy(logits, target)
        reconstruction, parts = beta_vae_loss(
            self.vae.decode(z), obs_query, mu, logvar, self.beta
        )
        loss = (
            self.temporal_weight * contrastive
            + (1 - self.temporal_weight) * reconstruction
        )

        metrics = {
            "loss": loss.item(),
            "contrastive": contrastive.item(),
            "recon": parts["recon"],
            "kl": parts["kl"],
            # At w=1 the KL carries no weight, leaving logvar unregularized: sigma then reports
            # whether the encoder has drifted to an effectively deterministic one.
            "sigma": torch.exp(0.5 * logvar).mean().item(),
        }
        return loss, metrics

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)
        loss, parts = self.contrastive_vae_loss(obs, obs_next, action, device)
        return loss, {f"train_{name}": v for name, v in parts.items()}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)
        loss, parts = self.contrastive_vae_loss(obs, obs_next, action, device)
        return loss, {f"val_{name}": v for name, v in parts.items()}

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.contrastive.lr,
            "weight_decay": cfg.train.contrastive.weight_decay,
            "gradient_clip": cfg.train.contrastive.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "contrastive_vae"

    def get_encoder_fn(self):
        """Return a function that encodes input states to latent representations: the posterior
        mean, which downstream dynamics and probe training use."""
        return self.vae.get_encoder_fn()

    def get_decoder_fn(self):
        """Return a function that maps latent representations back to observations"""
        return self.vae.get_decoder_fn()


class DynamicsModel(TrainableModel):
    def __init__(
        self,
        z_dim: int,
        num_actions: int,
        widths,
        repr_method: str,
        activation: str = "relu",
    ):
        super().__init__()
        self.repr_method = repr_method
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.net = make_mlp([z_dim + num_actions] + list(widths) + [z_dim], activation)
        self.mse = nn.MSELoss()
        self.encoder_fn = None  # Will be set during training setup

    def forward(self, z, a_onehot):
        return self.net(torch.cat([z, a_onehot], dim=-1))

    def set_encoder(self, encoder_fn):
        """Set the encoder function to use for getting latent representations"""
        self.encoder_fn = encoder_fn

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore
            z_next_true = self.encoder_fn(obs_next)  # type: ignore

        z_next_pred = self(z, action)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore
            z_next_true = self.encoder_fn(obs_next)  # type: ignore

        z_next_pred = self(z, action)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.dynamics.lr,
            "weight_decay": cfg.train.dynamics.weight_decay,
            "gradient_clip": cfg.train.dynamics.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "dynamics"


class Probe(TrainableModel):
    def __init__(
        self,
        z_dim: int,
        signal_dim: int,
        repr_method: str,
        widths=(64, 64),
        activation: str = "relu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.signal_dim = signal_dim
        self.repr_method = repr_method
        self.net = make_mlp([z_dim] + list(widths) + [signal_dim], activation)
        self.mse = nn.MSELoss()
        self.encoder_fn = None  # Will be set during training setup

    def forward(self, z):
        return self.net(z)

    def set_encoder(self, encoder_fn):
        """Set the encoder function to use for getting latent representations"""
        self.encoder_fn = encoder_fn

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)
        signal_true = batch["signal"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        signal_pred = self(z)
        loss = self.mse(signal_pred, signal_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)
        signal_true = batch["signal"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        signal_pred = self(z)
        loss = self.mse(signal_pred, signal_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.probe.lr,
            "weight_decay": cfg.train.probe.weight_decay,
            "gradient_clip": cfg.train.probe.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "probe"


class ReadoutDecoder(TrainableModel):
    """Probe onto the observation instead of the signal: how much of x the latent still carries.

    A method's OWN decoder cannot serve as that axis. Only contrastive_vae has one, and at
    temporal_weight=1 it receives zero gradient, so its reconstruction error is meaningless at
    exactly the endpoint the frontier most needs to characterize. Training a decoder afterwards on
    a frozen encoder, with the latent detached, gives a reconstruction number for every method --
    identity included -- that measures information the representation retained without having
    shaped it.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        repr_method: str,
        widths=(256,),
        activation: str = "relu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.repr_method = repr_method
        self.net = make_mlp([z_dim] + list(widths) + [x_dim], activation)
        self.mse = nn.MSELoss()
        self.encoder_fn = None  # Will be set during training setup

    def forward(self, z):
        return self.net(z)

    def set_encoder(self, encoder_fn):
        """Set the encoder function to use for getting latent representations"""
        self.encoder_fn = encoder_fn

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        obs_pred = self(z)
        loss = self.mse(obs_pred, obs)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        obs_pred = self(z)
        loss = self.mse(obs_pred, obs)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.readout_decoder.lr,
            "weight_decay": cfg.train.readout_decoder.weight_decay,
            "gradient_clip": cfg.train.readout_decoder.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "readout_decoder"
