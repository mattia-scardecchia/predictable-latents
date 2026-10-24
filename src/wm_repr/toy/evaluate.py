import os
import torch
import logging
from torch.utils.data import DataLoader

from .data import TransitionDataset
from .models import (
    ContrastiveVAE,
    DynamicsModel,
    Identity,
    Probe,
    ReadoutDecoder,
    _unpack_transition,
)
from .utils import AverageMeter


def load_components(cfg, repr_method: str, device, ckpt_dir: str):
    """Load the frozen encoder, dynamics model, probe and readout decoder for a method.

    decode_fn is the method's OWN decoder and is None for methods that have none, which is what
    gates the own-decoder observation-reconstruction metrics in evaluate(). The readout decoder is
    trained separately for every method (see ReadoutDecoder), so it is never None.
    """
    logging.info(f"Loading {repr_method} components from {ckpt_dir}")
    D = cfg.data.state_dim
    decode_fn = None

    if repr_method == "contrastive_vae":
        hybrid_cfg = cfg.model.contrastive_vae
        z_dim = hybrid_cfg.z_dim
        hybrid_model = ContrastiveVAE(
            D,
            z_dim,
            cfg.data.num_actions,
            hybrid_cfg.enc_widths,
            hybrid_cfg.proj_widths,
            hybrid_cfg.dec_widths,
            temperature=hybrid_cfg.temperature,
            beta=hybrid_cfg.beta,
            temporal_weight=hybrid_cfg.temporal_weight,
            activation=hybrid_cfg.activation,
        )
        hybrid_model.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, f"{repr_method}.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        hybrid_model.to(device).eval()
        encode_fn = hybrid_model.get_encoder_fn()
        decode_fn = hybrid_model.get_decoder_fn()

    elif repr_method == "identity":
        # Not learned, so there is no phase-1 checkpoint to load.
        z_dim = cfg.model.identity.z_dim
        encode_fn = Identity().get_encoder_fn()

    else:
        raise ValueError(
            f"Unknown repr_method: {repr_method}. Must be 'contrastive_vae' or 'identity'"
        )

    dyn = DynamicsModel(
        z_dim,
        cfg.data.num_actions,
        cfg.model.dynamics.dyn_widths,
        repr_method=repr_method,
        activation=cfg.model.dynamics.activation,
    )
    dyn.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "dynamics.pt"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    dyn.to(device).eval()

    probe = Probe(
        z_dim,
        cfg.data.signal_dim,
        repr_method=repr_method,
        widths=cfg.model.probe.probe_widths,
        activation=cfg.model.probe.activation,
    )
    probe.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "probe.pt"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    probe.to(device).eval()

    readout = ReadoutDecoder(
        z_dim,
        D,
        repr_method=repr_method,
        widths=cfg.model.readout.dec_widths,
        activation=cfg.model.readout.activation,
    )
    readout.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "readout_decoder.pt"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    readout.to(device).eval()

    return encode_fn, decode_fn, dyn, probe, readout


@torch.inference_mode()
def evaluate(cfg, repr_method: str, device, ckpt_dir: str, split: str = "test"):
    dd = cfg.data.out_dir
    eval_batch_size = cfg.train.eval_batch_size
    num_workers = cfg.train.num_workers

    loader = DataLoader(
        TransitionDataset(os.path.join(dd, f"{split}.npz")),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    encode, decode, dyn, probe, readout = load_components(
        cfg, repr_method, device, ckpt_dir
    )

    meter = AverageMeter()
    mse = torch.nn.MSELoss()

    for batch in loader:
        obs, obs_next, action = _unpack_transition(batch, device, cfg.data.num_actions)
        signal, signal_next = (
            batch["signal"].to(device),
            batch["signal_next"].to(device),
        )
        batch_size = obs.size(0)

        with torch.no_grad():
            z = encode(obs)
            z_next = encode(obs_next)
            z_next_pred = dyn(z, action)

            signal_pred = probe(z)
            signal_next_pred = probe(z_next_pred)

            mse_signal_next = mse(signal_next_pred, signal_next).item()
            mse_signal = mse(signal_pred, signal).item()
            mse_znext_znextpred = mse(z_next, z_next_pred).item()
            mse_z_znext = mse(z, z_next).item()

            z_expanded1 = z.unsqueeze(1)  # [B, 1, D]
            z_expanded2 = z.unsqueeze(0)  # [1, B, D]
            pairwise_diffs = z_expanded1 - z_expanded2  # [B, B, D]
            pairwise_mse = torch.mean(pairwise_diffs**2, dim=2)  # [B, B]
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            inter_batch_mse = torch.mean(pairwise_mse[mask]).item()

            metrics = {
                "mse_signal_reconstruction": mse_signal,
                "mse_next_signal_prediction": mse_signal_next,
                "mse_consecutive_embeddings": mse_z_znext,
                "mse_unrelated_embeddings": inter_batch_mse,
                "mse_dynamics_prediction": mse_znext_znextpred,
                # How much of the observation the latent still carries, and how much of it
                # survives a step of latent dynamics -- the two axes the signal metrics above are
                # traded against. Read through a decoder trained after the fact on the frozen
                # representation, so these are defined for every method and are not shaped by the
                # objective that produced the representation.
                "mse_readout_reconstruction": mse(readout(z), obs).item(),
                "mse_next_readout_reconstruction": mse(
                    readout(z_next_pred), obs_next
                ).item(),
            }
            if decode is not None:
                # The same two quantities read through the method's own decoder, which only some
                # methods have and which its objective trained (or, at temporal_weight=1, did not).
                metrics["mse_observation_reconstruction"] = mse(decode(z), obs).item()
                metrics["mse_next_observation_reconstruction"] = mse(
                    decode(z_next_pred), obs_next
                ).item()
            meter.update(metrics, {k: batch_size for k in metrics})

    final_metrics = meter.avg
    final_metrics.update(
        {k.replace("mse", "rmse"): v**0.5 for k, v in final_metrics.items()}
    )
    for metric_name, value in final_metrics.items():
        logging.info(f"[EVAL-{repr_method}] {metric_name} = {value:.4f}")
    return final_metrics
