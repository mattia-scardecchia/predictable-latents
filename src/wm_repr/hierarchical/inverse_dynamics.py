"""Single-action inverse supervision with the matched past-conditioned reconstruction path."""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from wm_repr.hierarchical.jepa import LitHierarchicalJEPA
from wm_repr.jepa.predictor import InverseDynamicsHead
from wm_repr.jepa.registry import register


def inverse_action_loss(prediction, actions, mean, std):
    """Mean over the seven action components: squared error of the six standardized motion
    components and binary cross-entropy of the gripper."""
    seq, mask = actions["seq"], actions["mask"]
    if seq.ndim != 3 or seq.shape[1] != 1 or seq.shape[-1] < 7:
        raise ValueError("Inverse dynamics requires exactly one action with at least seven components")
    if mask.shape != seq.shape[:2] or not torch.all(mask):
        raise ValueError("Every inverse-dynamics pair must contain one valid action")
    if prediction.shape != (seq.shape[0], 7):
        raise ValueError("Inverse head must output six motion values and one gripper logit")
    target = seq[:, 0, :7].float()
    if not torch.isfinite(target).all():
        raise ValueError("Inverse action targets must be finite")
    if not torch.all((target[:, 6] == -1) | (target[:, 6] == 1)):
        raise ValueError("Bridge gripper targets must be -1 or +1")
    standardized = (target[:, :6] - mean) / std
    errors = prediction[:, :6].float() - standardized
    gripper = (target[:, 6] + 1) / 2
    bce = F.binary_cross_entropy_with_logits(prediction[:, 6].float(), gripper, reduction="none")
    return (errors.square().sum(dim=-1) + bce).mean() / 7


@register
class LitHierarchicalInverseDynamics(LitHierarchicalJEPA):
    """Predict the action between two frames from both latents; reconstruct as LitHierarchicalJEPA."""

    def __init__(self, cfg, **kwargs):
        cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        weight = float(cfg.model.encoder.inverse_weight)
        if not 0 <= weight <= 1:
            raise ValueError("inverse_weight must lie in [0, 1]")
        if cfg.model.encoder.kl_weight:
            raise ValueError("Hierarchical inverse dynamics requires zero KL")
        if cfg.model.encoder.future_pred_weight != 0:
            raise ValueError("The forward latent-prediction loss must be disabled")
        mean = torch.tensor(list(cfg.model.inverse.action_mean), dtype=torch.float32)
        std = torch.tensor(list(cfg.model.inverse.action_std), dtype=torch.float32)
        if mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all():
            raise ValueError("Inverse dynamics requires six finite training-action means and scales")
        if not torch.isfinite(std).all() or not torch.all(std > 0):
            raise ValueError("Invalid training-action scales")
        if cfg.model.predictor.type != "conv":
            raise ValueError("Baseline predictor construction must precede decoder initialization")
        super().__init__(cfg, **kwargs)
        # Preserve the baseline decoder initialization and the RNG seen by downstream modules.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(int(cfg.seed)).get_state())
            self.predictor = InverseDynamicsHead(self.encoder.latent_channels, cfg.model.predictor)
        self.auxiliary_weight = weight
        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)

    def _auxiliary_loss(self, z, z_future, batch, split):
        prediction = self.predictor(z, z_future)
        loss = inverse_action_loss(prediction, batch["actions"], self.action_mean, self.action_std)
        return loss, {f"{split}/inverse_loss": loss}
