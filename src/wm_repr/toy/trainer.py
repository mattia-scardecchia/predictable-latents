"""Epoch-based AdamW loop used by the synthetic study."""
import torch
from .utils import AverageMeter


class GenericTrainer:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

    def train(self, model, train_loader, val_loader, num_steps):
        config = model.get_optimizer_config(self.cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                      weight_decay=config['weight_decay'])
        clip = config['gradient_clip']
        # The training budget counts complete epochs.
        epochs = num_steps // len(train_loader)
        if epochs == 0:
            raise ValueError(f'Request at least {len(train_loader)} steps for one complete epoch')
        for epoch in range(epochs):
            model.train()
            meter = AverageMeter()
            for batch in train_loader:
                loss, metrics = model.training_step(batch, self.device)
                optimizer.zero_grad()
                loss.backward()
                if clip.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip.max_norm, clip.norm_type)
                optimizer.step()
                meter.update(metrics, {key: batch['state'].size(0) for key in metrics})
            train_metrics = meter.avg
            model.eval()
            meter = AverageMeter()
            with torch.inference_mode():
                for batch in val_loader:
                    _, metrics = model.validation_step(batch, self.device)
                    meter.update(metrics, {key: batch['state'].size(0) for key in metrics})
            val_metrics = meter.avg
        model.save_checkpoint(self.cfg.train.ckpt_dir, self.cfg)
        result = {f'final_{model.id}_{split}_{k}': v
                  for split, metrics in [('train', train_metrics), ('val', val_metrics)]
                  for k, v in metrics.items()}
        result.update(requested_steps=num_steps, actual_steps=epochs * len(train_loader))
        return result
