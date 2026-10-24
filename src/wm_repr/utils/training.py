"""Training helpers shared by the paper implementations."""
import importlib
import logging
from collections import OrderedDict

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _worker_init_fn(worker_id: int):
    """Re-seed per-instance numpy RNGs so forked workers don't share state."""
    import numpy as np

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, "rng"):
        dataset.rng = np.random.default_rng(worker_info.seed % (2**32))


def prepare_dataloaders(cfg: DictConfig):
    data_cfg: dict = OmegaConf.to_container(cfg.data, resolve=True)  # type: ignore
    data_cfg["params"]["split"] = "train"
    train_dataset = instantiate_from_config(data_cfg)
    data_cfg["params"]["split"] = "test"
    test_dataset = instantiate_from_config(data_cfg)

    # By default, shuffling and worker-local dataset RNGs are independent of the
    # random numbers consumed while the model is constructed. With `use_global_rng`,
    # they draw from the global torch RNG instead: the matched-reconstruction recipes
    # shuffle as their original runs did, while their frame draws match only in distribution.
    train_generator = eval_generator = None
    if not cfg.training.dataloader.get("use_global_rng", False):
        train_generator = torch.Generator().manual_seed(int(cfg.seed))
        eval_generator = torch.Generator().manual_seed(
            int(data_cfg["params"].get("seed", 42))
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.dataloader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.dataloader.num_workers,
        pin_memory=cfg.training.dataloader.pin_memory,
        persistent_workers=bool(cfg.training.dataloader.persistent_workers and cfg.training.dataloader.num_workers > 0),
        collate_fn=train_dataset.get_collate_fn(),
        worker_init_fn=_worker_init_fn,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.dataloader.num_workers,
        pin_memory=cfg.training.dataloader.pin_memory,
        persistent_workers=bool(cfg.training.dataloader.persistent_workers and cfg.training.dataloader.num_workers > 0),
        collate_fn=test_dataset.get_collate_fn(),
        worker_init_fn=_worker_init_fn,
        generator=eval_generator,
    )
    logging.info(f"Train size: {len(train_dataset)}  |  Val size: {len(test_dataset)}")
    return train_dataset, test_dataset, train_loader, test_loader
