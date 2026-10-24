"""Train an autoencoder or frozen-representation world model from a paper recipe."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from wm_repr.config import load_config, recipe_names
from wm_repr.utils.provenance import source_revision
from wm_repr.utils.training import prepare_dataloaders


def write_provenance(output, cfg, dataset, attempt=None):
    destination = attempt or output
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (destination / "config.yaml").write_text(resolved)
    commit = source_revision()
    metadata = {"command": sys.argv, "commit": commit, "python": platform.python_version(),
                "torch": torch.__version__, "lightning": pl.__version__,
                "config_sha256": hashlib.sha256(resolved.encode()).hexdigest()}
    records = None
    if getattr(dataset, "fixed_eval_pairs", False):
        records = dataset.fixed_pair_manifest()
    elif getattr(dataset, "fixed_eval_clips", False):
        records = dataset.fixed_clip_manifest()
    if records is not None:
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        (destination / "validation_manifest.jsonl").write_text(payload)
        metadata["validation_manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if destination != output:
        for name in ("config.yaml", "provenance.json", "validation_manifest.jsonl"):
            if (destination / name).exists():
                shutil.copyfile(destination / name, output / name)


def begin_attempt(output, stage, resume=None, stats_batches=100):
    """Keep every invocation's metadata and CSV logs when continuing a run."""
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    for index in itertools.count(1):
        attempt = attempts / f"{index:04d}"
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            continue
    record = {"stage": stage, "command": sys.argv, "status": "started",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "resume_checkpoint": str(Path(resume).resolve()) if resume else None,
              "stats_batches": stats_batches}
    save_attempt(output, attempt, record)
    return attempt, record


def save_attempt(output, attempt, record):
    (attempt / "attempt.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "latest_attempt.json").write_text(json.dumps(
        {"attempt": str(attempt.relative_to(output)), "status": record["status"]}, indent=2) + "\n")


def run(cfg, stage, resume=None, stats_batches=100):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.ckpt").exists() and resume is None:
        raise FileExistsError(f"{output}/last.ckpt exists; use --resume or a new output directory")
    attempt, record = begin_attempt(output, stage, resume, stats_batches)
    try:
        if resume:
            with Path(resume).open("rb") as stream:
                record["resume_checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            save_attempt(output, attempt, record)
        summary = _fit(cfg, stage, output, attempt, resume, stats_batches)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      ended_utc=datetime.now(timezone.utc).isoformat())
        save_attempt(output, attempt, record)
        raise
    record.update(status="completed", global_step=summary["global_step"],
                  ended_utc=datetime.now(timezone.utc).isoformat())
    save_attempt(output, attempt, record)
    return summary


def _fit(cfg, stage, output, attempt, resume, stats_batches):
    pl.seed_everything(int(cfg.seed), workers=True)
    _, test_dataset, train_loader, test_loader = prepare_dataloaders(cfg)
    write_provenance(output, cfg, test_dataset, attempt)
    if stage == "ae":
        import wm_repr.hierarchical  # noqa: F401  (registers every autoencoder class)
        from wm_repr.jepa.registry import get_by_class_name
        model = get_by_class_name(cfg.model.class_name)(cfg)
    else:
        from wm_repr.wm.world_model import LitWorldModel
        model = LitWorldModel(cfg, learning_rate=cfg.training.base_learning_rate)
        if resume is None:
            device = "cuda" if cfg.training.trainer.accelerator == "gpu" else "cpu"
            model.ae.to(device).eval()
            model.ae.compute_embedding_stats(itertools.islice(train_loader, stats_batches))
            model.ae.cpu()
    checkpoint_steps = cfg.training.get("checkpoint_every_n_steps")
    if checkpoint_steps:
        checkpoint = ModelCheckpoint(dirpath=output, filename="step-{step}", save_last=True,
                                     save_top_k=-1, every_n_train_steps=int(checkpoint_steps),
                                     every_n_epochs=0, save_on_train_epoch_end=False)
    else:
        checkpoint = ModelCheckpoint(dirpath=output, filename="step-{step}", save_last=True,
                                     save_top_k=0, every_n_epochs=1)
    callbacks = [checkpoint]
    threshold = cfg.get("validation", {}).get("reconstruction_threshold")
    if stage == "ae" and threshold is not None:
        callbacks.append(EarlyStopping(monitor="val/recon_loss", mode="min", patience=100000,
                                       stopping_threshold=float(threshold), check_on_train_epoch_end=False))
    kwargs = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    kwargs.setdefault("enable_progress_bar", sys.stderr.isatty())
    trainer = pl.Trainer(**kwargs, callbacks=callbacks, default_root_dir=output,
                         logger=CSVLogger(str(attempt), name="metrics", version=""))
    if stage == "ae" and resume is None and kwargs.get("limit_val_batches", 1) != 0:
        trainer.validate(model, test_loader)
    trainer.fit(model, train_loader, test_loader, ckpt_path=resume)
    trainer.save_checkpoint(output / "last.ckpt")
    summary = {"stage": stage, "global_step": trainer.global_step,
               "attempt": str(attempt.relative_to(output)),
               "checkpoint": str(output / "last.ckpt"),
               "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()
                           if isinstance(v, torch.Tensor) and v.numel() == 1}}
    result = json.dumps(summary, indent=2) + "\n"
    (attempt / "result.json").write_text(result)
    (output / "result.json").write_text(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=recipe_names(), required=True)
    parser.add_argument("--stage", choices=["ae", "wm"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stats-batches", type=int, default=100)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.recipe, args.stage, data_root=args.data_root, assets_root=args.assets,
                      output_dir=args.output, seed=args.seed, ae_checkpoint=args.ae_checkpoint,
                      overrides=args.set)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    os.environ.setdefault("WM_LPIPS_CACHE_ROOT", str(args.assets.resolve() / "lpips"))
    os.environ.setdefault("TORCH_HOME", str(args.assets.resolve() / "torch"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(cfg, args.stage, str(args.resume) if args.resume else None, args.stats_batches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
