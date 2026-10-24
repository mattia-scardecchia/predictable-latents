"""Generate WorldGym rollout cells with policy-specific external dependencies."""
import argparse
import hashlib
import importlib
import json
import logging
import math
import sys
from pathlib import Path
import numpy as np
import torch
from .wrapper import WMReprWorldModel
from wm_repr.utils.provenance import source_revision

log = logging.getLogger(__name__)
CHUNK_SIZE = {"openvla": 1, "rt1": 1, "octo": 4}
DEFAULT_MODEL_NAME = {"openvla": "openvla-7b", "octo": "octo-base-1.5", "rt1": None}
MODEL_NAME_PREFIX = {"openvla": "openvla/", "octo": "hf://rail-berkeley/"}
RT1_MODEL_CONFIG = dict(num_image_tokens=81, num_action_tokens=11, layer_size=256,
                       vocab_size=512, use_token_learner=True, world_vector_range=(-2.0, 2.0))

def _qualified_model_name(policy: str, raw: str) -> str:
    prefix = MODEL_NAME_PREFIX.get(policy, "")
    return raw if ("/" in raw or raw.startswith("hf://")) else prefix + raw

def run_openvla(wm, args, trials, rollout_length):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from world_model_eval.run_openvla import evaluate_openvla

    model_name = _qualified_model_name("openvla", args.model_name or DEFAULT_MODEL_NAME["openvla"])
    log.info(f"Loading OpenVLA: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vla = (
        AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )
    return evaluate_openvla(
        wm, vla, processor, trials,
        retries=args.retries,
        rollout_length=rollout_length,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
        root_dir=args.root_dir,
    ), model_name

def run_octo(wm, args, trials, rollout_length):
    from octo.model.octo_model import OctoModel
    from world_model_eval.run_octo import evaluate_octo

    model_name = _qualified_model_name("octo", args.model_name or DEFAULT_MODEL_NAME["octo"])
    log.info(f"Loading Octo: {model_name}")
    vla = OctoModel.load_pretrained(model_name)
    return evaluate_octo(
        wm, vla, trials,
        rollout_length=rollout_length,
        retries=args.retries,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
        root_dir=args.root_dir,
    ), model_name

def run_rt1(wm, args, trials, rollout_length):
    if not args.rt1_checkpoint_path:
        raise ValueError("--rt1-checkpoint-path is required for --policy=rt1")
    from world_model_eval.rt1 import rt1
    from world_model_eval.run_rt1 import RT1Policy, evaluate_rt1

    log.info(f"Loading RT-1-X from {args.rt1_checkpoint_path}")
    rt1x_model = rt1.RT1(**RT1_MODEL_CONFIG)
    policy = RT1Policy(
        checkpoint_path=args.rt1_checkpoint_path,
        model=rt1x_model,
        seqlen=args.history_len,
    )
    model_name = f"rt1x:{Path(args.rt1_checkpoint_path).name}"
    return evaluate_rt1(
        wm, policy, trials,
        rollout_length=rollout_length,
        retries=args.retries,
        history_len=args.history_len,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
        root_dir=args.root_dir,
    ), model_name

class RolloutRecorder:
    """Stands in for WorldGym's scorer: saves each (trial, retry) video for offline judging and
    returns no score. Retries are counted per start frame, in the order the policy loop runs them."""

    def __init__(self, out_dir, root_dir, retries):
        self.out_dir, self.root = Path(out_dir), Path(root_dir).resolve()
        (self.out_dir / "videos").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "rollouts.jsonl"
        rows = []
        if self.manifest_path.exists():
            rows = [json.loads(l) for l in self.manifest_path.read_text().splitlines()]
        per_trial = {}
        for r in rows:
            per_trial.setdefault(r["trial_png"], []).append(r)
        # A trial interrupted mid-retries is redone from its first retry.
        self.complete = {t for t, rs in per_trial.items() if len(rs) == retries}
        kept = [r for r in rows if r["trial_png"] in self.complete]
        self.manifest_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        self.counts = {}

    def __call__(self, video, trial):
        import mediapy as media

        retry = self.counts.get(trial["trial_png"], 0)
        self.counts[trial["trial_png"]] = retry + 1
        rel = Path(trial["trial_png"]).resolve().relative_to(self.root).with_suffix("")
        stem = f"{str(rel).replace('/', '__')}__retry{retry}"
        video = np.ascontiguousarray(video, dtype=np.uint8)
        np.savez_compressed(self.out_dir / "videos" / f"{stem}.npz", video=video)
        media.write_video(str(self.out_dir / "videos" / f"{stem}.mp4"), video, fps=20)
        row = {**trial, "retry_idx": retry, "video": f"videos/{stem}.npz", "mp4": f"videos/{stem}.mp4",
               "shape": list(video.shape), "sha256": hashlib.sha256(video.tobytes()).hexdigest()}
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return float("nan")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root-dir", required=True, help="WorldGym starting images and task metadata")
    p.add_argument("--assets")
    p.add_argument("--ae-checkpoint")
    p.add_argument("--policy", choices=list(CHUNK_SIZE), required=True)
    p.add_argument("--model-name")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--wm-frames", type=int, default=40)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--rt1-checkpoint-path")
    p.add_argument("--history-len", type=int, default=15)
    p.add_argument("--task-subset", help="Comma-separated task keys")
    p.add_argument("--trial-subset", help="Comma-separated trial indices within every task")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    from world_model_eval.utils import discover_trials
    wm = WMReprWorldModel(args.checkpoint, device=args.device, cfg_scale=args.cfg_scale,
                         ae_ckpt_override=args.ae_checkpoint, assets=args.assets,
                         sampling_steps=args.steps)
    trials = discover_trials(args.root_dir)
    if args.task_subset:
        trials = [t for t in trials if t["task_key"] in args.task_subset.split(",")]
    if args.trial_subset:
        indices, counts, selected = set(map(int, args.trial_subset.split(","))), {}, []
        for trial in trials:
            key = trial["task_key"]
            index = counts.get(key, 0)
            counts[key] = index + 1
            if index in indices:
                selected.append(trial)
        trials = selected
    if not trials:
        raise ValueError("No WorldGym trials match this directory and selection")
    recorder = RolloutRecorder(args.output, args.root_dir, args.retries)
    trials = [t for t in trials if t["trial_png"] not in recorder.complete]
    importlib.import_module(f"world_model_eval.run_{args.policy}").predict = recorder
    chunk_size = CHUNK_SIZE[args.policy]
    length = math.ceil(args.wm_frames / chunk_size)
    meta = dict(vars(args), chunk_size=chunk_size, rollout_length=length,
                sampling_steps_used=wm.sampling_steps, cfg_scale_used=wm.cfg_scale,
                commit=source_revision(), command=sys.argv)
    (Path(args.output) / "rollout_meta.json").write_text(json.dumps(meta, indent=2))
    args.save_video, args.video_out_dir = False, None
    if trials:
        {"openvla": run_openvla, "octo": run_octo, "rt1": run_rt1}[args.policy](wm, args, trials, length)
    log.info("Rollouts saved to %s", args.output)


if __name__ == "__main__":
    main()
