"""Run the synthetic study's 93-cell grid."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from wm_repr.utils.provenance import source_revision
from . import phases
from .data import TransitionDataset, generate_data
from .evaluate import evaluate
from .utils import seed_all


LATENT_SIZES = (10, 20, 30, 50, 100)
WEIGHTS = (1.0, 0.8, 0.5, 0.2, 0.05, 0.0)


def configuration(output, seed=0, latent_size=30, weight=0.5, method='contrastive_vae', device='cpu'):
    output = Path(output).resolve()
    steps, n_train, n_eval, batch_size = 7500, 10000, 1000, 256
    clipping = dict(enabled=False, max_norm=10.0, norm_type=2.0)
    opt = dict(lr=0.001, batch_size=batch_size, weight_decay=0.0, gradient_clip=clipping)
    encoder = dict(z_dim=latent_size, enc_widths=[256], dec_widths=[256],
                   beta=1e-4, activation='relu')
    cfg = OmegaConf.create(dict(
        method=method,
        data=dict(out_dir=str(output/'data'),
                  signal_dim=2, noise_dim=100, state_dim=102, traj_len=2, num_actions=4,
                  seed=seed, step_size=0.3,
                  n_train=n_train, n_val=n_eval, n_test=n_eval),
        model=dict(contrastive_vae=dict(**encoder, proj_widths=[256],
                   temperature=1.0, temporal_weight=weight),
                   repr=dict(z_dim=102 if method == 'identity' else latent_size),
                   identity=dict(z_dim=102),
                   dynamics=dict(dyn_widths=[256], activation='relu'),
                   probe=dict(probe_widths=[64], activation='relu'),
                   readout=dict(dec_widths=[256], activation='relu')),
        train=dict(device=device, ckpt_dir=str(output/'checkpoints'), num_workers=0,
                   eval_batch_size=512, steps_phase1=steps, steps_phase2=steps,
                   steps_probe=steps, steps_readout=steps, contrastive=opt,
                   dynamics=opt, probe=opt, readout_decoder=opt)))
    cfg.train.contrastive.gradient_clip = dict(enabled=True, max_norm=1.0, norm_type=2.0)
    return cfg


def run(cfg):
    output = Path(cfg.train.ckpt_dir).parent
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    (output/'provenance.json').write_text(json.dumps({
        'commit': source_revision(), 'command': sys.argv,
    }, indent=2)+'\n')
    device = torch.device(cfg.train.device)
    seed_all(cfg.data.seed)
    generate_data(cfg)
    training = {}
    if cfg.method == 'contrastive_vae':
        training['representation'] = phases.train_phase1_contrastive(
            cfg, device, phases.create_contrastive_vae(cfg, device))
    elif cfg.method != 'identity':
        raise ValueError('Supported paper methods: contrastive_vae, identity')
    for name, function in [('dynamics', phases.train_phase2_dynamics),
                           ('probe', phases.train_probes),
                           ('readout', phases.train_readout_decoder)]:
        _, training[name] = function(cfg, device, cfg.method, cfg.train.ckpt_dir)
    seed_all(cfg.data.seed)
    raw = {split: evaluate(cfg, cfg.method, device, cfg.train.ckpt_dir, split)
           for split in ['train', 'val', 'test']}
    ds = TransitionDataset(Path(cfg.data.out_dir)/'test.npz')
    variance = lambda x: float(np.var(x.numpy(), axis=0).mean())
    test = raw['test']
    denominators = dict(observation=variance(ds.observation), signal=variance(ds.signal),
                        next_signal=variance(ds.signal_next),
                        next_latent=test['mse_unrelated_embeddings']/2)
    standardized = dict(recon=test['mse_readout_reconstruction']/denominators['observation'],
                        signal=test['mse_signal_reconstruction']/denominators['signal'],
                        next_signal=test['mse_next_signal_prediction']/denominators['next_signal'],
                        next_latent=test['mse_dynamics_prediction']/denominators['next_latent'])
    result = dict(seed=cfg.data.seed, latent_size=cfg.model.repr.z_dim,
                  weight=cfg.model.contrastive_vae.temporal_weight, method=cfg.method,
                  training=training, raw=raw, target_variances=denominators,
                  normalized_errors=standardized)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), normalized_errors=standardized)), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    # A fixed thread count keeps CPU reductions, and so the results, the same on every machine.
    torch.set_num_threads(4)
    cells = [(s,z,w,'contrastive_vae') for s in range(3) for z in LATENT_SIZES for w in WEIGHTS]
    cells += [(s,102,0,'identity') for s in range(3)]
    for seed,z,w,method in cells:
        run(configuration(args.output/f'{method}-z{z}-w{w:g}-seed{seed}',seed,z,w,method,args.device))


if __name__ == '__main__':
    main()
