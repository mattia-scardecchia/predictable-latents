from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from omegaconf import DictConfig


class VectorWorld:
    """State s = [signal(2), noise(>=1)]. Actions move only the 2-D signal; noise is resampled i.i.d.
    Actions: 0:+x, 1:+y, 2:-x, 3:-y (scaled by step).
    """

    def __init__(
        self,
        signal_dim: int,
        noise_dim: int,
        step: float,
        seed: int,
    ):
        assert signal_dim == 2
        self.seed = seed
        self.D = signal_dim + noise_dim
        self.signal_dim = signal_dim
        self.noise_dim = noise_dim
        self.step_size = step
        self.rng = np.random.RandomState(seed)

    @classmethod
    def from_config(cls, cfg: Any) -> "VectorWorld":
        return cls(
            signal_dim=cfg.signal_dim,
            noise_dim=cfg.noise_dim,
            step=cfg.step_size,
            seed=cfg.seed,
        )

    def init_state(self, N: int) -> Dict[str, np.ndarray]:
        signal = self.rng.randn(N, self.signal_dim)
        noise = self.rng.randn(N, self.noise_dim)
        state = np.concatenate([signal, noise], axis=1)
        out = {"state": state, "observation": state.copy(), "signal": signal.copy()}
        return out

    def step(self, state: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance state given action. Returns a dictionary with keys "state", "observation", "signal"."""
        signal = state[:, : self.signal_dim].copy()

        signal_next = signal + action
        noise_next = self.rng.randn(state.shape[0], self.noise_dim)
        state_next = np.concatenate([signal_next, noise_next], axis=1)
        out = {
            "state": state_next,
            "observation": state_next.copy(),
            "signal": signal_next.copy(),
        }
        return out


def discrete2continuous_action(
    a_discrete: np.ndarray, signal_dim: int, step_size: float
):
    B = int(a_discrete.shape[0])
    a_cont = np.zeros((B, signal_dim), dtype=np.float32)
    coords = a_discrete.astype(np.int32) % signal_dim
    signs = np.where(a_discrete < signal_dim, 1.0, -1.0)
    coords = coords.reshape(B)
    signs = signs.reshape(B)
    a_cont[np.arange(B), coords] = signs * step_size
    return a_discrete, a_cont


def sample_trajectories(
    env: VectorWorld,
    n_traj: int,
    traj_len: int,
) -> Dict[str, np.ndarray]:
    """Trajectory generator using env.init_state and env.step, which return dictionaries with keys
    "state", "observation" and "signal". Actions are drawn uniformly from the four moves."""
    st_list, obs_list, sig_list, ac_list, snxt_list, obsnxt_list, signxt_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    init_dict = env.init_state(n_traj)
    state, signal, obs = (
        init_dict["state"],
        init_dict["signal"],
        init_dict["observation"],
    )

    for _ in range(traj_len):
        num_actions = 2 * env.signal_dim
        a_discrete = env.rng.randint(0, num_actions, size=(n_traj,))
        a_to_store, a_to_pass = discrete2continuous_action(
            a_discrete=a_discrete,
            signal_dim=env.signal_dim,
            step_size=env.step_size,
        )
        next_dict = env.step(state, a_to_pass)

        st_list.append(state)
        sig_list.append(signal)
        obs_list.append(obs)
        ac_list.append(a_to_store)
        snxt_list.append(next_dict["state"])
        signxt_list.append(next_dict["signal"])
        obsnxt_list.append(next_dict["observation"])

        state = next_dict["state"]
        signal = next_dict["signal"]
        obs = next_dict["observation"]

    out = {
        "state": np.stack(st_list, axis=0).astype(np.float32),
        "action": np.stack(ac_list, axis=0).astype(np.float32),
        "observation": np.stack(obs_list, axis=0).astype(np.float32),
        "state_next": np.stack(snxt_list, axis=0).astype(np.float32),
        "observation_next": np.stack(obsnxt_list, axis=0).astype(np.float32),
        "signal": np.stack(sig_list, axis=0).astype(np.float32),
        "signal_next": np.stack(signxt_list, axis=0).astype(np.float32),
    }
    return out  # T, B, ...


class TransitionDataset(Dataset):
    def __init__(self, path: str | Path):
        data = np.load(path)
        T, B, _ = torch.from_numpy(data["state"]).shape

        self.state = torch.from_numpy(data["state"]).reshape(B * T, -1).float()
        self.action = torch.from_numpy(data["action"]).reshape(B * T, -1).float()
        self.observation = (
            torch.from_numpy(data["observation"]).reshape(B * T, -1).float()
        )
        self.state_next = (
            torch.from_numpy(data["state_next"]).reshape(B * T, -1).float()
        )
        self.observation_next = (
            torch.from_numpy(data["observation_next"]).reshape(B * T, -1).float()
        )
        self.signal = torch.from_numpy(data["signal"]).reshape(B * T, -1).float()
        self.signal_next = (
            torch.from_numpy(data["signal_next"]).reshape(B * T, -1).float()
        )

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, i: int):
        return {
            "state": self.state[i],
            "action": self.action[i],
            "observation": self.observation[i],
            "state_next": self.state_next[i],
            "observation_next": self.observation_next[i],
            "signal": self.signal[i],
            "signal_next": self.signal_next[i],
        }


def generate_data(cfg: DictConfig) -> None:
    """Write the train, validation and test transitions of one run to ``cfg.data.out_dir``."""
    data_dir = Path(cfg.data.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = VectorWorld.from_config(cfg.data)
    for split_name, n_traj in (
        ("train", cfg.data.n_train),
        ("val", cfg.data.n_val),
        ("test", cfg.data.n_test),
    ):
        trajs = sample_trajectories(env, n_traj=n_traj, traj_len=cfg.data.traj_len)
        np.savez_compressed(str(data_dir / f"{split_name}.npz"), **trajs)


def data_loaders(cfg: DictConfig, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the transitions written by `generate_data`."""
    data_dir = Path(cfg.data.out_dir)
    num_workers = cfg.train.num_workers
    train_loader = DataLoader(
        TransitionDataset(str(data_dir / "train.npz")),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        TransitionDataset(str(data_dir / "val.npz")),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )
    return train_loader, val_loader
