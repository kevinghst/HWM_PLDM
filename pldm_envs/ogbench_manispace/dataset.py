from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import numpy as np

default_path = "/volume/data/ogbench/visual-cube-single-play-v0-val.npz"


@dataclass
class ManispaceDatasetConfig:
    path: str = default_path
    val_path: Optional[str] = None
    lazy_load: bool = False
    device: str = "cuda"
    n_steps: Optional[int] = None  # if None, we use the full trajectory length
    batch_size: int = 64
    l2_step_skip: int = 10
    image_based: bool = True
    chunked_actions: bool = False


class ManispaceSample(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    locations: torch.Tensor
    proprio_pos: torch.Tensor
    proprio_vel: torch.Tensor


def load_npz(path, lazy_load=False):
    if lazy_load:
        npz = np.load(path, allow_pickle=True, mmap_mode="r")
    else:
        npz = np.load(path, allow_pickle=True)

    observations = npz["observations"]
    actions = npz["actions"]
    terminals = npz["terminals"]
    qpos = npz["qpos"]
    qvel = npz["qvel"]

    # observations: NxHxWxC -> NxCxHxW
    if len(observations.shape) > 2:
        observations = observations.transpose(0, 3, 1, 2)

    # we assume that all trajectories are the same length
    # we use terminals to find the length of the trajectories

    # find the length of the trajectories
    traj_ends = np.where(terminals)[0]
    lengths = np.unique(traj_ends[1:] - traj_ends[:-1])
    assert len(lengths) == 1, "All trajectories must be the same length"
    # get the length of the trajectories
    length = lengths[0]
    print("Found trajectories of length", length)

    # now reshape observations, actions, and locations
    # to have shape (n_trajectories, length, ...)

    n_trajectories = len(traj_ends)
    observations = observations.reshape(n_trajectories, length, *observations.shape[1:])
    actions = actions.reshape(n_trajectories, length, *actions.shape[1:])
    qpos = qpos.reshape(n_trajectories, length, *qpos.shape[1:])
    qvel = qvel.reshape(n_trajectories, length, *qvel.shape[1:])

    return observations, actions, qpos, qvel


class ManispaceDataset(torch.utils.data.Dataset):
    def __init__(self, config: ManispaceDatasetConfig, train: bool = True):
        """
        Loads the NPZ data and slices it into episodes (and sub-episodes if n_steps < full length).
        """
        self.config = config
        self.train = train

        data_path = config.path if train else config.val_path
        # Load the data
        self.observations, self.actions, self.qpos, self.qvel = load_npz(
            data_path, self.config.lazy_load
        )
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        # Full trajectory length in the dataset
        self.data_ep_length = self.observations.shape[1]
        # The length each sample will have
        self.ep_length = (
            self.config.n_steps
            if self.config.n_steps is not None
            else self.data_ep_length
        )

        # How many distinct sub-episodes per trajectory
        self.slices_per_traj = self.data_ep_length - self.ep_length + 1

        # Print some info
        print(f"Number of trajectories: {len(self.observations)}")
        print(f"Data episode length: {self.data_ep_length}")
        print(f"Chosen sub-episode length: {self.ep_length}")
        print(f"Slices (sub-episodes) per trajectory: {self.slices_per_traj}")
        print(f"Total samples in dataset: {len(self)}")

    def __len__(self):
        return len(self.observations) * self.slices_per_traj

    def __getitem__(self, idx):
        """
        Returns a slice of length self.ep_length from a particular trajectory.
        """
        traj_idx = idx // self.slices_per_traj
        slice_idx = idx % self.slices_per_traj

        obs = self.observations[traj_idx, slice_idx : slice_idx + self.ep_length]
        act = self.actions[traj_idx, slice_idx : slice_idx + self.ep_length - 1]
        qpos = self.qpos[traj_idx, slice_idx : slice_idx + self.ep_length]
        qvel = self.qvel[traj_idx, slice_idx : slice_idx + self.ep_length]

        # Convert to torch Tensors on the correct device
        obs = torch.from_numpy(obs).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device)
        qpos = torch.from_numpy(qpos).float().to(self.device)
        qvel = torch.from_numpy(qvel).float().to(self.device)

        if not self.config.image_based:
            # qpos will contain the proprio info
            qpos = obs[..., :19]
            # qvel will contain nothing
            qvel = torch.empty(0)
            # location will be the location of first cube
            locations = obs[..., 19:22]
            # obs will contain the info of cubes
            obs = obs[..., 19:]

        sample = ManispaceSample(
            states=obs,
            actions=act,
            locations=locations,
            proprio_pos=qpos,
            proprio_vel=qvel,
        )
        return sample
