from typing import Optional
import torch
import numpy as np
from pldm_envs.wall.data.wall import WallSample
from dataclasses import dataclass


@dataclass
class OfflineWallDatasetConfig:
    batch_size: int = 64
    img_size: int = 65
    train: bool = True
    device: str = "cuda"
    repeat_actions: int = 1
    image_based: bool = True
    use_offline: bool = False
    offline_data_path: str = ""
    lazy_load: bool = False
    l2_step_skip: int = 4
    n_steps: Optional[int] = None
    l2_n_steps: Optional[int] = None
    chunked_actions: bool = False


def load_np(offline_data_path, lazy_load=False):
    if lazy_load:
        states = np.load(f"{offline_data_path}/states.npy", mmap_mode="r")
    else:
        states = np.load(f"{offline_data_path}/states.npy")

    actions = np.load(f"{offline_data_path}/actions.npy")
    locations = np.load(f"{offline_data_path}/locations.npy")

    return states, actions, locations


def load_npz(path, lazy_load=False):
    if lazy_load:
        npz = np.load(path, allow_pickle=True, mmap_mode="r")
    else:
        npz = np.load(path, allow_pickle=True)

    observations = npz["observations"]
    actions = npz["actions"]
    terminals = npz["terminals"]
    locations = npz["locations"]

    # observations: NxHxWxC -> NxCxHxW
    observations = observations.transpose(0, 3, 1, 2)

    # we assume that all trajectories are the same length
    # we use terminals to find the length of the trajectories

    # find the length of the trajectories
    traj_ends = np.where(terminals)[0]
    lengths = np.unique(traj_ends[1:] - traj_ends[:-1])
    assert len(lengths) == 1, "All trajectories must be the same length"
    # get the length of the trajectories
    length = lengths[0]

    # now reshape observations, actions, and locations
    # to have shape (n_trajectories, length, ...)

    n_trajectories = len(traj_ends)
    observations = observations.reshape(n_trajectories, length, *observations.shape[1:])
    actions = actions.reshape(n_trajectories, length, *actions.shape[1:])
    locations = locations.reshape(n_trajectories, length, *locations.shape[1:])

    # we drop the last action and location since we don't have the next state
    actions = actions[:, :-1]
    locations = locations[:, :-1]

    return observations, actions, locations


class OfflineWallDataset:
    def __init__(
        self,
        config: OfflineWallDatasetConfig,
        probing=False,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "npz" in config.offline_data_path:
            self.states, self.actions, self.locations = load_npz(
                config.offline_data_path, config.lazy_load
            )
        else:
            self.states, self.actions, self.locations = load_np(
                config.offline_data_path, config.lazy_load
            )

        self.data_ep_length = self.states.shape[1]
        self.ep_length = (
            config.n_steps if config.n_steps is not None else self.data_ep_length
        )
        self.slices_per_traj = self.data_ep_length - self.ep_length + 1

        print(f"Data episode length: {self.ep_length}")
        print(f"Episode length: {self.ep_length}")
        print(f"Slices per traj: {self.slices_per_traj}")

        if self.config.l2_n_steps:
            self.l2_ep_length = self.config.l2_n_steps * self.config.l2_step_skip + 1
            self.l2_slices_per_traj = self.data_ep_length - self.l2_ep_length + 1

            print(f"L2 Data episode length: {self.config.l2_n_steps}")
            print(f"L2 Episode length: {self.l2_ep_length}")
            print(f"L2 Slices per traj: {self.l2_slices_per_traj}")
        else:
            self.l2_ep_length = 0
            self.l2_slices_per_traj = None

        print(f"Loaded offline data with {len(self)} samples")

    def __len__(self):
        if self.l2_slices_per_traj is not None:
            return len(self.states) * self.l2_slices_per_traj
        else:
            return len(self.states) * self.slices_per_traj

    def _load_traj(self, i, slices_per_traj, ep_length):
        traj_idx = i // (slices_per_traj)
        slice_idx = i % (slices_per_traj)

        states = torch.from_numpy(
            self.states[traj_idx, slice_idx : slice_idx + ep_length]
        ).to(self.device)

        locations = (
            torch.from_numpy(
                self.locations[traj_idx, slice_idx : slice_idx + ep_length - 1]
            )
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                self.actions[traj_idx, slice_idx : slice_idx + ep_length - 1]
            )
            .float()
            .to(self.device)
        )

        return states, locations, actions

    def __getitem__(self, i):
        states, locations, actions = self._load_traj(
            i, self.slices_per_traj, self.ep_length
        )

        if self.l2_ep_length:
            l2_states, l2_locations, l2_actions = self._load_traj(
                i, self.l2_slices_per_traj, self.l2_ep_length
            )

            l2_states = l2_states[:: self.config.l2_step_skip]
            l2_locations = l2_locations[:: self.config.l2_step_skip]
            # return chunked actions
            if self.config.chunked_actions:
                # (t, 2) -> (t//skip, skip, 2)
                chunks = l2_actions.split(self.config.l2_step_skip)
                l2_actions = torch.stack(chunks, dim=0)
            else:
                raise NotImplementedError
        else:
            l2_states = torch.empty(0)
            l2_locations = torch.empty(0)
            l2_actions = torch.empty(0)

        sample = WallSample(
            states=states,
            locations=locations,
            actions=actions,
            bias_angle=torch.empty(0).to(self.device),
            wall_x=torch.empty(0).to(self.device),
            door_y=torch.empty(0).to(self.device),
            l2_states=l2_states,
            l2_locations=l2_locations,
            l2_actions=l2_actions,
        )

        return sample
