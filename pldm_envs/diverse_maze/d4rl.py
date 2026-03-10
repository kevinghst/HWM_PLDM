import torch
import numpy as np
import zarr
import torchvision.transforms as transforms

from pldm_envs.diverse_maze.enums import D4RLSample, D4RLDatasetConfig


def get_eval_env_name(env_name: str):
    split_name = env_name.split("-")
    if "ant" in env_name:
        split_name[-1] = "v0"
    split_name.insert(1, "eval")
    return "-".join(split_name)


class D4RLDataset(torch.utils.data.Dataset):
    def __init__(self, config: D4RLDatasetConfig, images_tensor=None, load_l1=True):
        self.config = config
        self.load_l1 = load_l1

        if self.config.l2_n_steps:
            self.l2_n_steps_total = (
                self.config.l2_n_steps * self.config.l2_step_skip + 1
            )
        else:
            self.l2_n_steps_total = 0

        self._prepare_saved_ds()

        if self.config.images_path is not None:
            print("states will contain images")
            if self.config.images_path.endswith("zarr"):
                print(f"loading zarr {self.config.images_path} into memory")
                if images_tensor is None:
                    self.images_tensor = zarr.open(self.config.images_path, "r")[:]
                else:
                    self.images_tensor = images_tensor
                print("using zarr format, shape of zarr is:", self.images_tensor.shape)
            elif self.config.images_path.endswith("npy"):
                print(f"loading {self.config.images_path}")
                if images_tensor is None:
                    self.images_tensor = np.load(self.config.images_path, mmap_mode="r")
                else:
                    self.images_tensor = images_tensor
                print("shape of images is:", self.images_tensor.shape)
            else:
                if "ant" not in self.config.env_name:
                    self.image_transform = transforms.Compose(
                        [
                            transforms.CenterCrop(200),
                            transforms.Resize(64),
                            transforms.ToTensor(),
                        ]
                    )
        else:
            print("states will contain proprioceptive info")

    def _prepare_saved_ds(self):
        assert self.config.path is not None
        print("loading saved dataset from", self.config.path)
        self.splits = torch.load(self.config.path)

        max_n_steps = max(
            self.l2_n_steps_total,
            self.config.n_steps,
        )

        self.cum_lengths = np.cumsum(
            [
                len(x["observations"]) - max_n_steps - (self.config.stack_states - 1)
                for x in self.splits
            ]
        )

        self.cum_lengths_total = np.cumsum(
            [len(x["observations"]) for x in self.splits]
        )

    def __len__(self):
        if self.config.crop_length is not None:
            return min(self.config.crop_length, self.cum_lengths[-1])
        else:
            return self.cum_lengths[-1]

    def _load_images_tensor(self, episode_idx, start_idx, length, skip_frame=1):
        if episode_idx == 0:
            index = start_idx
        else:
            index = self.cum_lengths_total[episode_idx - 1] + start_idx
        return (
            torch.from_numpy(self.images_tensor[index : index + length : skip_frame])
            .permute(0, 3, 1, 2)
            .float()
        )

    def sample_location(self):
        idx = np.random.randint(0, len(self.splits))
        idx_2 = np.random.randint(0, len(self.splits[idx]["observations"]))
        return self.splits[idx]["observations"][idx_2]

    def _load_locations(self, episode_idx, start_idx, end_idx, skip_frame=1):
        return self.splits[episode_idx]["observations"][
            start_idx:end_idx:skip_frame, :2
        ]

    def _load_qvel(self, episode_idx, start_idx, end_idx, skip_frame=1):
        obs = torch.from_numpy(
            self.splits[episode_idx]["observations"][start_idx:end_idx:skip_frame]
        ).float()

        if "ant" in self.config.env_name:
            proprio_vel = obs[:, 15:]
        else:
            proprio_vel = obs[:, 2:]

        return proprio_vel

    def _load_qpos(self, episode_idx, start_idx, end_idx, skip_frame=1):
        obs = torch.from_numpy(
            self.splits[episode_idx]["observations"][start_idx:end_idx:skip_frame]
        ).float()

        if "ant" in self.config.env_name:
            if self.config.location_in_proprio_component:
                proprio_pos = obs[:, :15]
            else:
                proprio_pos = obs[:, 2:15]
        else:
            proprio_pos = torch.empty(0)

        return proprio_pos

    def _load_data_from_start_idx(self, episode_idx, start_idx, length, skip_frame=1):
        end_idx = start_idx + length

        proprio_vel = self._load_qvel(
            episode_idx, start_idx, end_idx, skip_frame=skip_frame
        )
        proprio_pos = self._load_qpos(
            episode_idx, start_idx, end_idx, skip_frame=skip_frame
        )

        locations = self._load_locations(
            episode_idx, start_idx, end_idx, skip_frame=skip_frame
        )

        if self.config.images_path is not None:
            states = self._load_images_tensor(
                episode_idx,
                start_idx,
                length,
                skip_frame=skip_frame,
            )
        else:
            if self.config.location_only:
                states = locations
            else:
                states = self.splits[episode_idx]["observations"][start_idx:end_idx]
            states = torch.from_numpy(states).float()

        actions = torch.from_numpy(
            self.splits[episode_idx]["actions"][start_idx : end_idx - 1]
        ).float()
        # to be compatible with other datasets with the dot.

        if self.config.stack_states > 1:
            states = torch.stack(
                [
                    states[i : i + self.config.stack_states]
                    for i in range(self.config.n_steps)
                ],
                dim=0,
            )
            states = states.flatten(1, 2)  # (n_steps, stack_states * state_dim)
            locations = locations[(self.config.stack_states - 1) :]
            actions = actions[(self.config.stack_states - 1) :]
            proprio_vel = proprio_vel[(self.config.stack_states - 1) :]

            if bool(proprio_pos.shape[-1]):
                # If proprio_pos is not empty
                proprio_pos = proprio_pos[(self.config.stack_states - 1) :]

        if self.config.random_actions:
            # uniformly sample values from -1 to 1
            actions = torch.rand_like(actions) * 2 - 1

        locations = torch.from_numpy(locations).float()

        return states, locations, actions, proprio_vel, proprio_pos

    def __getitem__(self, idx):
        """
        Return:
            - states: [n_steps, ...]
            - locations: [n_steps, 2]
            - actions: [n_steps - 1, ...]
            - indices: [n_steps, ]
            - proprio_vel: [n_steps, ...]
            - proprio_pos: [n_steps, ...]

            if l2_n_steps > 0:
            - l2_states: [l2_n_steps + 1, ...]
            - l2_locations: [l2_n_steps + 1, 2]
            - l2_proprio_vel: [l2_n_steps + 1, ...]
            - l2_proprio_pos: [l2_n_steps + 1, ...]
        """
        episode_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        start_idx = idx - self.cum_lengths[episode_idx - 1] if episode_idx > 0 else idx

        if self.load_l1:
            states, locations, actions, proprio_vel, proprio_pos = (
                self._load_data_from_start_idx(
                    episode_idx=episode_idx,
                    start_idx=start_idx,
                    length=self.config.n_steps + self.config.stack_states - 1,
                )
            )
        else:
            states = torch.empty(0)
            locations = torch.empty(0)
            actions = torch.empty(0)
            proprio_vel = torch.empty(0)
            proprio_pos = torch.empty(0)

        if self.l2_n_steps_total > 0:
            l2_states, l2_locations, l2_actions1, l2_proprio_vel, l2_proprio_pos = (
                self._load_data_from_start_idx(
                    episode_idx=episode_idx,
                    start_idx=start_idx,
                    length=self.l2_n_steps_total + self.config.stack_states - 1,
                    skip_frame=self.config.l2_step_skip,
                )
            )
            # print(l2_actions1.shape)
            if self.config.chunked_actions:
                chunks = l2_actions1.split(self.config.l2_step_skip)
                l2_actions = torch.stack(chunks, dim=0)
            else:
                raise NotImplementedError
        else:
            l2_states = torch.empty(0)
            l2_locations = torch.empty(0)
            l2_proprio_vel = torch.empty(0)
            l2_proprio_pos = torch.empty(0)
            l2_actions = torch.empty(0)

        return D4RLSample(
            states=states,
            locations=locations,
            actions=actions,
            indices=idx,
            proprio_vel=proprio_vel,
            proprio_pos=proprio_pos,
            l2_states=l2_states,
            l2_locations=l2_locations,
            l2_proprio_vel=l2_proprio_vel,
            l2_proprio_pos=l2_proprio_pos,
            l2_actions=l2_actions,
        )
