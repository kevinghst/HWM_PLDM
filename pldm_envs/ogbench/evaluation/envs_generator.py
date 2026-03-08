from pldm_envs.utils.normalizer import Normalizer
import gymnasium
from pldm_envs.ogbench.wrappers import NormEvalWrapper
from pldm_envs.diverse_maze.enums import D4RLDatasetConfig
from pldm_envs.ogbench.dataset import LocoMazeDataset
import dataclasses
from pldm.data.utils import make_dataloader
import torch
import ogbench.locomaze


class SimpleObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EnvsGenerator:
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        level: str,
        stack_states: int,
        offline_T: int,
        seed: int = 42,
        set_full_states: bool = False,
        image_obs: bool = False,
        data_path: str = None,
        normalizer: Normalizer = None,
        trials_path: str = None,
    ):
        self.env_name = env_name
        self.n_envs = n_envs
        self.level = level
        self.stack_states = stack_states
        self.offline_T = offline_T
        self.seed = seed
        self.set_full_states = set_full_states
        self.image_obs = image_obs
        self.data_path = data_path
        self.normalizer = normalizer
        self.trials_path = trials_path

    def _make_env(self, idx):
        T = self.offline_T

        env = gymnasium.make(
            self.env_name,
            terminate_at_goal=False,
            max_episode_steps=9999,
        )

        env = NormEvalWrapper(
            env=env,
            normalizer=self.normalizer,
            stack_states=self.stack_states,
        )

        init_xy = tuple(self.trials["init_xy"][idx].tolist())
        goal_xy = tuple(self.trials["goal_xy"][idx].tolist())

        if self.set_full_states:
            start_qpos = self.trials["proprio_pos"][idx][0].numpy().astype("float64")
            start_qvel = self.trials["proprio_vel"][idx][0].numpy().astype("float64")

            goal_qpos = self.trials["proprio_pos"][idx][T].numpy().astype("float64")
            goal_qvel = self.trials["proprio_vel"][idx][T].numpy().astype("float64")

            env_options = {
                "reset_exact": True,
                "start_qpos": start_qpos,
                "start_qvel": start_qvel,
                "goal_qpos": goal_qpos,
                "goal_qvel": goal_qvel,
                "render_goal": env.unwrapped._ob_type == "pixels",
            }
        else:
            env_options = {
                "init_xy": init_xy,
                "goal_xy": goal_xy,
                "render_goal": env.unwrapped._ob_type == "pixels",
            }

        _, info = env.reset(seed=self.seed + idx, options=env_options)

        return env

    def _make_ogbench_envs(self):
        # we know ogbench has 5 tasks. make sure they are evenly distributed across envs
        envs = []

        for i in range(self.n_envs):
            env = gymnasium.make(
                self.env_name,
                terminate_at_goal=False,
                max_episode_steps=9999,
            )

            env = NormEvalWrapper(
                env=env,
                normalizer=self.normalizer,
                stack_states=self.stack_states,
            )

            num_tasks = len(env.unwrapped.task_infos)

            task_id = i % num_tasks + 1
            env_options = {"task_id": task_id}
            _, info = env.reset(seed=self.seed + i, options=env_options)
            envs.append(env)

        return envs

    def _make_envs_from_offline_data(self):
        envs = []
        if self.trials_path:
            self.trials = torch.load(self.trials_path)
            assert self.trials["init_xy"].shape[0] >= self.n_envs
        elif self.data_path:
            self.trials = self._create_trials_from_offline_data()
            assert self.trials["init_xy"].shape[0] >= self.n_envs

            """
            EITHER has following structure:
            {
                states: (BS, T, ch, w, h),
                actions: (BS, A, 1, a_dim),
                proprio_pos: (BS, T, p_dim),
                proprio_vel: (BS, T, v_dim),
                top_down_view_states: (BS, T, ch, w, h)
            }
            OR has the following structure (ogbench format):
            {
                starts: (BS, 2),
                targets: (BS, 2),
            }
            """
        else:
            raise NotImplementedError

        for i in range(self.n_envs):
            env = self._make_env(idx=i)
            envs.append(env)

        return envs

    def _create_trials_from_offline_data(self):
        sample_length = self.offline_T * 10  # TODO: think of a better way
        num_trajs = self.n_envs

        data_config = D4RLDatasetConfig()

        ds = LocoMazeDataset(
            dataclasses.replace(
                data_config,
                env_name=self.env_name,
                batch_size=num_trajs,
                path=self.data_path,
                load_top_down_view=True,
                num_workers=1,
                sample_length=sample_length,
                stack_states=self.stack_states,
                image_based=self.image_obs,
            )
        )

        loader_config = SimpleObject(
            num_workers=0,
            quick_debug=False,
            normalize=False,
            min_max_normalize_state=False,
            normalizer_hardset=False,
        )

        ds = make_dataloader(ds=ds, loader_config=loader_config, train=False)
        datum = next(iter(ds))

        (
            states,
            locations,
            actions,
            indices,
            proprio_pos,
            proprio_vel,
            top_down_view_states,
            directions,
            chunked_locations,
            chunked_proprio_pos,
            chunked_proprio_vel,
        ) = datum

        proprio_pos_w_loc = torch.cat([locations.squeeze(-2), proprio_pos], dim=-1)

        trials = {
            "states": states,
            "actions": actions,
            "indices": indices,
            "proprio_pos": proprio_pos_w_loc,
            "proprio_vel": proprio_vel,
            "top_down_view_states": top_down_view_states,
            "init_xy": states[:, 0],
            "goal_xy": states[:, self.offline_T],
        }

        return trials

    def __call__(self):
        if self.level == "ogbench":
            envs = self._make_ogbench_envs()
        else:
            envs = self._make_envs_from_offline_data()

        return envs
