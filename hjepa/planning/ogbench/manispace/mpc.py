from typing import Optional, NamedTuple
import time
import re

import torch
import gymnasium
import ogbench.manipspace  # noqa : registers the environment
import wandb
import numpy as np

from hjepa.models.jepa import JEPA

from hjepa.planning.enums import PooledMPCResult, MPCConfig
from hjepa.planning.mpc import MPCEvaluator

from environments.utils.normalizer import Normalizer
from environments.ogbench.utils import PixelMapper


def reset_arm(env, qpos, qvel, options={}):
    _, og_info = env.reset(options=options)
    env.unwrapped.set_state(qpos, qvel)
    obs, _, _, _, info = env.step(np.zeros(env.action_space.shape))

    for key in og_info:
        if key not in info:
            info[key] = og_info[key]

    return obs, info


def reset_arm_holding_cube(env, options={}):
    qpos = np.array(
        [
            -1.90516281e00,
            -1.90617371e00,
            1.95583856e00,
            -1.62234735e00,
            -1.57091224e00,
            2.28881359e-01,
            4.50547516e-01,
            3.47839174e-04,
            4.47680086e-01,
            -4.41072851e-01,
            4.50569421e-01,
            4.81896597e-04,
            4.47749972e-01,
            -4.42742437e-01,
            3.75314951e-01,
            1.18552195e-02,
            2.90179759e-01,
            -9.60629821e-01,
            4.91811545e-04,
            5.16511826e-03,
            2.77783036e-01,
        ]
    )
    qvel = np.array(
        [
            -4.1451257e-01,
            -2.7817622e-01,
            7.8886837e-01,
            -7.1419871e-01,
            -1.1793452e-03,
            -1.0085071e00,
            -1.9280093e-03,
            1.1752951e-03,
            -2.0191273e-04,
            -6.2580048e-03,
            -2.2577052e-03,
            -5.9142360e-04,
            -2.2483133e-03,
            9.1662025e-03,
            -6.9153324e-02,
            -1.3006519e-01,
            -2.1455431e-01,
            3.1601492e-02,
            -1.5472466e-01,
            5.8919394e-01,
        ],
    )
    return reset_arm(env, qpos, qvel, options=options)


class ManispaceMPCReport(NamedTuple):
    planning_time: float
    success_rate: float
    episode_success: torch.Tensor
    video: Optional[wandb.Video] = None

    def build_log_dict(self, prefix: str = ""):
        result = {
            f"{prefix}_success_rate": self.success_rate,
            f"{prefix}_planning_time": self.planning_time,
        }
        if self.video is not None:
            result[f"{prefix}_video"] = self.video
        return result


class PlanningWrapper(gymnasium.Wrapper):
    def __init__(self, env, normalizer, level, visualize=False):
        super().__init__(env)
        self._target = None
        self._obs = None
        self._full_obs = None
        self._info = None
        self.normalizer = normalizer
        self.level = level
        self.visualize = visualize
        self.image_based = env.unwrapped._ob_type == "pixels"

    def _adjust_obs(self, obs):
        if self.image_based:
            obs = self.normalizer.normalize_state(
                torch.from_numpy(obs).permute(2, 0, 1)
            ).contiguous()
        else:
            obs = self.normalizer.normalize_state(torch.from_numpy(obs[19:]))

        return obs

    def _update_obs_and_target(self, obs, info):
        if "goal" in info:
            self._target = self._adjust_obs(info["goal"])
        self._full_obs = obs
        self._obs = self._adjust_obs(obs.copy())
        self._info = {k: v for k, v in info.items() if "goal" not in k}

    def reset(self, qpos=None, qvel=None, options={}):
        self._target = None
        self._obs = None
        self._info = None
        self._target_visual = None

        if qpos is not None and qvel is not None:
            options = {"render_goal": self.visualize}
            obs, info = reset_arm(
                self.env, qpos, qvel, options
            )  # Not sure if need to render goal here

        elif self.level.startswith("ogbench"):
            # level can be just "ogbench" or "ogbench_<task_id>"
            # task id is integer
            regex = re.compile(r"ogbench(?:_(\d+))?")
            match = regex.match(self.level)
            if match is None:
                raise ValueError(f"Invalid level {self.level}")
            options = {}
            if match.group(1) is not None:
                options["task_id"] = int(match.group(1))
            if self.visualize:
                options["render_goal"] = True
            obs, info = self.env.reset(options=options)
        elif self.level.startswith("holding_cube_init"):
            options = {"render_goal": self.visualize}
            obs, info = reset_arm_holding_cube(
                self.env, options
            )  # Not sure if need to render goal here

        if self.visualize:
            self._target_visual = info["goal_rendered"].transpose((2, 0, 1))
            info["visual_obs"] = self.env.render().transpose((2, 0, 1))

        self._update_obs_and_target(obs, info)

        return self._obs, info

    def step(self, action):
        if self.level == "holding_cube_init_fix_gripper":
            action = action.copy()
            action[..., 3:] = 0  # disable gripper to not drop the cube
        obs, reward, done, truncated, info = self.env.step(action)
        self._update_obs_and_target(obs, info)
        if self.visualize:
            info["visual_obs"] = self.env.render().transpose((2, 0, 1))
        return self._obs, reward, done, truncated, info

    def get_target(self):
        return self.normalizer.unnormalize_state(self._target)

    def get_target_visual(self):
        return self._target_visual

    def get_target_obs(self):
        return self._target.float()

    def get_obs(self):
        return self._obs.float()

    def get_info(self):
        return self._info

    def get_proprio_vel(self, normalized: bool = False):
        if normalized:
            return self.normalizer.normalize_proprio_vel(
                torch.from_numpy(self._info["qvel"])
            )
        else:
            return self._info["qvel"]

    def get_proprio_pos(self, normalized: bool = False):
        if self.image_based:
            qpos = self._info["qpos"]
        else:
            qpos = self._full_obs[:19]

        qpos = torch.from_numpy(qpos)

        if normalized:
            qpos = self.normalizer.normalize_proprio_pos(qpos)

        return qpos


class OgbenchManispaceMPCEvaluator(MPCEvaluator):
    def __init__(
        self,
        config: MPCConfig,
        normalizer: Normalizer,
        jepa: JEPA,
        pixel_mapper: PixelMapper,
        prober: Optional[torch.nn.Module] = None,
        prefix: str = "",
        quick_debug: bool = False,
    ):
        super().__init__(
            config=config,
            model=jepa,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
            pixel_mapper=pixel_mapper,
            image_based="visual" in config.env_name,
        )

        self.envs = [
            PlanningWrapper(
                self._make_env(),
                self.normalizer,
                self.config.level,
                visualize=(
                    "visual" not in config.env_name and self.config.visualize_planning
                ),
            )
            for _ in range(self.config.n_envs)
        ]
        for env in self.envs:
            env.reset()

    def _make_env(self):
        return gymnasium.make(self.config.env_name)

    def _construct_planning_envs(self, n_envs: int):
        envs = [self._make_env() for _ in range(n_envs)]
        return envs

    def _construct_report(self, data: PooledMPCResult, eval_time: float):
        stacked_successes = torch.stack(data.success_history, dim=0)
        # shape is TxN_envs
        success = stacked_successes.any(dim=0)

        if self.config.visualize_planning:
            self._visualize_planning(data)

        return ManispaceMPCReport(
            planning_time=eval_time,
            success_rate=success.float().mean().item(),
            episode_success=success,
            video=(
                self._visualize_planning(data)
                if self.config.visualize_planning
                else None
            ),
        )

    def _visualize_planning(self, data: PooledMPCResult):
        observations_t = (
            torch.stack(data.observations, dim=0)
            if data.visual_observations is None
            else torch.stack(data.visual_observations, dim=0)
        )
        targets = (
            data.targets if data.visual_targets is None else data.visual_targets
        )  # shape is N_envs x C x H x W
        # observations_t is of shape T x N_envs x C x H x W
        # only show 5, and show every 3rd frame
        frame_skip = 3
        n_envs = 5
        observations_t = observations_t[::frame_skip, :n_envs]
        targets = targets[:n_envs]

        observations_t = observations_t.byte()
        targets = targets.byte()

        # concat observations and targets along width
        obs_with_target = torch.cat(
            [
                observations_t,
                targets.unsqueeze(0).repeat(observations_t.shape[0], 1, 1, 1, 1),
            ],
            dim=-1,
        )

        # obs target is now of shape T x N_envs x C x H x 2W
        # concat envs along height to get T x C x (H x N_envs) x 2W

        obs_with_target = obs_with_target.permute(
            0, 2, 1, 3, 4
        )  # T x C x N_envs x H x 2W
        obs_with_target = obs_with_target.reshape(
            obs_with_target.shape[0],  # T
            obs_with_target.shape[1],  # C
            obs_with_target.shape[2] * obs_with_target.shape[3],  # N_envs*H
            obs_with_target.shape[4],  # 2W
        )

        video = wandb.Video(
            obs_with_target.cpu().numpy(),
            fps=10 / frame_skip,
            format="gif",
        )
        return video

    def evaluate(self):
        start_time = time.time()
        mpc_data = self._perform_mpc_in_chunks()
        elapsed_time = time.time() - start_time

        report = self._construct_report(mpc_data, elapsed_time)

        return mpc_data, report
