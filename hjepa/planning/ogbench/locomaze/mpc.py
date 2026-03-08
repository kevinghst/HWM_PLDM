from typing import Optional

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from hjepa.models.hjepa import HJEPA

from hjepa.planning.utils import *
from hjepa.planning.plotting import log_planning_plots, log_l1_planning_loss
from hjepa.planning.d4rl.enums import D4RLMPCConfig
from hjepa.planning.enums import PooledMPCResult
from hjepa.planning.ogbench.locomaze.enums import MPCReport
from hjepa.planning.mpc import MPCEvaluator
import gymnasium

from environments.utils.normalizer import Normalizer
from environments.ogbench.utils import PixelMapper
from environments.ogbench.evaluation.envs_generator import EnvsGenerator


class LocoMazeMPCEvaluator(MPCEvaluator):
    def __init__(
        self,
        config: D4RLMPCConfig,
        normalizer: Normalizer,
        model: HJEPA,
        pixel_mapper: PixelMapper,
        prober: Optional[torch.nn.Module] = None,
        prefix: str = "",
        quick_debug: bool = False,
    ):
        super().__init__(
            config=config,
            model=model,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
            pixel_mapper=pixel_mapper,
            image_based="visual" in config.env_name,
        )

        envs_generator = EnvsGenerator(
            env_name=self.config.env_name,
            n_envs=self.config.n_envs,
            level=self.config.level,
            stack_states=self.config.stack_states,
            offline_T=self.config.offline_T,
            seed=self.config.seed,
            set_full_states=self.config.set_full_states,
            image_obs=self.config.image_obs,
            data_path=self.config.data_path,
            normalizer=self.normalizer,
            trials_path=config.set_start_target_path,
        )

        self.envs = envs_generator()

    def close(self):
        print(f"closing {self.prefix}")
        """Manually close environments."""
        for env in self.envs:
            if env is not None:
                env.close()

    def _render_top_down_img(self, drawer, qpos, qvel=None):
        drawer.reset()
        drawer.unwrapped.set_state(qpos=qpos, qvel=np.zeros(14))
        obs = drawer.render()
        obs = Image.fromarray(obs).resize((81, 81)).convert("L")
        transform = ToTensor()
        obs_t = transform(obs)
        return obs_t

    def evaluate(self):
        mpc_data = self._perform_mpc_in_chunks()

        report = self._construct_report(mpc_data)

        log_l1_planning_loss(result=mpc_data, prefix=self.prefix)

        if mpc_data.ensemble_var_history:
            self._log_ensemble_var(
                ensemble_var_history=mpc_data.ensemble_var_history,
                plot_prefix=self.prefix,
            )

        mpc_data.targets = mpc_data.targets[:, :2]  # only keep (pos_x, pos_y)

        # for rendering top down view
        drawer = gymnasium.make(
            self.config.env_name,
            terminate_at_goal=False,
            max_episode_steps=1001,
        )
        drawer.reset()

        top_down_obs = []

        for i, batch_qpos in enumerate(mpc_data.qpos_history):
            batch_top_down_obs = []

            if i % self.config.plot_every == 0:
                for qpos in batch_qpos:
                    top_down = self._render_top_down_img(
                        drawer, qpos=qpos.cpu().numpy()
                    )
                    batch_top_down_obs.append(top_down)
                batch_top_down_obs = torch.stack(batch_top_down_obs)

            top_down_obs.append(batch_top_down_obs)

        mpc_data.observations = top_down_obs

        drawer.close()

        # print(top_down_obs[0][0])

        if self.config.visualize_planning:
            log_planning_plots(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)),
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                xy_action=True,
                plot_every=self.config.plot_every,
                quick_debug=self.quick_debug,
                pixel_mapper=self.pixel_mapper,
                plot_failure_only=self.config.plot_failure_only,
                log_pred_dist_every=self.config.log_pred_dist_every,
                mark_action=False,
            )

        return mpc_data, report

    def _construct_report(self, data: PooledMPCResult):
        # Determine termination indices
        T = len(data.reward_history)
        B = data.reward_history[0].shape[0]

        terminations = [T] * B

        for b_i in range(B):
            for t_i in range(T):
                if data.reward_history[t_i][b_i]:
                    terminations[b_i] = t_i
                    break

        successes = [int(x < T) for x in terminations]
        success_rate = sum(successes) / len(successes)

        avg_steps_to_goal = calc_avg_steps_to_goal(data.reward_history)

        median_steps_to_goal = calc_avg_steps_to_goal(
            data.reward_history, reduce_type="median"
        )

        # proportion of blocks traveled towards goal
        start_goal_blocks_dist = np.array(
            [
                e.unwrapped.shortest_block_dist_btw_xy(
                    start_xy=e.unwrapped.init_xy,
                    goal_xy=e.get_target(),
                )
                for e in self.envs
            ]
        )

        curr_goal_blocks_dist = np.array(
            [
                e.unwrapped.shortest_block_dist_btw_xy(
                    start_xy=e.get_pos(),
                    goal_xy=e.get_target(),
                )
                for e in self.envs
            ]
        )

        blocks_traveled = np.maximum(start_goal_blocks_dist - curr_goal_blocks_dist, 0)
        proportion_traveled = blocks_traveled / start_goal_blocks_dist
        avg_proportion_traveled = np.nanmean(proportion_traveled)
        block_success_rate = np.mean(proportion_traveled == 1)

        report = MPCReport(
            success_rate=success_rate,
            block_success_rate=block_success_rate,
            success=successes,
            avg_steps_to_goal=avg_steps_to_goal,
            median_steps_to_goal=median_steps_to_goal,
            terminations=terminations,
            avg_proportion_traveled=avg_proportion_traveled,
        )

        return report
