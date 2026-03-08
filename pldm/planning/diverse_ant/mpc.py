from typing import Optional

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from pldm.models.hjepa import HJEPA

from pldm.planning.utils import *
from pldm.planning.plotting import log_planning_plots, log_l1_planning_loss
from pldm.planning.d4rl.enums import D4RLMPCConfig
from pldm.planning.enums import PooledMPCResult
from pldm.planning.ogbench.locomaze.enums import MPCReport
from pldm.planning.mpc import MPCEvaluator
import gymnasium

from pldm_envs.utils.normalizer import Normalizer
from pldm_envs.ogbench.utils import PixelMapper
from pldm_envs.diverse_ant.ant_envs_generator import AntEnvsGenerator


class DiverseAntMPCEvaluator(MPCEvaluator):
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
        )

        level_cfg = getattr(config, config.level)

        envs_generator = AntEnvsGenerator(
            env_name=config.env_name,
            n_envs=config.n_envs,
            min_block_radius=level_cfg.min_block_radius,
            max_block_radius=level_cfg.max_block_radius,
            seed=config.seed,
            stack_states=config.stack_states,
            image_obs=config.image_obs,
            data_path=config.data_path,
            trials_path=config.set_start_target_path,
            unique_shortest_path=config.unique_shortest_path,
            normalizer=self.normalizer,
        )

        self.envs = envs_generator()

    def close(self):
        print(f"closing {self.prefix}")
        """Manually close environments."""
        for env in self.envs:
            if env is not None:
                env.close()

    def evaluate(self):
        mpc_data = self._perform_mpc_in_chunks()

        report = self._construct_report(mpc_data)

        log_l1_planning_loss(result=mpc_data, prefix=self.prefix)

        if mpc_data.ensemble_var_history:
            self._log_ensemble_var(
                ensemble_var_history=mpc_data.ensemble_var_history,
                plot_prefix=self.prefix,
            )

        if mpc_data.ensemble_proprio_var_history:
            self._log_ensemble_var(
                ensemble_var_history=mpc_data.ensemble_proprio_var_history,
                plot_prefix=self.prefix,
                var_type="proprio",
            )

        mpc_data.targets = mpc_data.targets[:, :2]  # only keep (pos_x, pos_y)

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
