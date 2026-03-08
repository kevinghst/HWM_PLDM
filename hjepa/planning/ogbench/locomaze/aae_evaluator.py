import torch
import gymnasium
from torch import nn
from hjepa.data.enums import DatasetType
from tqdm import tqdm
from environments.utils.normalizer import Normalizer
from hjepa.logger import Logger
from hjepa.planning.plotting import plot_aae_traj


class LocoMazeAAEEvaluator:
    def __init__(
        self,
        epoch: int,
        aae: nn.Module,
        normalizer: Normalizer,
        env_name: str,
        ds: DatasetType,
        pixel_mapper,
        quick_debug: bool = False,
    ):
        self.epoch = epoch
        self.aae = aae
        self.env_name = env_name
        self.normalizer = normalizer
        self.ds = ds
        self.pixel_mapper = pixel_mapper
        self.quick_debug = quick_debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gymnasium.make(
            env_name, terminate_at_goal=False, max_episode_steps=1001
        )

    def _roll_out_w_env(
        self,
        latents: torch.Tensor,
        loc: torch.Tensor,
        pp: torch.Tensor,
        pv: torch.Tensor,
        T: int,
    ):
        """
        Args:
            latents: (BS, LD)
            loc: (BS, 2)
            pp: (BS, 2 + pp_dim)
            pv: (BS, pv_dim)
            T: int
        Output:
            r_loc: (T, BS, 2)
            r_pp: (T, BS, pp_dim)
            r_pv: (T, BS, pv_dim)
            actions: (T, BS, AD)
        """

        BS = latents.shape[0]

        b_r_locs = []
        b_r_pps = []
        b_r_pvs = []
        b_r_actions = []

        for i in tqdm(range(BS), desc="Unrolling AAE"):
            self.env.reset()
            self.env.unwrapped.set_state(
                qpos=torch.cat([loc[i], pp[i]], dim=-1).cpu().numpy(),
                qvel=pv[i].cpu().numpy(),
            )

            r_locs = []
            r_pps = []
            r_pvs = []
            r_actions = []

            for t in range(T):
                states = torch.cat(
                    [
                        self.normalizer.normalize_proprio_pos(pp[i]),
                        self.normalizer.normalize_proprio_vel(pv[i]),
                    ],
                    dim=-1,
                )

                r_action, _, _ = self.aae.decoder.forward_single_t(
                    states=states.unsqueeze(0),
                    latents=latents[i].unsqueeze(0),
                    timestep=t,
                )
                r_action = r_action[0]

                r_action = self.normalizer.unnormalize_action(r_action)

                # for debugging
                # rand_action = self.env.action_space.sample()
                # obs, _, _, _, _ = self.env.step(rand_action)

                obs, _, _, _, _ = self.env.step(r_action.cpu().numpy())
                r_locs.append(torch.from_numpy(obs[:2]))
                r_pps.append(torch.from_numpy(obs[2:15]))
                r_pvs.append(torch.from_numpy(obs[15:]))
                r_actions.append(r_action)

            r_locs = torch.stack(r_locs)
            r_pps = torch.stack(r_pps)
            r_pvs = torch.stack(r_pvs)
            r_actions = torch.stack(r_actions)

            b_r_locs.append(r_locs)
            b_r_pps.append(r_pps)
            b_r_pvs.append(r_pvs)
            b_r_actions.append(r_actions)

        r_locs = torch.stack(b_r_locs).transpose(0, 1)
        r_pps = torch.stack(b_r_pps).transpose(0, 1)
        r_pvs = torch.stack(b_r_pvs).transpose(0, 1)
        r_actions = torch.stack(b_r_actions).transpose(0, 1)

        return r_locs, r_pps, r_pvs, r_actions

    @torch.no_grad()
    def evaluate(self):
        all_locs = []
        all_pv = []
        all_pp = []
        all_actions = []
        all_latents = []
        all_view_states = []

        all_pred_actions = []

        for idx, batch in enumerate(tqdm(self.ds, desc="Evaluating AAE")):
            # Put time first
            actions = batch.actions.to(self.device).transpose(0, 1)
            view_states = batch.view_states.transpose(0, 1)

            if self.aae.config.chunk_on_fly:
                locs = batch.locations.to(self.device).transpose(0, 1)
                pv = batch.proprio_vel.to(self.device).transpose(0, 1)
                pp = batch.proprio_pos.to(self.device).transpose(0, 1)
            else:
                locs = batch.chunked_locations.to(self.device).transpose(0, 1)
                pp = batch.chunked_proprio_pos.to(self.device).transpose(0, 1)
                pv = batch.chunked_proprio_vel.to(self.device).transpose(0, 1)

            proprio_states = torch.cat([pp, pv], dim=-1)

            aae_output = self.aae(
                actions=actions,
                proprio_states=proprio_states,
            )

            all_locs.append(self.normalizer.unnormalize_location(locs))
            all_pv.append(self.normalizer.unnormalize_proprio_vel(pv))
            all_pp.append(self.normalizer.unnormalize_proprio_pos(pp))
            all_actions.append(self.normalizer.unnormalize_action(actions))
            all_pred_actions.append(
                self.normalizer.unnormalize_action(aae_output.pred_actions)
            )
            all_latents.append(aae_output.latents)
            all_view_states.append(view_states)

        all_locs = torch.cat(all_locs, dim=1)
        all_pv = torch.cat(all_pv, dim=1)
        all_pp = torch.cat(all_pp, dim=1)
        all_actions = torch.cat(all_actions, dim=1)
        all_pred_actions = torch.cat(all_pred_actions, dim=1)
        all_latents = torch.cat(all_latents, dim=1)
        all_view_states = torch.cat(all_view_states, dim=1)

        # get val loss for pred actions
        action_loss = torch.sqrt(
            torch.sum((all_pred_actions - all_actions) ** 2, dim=-1)
        ).mean()
        # mean over batch and time
        action_loss = action_loss.mean().item()

        # unroll out in the real environment

        if self.aae.config.chunk_on_fly:
            latents = all_latents[0]
            loc = all_locs[0]
            pp = all_pp[0]
            pv = all_pv[0]
        else:
            latents = all_latents[0]
            loc = all_locs[0, :, 0]
            pp = all_pp[0, :, 0]
            pv = all_pv[0, :, 0]

        r_loc, r_pp, r_pv, r_actions = self._roll_out_w_env(
            latents=latents,
            loc=loc,
            pp=pp,
            pv=pv,
            T=actions.shape[0],
        )

        if self.aae.config.chunk_on_fly:
            loc_target = all_locs[1:]
            pp_target = all_pp[1:]
            pv_target = all_pv[1:]
            action_target = all_actions
            source_target = all_locs[0]
        else:
            loc_target = all_locs[0].transpose(0, 1)[1:]
            pp_target = all_pp[0].transpose(0, 1)[1:]
            pv_target = all_pv[0].transpose(0, 1)[1:]
            action_target = all_actions[0].transpose(0, 1)
            source_target = all_locs[0].transpose(0, 1)[0]

            r_loc = r_loc[:-1]
            r_pp = r_pp[:-1]
            r_pv = r_pv[:-1]

        # take difference between GT and unrolled
        r_loc_loss = torch.sqrt(
            torch.sum((r_loc.to(self.device) - loc_target) ** 2, dim=-1)
        ).mean(-1)
        r_pp_loss = torch.sqrt(
            torch.sum((r_pp.to(self.device) - pp_target) ** 2, dim=-1)
        ).mean(-1)
        r_pv_loss = torch.sqrt(
            torch.sum((r_pv.to(self.device) - pv_target) ** 2, dim=-1)
        ).mean(-1)
        r_action_loss = torch.sqrt(
            torch.sum((r_actions.to(self.device) - action_target) ** 2, dim=-1)
        ).mean(-1)

        logger = Logger.run()
        logger.log_across_t(
            data=r_loc_loss, name=f"aae_evaluator_epoch_{self.epoch}_/r_loc_loss"
        )
        logger.log_across_t(
            data=r_pp_loss, name=f"aae_evaluator_epoch_{self.epoch}_/r_pp_loss"
        )
        logger.log_across_t(
            data=r_pv_loss, name=f"aae_evaluator_epoch_{self.epoch}_/r_pv_loss"
        )
        logger.log_across_t(
            data=r_action_loss, name=f"aae_evaluator_epoch_{self.epoch}_/r_action_loss"
        )

        log_dict = {f"aae_evaluator_epoch_{self.epoch}_/action_loss": action_loss}
        logger.log(log_dict)

        plot_n = 5 if self.quick_debug else 64
        plot_states = all_view_states[0, :plot_n].cpu()

        gt_loc_xy = self.pixel_mapper.obs_coord_to_pixel_coord(
            loc_target[:, :plot_n].cpu()
        )
        r_loc_xy = self.pixel_mapper.obs_coord_to_pixel_coord(r_loc[:, :plot_n].cpu())

        source_xy = self.pixel_mapper.obs_coord_to_pixel_coord(
            source_target[:plot_n].cpu()
        )

        plot_aae_traj(
            states=plot_states,
            source_loc=source_xy,
            gt_loc=gt_loc_xy,
            r_loc=r_loc_xy,
        )
