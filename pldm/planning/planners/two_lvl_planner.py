from typing import NamedTuple, Optional
from .planner import Planner, PlanningResult
import torch


class TwoLvlPlanningResult(NamedTuple):
    level1: PlanningResult
    level2: PlanningResult


class TwoLvlPlanner:
    def __init__(
        self,
        l1_planner: Planner,
        l2_planner: Planner,
        l2_step_skip: int,
    ):
        self.l1_planner = l1_planner
        self.l2_planner = l2_planner
        self.l2_step_skip = l2_step_skip

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.l2_planner.reset_targets(targets, repr_input=repr_input)

    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        repr_input: bool = False,
        mock_l1: bool = False,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        batch_size = current_state.shape[0]

        proprio_l1 = None
        if curr_proprio_pos is not None and curr_proprio_vel is not None:
            proprio_l1 = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
        elif curr_proprio_pos is not None:
            proprio_l1 = curr_proprio_pos.cuda()
        elif curr_proprio_vel is not None:
            proprio_l1 = curr_proprio_vel.cuda()

        locations_cuda = curr_locations.cuda() if curr_locations is not None else None

        backbone_output = self.l1_planner.model.backbone(
            current_state.cuda(), proprio=proprio_l1, locations=locations_cuda
        )
        l2_result = self.l2_planner.plan(
            current_state=backbone_output,
            plan_size=plan_size,
            repr_input=True,
            # diff_loss_idx=diff_loss_idx.to(enc2.device),
        )

        if mock_l1:
            # ONLY MAKES SENSE FOR WALL DATASET
            actions_l1 = (
                l2_result.actions[:, 0]
                .detach()
                .unsqueeze(1)
                .repeat(1, self.l2_step_skip, 1)
                .to(encs2.device)
            )
            actions_l1 = actions_l1 / self.l2_step_skip * 2
            locations_l1 = torch.zeros(
                (self.l2_step_skip + 1, batch_size, 1, 2),
                device=encs2.device,
            )
        else:
            self.l1_planner.reset_targets(
                l2_result.pred_obs[1].detach(), repr_input=True
            )

            l1_result = self.l1_planner.plan(
                current_state=backbone_output,
                plan_size=self.l2_step_skip,
                repr_input=True,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
            )

        return TwoLvlPlanningResult(level1=l1_result, level2=l2_result)
