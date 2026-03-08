from typing import NamedTuple, Optional
from dataclasses import dataclass
import random
import math
import numpy as np

import torch
from scipy.stats import truncnorm

from pldm_envs.wall.data.single import DotDataset, DotDatasetConfig
from pldm_envs.wall.data.wall_utils import (
    generate_wall_layouts,
    sample_uniformly_between,
    sample_truncated_norm,
)


class WallSample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]
    wall_x: torch.Tensor  # [(batch_size), 1]
    door_y: torch.Tensor  # [(batch_size), 1]
    l2_states: torch.Tensor
    l2_locations: torch.Tensor
    l2_actions: torch.Tensor


@dataclass
class WallDatasetConfig(DotDatasetConfig):
    fix_wall: bool = True
    fix_wall_batch_k: Optional[int] = None

    l2_step_skip: int = 4
    fix_wall: bool = True
    wall_padding: int = 20
    door_padding: int = 10
    wall_width: int = 3
    door_space: int = 4
    cross_wall_rate: float = 0.1
    expert_cross_wall_rate: float = 0.0
    expert_action_step_mean: float = 0.9
    expert_action_step_std: float = 0
    expert_action_lower_bd: float = 0.9
    expert_action_upper_bd: float = 0.9
    expert_traj_door_padding: float = 2
    exclude_wall_train: str = ""  # don't generate wall at these x-axis values
    exclude_door_train: str = ""  # don't generate door at these y-axis values
    only_wall_val: str = ""  # only evalaute wall at these x-axis values
    only_door_val: str = ""  # only evaluate door at these y-axis values
    fix_wall_location: Optional[int] = 32
    fix_door_location: Optional[int] = 10
    num_train_layouts: Optional[int] = -1
    image_based: bool = True


class WallDataset(DotDataset):
    def __init__(
        self,
        config: WallDatasetConfig,
        *args,
        **kwargs,
    ):
        layouts, other_layouts = generate_wall_layouts(config)
        self.layouts = layouts
        super().__init__(config, *args, **kwargs)

    def render_location(self, locations):
        states = super().render_location(locations)
        return states

    def generate_actions_to_goal(self, start, goal, eps=1e-7):
        """
        Parameters:
            start: tensor of shape (2,)
            goal: tensor of shape (2,)
        Return:
            actions: tensor of shape (n, 2)
        Description:
            generates actions towards the goal in straight line. stop short of the goal.
        """
        # Calculate the direction vector from start to goal
        direction = goal - start
        direction_norm = direction.norm()

        # Handle the case where the start and goal are the same
        if direction_norm == 0:
            return torch.empty(0, 2)  # Return an empty tensor if no movement is needed

        # Normalize the direction vector
        unit_direction = direction / direction_norm

        # List to store the actions
        actions = []

        # Current position starts at the start
        current_position = start.clone()

        # define truncated normal dist for sampling expert action
        a = (
            self.config.expert_action_lower_bd - self.config.expert_action_step_mean
        ) / (self.config.expert_action_step_std + eps)
        b = (
            self.config.expert_action_upper_bd - self.config.expert_action_step_mean
        ) / (self.config.expert_action_step_std + eps)

        truncated_normal_dist = truncnorm(
            a,
            b,
            loc=self.config.expert_action_step_mean,
            scale=self.config.expert_action_step_std,
        )

        # Generate actions until the stopping condition is met
        reached_goal = False
        while direction_norm > 0:
            # Sample an action magnitude
            if self.config.expert_action_lower_bd == self.config.expert_action_upper_bd:
                action_norm = self.config.expert_action_step_mean
            else:
                action_norm = truncated_normal_dist.rvs()

            # If the action norm is greater than the remaining distance to the goal, adjust it
            if action_norm > direction_norm:
                action_norm = direction_norm
                reached_goal = True

            # Create the action
            action = unit_direction * action_norm
            actions.append(action)

            # Update current position and calculate new direction
            current_position += action
            direction = goal - current_position
            direction_norm = direction.norm()

            if reached_goal:
                break

        if actions[-1].norm() < self.config.action_lower_bd:
            actions.pop()

        # Stack the list of actions into a tensor
        if actions:
            actions_tensor = torch.stack(actions)
        else:
            actions_tensor = torch.empty(0, 2).to(start.device)

        return actions_tensor

    def calculate_expert_traj_door_padding(self):
        seg_length = self.config.l2_step_skip * self.config.expert_action_step_mean

        # use below for calculating expert_traj_door_padding
        def find_y_at_x(a, b, c, d, x_value):
            if c == a:  # Check if line is vertical
                return (
                    "Line is vertical; no unique intersection with vertical line at x="
                    + str(x_value)
                )
            else:
                m = (d - b) / (c - a)  # Calculate the slope
                y_at_x = b + m * (
                    x_value - a
                )  # Calculate the y-value at given x_value using point (a, b)
                return y_at_x

        max_y_above_seg = 0
        for x in range(0, int(seg_length / 2) + 1):
            a = -seg_length / 2
            b = seg_length / 2 - x
            c = x
            d = 0
            x_value = -(self.config.wall_width // 2)
            max_y_above_seg = max(
                max_y_above_seg, find_y_at_x(a, b, c, d, x_value=x_value)
            )

        # we need to make sure segment is this much space away from door edge
        return max_y_above_seg

    def generate_cross_wall_points(self, wall_locs, action_padding=0):
        """
        Parameters:
            wall_locs (bs)
        Output:
            starts (bs, 2)
            goal (bs, 2)
        Description:
            Generate left and right points on opposite sides of the wall
        """
        bs = wall_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01

        left_x = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            left_wall_locs - action_padding,
        )

        right_x = sample_uniformly_between(
            right_wall_locs + action_padding,
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        left_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        right_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        left_pos = torch.stack([left_x, left_y]).transpose(0, 1)
        right_pos = torch.stack([right_x, right_y]).transpose(0, 1)

        return left_pos, right_pos

    def generate_expert_cross_wall_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        n_steps=17,
    ):
        """
        Parameters:
            wall_locs (bs)
            door_locs (bs)
            n_steps: int
        Output:
            location (bs, 2)
            actions (bs, n_steps-1, 2)
            valid_trajs_idxs list of valid indices
        Description:
            Generate expert trajectory consisting of 3 segments:
            From point A to left of door
            from left to door to right of door
            from right of door to point B on other side
        """
        bs = wall_locs.size(0)

        door_top = door_locs + self.config.door_space
        door_bot = door_locs - self.config.door_space

        # define middle segment left and right points
        seg_length = self.config.l2_step_skip * self.config.expert_action_step_mean
        seg_left_x = wall_locs - seg_length / 2
        seg_right_x = wall_locs + seg_length / 2

        seg_top_y = door_top - self.config.expert_traj_door_padding
        seg_bot_y = door_bot + self.config.expert_traj_door_padding

        seg_y = sample_uniformly_between(seg_bot_y, seg_top_y)

        seg_left = torch.stack((seg_left_x, seg_y), dim=1)
        seg_right = torch.stack((seg_right_x, seg_y), dim=1)

        # generate start and goal
        start_pos, goal_pos = self.generate_cross_wall_points(wall_locs)

        # invert 50% the direction.
        half_bs = bs // 2
        start_pos[half_bs:], goal_pos[half_bs:] = (
            goal_pos[half_bs:],
            start_pos[half_bs:].clone(),
        )
        seg_left[half_bs:], seg_right[half_bs:] = (
            seg_right[half_bs:],
            seg_left[half_bs:].clone(),
        )

        """
        1) We start from start, take steps move towards seg_left, stop as soon as it pasts it
        2) we move horizontally right until it move pasts by seg_right[0]
        3) we move to end, stop right before passing it
        
        Exceptions: 
        if start was already within "middle box", we skip 1)
        if end is already within "middle box", we skip 2)
        """

        def in_middle(x, y, left, right, top, bot):
            if left <= x and x <= right and bot <= y and y <= top:
                return True
            return False

        expert_actions = torch.zeros((bs, self.config.traj_n_steps - 1, 2)).to(
            wall_locs.device
        )

        for i in range(bs):
            curr_pos = start_pos[i]

            actions = []
            # 1)
            if in_middle(
                x=start_pos[i][0],
                y=start_pos[i][1],
                left=seg_left[i][0],
                right=seg_right[i][0],
                top=seg_top_y[i],
                bot=seg_bot_y[i],
            ):
                # we make goal in step 2) parallel to start position
                seg_right[i][1] = curr_pos[1]
            else:
                actions_to_left_seg = self.generate_actions_to_goal(
                    curr_pos, seg_left[i]
                )
                actions.append(actions_to_left_seg)
                curr_pos = seg_left[i]

            # 2)
            if not in_middle(
                x=goal_pos[i][0],
                y=goal_pos[i][1],
                left=seg_left[i][0],
                right=seg_right[i][0],
                top=seg_top_y[i],
                bot=seg_bot_y[i],
            ):
                # we travel to right segment
                actions_to_right_seg = self.generate_actions_to_goal(
                    curr_pos, seg_right[i]
                )
                actions.append(actions_to_right_seg)
                curr_pos = seg_right[i]

            # 3)
            actions_to_goal = self.generate_actions_to_goal(curr_pos, goal_pos[i])
            actions.append(actions_to_goal)

            actions = torch.cat(actions)
            expert_actions[i][: actions.shape[0]] = actions

        valid_trajs_idxs = torch.tensor(list(range(0, bs))).to(wall_locs.device)

        return start_pos, expert_actions, valid_trajs_idxs

    def generate_cross_wall_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        n_steps=17,
    ):
        """
        Parameters:
            wall_locs (bs)
            door_locs (bs)
            actions (bs)
            n_steps: int
        Output:
            location (bs, 2)
            actions (bs, n_steps-1, 2)
            bias_angle (bs, 2)
        """
        bs = door_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        # sample point at door
        x = sample_uniformly_between(
            left_wall_locs,
            right_wall_locs,
        )

        # we define a truncated normal distribution to sample y centered at the door center
        y = sample_truncated_norm(
            upper_bound=door_locs + self.config.door_space,
            lower_bound=door_locs - self.config.door_space,
            mean=door_locs,
        ).to(door_locs.device)

        # y = sample_uniformly_between(
        #     door_locs - self.config.door_space,
        #     door_locs + self.config.door_space
        # )
        loc_at_door = torch.stack([x, y]).transpose(0, 1)
        # sample the step which this point refers to
        step_idxs = torch.randint(1, n_steps, size=x.shape)

        # determine the angles for points above
        angles = torch.empty(bs)

        # Sample angles pointing left
        for i in range(bs):
            angles[i] = torch.pi + (torch.rand(1) - 0.5) * torch.pi / 2

        angles = self.angle_to_vec(angles).to(self.device)
        actions_dir_left, _ = self.generate_actions(n_steps, bias_angle=angles)
        actions_dir_right, _ = self.generate_actions(n_steps, bias_angle=-1 * angles)

        cw_actions = torch.zeros((bs, n_steps - 1, 2))
        cw_start_loc = torch.zeros((bs, 2))

        for i in range(bs):
            step = step_idxs[i]

            # right pointing trajectory
            traj = torch.cat(
                [
                    torch.flip(actions_dir_left[i][:step], dims=[0]) * -1,
                    actions_dir_right[i][1 : n_steps - step],
                ]
            )
            step_sum_before_door = traj[:step].sum(dim=0)

            if random.random() < 0.5:
                # turn it into left pointing trajectory
                traj = torch.flip(traj, dims=[0]) * -1
                step_sum_before_door = traj[: n_steps - step].sum(dim=0)

            cw_actions[i] = traj

            # calcualte start position such that action at given step will reach loc at door
            cw_start_loc[i] = loc_at_door[i] - step_sum_before_door

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01
        cw_start_loc = torch.clamp(cw_start_loc, min=min_val, max=max_val)

        return cw_start_loc, cw_actions, torch.zeros_like(cw_actions)

    def generate_state_and_actions(
        self, wall_locs=None, door_locs=None, size=None, n_steps=17
    ):
        location, actions, bias_angle = super().generate_state_and_actions(
            wall_locs=wall_locs, door_locs=door_locs, size=size, n_steps=n_steps
        )

        # select a proportion of these and make sure they are cross wall samples
        if self.config.cross_wall_rate:
            cw_count = math.ceil(self.config.batch_size * self.config.cross_wall_rate)
            (cw_locations, cw_actions, _) = self.generate_cross_wall_state_and_actions(
                wall_locs=wall_locs[:cw_count],
                door_locs=door_locs[:cw_count],
                n_steps=n_steps,
            )
            location[:cw_count] = cw_locations
            actions[:cw_count] = cw_actions

        # select a proportion of these and make sure they are expert cross wall samples
        if self.config.expert_cross_wall_rate:
            (
                ecw_locations,
                ecw_actions,
                valid_traj_idxs,
            ) = self.generate_expert_cross_wall_state_and_actions(
                wall_locs=wall_locs,
                door_locs=door_locs,
                n_steps=n_steps,
            )
            max_ecw_count = math.ceil(
                self.config.batch_size * self.config.expert_cross_wall_rate
            )
            valid_traj_idxs = valid_traj_idxs[:max_ecw_count]
            if valid_traj_idxs.shape[0]:
                location[valid_traj_idxs] = ecw_locations[valid_traj_idxs]
                actions[valid_traj_idxs] = ecw_actions[valid_traj_idxs]

        return location, actions, bias_angle

    def check_wall_intersection(self, current_location, next_location, walls):
        """
        Args:
            current_location (bs, 2):
            next_location (bs, 2):
            walls (bs): x coordinate of walls
        """

        # Calculate the half width to determine the range of the wall
        half_width = self.config.wall_width // 2

        # Calculate the left and right boundaries of the walls
        wall_left = walls - half_width
        wall_right = walls + half_width

        # Determine the relative positions of the current and next locations to the wall's boundaries
        # Check if the x-coordinates are less than the right boundary of the wall
        current_right = current_location[:, 0] <= wall_right
        next_right = next_location[:, 0] <= wall_right

        # Check if the x-coordinates are more than the left boundary of the wall
        current_left = current_location[:, 0] >= wall_left
        next_left = next_location[:, 0] >= wall_left

        # Evaluate intersection conditions:
        # Case 1: One point is inside the wall boundaries and the other is outside
        # Case 2: Both points are outside the wall boundaries but on opposite sides of the wall
        inside_wall = (current_right & current_left) != (next_right & next_left)
        across_wall = (current_right != next_right) & (current_left != next_left)

        check_wall_intersection = inside_wall | across_wall

        return check_wall_intersection

    def check_pass_through_door(
        self, current_location, next_location, wall_loc, door_loc
    ):
        """
        Args:
            current_location (2,):
            next_location (2,):
            wall_loc (1,):
            door_loc (1,):
        Summary:
            By this point we assume the path intersects a wall
            This function finds out whether if the intersection happens at the door
        """
        half_width = self.config.wall_width // 2

        # Calculate intersection points with the left and right boundaries of the wall
        left_wall = wall_loc - half_width
        right_wall = wall_loc + half_width

        # Get the displacement vector
        d = next_location - current_location

        # Calculate the slope (a) and intercept (b) of the line
        a = d[1] / d[0]
        b = current_location[1] - a * current_location[0]

        # if path intersects left wall
        if (
            torch.sign(left_wall - current_location[0])
            * torch.sign(left_wall - next_location[0])
            < 0
        ):
            # calcualte y coordinate of intersection point with left wall
            y_left = a * left_wall + b
            pass_left_wall = (
                door_loc - self.config.door_space
                <= y_left
                <= door_loc + self.config.door_space
            )
        else:
            pass_left_wall = True

        # if path intersects right wall
        if (
            torch.sign(right_wall - current_location[0])
            * torch.sign(right_wall - next_location[0])
            < 0
        ):
            # calculate y coordinate of intersection point with right wall
            y_right = a * right_wall + b
            pass_right_wall = (
                door_loc - self.config.door_space
                <= y_right
                <= door_loc + self.config.door_space
            )
        else:
            pass_right_wall = True

        return pass_left_wall and pass_right_wall

    @staticmethod
    def segments_intersect(A, B):
        """
        Input:
            A: (bs, 2, 2)
            B: (bs, 2, 2)
        Summary:
            Test whether if the segment from A[i][0] to A[i][1] intersects
            the segment from B[i][0] to B[i][1]
        """
        # Extract points
        A0, A1 = A[:, 0], A[:, 1]  # Endpoints of segment A
        B0, B1 = B[:, 0], B[:, 1]  # Endpoints of segment B

        # Direction vectors
        dA = A1 - A0  # Direction vector of segment A
        dB = B1 - B0  # Direction vector of segment B

        # Helper function to compute cross product of 2D vectors
        def cross_2d(v, w):
            return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

        # Translate points to origin based on one endpoint of each segment
        # Check orientation of other segment's endpoints relative to this segment
        B0_to_A0 = B0 - A0
        B1_to_A0 = B1 - A0
        A0_to_B0 = A0 - B0
        A1_to_B0 = A1 - B0

        # Cross products to determine the relative positions
        cross_A_B0 = cross_2d(dA, B0_to_A0)
        cross_A_B1 = cross_2d(dA, B1_to_A0)
        cross_B_A0 = cross_2d(dB, A0_to_B0)
        cross_B_A1 = cross_2d(dB, A1_to_B0)

        # Intersection condition: opposite signs of cross products indicate the points are on opposite sides
        intersect_A = cross_A_B0 * cross_A_B1 < 0  # B endpoints on opposite sides of A
        intersect_B = cross_B_A0 * cross_B_A1 < 0  # A endpoints on opposite sides of B

        # Combine conditions for full intersection test
        # Use logical AND: both conditions must be true for segments to intersect
        intersection = intersect_A & intersect_B

        # Return results as 1 (intersect) or 0 (no intersect)
        return intersection.long()

    def check_wall_width_intersection(
        self,
        locations,
        next_locations,
        walls,
        doors,
    ):
        disp = torch.stack([locations, next_locations], dim=1)

        # check if the action is pointing upwards or downwards
        deltas = next_locations - locations
        upwards = deltas[:, 1] > 0
        downwards = deltas[:, 1] < 0

        left_wall = walls - self.config.wall_width // 2
        right_wall = walls + self.config.wall_width // 2

        door_bot = doors - self.config.door_space
        door_top = doors + self.config.door_space

        top_left = torch.stack([left_wall, door_top], dim=1)
        top_right = torch.stack([right_wall, door_top], dim=1)

        bot_left = torch.stack([left_wall, door_bot], dim=1)
        bot_right = torch.stack([right_wall, door_bot], dim=1)

        top_seg = torch.stack([top_left, top_right], dim=1)
        bot_seg = torch.stack([bot_left, bot_right], dim=1)

        top_intersect = self.segments_intersect(disp, top_seg)
        bot_intersect = self.segments_intersect(disp, bot_seg)

        output = (top_intersect & upwards) | (bot_intersect & downwards)

        return output

    def generate_transitions(
        self,
        location,
        actions,
        bias_angle,
        walls,
    ):
        """
        Parameters:
            location: [bs, 2]
            actions: [bs, n_steps-1, 2]
            bias_angle: [bs, 2]
            walls: tuple([bs], [bs])
        """
        # print("walls", walls)
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            # print("next_location", next_location)

            left_border = torch.zeros_like(walls[0])
            left_border[:] = self.config.border_wall_loc - 1
            right_border = torch.zeros_like(walls[0])
            right_border[:] = self.config.img_size - self.config.border_wall_loc
            top_border, bot_border = left_border, right_border

            check_border_intersection = (
                (
                    (
                        torch.sign(locations[-1][:, 0] - left_border)
                        * torch.sign(next_location[:, 0] - left_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 0] - right_border)
                        * torch.sign(next_location[:, 0] - right_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - top_border)
                        * torch.sign(next_location[:, 1] - top_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - bot_border)
                        * torch.sign(next_location[:, 1] - bot_border)
                    )
                    <= 0
                )
            )

            check_wall_intersection = self.check_wall_intersection(
                locations[-1], next_location, walls[0]
            )
            # print("check_intersection", check_intersection.shape)
            # print("next_location", next_location.shape)

            check_wall_width_intersection = self.check_wall_width_intersection(
                locations=locations[-1],
                next_locations=next_location,
                walls=walls[0],
                doors=walls[1],
            )

            check_intersection = (
                check_border_intersection
                | check_wall_intersection
                | check_wall_width_intersection
            )

            for j in check_intersection.nonzero():
                if check_border_intersection[j] or check_wall_width_intersection[j]:
                    next_location[j] = locations[-1][j].clone()
                else:
                    if not self.check_pass_through_door(
                        current_location=locations[-1][j][0],
                        next_location=next_location[j][0],
                        wall_loc=walls[0][j],
                        door_loc=walls[1][j],
                    ):
                        next_location[j] = locations[-1][j].clone()

            locations.append(next_location)

        # Unsqueeze for compatibility with multi-dot dataset
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        walls = self.render_walls(*walls).unsqueeze(1).unsqueeze(1)
        walls = walls.repeat(1, states.shape[1], 1, 1, 1)
        # print('walls are', walls.shape)
        # print('states are', states.shape)
        # walls *= states.max()
        states_with_walls = torch.cat([states, walls], dim=-3)
        # print('states with walls are', states_with_walls.shape)

        if self.config.traj_n_steps_reduce_factor > 1:
            states_with_walls = states_with_walls[
                :, :: self.config.traj_n_steps_reduce_factor
            ]
            locations = locations[:, :: self.config.traj_n_steps_reduce_factor]
            reduced_chunks = actions.shape[1] // self.config.traj_n_steps_reduce_factor
            action_chunks = torch.chunk(actions, chunks=reduced_chunks, dim=1)
            actions = torch.cat(
                [torch.sum(chunk, dim=1, keepdim=True) for chunk in action_chunks],
                dim=1,
            )

        # calculate what proportion of samples crossed walls
        # cross_wall_counts = 0
        # for i, wall_x in enumerate(walls_x):
        #     start = locations[i][0][0][0]
        #     end = locations[i][-1][0][0]
        #     if torch.sign(start-wall_x) != torch.sign(end-wall_x):
        #         cross_wall_counts += 1
        # print(cross_wall_counts/walls_x.shape[0])

        # get rid of agent dimensions
        locations = locations.squeeze(2)
        actions = actions.squeeze(2)

        if self.config.return_l2:
            l2_states = states_with_walls[:, :: self.config.l2_step_skip]
            l2_locations = locations[:, :: self.config.l2_step_skip]

            if self.config.chunked_actions:
                # (bs, t, 2) --> (bs, t//skip, skip, 2)
                l2_actions = actions.reshape(
                    actions.shape[0],
                    actions.shape[1] // self.config.l2_step_skip,
                    self.config.l2_step_skip,
                    2,
                )
        else:
            l2_states, l2_locations, l2_actions = (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
            )

        if self.config.n_steps != self.config.traj_n_steps:
            if self.config.return_l2:
                raise NotImplementedError(
                    "L2 not implemented when n_steps != traj_n_steps"
                )
            start = np.random.randint(0, self.config.traj_n_steps - self.config.n_steps)
            end = start + self.config.n_steps
            states_with_walls = states_with_walls[:, start:end]
            locations = locations[:, start:end]
            actions = actions[:, start : end - 1]

        return WallSample(
            states=states_with_walls,
            locations=locations,
            actions=actions,
            bias_angle=bias_angle,
            wall_x=None,
            door_y=None,
            l2_states=l2_states,
            l2_locations=l2_locations,
            l2_actions=l2_actions,
        )

    def sample_walls(self):
        """
        Returns:
        wall_x: Tensor (bs). x coordinate of the wall
        door_y: Tensor (bs). y coordinate of the door
        """
        layout_codes = list(self.layouts.keys())
        if self.config.fix_wall_batch_k is not None:
            layout_codes = random.sample(layout_codes, self.config.fix_wall_batch_k)

        weights = [1] * len(layout_codes)
        sampled_codes = random.choices(
            layout_codes, weights=weights, k=self.config.batch_size
        )
        wall_locs = []
        door_locs = []
        types = []

        for code in sampled_codes:
            attr = self.layouts[code]
            wall_locs.append(attr["wall_pos"])
            door_locs.append(attr["door_pos"])
            types.append(attr["type"])

        wall_locs = torch.tensor(wall_locs, device=self.device)
        door_locs = torch.tensor(door_locs, device=self.device)
        return (wall_locs, door_locs)

    def render_walls(self, wall_locs, hole_locs):
        """
        Params:
            wall_locs: torch tensor size (batch_size,)
                holds x coordinates of walls for each batch index
            hole_locs: torch tensor size (batch_size,)
                holds y coordinates of doors for each batch index
        """
        x = torch.arange(0, self.config.img_size, device=self.device)
        y = torch.arange(0, self.config.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_x = grid_x.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(self.config.batch_size, 1, 1)

        wall_locs_r = wall_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )
        hole_locs_r = hole_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )

        # Calculate offsets for wall width
        offset = self.config.wall_width // 2
        wall_mask = (wall_locs_r - offset <= grid_x) & (grid_x <= wall_locs_r + offset)

        res = (
            wall_mask
            * (
                (hole_locs_r < grid_y - self.config.door_space)
                + (hole_locs_r > grid_y + self.config.door_space)
            )
        ).float()

        # set border walls
        border_wall_loc = self.config.border_wall_loc
        res[:, :, border_wall_loc - 1] = 1
        res[:, :, -border_wall_loc] = 1
        res[:, border_wall_loc - 1, :] = 1
        res[:, -border_wall_loc, :] = 1

        # to bytes

        res = (res * 255).clamp(0, 255).to(torch.uint8)

        return res
