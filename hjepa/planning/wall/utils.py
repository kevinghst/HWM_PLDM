import torch
import random

from environments.wall.wall import DotWall


def determine_terminations(location_history, targets, error_threshold):
    """
    Return the index of location history at which the target is reached
    """
    terminations = [len(location_history)] * targets.shape[0]

    for i, target in enumerate(targets):
        j = 0
        while j < len(location_history):
            error = (
                (location_history[j][i] - target).cpu().pow(2).mean(dtype=torch.float32)
            )
            if error < error_threshold:
                terminations[i] = j
                break
            j += 1

    return terminations


def determine_proximities(observation_embeds, target_repr_l1, final_trans_norm_cutoff):
    """
    Return the first timestep in location history where l1 embed norm distance to target
    is smaller than threshold
    """

    l1_embed_dists = torch.norm(
        torch.stack(observation_embeds) - target_repr_l1.unsqueeze(0), dim=2
    ).transpose(0, 1)
    proximities = []
    for row in l1_embed_dists:
        index = torch.nonzero(row < final_trans_norm_cutoff, as_tuple=True)[0]
        index = index[0].item() if len(index) > 0 else (len(row) - 1)
        proximities.append(index)
    return proximities


def analyze_norm_angle_diff(loc1, loc2, wall_locs, door_locs):
    """
    Args:
        loc1 (bs, 2):
        loc2 (bs, 2):
        wall_locs (bs,)
        door_locs (bs,)
    Returns:
        norm_diff (bs,)
        angle_diff (bs, )
    Description:
        norm_diff[i] = the distance btw loc1[i] and loc2[i] via valid path through door
        angle_diff[i] = the diff between the angle of loc1[i] and loc2[i] w/ respect to door
    """
    if not wall_locs.shape[0]:
        return torch.tensor([]).to(loc1.device), torch.tensor([]).to(loc1.device)

    midpoints = torch.stack([wall_locs, door_locs], dim=1)
    norm_diff = torch.norm(loc1 - midpoints, dim=1) + torch.norm(
        midpoints - loc2, dim=1
    )

    vec_to_loc1 = loc1 - midpoints
    vec_to_loc2 = loc2 - midpoints

    vec_from_loc1 = -vec_to_loc1

    vec_to_loc2_norm = torch.nn.functional.normalize(vec_to_loc2, dim=1)
    vec_from_loc1_norm = torch.nn.functional.normalize(vec_from_loc1, dim=1)

    cos_theta = (vec_to_loc2_norm * vec_from_loc1_norm).sum(dim=1)

    angle_diff = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return norm_diff, angle_diff


def calculate_num_succ_plans(
    pred_locations,
    wall_locs,
    door_locs,
    targets,
    goal_threshold,
    wall_config,
    first_n_steps=20,
    door_padding=1.5,
    cross_wall_threshold=1,
):
    """_summary_

    Args:
        pred_locations (list(n_steps) of (pred_depth, bs, 2)): predicted probed coordinates
        wall_locs (bs,): wall locations
        door_locs (bs,): door locations
        goal_threshold (float): count as success if prediction and target coordinates are within this
            number of pixels within each other
        targets (bs, 2): location of target positions
        first_n_steps (int): only perform this calculation for a subset of the actions steps
        door_padding (float): we allow this much space above/below door for legitimate wall passing
        cross_wall_threshold (float): the difference in x-axis to qualify as wall crossing

    Returns:
        reach_target_plans: (n_steps, bs) number of plans that get model close enuf to goal
        succ_plans: (n_steps, bs) number of plans that get model close enuf to goal while NOT crossing walls
        Both as binary tensors
    """
    pred_locations = pred_locations[:first_n_steps]
    pred_locations = [pred.transpose(0, 1) for pred in pred_locations]

    n_steps = len(pred_locations)
    bs = pred_locations[0].shape[0]

    # calculate plans that don't go through walls

    door_top = door_locs + wall_config.door_space + door_padding
    door_bot = door_locs - wall_config.door_space - door_padding

    legal_plans = torch.zeros((n_steps, bs)).to(wall_locs.device)
    reach_target_plans = torch.zeros((n_steps, bs)).to(wall_locs.device)

    for i in range(n_steps):
        for j in range(bs):
            traj = pred_locations[i][j]

            traj_success = True
            for z in range(traj.shape[0] - 1):
                curr_pos, next_pos = traj[z], traj[z + 1]

                # if they are on different side of the wall center.
                # and at least 1 pixel apart on x-axis
                if (
                    curr_pos[0] < wall_locs[j]
                    and next_pos[0] > wall_locs[j]
                    and abs(curr_pos[0] - next_pos[0]) >= cross_wall_threshold
                    or next_pos[0] < wall_locs[j]
                    and curr_pos[0] > wall_locs[j]
                    and abs(curr_pos[0] - next_pos[0]) >= cross_wall_threshold
                ):
                    # if they don't pass through the door
                    if (
                        curr_pos[1] < door_bot[j]
                        or curr_pos[1] > door_top[j]
                        or next_pos[1] < door_bot[j]
                        or next_pos[1] > door_top[j]
                    ):
                        traj_success = False
                        break

            if traj_success:
                legal_plans[i][j] = 1

            last_pred = pred_locations[i][j][-1]
            dist = torch.norm(last_pred - targets[j])
            if dist < goal_threshold:
                reach_target_plans[i][j] = 1

    return reach_target_plans.int(), legal_plans.int()


def calculate_cross_wall_rate(
    starts,
    ends,
    targets,
    wall_locs,
    wall_config,
):
    """
    parameters:
        starts (bs, 2): locations of start positions
        ends: (bs, 2): locations of end positions
        targets: (bs, 2): locations of target positions
        wall_locs: (bs,): location of walls

    Calculates the percentage of trials where agent gets to the other side
    of the wall, given it started from a different side than the target
    """
    # TODO: this function is no longer compatible with non-fixed wall setting
    wall_padding = 1
    wall_left_pos = wall_locs - wall_padding
    wall_right_pos = wall_locs + wall_padding

    start_diff_side = (starts[:, 0] < wall_left_pos) & (
        targets[:, 0] > wall_right_pos
    ) | (starts[:, 0] > wall_right_pos) & (targets[:, 0] < wall_left_pos)
    end_same_side = (ends[:, 0] < wall_left_pos) & (targets[:, 0] < wall_left_pos) | (
        ends[:, 0] > wall_right_pos
    ) & (targets[:, 0] > wall_right_pos)
    cross_wall_successes = start_diff_side & end_same_side
    cross_wall_success_rate = (sum(cross_wall_successes) / sum(start_diff_side)).item()

    return cross_wall_success_rate


def calculate_embed_dists(
    encoder_l1, encoder_l2, coord_dists_norms, wall_config, normalizer, n_envs=100
):
    """
    This method calcualtes the embedding latent l2 distances between images with
    corresponding coordinate (x,y) l2 distances.

    Parameters:
        coord_dists_norms: List of tuples of (norm_lower_bound, norm_upper_bound)
    """
    output_dict = {}

    for x in coord_dists_norms:
        norm_lower_bd, norm_upper_bd = x

        start_ends = generate_sampled_norm_diff_start_ends(
            n_envs=n_envs,
            wall_pos=14,
            norm_lower_bd=norm_lower_bd,
            norm_upper_bd=norm_upper_bd,
        ).to("cuda")

        # generate env based on start_ends coords
        envs = [
            DotWall(fix_wall=True, wall_config=wall_config)
            for _ in range(start_ends.shape[0])
        ]
        start_obs = torch.stack(
            [
                env.reset(location=start_loc)
                for env, start_loc in zip(envs, start_ends[:, 0])
            ]
        )
        targets = start_ends[:, 1]

        target_obs = torch.stack(
            [env._render_dot_and_wall_target(targets[i]) for i, env in enumerate(envs)]
        )

        start_repr = encoder_l1(start_obs)
        target_repr = encoder_l1(target_obs)

        dist = torch.norm(start_repr - target_repr, dim=1)

        output_dict[f"{norm_lower_bd}_{norm_upper_bd}"] = {
            "mean_norm": torch.mean(dist).item(),
            "max_norm": torch.max(dist).item(),
            "min_norm": torch.min(dist).item(),
        }

        if encoder_l2 is not None:
            start_repr_l2 = encoder_l2(start_repr)
            target_repr_l2 = encoder_l2(target_repr)

            l2_dist = torch.norm(start_repr_l2 - target_repr_l2, dim=1)
            key = f"{norm_lower_bd}_{norm_upper_bd}"
            output_dict[key]["l2_mean_norm"] = torch.mean(l2_dist).item()
            output_dict[key]["l2_max_norm"] = torch.max(l2_dist).item()
            output_dict[key]["l2_min_norm"] = torch.min(l2_dist).item()

    # print('l1 stats:')
    # for x in coord_dists_norms:
    #     key = f'{x[0]}_{x[1]}'
    #     print(key)
    #     print(f'mean: {output_dict[key]["mean_norm"]}')
    #     print(f'max: {output_dict[key]["max_norm"]}')
    #     print(f'min: {output_dict[key]["min_norm"]}')

    # if encoder_l2 is not None:
    #     print('l2 stats:')
    #     for x in coord_dists_norms:
    #         key = f'{x[0]}_{x[1]}'
    #         print(key)
    #         print(f'mean: {output_dict[key]["l2_mean_norm"]}')
    #         print(f'max: {output_dict[key]["l2_max_norm"]}')
    #         print(f'min: {output_dict[key]["l2_min_norm"]}')

    return output_dict


def generate_sampled_norm_diff_start_ends(
    n_envs, wall_pos, norm_lower_bd, norm_upper_bd
):
    # Initialize an empty tensor with shape (n, 2, 2)
    x = torch.zeros((n_envs, 2, 2), dtype=torch.float32)

    for i in range(n_envs):
        # Sample random start coordinates
        start_x = (
            random.uniform(0, 13)
            if torch.rand(1).item() < 0.5
            else random.uniform(15, 28)
        )
        start_y = random.uniform(0, 28)

        # Sample random angle and distance for end coordinates
        angle = torch.tensor(random.uniform(0, 2 * 3.14159))
        distance = torch.tensor(random.uniform(norm_lower_bd, norm_upper_bd))

        # Calculate end coordinates based on polar coordinates
        end_x = start_x + distance * torch.cos(angle)
        end_y = start_y + distance * torch.sin(angle)

        # Ensure end coordinates are within the specified range (0 < end_x < 28 and 0 < end_y < 28)
        end_x = max(0, min(27, end_x))
        end_y = max(0, min(27, end_y))

        start_tensor = torch.tensor([start_x, start_y])
        end_tensor = torch.tensor([end_x, end_y])

        norm = torch.norm(start_tensor - end_tensor, p=2)

        # Update the tensor
        x[i][0] = start_tensor
        x[i][1] = end_tensor

    # remove those with norm < norm_lower_bd
    l2_norms = torch.norm(x[:, 0, :] - x[:, 1, :], dim=1)
    mask = l2_norms >= norm_lower_bd
    x = x[mask]

    return x
