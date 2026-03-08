from environments.diverse_maze import ant_draw
from environments.diverse_maze.wrappers import NavigationWrapper, RenderImageWrapper

import numpy as np
import torch
import matplotlib.pyplot as plt

"""
This script generates test maps / trials that test the shortest path planning.
Maps are symmetrical. Original starts and goals are symmetrical as well.
Goals are gradually shifted so that 1 path is the shortest path and the other is not.
"""


def concat_and_save(x, y, name):
    # Normalize the tensors to [0, 1] if not already in this range
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Concatenate x and y side by side along the width dimension
    combined = torch.cat((x, y), dim=2)  # Concatenate along the width

    # Convert tensor to numpy array for plotting
    combined_np = combined.permute(1, 2, 0).cpu().numpy()  # Change shape to (H, W, C)

    # Save the image using matplotlib
    plt.imsave(f"{name}.png", combined_np)


map_layouts = [
    ["######", "#OOOO#", "#OOOO#", "#O#OO#", "#OOOO#", "######"],
    [
        "######",
        "#OOOO#",
        "#O##O#",
        "#O##O#",
        "#OOOO#",
        "######",
    ],
]

starts = [
    (4, 1),
    (4, 1),
]

goals = [
    (2, 3),
    (1, 4),
]

# represents shift leftwawrds to the block on the right
# 0 means it is at center of goal block
# 1 means it is at center of block left of goal block
# 0 < x < 1 means interpolated position
goal_shifts = [0.25, 0.5, 0.75, 1]

trials = {
    "map_layouts": [],
    "starts": [],
    "targets": [],
    "shortest_path": [],
    "shifts": [],
}

for i in range(len(map_layouts)):

    map_layout = map_layouts[i]
    map_key = "\\".join(map_layout)

    start = starts[i]
    goal = goals[i]

    env = ant_draw.load_environment(
        name="maze2d_small_diverse",
        map_key=map_key,
    )

    env = RenderImageWrapper(env, None)
    env = NavigationWrapper(env)

    env.reset()

    start_xy = env.ij_to_xy(start)
    goal_xy = env.ij_to_xy(goal)

    for goal_shift in goal_shifts:
        goal_left_ij = (goal[0], goal[1] - 1)
        goal_left_xy = env.ij_to_xy(goal_left_ij)
        shifted_goal = goal_left_xy * goal_shift + goal_xy * (1 - goal_shift)

        env.set_state(qpos=start_xy, qvel=np.array([0, 0]))
        env.set_target(shifted_goal)

        goal_obs = env.get_goal_rendered()
        obs = env.env._get_obs()

        shortest_path, _ = env.shortest_path(
            current_ij=start,
            target_ij=goal_left_ij,
        )
        shortest_path = [start] + shortest_path
        shifted_goal_ij = env.xy_to_ij(shifted_goal)
        if shortest_path[-1] != shifted_goal_ij:
            shortest_path.append(shifted_goal_ij)

        trials["map_layouts"].append(map_key)
        trials["starts"].append(start_xy)
        trials["targets"].append(shifted_goal)
        trials["shortest_path"].append(shortest_path)
        trials["shifts"].append(goal_shift)

        # concat_and_save(obs, goal_obs, f'/scratch/wz1232/HJEPA/imgs/trial_{i}_shift_{goal_shift}')


torch.save(trials, "/vast/wz1232/maze2d_small_diverse/opt_trials.pt")
