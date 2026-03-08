import torch
import random
import numpy as np
import statistics

"""
prepare the following:

trials = {
    'states': list of (T, 29,)
    'actions': list of (T-1, 8,)
    'map_layouts': list of str
}
"""


offline_data_path = "/vast/wz1232/ant_diverse/ant_medium_diverse_probe/"

data = torch.load(offline_data_path + "data.p")

maps = torch.load(offline_data_path + "train_maps.pt")

T = 25

num_trials = 40

trials = {
    "observations": [],
    "actions": [],
    "map_layouts": [],
    "xy_dist_start_goal": [],
}


idxs = random.sample(range(len(data)), num_trials)

xy_dist_min = 1.5

for idx in idxs:
    datum = data[idx]

    xy_dist = 0
    while xy_dist < xy_dist_min:

        max_start = len(datum["actions"]) - T + 1
        i = random.randint(0, max_start - 1)

        traj = datum["observations"][i : i + T]
        actions = datum["actions"][i : i + T - 1]

        xy_dist = np.linalg.norm(traj[0][:2] - traj[-1][:2])

    trials["observations"].append(traj)
    trials["actions"].append(actions)
    trials["xy_dist_start_goal"].append(xy_dist)
    trials["map_layouts"].append(maps[datum["map_idx"]])


median_xy_dist = statistics.median(trials["xy_dist_start_goal"])
print(f"median xy dist: {median_xy_dist}")
print(f"min xy dist: {min(trials['xy_dist_start_goal'])}")


torch.save(trials, offline_data_path + f"trials_{T}.pt")
