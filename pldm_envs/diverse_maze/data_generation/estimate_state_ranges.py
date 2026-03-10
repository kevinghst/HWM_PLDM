import torch
from tqdm import tqdm

# data_path = '/vast/wz1232/maze2d_small_diverse/'
data_path = "/vast/wz1232/maze2d_medium_diverse_15maps/"
# data_path = "/vast/wz1232/maze2d_large_diverse_25maps/"

data = torch.load(f"{data_path}/data.p")

min_state_x = float("inf")
max_state_x = float("-inf")
min_state_y = float("inf")
max_state_y = float("-inf")


for datum in tqdm(data):
    obs = datum["observations"][:, :2]
    min_state_x = min(min_state_x, obs[:, 0].min())
    max_state_x = max(max_state_x, obs[:, 0].max())
    min_state_y = min(min_state_y, obs[:, 1].min())
    max_state_y = max(max_state_y, obs[:, 1].max())


print("State ranges:", [min_state_x, min_state_y], [max_state_x, max_state_y])

"""
medium: State ranges: [0.35307181546022387, 0.35314092641169464] [6.246377909065589, 6.246281227643531]
large: State ranges: [0.35104091093137213, 0.3509251577427265] [8.247745348017226, 8.247219015402822]
"""
