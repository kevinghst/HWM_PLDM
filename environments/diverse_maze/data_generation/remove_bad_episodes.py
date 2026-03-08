import torch
import os

data_path = "/vast/wz1232/maze2d_medium_diverse_v2/"

data = torch.load(f"{data_path}/data.p")

assert not os.path.exists(f"{data_path}/bad_data.pt")

bad_episodes = torch.load(f"{data_path}/bad_episodes.pt")

bad_episodes = list(bad_episodes.keys())

filtered_data = [data[i] for i in range(len(data)) if i not in bad_episodes]

torch.save(data, f"{data_path}/bad_data.p")
torch.save(filtered_data, f"{data_path}/data.p")
