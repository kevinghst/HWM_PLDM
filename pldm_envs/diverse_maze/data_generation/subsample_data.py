"""
prepare a smaller subset of the original data and save it.
"""

import torch
import numpy as np
import os

# SET THIS
reduce_factor = 2
data_path = "/vast/wz1232/maze2d_small_diverse"
save_path = "/vast/wz1232/maze2d_small_diverse_40_maps"
reduce_type = "n_maps"  # n_episodes or n_maps
assert reduce_type in ["n_episodes", "n_maps"]

if not os.path.exists(save_path):
    os.makedirs(save_path)

metadata = torch.load(f"{data_path}/metadata.pt")
data = torch.load(f"{data_path}/data.p")

images = np.load(f"{data_path}/images.npy", mmap_mode="r")

n_episodes = metadata["n_episodes"]
train_maps_n = metadata["train_maps_n"]
episode_length = metadata["episode_length"]

if reduce_type == "n_episodes":
    new_n_episodes = n_episodes // reduce_factor
    new_n_maps = train_maps_n
else:
    new_n_episodes = n_episodes
    new_n_maps = train_maps_n // reduce_factor

# reduce data
p_data = [data[i : i + n_episodes] for i in range(0, len(data), n_episodes)]
if reduce_type == "n_episodes":
    new_data = [x[:new_n_episodes] for x in p_data]
else:
    new_data = p_data[:new_n_maps]

new_data = np.concatenate(new_data, axis=0)


# reduce images
p_images = images.reshape(
    train_maps_n, n_episodes, episode_length + 1, *metadata["img_size"]
)
if reduce_type == "n_episodes":
    new_images = p_images[:, :new_n_episodes]
else:
    new_images = p_images[:new_n_maps]

new_images = new_images.reshape(
    new_n_maps * new_n_episodes * (episode_length + 1), *metadata["img_size"]
)


# save data
torch.save(new_data, f"{save_path}/data.p")
print(f"length of new data: {len(new_data)}")

np.save(f"{save_path}/images.npy", new_images)
print(f"shape of new image: {new_images.shape}")

metadata["n_episodes"] = new_n_episodes
metadata["train_maps_n"] = new_n_maps
torch.save(metadata, f"{save_path}/metadata.pt")
