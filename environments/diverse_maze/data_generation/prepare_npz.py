"""
This script prepares the npz file for the ogbench repo
"""

import torch
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset into npz format compatible with OGbench API"
    )
    parser.add_argument("--data_path", type=str, help="Path to the data")

    args = parser.parse_args()

    data = torch.load(f"{args.data_path}/data.p")

    images = np.load(f"{args.data_path}/images.npy")

    all_obs = [x["observations"] for x in data]

    # repeat last action to match the length of observations
    all_actions = [np.append(x["actions"], [x["actions"][-1]], axis=0) for x in data]

    num_episodes = len(all_obs)
    episode_len = all_obs[0].shape[0]

    def create_terminals(num_episodes, episode_len):
        # Create a list of N numpy arrays
        tensors = [np.zeros(episode_len, dtype=int) for _ in range(num_episodes)]
        # Set the last element of each tensor to 1
        for tensor in tensors:
            tensor[-1] = 1
        return tensors

    all_terminals = create_terminals(num_episodes, episode_len)

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_terminals = np.concatenate(all_terminals, axis=0)

    data_dict = {
        "observations": images,
        "proprio_states": all_obs[:, 2:],
        "actions": all_actions,
        "terminals": all_terminals,
        "locations": all_obs[:, :2],
    }

    np.savez(f"{args.data_path}/data.npz", **data_dict)


if __name__ == "__main__":
    main()
