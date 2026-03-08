import argparse
from collections import defaultdict
import numpy as np


def parse_args():
    """A function to parse arguments with argparse:
    - data_paths: a list of paths to the data files
    - wc_rate: target wall crossing rate in the new dataset
    - output_path: path to save the new dataset
    """

    parser = argparse.ArgumentParser(
        description="Create a new dataset with a specified wall crossing rate"
    )
    parser.add_argument(
        "data_paths", type=str, nargs="+", help="Paths to the data files"
    )
    parser.add_argument(
        "--wc_rate", type=float, default=0.5, help="Target wall crossing rate"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Number of trajectories to sample",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="new_dataset.npz",
        help="Path to save the new dataset",
    )

    return parser.parse_args()


def get_wall_passing_percentate(locations):
    # let's calculate percentage of the wall passing episodes
    x_coords = locations[:, :, 0]
    x_coords_min = np.min(x_coords, axis=1)
    x_coords_max = np.max(x_coords, axis=1)
    wall_passing = (x_coords_min < 31) * (x_coords_max > 33)
    return wall_passing.mean(), wall_passing.sum()


def check_traj_length(data):
    terminals = data["terminals"]

    # nonzero terminals
    terminal_idxs = np.where(terminals)[0]
    traj_lengths = np.diff(np.concatenate([[-1], terminal_idxs]))
    # unique traj lengths
    traj_lengths = np.unique(traj_lengths)
    assert (
        len(traj_lengths) == 1
    ), f"All trajectories should have the same length, found {traj_lengths}"

    return traj_lengths[0]


def find_wall_passing_idxs(locations):
    x_coords = locations[:, :, 0]
    x_coords_min = np.min(x_coords, axis=1)
    x_coords_max = np.max(x_coords, axis=1)
    wall_passing = (x_coords_min < 31) * (x_coords_max > 33)

    return np.where(wall_passing)[0], np.where(~wall_passing)[0]


def main():
    args = parse_args()
    data_paths = args.data_paths
    datasets = []

    for data_path in data_paths:
        data = np.load(data_path, mmap_mode="r")
        datasets.append(data)

    # reshape locations to (N, T, 2)

    T = check_traj_length(datasets[0])
    assert check_traj_length(datasets[1]) == T, "Trajectory lengths should be the same"

    wall_passing_episodes = []
    non_wall_passing_episodes = []

    traj_nums = []

    for i, data in enumerate(datasets):
        terminals = data["terminals"]
        locations = data["locations"]

        N = len(terminals) // T
        traj_nums.append(N)
        locations = locations.reshape(N, T, -1)

        wall_passing, non_wall_passing = find_wall_passing_idxs(locations)

        wall_passing_episodes.extend(zip([i] * len(wall_passing), wall_passing))
        non_wall_passing_episodes.extend(
            zip([i] * len(non_wall_passing), non_wall_passing)
        )

    print("Found wall crossing episodes: ", len(wall_passing_episodes))
    print("Found non-wall crossing episodes: ", len(non_wall_passing_episodes))

    # we want to build a new dataset with the same number of trajectories
    if args.num_trajectories is not None:
        target_N = args.num_trajectories
    else:
        target_N = traj_nums[0]  # pick the first dataset
    wc_N = int(target_N * args.wc_rate)
    non_wc_N = target_N - wc_N

    wc_idxs = np.random.choice(
        np.arange(len(wall_passing_episodes)), wc_N, replace=False
    )
    non_wc_idxs = np.random.choice(
        np.arange(len(non_wall_passing_episodes)), non_wc_N, replace=False
    )
    wc_idxs = [wall_passing_episodes[idx] for idx in wc_idxs]
    non_wc_idxs = [non_wall_passing_episodes[idx] for idx in non_wc_idxs]

    if len(wc_idxs) == 0:
        all_idxs = non_wc_idxs
    else:
        all_idxs = np.concatenate([wc_idxs, non_wc_idxs])

    all_idxs = np.random.permutation(all_idxs)

    all_episodes = defaultdict(list)

    keys = list(datasets[0].keys())

    # view data with N by T

    new_datasets = []

    for dataset in datasets:
        new_dataset = {}
        for key in keys:
            new_dataset[key] = dataset[key].reshape(N, T, *dataset[key].shape[1:])

        new_datasets.append(new_dataset)

    for data_idx, episode_idx in all_idxs:
        for key in keys:
            all_episodes[key].append(new_datasets[data_idx][key][episode_idx])

    for key in keys:
        all_episodes[key] = np.concatenate(all_episodes[key], axis=0)

    np.savez(args.output_path, **all_episodes)


if __name__ == "__main__":
    main()
