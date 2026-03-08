import torch
import numpy as np

from tqdm import tqdm
import argparse

from pldm_envs.diverse_maze.utils import sample_nearby_grid_location_v2
from pldm_envs.diverse_maze.maze_draw import NavigationWrapper
from pldm_envs.diverse_maze import ant_draw
from pldm_envs.diverse_maze.data_generation.maze_stats import RENDER_STATS
from pldm_envs.diverse_maze.data_generation.map_generator import MapGenerator
import editdistance


def generate_trials_for_maps(maps: dict, trials_per_map: int, env_name: str):
    trials = {
        "starts": [],
        "targets": [],
        "map_layouts": [],
        "block_dists": [],
        "ood_distance": [],
        "mode": [],
    }

    max_attempts = 10
    min_block_radius = 5

    for idx, attr in tqdm(maps.items()):
        layout = attr["layout"]

        env = ant_draw.load_environment(
            name=env_name,
            map_key=layout,
        )

        env.reset()

        env = NavigationWrapper(env)

        for i in range(trials_per_map):
            block_dist = 1
            attempts = 0
            while block_dist < min_block_radius:
                start_xy = env.sample_xy()

                obs_coord, block_dist, _, _ = sample_nearby_grid_location_v2(
                    anchor=start_xy,
                    map_key=layout,
                    min_block_radius=min_block_radius,
                    max_block_radius=99999999,
                    obs_range_total=RENDER_STATS[env_name]["obs_range_total"],
                    obs_min_total=RENDER_STATS[env_name]["obs_min_total"],
                    unique_shortest_path=False,
                )
                attempts += 1
                if attempts > max_attempts:
                    break

            trials["starts"].append(start_xy)
            trials["targets"].append(obs_coord)
            trials["map_layouts"].append(layout)
            trials["block_dists"].append(block_dist)
            trials["ood_distance"].append(attr["ood_distance"])
            if "mode" not in attr:
                trials["mode"].append("train")
            else:
                trials["mode"].append(attr["mode"])

    return trials


def calc_ed(test_map, train_maps: list, mode="avg"):
    """
    test_map: string
    train_maps: list of strings
    Description: Calculate the average edit distance between test_map and train_maps
    """
    ed_dists = []

    for train_map in train_maps:
        ed = editdistance.eval(test_map, train_map)
        ed_dists.append(ed)

    if mode == "avg":
        return np.mean(ed_dists)
    else:
        return np.min(ed_dists)


def generate_ood_maps(train_maps: list, metadata: dict, maps_per_ood_value: int):
    test_maps_n = 1000

    map_generator = MapGenerator(
        width=metadata["num_blocks_width_in_img"] - 2,
        height=metadata["num_blocks_width_in_img"] - 2,
        num_maps=test_maps_n,
        sparsity_low=metadata.get("sparsity_low", 53),
        sparsity_high=metadata.get("sparsity_high", 88),
        max_path_len=metadata.get("max_path_len", 13),
        # exclude_map_path=config.exclude_map_path,
        wall_coords=metadata.get("wall_coords", []),
        space_coords=metadata.get("space_coords", []),
    )

    test_maps = map_generator.generate_diverse_maps()

    min_eds = {}

    for _, test_map in test_maps.items():

        min_ed = calc_ed(test_map, train_maps, mode="min")

        if min_ed in min_eds:
            min_eds[min_ed].append(test_map)
        else:
            min_eds[min_ed] = [test_map]

    min_eds = {
        key: value[:maps_per_ood_value]  # Take the first max_length elements
        for key, value in min_eds.items()
        if len(value) >= maps_per_ood_value  # Keep only lists with length >= min_length
    }

    pooled_maps = {}

    counter = 0

    for min_ed, maps in min_eds.items():
        for map in maps:
            pooled_maps[counter] = {
                "layout": map,
                "ood_distance": min_ed,
                "mode": "min",
            }
            counter += 1

    return pooled_maps


def main():
    """
    Given some training maps, generate new maps with varying OOD distance. Then generate trials for these OOD maps.
    """

    parser = argparse.ArgumentParser(description="Generate trials for the maps")
    parser.add_argument("--data_path", type=str, help="Path to the train data")
    parser.add_argument("--maps_per_ood_value", type=int, default=5)
    parser.add_argument("--trials_per_map", type=int, default=4)

    args = parser.parse_args()

    metadata = torch.load(f"{args.data_path}/metadata.pt")
    env_name = metadata["env"]

    train_maps = torch.load(f"{args.data_path}/train_maps.pt")

    ood_maps = generate_ood_maps(
        list(train_maps.values()),
        metadata,
        maps_per_ood_value=args.maps_per_ood_value,
    )

    trials = generate_trials_for_maps(
        ood_maps,
        trials_per_map=args.trials_per_map,
        env_name=env_name,
    )

    torch.save(ood_maps, f"{args.data_path}/ood_maps.pt")
    torch.save(trials, f"{args.data_path}/ood_trials.pt")


if __name__ == "__main__":
    main()
