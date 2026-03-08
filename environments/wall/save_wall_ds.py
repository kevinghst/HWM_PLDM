import yaml
from dataclasses import fields
from types import SimpleNamespace
from pathlib import Path
import argparse

from tqdm import tqdm
import torch
import numpy as np

from environments.wall.data.wall import WallDataset, WallDatasetConfig


def dict_to_namespace(d):
    """
    # Function to convert dictionary to SimpleNamespace
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)  # Recursively handle nested dictionaries
    return SimpleNamespace(**d)


def update_config_from_yaml(config_class, yaml_data):
    """
    Create an instance of `config_class` using default values, but override
    fields with those provided in `yaml_data`.
    """
    config_field_names = {f.name for f in fields(config_class)}
    relevant_yaml_data = {
        key: value for key, value in yaml_data.items() if key in config_field_names
    }
    return config_class(**relevant_yaml_data)


def get_wall_passing_percentate(locations):
    # let's calculate percentage of the wall passing episodes
    x_coords = locations[:, :, 0]
    signs = np.sign(x_coords - 32)
    wall_passing = signs[:, 0] != signs[:, -1]
    return wall_passing.float().mean().item()


def create_dataset(states, actions, locations):
    # Assuming 'states' is of shape Nx(T+1)x(*obs_dims) and 'actions' is NxTx(action_dim)
    N, T, *obs_dims = states.shape
    T -= 1  # The last state is not used as an observation
    action_dim = actions.shape[-1]

    # Prepare observations
    observations = states[:, :-1].reshape(N * T, *obs_dims)

    # Prepare next_observations
    next_observations = states[:, 1:].reshape(N * T, *obs_dims)

    # Prepare actions
    actions = actions.reshape(N * T, action_dim)

    # Prepare locations
    # we drop the last location to be consistent with the observations

    print(f"Number of transitions: {N * T}")
    print(f"Wall passing percentage: {get_wall_passing_percentate(locations):.3f}")

    locations = locations[:, :-1].reshape(N * T, 2)

    # Prepare terminals (assuming the last state in each sequence is terminal)
    terminals = np.zeros((N, T), dtype=np.float32)
    terminals[:, -1] = 1.0  # Mark the last timestep as terminal
    terminals = terminals.reshape(N * T)

    return observations, actions, locations, terminals, next_observations


def save_dataset(train_data, val_data, dataset_path):
    train_observations, train_actions, train_locations, train_terminals, _ = train_data
    val_observations, val_actions, val_locations, val_terminals, _ = val_data

    # Save the train dataset
    np.savez(
        dataset_path,
        observations=train_observations,
        actions=train_actions,
        terminals=train_terminals,
        locations=train_locations,
    )

    # Save the validation dataset
    val_path = dataset_path.replace(".npz", "-val.npz")
    np.savez(
        val_path,
        observations=val_observations,
        actions=val_actions,
        terminals=val_terminals,
        locations=val_locations,
    )


def make_dataset(ds, n_batches=1000):
    states = []
    actions = []
    locations = []

    for i in tqdm(range(n_batches), desc="Building a wall dataset"):
        b = ds.generate_multistep_sample()
        images = b.states

        shape = images.shape

        images -= images.amin(dim=(3, 4)).view(*shape[:3], 1, 1)
        images /= images.amax(dim=(3, 4)).view(*shape[:3], 1, 1)
        images = (images * 255).to(torch.uint8)

        states.append(images.cpu())
        actions.append(b.actions.cpu())
        locations.append(b.locations.squeeze().cpu())

    states = torch.cat(states, dim=0)

    # put channels last
    # states shape is (N_bathes, N_steps, C, H, W)
    states = states.permute(0, 1, 3, 4, 2)
    # states shape is (N_bathes, N_steps, H, W, C)

    actions = torch.cat(actions, dim=0)
    locations = torch.cat(locations, dim=0)

    return create_dataset(states, actions, locations)


def parse_args():
    # args are
    # config path: str
    # num_batches: int

    parser = argparse.ArgumentParser(
        description="Create a dataset from a wall environment"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--num_batches", type=int, help="Number of batches to sample", default=None
    )
    parser.add_argument(
        "--num_transitions",
        type=int,
        help="Number of transitions to sample",
        default=None,
    )

    args = parser.parse_args()
    return args


def build_name_suffix(args):
    if args.num_batches is not None:
        return f"B={args.num_batches}"
    elif args.num_transitions is not None:
        # if the number is round, return it like 3K 3M
        if args.num_transitions % 1_000_000 == 0:
            return f"{int(args.num_transitions / 1_000_000)}M"
        elif args.num_transitions % 1_000 == 0:
            return f"{int(args.num_transitions / 1_000)}K"
        else:
            return str(args.num_transitions)

    else:
        raise ValueError("Either num batches or num transitions must be provided")


def main():
    # yaml file for HJEPA experiment using random exploratory dataset
    args = parse_args()
    data_yaml_path = args.config_path
    config_name = Path(data_yaml_path).stem

    # Load the YAML configuration from a file (or string)
    with open(data_yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Create offline dataset
    data_yaml_data = yaml_data["wall_config"]
    data_config = update_config_from_yaml(WallDatasetConfig, data_yaml_data)

    ds = WallDataset(data_config)
    states, locations, actions, bias_angle, wall_x, door_y = ds[
        0
    ]  # All the samples are normalized

    N = args.num_batches

    assert (
        N is not None or args.num_transitions is not None
    ), "Either num_batches or num_transitions must be provided"

    if args.num_transitions is not None:
        transitions_in_batch = ds.config.batch_size * (ds.config.n_steps - 1)

        # calculate the number of batches needed to sample
        # the required number of transitions
        N = (args.num_transitions + transitions_in_batch - 1) // transitions_in_batch
        print("Using the number of transitions to calculate the number of batches")
        print(f"Sampling {args.num_transitions} transitions will require {N} batches")

    else:
        N = args.num_batches

    ds_train = make_dataset(ds, n_batches=N)
    ds_val = make_dataset(ds, n_batches=1)

    ds_name_suffix = build_name_suffix(args)

    # output_path = f'/scratch/us441/datasets/wall/wall-visual-{config_name}_{ds_name_suffix}-v0.npz'
    output_path = f"/volume/wall-visual-{config_name}_{ds_name_suffix}-v0.npz"

    save_dataset(ds_train, ds_val, output_path)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
