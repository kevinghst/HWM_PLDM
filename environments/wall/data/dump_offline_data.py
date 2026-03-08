from environments.wall.data.wall import WallDataset, WallDatasetConfig
import torch
import yaml
from dataclasses import fields
import dataclasses
from tqdm import tqdm
import numpy as np
import random
import os
import argparse


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def create_and_save_ds(ds, save_dir, suffix="", save_locations=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds.device = device

    all_states = []
    all_actions = []
    all_locations = []
    all_bias_angle = []

    for i in tqdm(range(len(ds))):
        states, locations, actions, bias_angle, _, _ = ds[i]

        all_states.append(states.cpu().numpy())
        all_actions.append(actions.cpu().numpy())

        all_locations.append(locations.cpu().numpy())
        all_bias_angle.append(bias_angle.cpu().numpy())

    all_actions = np.concatenate(all_actions, axis=0).squeeze(-2)
    all_states = np.concatenate(all_states, axis=0)

    all_locations = np.concatenate(all_locations, axis=0).squeeze(-2)
    all_bias_angle = np.concatenate(all_bias_angle, axis=0)

    if suffix:
        final_save_dir = f"{save_dir}/{suffix}"
    else:
        final_save_dir = f"{save_dir}/train"

    if not os.path.exists(final_save_dir):
        os.mkdir(final_save_dir)

    np.save(f"{final_save_dir}/actions.npy", all_actions)
    np.save(f"{final_save_dir}/states.npy", all_states)
    np.save(f"{final_save_dir}/locations.npy", all_locations)
    np.save(f"{final_save_dir}/bias_angle.npy", all_bias_angle)

    print(f"saved to {final_save_dir}")

    return all_states, all_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump Offline Data for HJEPA Wall experiment."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where datasets will be saved.",
    )
    parser.add_argument(
        "--fix_wall", action="store_true", help="Fix wall configuration in the dataset."
    )

    args = parser.parse_args()

    data_yaml_path = "/scratch/wz1232/HJEPA/hjepa/configs/wall/no_fixed_wall.yaml"

    # ===========================================================================================#

    # Load the YAML configuration from a file (or string)
    with open(data_yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Create offline dataset train
    data_yaml_data = yaml_data["data"]["wall_config"]

    data_config = update_config_from_yaml(WallDatasetConfig, data_yaml_data)
    ds = WallDataset(
        config=dataclasses.replace(
            data_config,
            size=188235,  # 188235 * 17 ~ 3.2M frames
            normalize=False,
            train=True,
            fix_wall=args.fix_wall,
        )
    )

    create_and_save_ds(ds=ds, save_dir=args.save_dir)
