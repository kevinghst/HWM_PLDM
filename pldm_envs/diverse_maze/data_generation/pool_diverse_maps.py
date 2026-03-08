import concurrent
import collections
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from functools import partial
import sys
import argparse

import zarr
import torch
from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser(
        description="pool data for diverse antmaze into a single file for training"
    )
    parser.add_argument("--data_path", type=str, help="Path to the data")

    args = parser.parse_args()

    data_path = args.data_path

    all_visual_obs = []

    all_data = []

    folders = [entry.name for entry in os.scandir(data_path) if entry.is_dir()]

    for folder in tqdm(folders):
        folder_path = os.path.join(data_path, folder)

        data = torch.load(os.path.join(folder_path, "data.p"))

        data_no_img = [
            {
                "actions": x["actions"],
                "observations": x["observations"],
                "map_idx": x["map_idx"],
            }
            for x in data
        ]

        all_data += data_no_img

        sub_visual_obs = [
            x["visual_obs"] for x in data
        ]  # each x['visual_obs'] is a numpy tensor of shape (N, 64, 64, 3)
        all_visual_obs.extend(sub_visual_obs)

    images = np.concatenate(all_visual_obs, axis=0)

    np.save(os.path.join(data_path, "images.npy"), images)
    torch.save(all_data, os.path.join(data_path, "data.p"))


if __name__ == "__main__":
    main()
