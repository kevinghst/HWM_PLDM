import os
import glob
import torch
import numpy as np
from tqdm import tqdm


"""
This happens after executing generate_dat_sbatch.py --render_only.
It ignores the workers that errored out (failing to output an image file)
It pools together the data and images of workers that successfully outputted images.
"""

root = "/vast/wz1232/ant_diverse/ant_medium_diverse_explore_2/maps_0_19"
save_root = root

all_splits = torch.load(os.path.join(root, "data.p"))

workers_num = 100

per_worker = len(all_splits) // workers_num

files = glob.glob(f"{root}/*images_worker_*")

missing_workers = []
filtered_splits = []


for i in tqdm(range(workers_num)):
    substring = f"images_worker_{i}_"

    start = per_worker * i
    end = start + per_worker

    # check if substring is in any of the file names in one line
    if not any(substring in file for file in files):
        missing_workers.append(i)
        print(f"Worker {i} is missing. Expected range: {start} to {end}")
        continue
    else:
        indices = np.arange(start, end)

        for idx in indices:
            filtered_splits.append(all_splits[idx])

# preallocate numpy array of images
print(f"Original length: {len(all_splits)}")
print(f"Filtered length: {len(filtered_splits)}")

if len(all_splits) != len(filtered_splits):
    raise "Need to generate metadata for the filtered_data!!"

torch.save(filtered_splits, f"{save_root}/filtered_data.p")


image_size = (64, 64, 3)
total_images = len(filtered_splits) * filtered_splits[0]["observations"].shape[0]

del filtered_splits

filtered_images = np.empty((total_images, *image_size), dtype=np.uint8)

image_offset = 0

for i in tqdm(range(workers_num)):
    start = per_worker * i
    end = start + per_worker

    if i not in missing_workers:
        images = torch.load(
            os.path.join(root, f"images_worker_{i}_start_{start}_end_{end}.pt")
        )
        images = np.concatenate(images, axis=0)
        filtered_images[image_offset : image_offset + images.shape[0]] = images
        image_offset += images.shape[0]
        del images

np.save(os.path.join(save_root, "filtered_images.npy"), filtered_images)
