from wall import DotWall
from pathlib import Path

import torch
import imageio
import torchvision.transforms as T

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
from pldm_envs.wall.data.offline_wall import (
    OfflineWallDataset,
    OfflineWallDatasetConfig,
)


OUTPUT_DIR = Path("debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_gif(frames, filename, fps=10):
    """
    Convert a list of PyTorch tensors into a GIF and save it to a file.

    Args:
        frames (list of torch.Tensor): List of image tensors (C, H, W) or (H, W).
        filename (str or Path): Path to save the GIF.
        fps (int): Frames per second for the GIF.
    """
    images = []
    transform = T.ToPILImage()

    for frame in frames:
        if (
            frame.ndimension() == 2
        ):  # Convert grayscale to 3-channel RGB for consistency
            frame = frame.unsqueeze(0).repeat(3, 1, 1)
        elif frame.shape[0] == 1:  # Single channel, expand to RGB
            frame = frame.repeat(3, 1, 1)

        image = transform(frame.cpu())  # Convert to PIL Image
        images.append(image)

    imageio.mimsave(filename, images, fps=fps)


def test_env():
    env = DotWall()
    obs, info = env.reset()
    obses = [obs]
    targets = [info["target_obs"]]

    done = truncated = False

    while not done and not truncated:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        obses.append(obs)
        targets.append(info["target_obs"])

    frames = []
    for obs, target in zip(obses, targets):
        joint_frame = torch.cat([obs, target], dim=2)
        joint_frame = joint_frame.max(dim=0)[0]  # Take the max value for each pixel
        frames.append(joint_frame)

    make_gif(frames, OUTPUT_DIR / "debug.gif")

    print(obs.sum())
    print(reward)
    print(done)
    print(truncated)
    print(info)


def test_ds():
    ds_wall = WallDataset(WallDatasetConfig())

    sample = ds_wall[0]

    # render the first trajectory
    traj = sample.states[0]
    traj = traj.max(dim=1)[0]  # Take the max value for each pixel

    make_gif(traj, OUTPUT_DIR / "debug_ds.gif")


def test_offline_ds():
    ds_wall = OfflineWallDataset(
        OfflineWallDatasetConfig(
            offline_data_path="/volume/data/code_release_test/len_17.npz"
        )
    )
    sample = ds_wall[0]

    # render the first trajectory
    traj = sample.states
    traj = traj.max(dim=1)[0]  # Take the max value for each pixel

    make_gif(traj, OUTPUT_DIR / "debug_offline_ds.gif")


def main():
    test_env()
    test_ds()
    test_offline_ds()


if __name__ == "__main__":
    main()
