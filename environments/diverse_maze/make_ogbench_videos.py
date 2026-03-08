# MAY NEED TO RUN THIS FILE FROM ROOT OF REPO TO WORK PROPERLY

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from environments.diverse_maze import ant_draw
from environments.diverse_maze import maze_draw

offline_trials_path = "/scratch/wz1232/data/maze2d_large_diverse_probe/starts_targets_13_16.pt"
data_path = Path("/scratch/wz1232/data/maze2d_large_diverse_probe")

offline_trials = torch.load(offline_trials_path)

trajs = {
    "gciql_new": "/scratch/wz1232/ogbench/impls/exp/OGBench_2026/Debug/1-15-1-7_sd000_s_4429085.0.20260115_115822",
    "hiql_new": "/scratch/wz1232/ogbench/impls/exp/OGBench_2026/Debug/1-17-2-12-seed2_sd000_s_4468057.0.20260117_221142",
    # "hilp_new": "/scratch/wz1232/ogbench/impls/exp/OGBench_2026/Debug/1-18-1-2-seed2_sd000_s_4470665.0.20260118_124402",
}

config = torch.load(data_path / "metadata.pt")
map_metadata = torch.load(data_path / "train_maps.pt") if "diverse" in config["env"] else None
_drawer_cache = {}
center_crop = transforms.CenterCrop(370)

def get_drawer(map_idx=None):
    """Create the same drawer as render_data.py for this dataset."""
    if "diverse" in config["env"]:
        if map_idx is None:
            raise ValueError("map_idx is required for diverse environments")
        map_idx = int(map_idx)
        if map_idx not in _drawer_cache:
            env = ant_draw.load_environment(
                name=f"{config['env']}_{map_idx}",
                map_key=map_metadata[map_idx],
            )
            _drawer_cache[map_idx] = maze_draw.create_drawer(env, env.name)
        return _drawer_cache[map_idx]

    if "default" not in _drawer_cache:
        env = ant_draw.load_environment(config["env"])
        _drawer_cache["default"] = maze_draw.create_drawer(env, env.name)
    return _drawer_cache["default"]


def extract_goal_xy(trial_targets):
    target_arr = np.asarray(trial_targets)
    if target_arr.ndim == 1:
        return target_arr[:2]
    return target_arr[0, :2]


def _maybe_int(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return int(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _extract_map_idx_from_trial_start(trial_start):
    """Best-effort extraction for map index stored in trial start metadata."""
    if isinstance(trial_start, dict):
        for key in ("map_idx", "maze_idx", "map_id", "maze_id"):
            idx = _maybe_int(trial_start.get(key))
            if idx is not None:
                return idx
    return None


def resolve_episode_map_idx(info, trial_start, traj_idx):
    """Resolve map index with clear priority and a safe fallback."""
    if info:
        first_step = info[0]
        if isinstance(first_step, dict):
            for key in ("map_idx", "maze_idx", "map_id", "maze_id"):
                idx = _maybe_int(first_step.get(key))
                if idx is not None:
                    return idx

    trial_idx = _extract_map_idx_from_trial_start(trial_start)
    if trial_idx is not None:
        return trial_idx

    num_maps = len(map_metadata) if map_metadata is not None else 20
    fallback_idx = (traj_idx - 1) % max(num_maps, 1)
    print(
        f"[warn] traj_{traj_idx}: missing map_idx in info/trial_start, "
        f"fallback to {fallback_idx}"
    )
    return fallback_idx


def center_crop_image(image):
    image_uint8 = np.asarray(image, dtype=np.uint8)
    image_pil = Image.fromarray(image_uint8)
    cropped = center_crop(image_pil)
    return np.asarray(cropped, dtype=np.uint8)

"""
data structure:

traj = {
    1: ...
    2: ...
}

traj[i][0] = {
    'observation': list of N
    'action':
    'reward':
    'done':
    'info': 
}



"""

total_episodes = 20

for model, traj_path in trajs.items():  # loop over models
    traj = torch.load(f"{traj_path}/large_config_default_eval_hard_all_trajs_step_1.pt")

    for traj_idx, traj_data in traj.items():  # loop over episodes
        if traj_idx >= total_episodes:
            break
        
        trial_start = offline_trials['starts'][traj_idx - 1]
        trial_targets = offline_trials['targets'][traj_idx - 1]

        info = traj_data[0]['info']

        # Prefer rollout map_idx, fallback to trial metadata, then index modulo.
        episode_map_idx = resolve_episode_map_idx(info, trial_start, traj_idx)
        drawer = get_drawer(episode_map_idx)

        goal_xy = extract_goal_xy(trial_targets)
        goal_obs = np.concatenate([goal_xy, np.zeros(2)])
        goal_image = center_crop_image(drawer.render_state(goal_obs))

        frames = []
        for i in range(len(info)):  # loop over steps
            curr_info = info[i]
            curr_location = curr_info['location'] # numpy float64
            # add 2 zeros at the end
            curr_obs = np.concatenate([curr_location, np.zeros(2)])
            curr_image = center_crop_image(drawer.render_state(curr_obs))
            frames.append(curr_image)

        success = len(info) < 500

        save_video_root = Path("videos") / model
        save_video_root.mkdir(parents=True, exist_ok=True)
        save_video_path = save_video_root / f"traj_{traj_idx - 1}_{success}.mp4"

        save_goal_path = save_video_root / f"traj_{traj_idx - 1}_{success}_goal.png"
        imageio.imwrite(save_goal_path, goal_image)

        if frames:
            with imageio.get_writer(save_video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
