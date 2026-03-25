from pathlib import Path
import random
import re
import os

import torch
import numpy as np
from dataclasses import fields
import matplotlib.pyplot as plt


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def pick_latest_model(path):
    # Finds the model in path with the largeest epoch value
    paths = list(Path(path).glob("epoch=*.ckpt"))
    rx = re.compile(".*epoch=(?P<epoch>\d+).*")
    max_epoch = 0
    max_p = None
    for p in paths:
        m = rx.match(str(p))
        epoch = int(m.group("epoch"))
        if epoch > max_epoch:
            max_epoch = epoch
            max_p = p
    return max_p


def fix_nvidia_ld_path():
    STR = "/usr/lib/nvidia"
    if STR not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            os.environ.get("LD_LIBRARY_PATH", "") + ":" + STR
        )
        print("Fixed LD_LIBRARY_PATH to have nvidia")
        return True
    return False


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def calculate_conv_out_dim(in_dim, stride, padding, kernel_size):
    return ((in_dim - kernel_size + 2 * padding) / stride) + 1


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


def normalize_for_vis(x, vmin, vmax):
    x = (x - vmin) / (vmax - vmin)
    return x.clamp(0.0, 1.0)


def plot_from_batch(x, samples=10, time_samples=5):
    """
    states: (T, B, C, H, W)
    """

    x = x[:time_samples, :samples].cpu()

    T, B, ch, h, w = x.shape
    x = normalize_for_vis(x, x.min(), x.max())

    fig, axes = plt.subplots(
        nrows=B, ncols=T, figsize=(1.8 * T, 1.8 * B), squeeze=False
    )

    for b in range(B):
        for t in range(T):
            img = x[t, b]

            if ch == 1:
                img = img[0]  # (h, w)
                axes[b, t].imshow(img, cmap="gray")
            else:
                img = img.permute(1, 2, 0)  # (h, w, ch)
                axes[b, t].imshow(img)

            axes[b, t].axis("off")

            if b == 0:
                axes[b, t].set_title(f"t={t}", fontsize=10)
        axes[b, 0].set_ylabel(f"traj {b}", fontsize=10)

    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
