# A standalone logger to json that optionally logs to weights and biases.
# The goal is to be completely independent of the rest of the codebase.

from collections import deque
from typing import Optional, Dict, Any, Tuple
import json
from pathlib import Path
import os
import tempfile

import wandb
from omegaconf import OmegaConf
import torch


class Logger:
    _instance = None

    @classmethod
    def run(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.file_path = None
        self.log_step = None
        self.current_log = None
        self.current_summary = None
        self.wandb_enabled = None
        self.output_path = None
        self.initialized = False

    def initialize(
        self,
        output_path: Optional[str] = None,
        *,
        wandb_enabled: bool = False,
        project: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if self.initialized:
            print("Logger already initialized, skipping")
            return
        if output_path is not None:
            self.output_path: Path = Path(output_path)
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = None

        self.wandb_enabled: bool = wandb_enabled
        if self.wandb_enabled:
            assert project is not None, "wandb requires project name"
            wandb.init(
                project=project,
                name=name,
                group=group,
                config=config,
                settings=wandb.Settings(start_method="fork"),
                dir=self.output_path,
            )
        self.config: Optional[Dict[str, Any]] = config
        self.save_config()
        self.log_step = 0
        self.current_log = {}
        self.current_summary = {}
        self.initialized = True

    def log(
        self,
        log_dict: Dict[str, Any],
        *,
        commit: bool = True,
    ):
        self.current_log.update(log_dict)
        self.log_step += 1
        if commit:
            self.commit()

    def log_across_t(
        self,
        data: torch.Tensor,  # shape (T,)
        name: str,
    ):
        for val in data:
            self.log({name: val.item()})

    def commit(self):
        if self.wandb_enabled:
            wandb.log(self.current_log)

        if self.output_path is not None:
            with (self.output_path / "log.json").open("a") as f:
                f.write(json.dumps(self.clean_dict(self.current_log)) + "\n")

        self.current_log = {}

    def log_summary(self, log_dict: Dict[str, Any], commit: bool = True):
        self.current_summary.update(log_dict)
        if commit:
            self.commit_summary()

    def commit_summary(self):
        if self.wandb_enabled:
            wandb.summary.update(self.current_summary)

        if self.output_path is not None:
            with (self.output_path / "summary.json").open("w") as f:
                f.write(json.dumps(self.clean_dict(self.current_summary), indent=4))

    def save_config(self):
        if self.config is not None and self.output_path is not None:
            p = self.output_path / "config.yaml"
            with p.open("w") as f:
                OmegaConf.save(config=self.config, f=f)

    def save_summary(self, filename):
        if self.output_path is not None:
            with (self.output_path / filename).open("w") as f:
                f.write(json.dumps(self.clean_dict(self.current_summary), indent=4))

    def clean_dict(self, log_dict: Dict[str, Any]):
        for k in log_dict.keys():
            if isinstance(log_dict[k], dict):
                self.clean_dict(log_dict[k])
            elif isinstance(log_dict[k], torch.Tensor):
                log_dict[k] = log_dict[k].item()
            elif isinstance(log_dict[k], wandb.Image) or isinstance(
                log_dict[k], wandb.Video
            ):
                log_dict[k] = "<media>"

        return log_dict

    def log_line_plot(
        self,
        data: list,
        plot_name: str,
    ):
        """
        Logs a single line plot with WandB.

        :param data: list of tuples [[x1, y1], [x2, y2], ...]
        :param plot_name: The name to assign to the plot in WandB.
        """
        if self.wandb_enabled:
            table = wandb.Table(data=data, columns=["x", "y"])
            wandb.log(
                {
                    f"Custom Plots/{plot_name}": wandb.plot.line(
                        table, "x", "y", title=plot_name
                    )
                }
            )

    def log_multiline_plot(
        self,
        xs: list,
        ys: list,
        plot_name: str,
    ):
        if self.wandb_enabled:
            wandb.log(
                {
                    f"Custom Plots/{plot_name}": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=["Pos Traj", "Neg Traj"],
                        title=plot_name,
                        xname="t",
                    )
                }
            )

    def log_figure(self, figure: Any, name: str):
        if self.output_path is not None:
            # This assumes that the image is already saved to disk
            filename = Path(self.output_path) / "media" / f"{name}"
            filename.parent.mkdir(parents=True, exist_ok=True)
        else:
            filename = tempfile.NamedTemporaryFile(suffix=".png").name[
                :-4
            ]  # remove .png
        figure.savefig(filename)

        # if "probe" not in str(filename):
        #     shutil.copy(f"{filename}.png", "/scratch/wz1232/HJEPA/imgs/")

        if self.wandb_enabled:
            # matplotlib automatically adds the extension
            wandb.log({name: wandb.Image(f"{filename}.png")})


class MetricTracker:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data: Dict[str, Tuple[deque, float, float, float, float]] = {}

    def update(self, key: str, value: float):
        if key not in self.data:
            self.data[key] = deque(maxlen=self.window_size), value, value, value, value
        else:
            values, mean, minimum, maximum, last = self.data[key]
            values.append(value)
            mean = sum(values) / len(values)
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            last = value
            self.data[key] = values, mean, minimum, maximum, last

    def build_log_dict(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key, (_values, mean, minimum, maximum, last) in self.data.items():
            result.update(
                {
                    f"{key}/mean": mean,
                    f"{key}/minimum": minimum,
                    f"{key}/maximum": maximum,
                    f"{key}/last": last,
                }
            )
        return result
