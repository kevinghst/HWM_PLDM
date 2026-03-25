import multiprocessing
import warnings

warnings.filterwarnings("ignore", message="Ill-formed record")

from typing import Optional
import os
import shutil
from dataclasses import dataclass
import dataclasses
from contextlib import contextmanager
import random
import time
from omegaconf import MISSING

import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib
from pldm.logger import Logger, MetricTracker

try:
    multiprocessing.set_start_method("fork")  # noqa
except:
    pass

from pldm.configs import ConfigBase
from pldm.data.enums import DataConfig
from pldm.data.dataset_factory import DatasetFactory
from pldm.data.utils import get_optional_fields
from pldm.optimizers.schedulers import Scheduler, LRSchedule
from pldm.optimizers.optimizer_factory import OptimizerFactory, OptimizerType
from pldm.evaluation.evaluator import EvalConfig, Evaluator
from pldm_envs.utils.normalizer import compute_l2_latent_bounds

# if "AMD" not in torch.cuda.get_device_name(0):

from pldm.models.hjepa import HJEPA, HJEPAConfig

from pldm.objectives import ObjectivesConfig
import pldm.utils as utils


def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


@contextmanager
def fixed_rng(seed):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class TrainConfig(ConfigBase):
    env_name: str = MISSING
    n_steps: int = 17
    l1_n_steps: int = 17
    l2_n_steps: int = 0
    l2_step_skip: int = 20
    val_n_steps: int = 17
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    run_project: str = "HJEPA-revived"
    output_root: Optional[str] = None
    output_dir: Optional[str] = None
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: int = 42
    load_checkpoint_path: Optional[str] = None
    load_l1_only: bool = False
    eval_only: bool = False
    train_only: bool = False
    epochs: int = 100
    base_lr: float = 0.2
    l1_to_l2_lr_ratio: float = 1.0
    disable_l2: bool = True
    l2_step_skip: int = 4
    optimizer_type: OptimizerType = OptimizerType.LARS
    optimizer_schedule: LRSchedule = LRSchedule.Cosine
    train_l1: bool = True
    root_path: Optional[str] = None

    data: DataConfig = DataConfig()

    objectives_l1: ObjectivesConfig = ObjectivesConfig()
    objectives_l2: ObjectivesConfig = ObjectivesConfig()

    eval_at_beginning: bool = False
    eval_during_training: bool = False

    save_every_n_epochs: int = 99999
    eval_every_n_epochs: int = 99999

    hjepa: HJEPAConfig = HJEPAConfig()

    resume_if_possible: bool = True
    compile_model: bool = True

    eval_cfg: EvalConfig = EvalConfig()

    def __post_init__(self):
        if self.quick_debug:
            self.data.quick_debug = True

            # D4RL stuff
            levels = ["easy", "medium", "hard"]
            self.data.d4rl_config.quick_debug = True
            self.data.d4rl_config.num_workers = 1
            self.eval_cfg.d4rl_planning.n_envs = 5
            self.eval_cfg.d4rl_planning.n_envs_batch_size = 2
            self.eval_cfg.d4rl_planning.replan_every = 1
            self.eval_cfg.d4rl_planning.n_steps = 6
            self.eval_cfg.d4rl_planning.plot_every = 1

            for level in levels:
                level_cfg = getattr(self.eval_cfg.d4rl_planning, level)
                level_cfg.n_envs = 5
                level_cfg.n_steps = 6
                level_cfg.max_plan_length = 3
                level_cfg.max_plan_length_l2 = 3
                level_cfg.plot_every = 1

            self.eval_cfg.h_d4rl_planning.n_envs = 5
            self.eval_cfg.h_d4rl_planning.n_envs_batch_size = 2
            self.eval_cfg.h_d4rl_planning.replan_every = 1
            self.eval_cfg.h_d4rl_planning.n_steps = 6
            self.eval_cfg.h_d4rl_planning.final_trans_steps = 3
            self.eval_cfg.h_d4rl_planning.plot_every = 1
            self.eval_cfg.h_d4rl_planning.level1.max_plan_length = 6
            self.eval_cfg.h_d4rl_planning.level2.max_plan_length = 3

            for level in levels:
                level_cfg = getattr(self.eval_cfg.h_d4rl_planning, level)
                level_cfg.n_envs = 5
                level_cfg.n_steps = 6
                level_cfg.max_plan_length = 6
                level_cfg.max_plan_length_l2 = 3
                level_cfg.plot_every = 1

        # D4RL stuff
        if self.hjepa.level1.backbone.arch in ["resnet18", "menet5"]:
            self.eval_cfg.d4rl_planning.image_obs = True
        # assert (
        #     self.eval_cfg.d4rl_planning.plot_every
        #     % self.eval_cfg.d4rl_planning.replan_every
        #     == 0
        # )
        self.eval_cfg.d4rl_planning.stack_states = self.data.d4rl_config.stack_states
        self.eval_cfg.d4rl_planning.img_size = self.data.d4rl_config.img_size

        # general
        self.val_n_steps = self.n_steps
        self.eval_cfg.eval_l2 = not self.hjepa.disable_l2
        self.eval_cfg.d4rl_planning.seed = self.seed
        self.eval_cfg.h_d4rl_planning.seed = self.seed
        self.eval_cfg.manispace_planning.seed = self.seed

        if self.hjepa.level2.predictor.z_discrete:
            self.objectives_l2.kl.discrete = True

        if self.output_root is not None:
            self.output_path = os.path.join(
                self.output_root.rstrip("/"), self.output_dir.lstrip("/")
            )
        else:
            self.output_path = None

        self.run_group = self.output_dir

        if self.train_only:
            self.eval_cfg.eval_l1 = False
            self.eval_cfg.probe_preds = False
            self.eval_cfg.probe_encoder = False
            self.eval_cfg.disable_planning = True

        if self.output_dir and "test" in self.output_dir:
            test_dir = os.path.join(self.output_root.rstrip("/"), "test")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        seed_everything(config.seed)

        Logger.run().initialize(
            output_path=self.config.output_path,
            wandb_enabled=self.config.wandb,
            project=config.run_project,
            name=config.run_name,
            group=config.run_group,
            config=dataclasses.asdict(config),
        )

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        # create data
        datasets = DatasetFactory(
            config.data,
            probing_cfg=config.eval_cfg.probing,
            disable_l2=config.hjepa.disable_l2,
        ).create_datasets()

        self.datasets = datasets

        self.ds = datasets.ds
        self.val_ds = datasets.val_ds

        # infer obs shape
        sample_data = next(iter(self.ds))

        load_l1 = getattr(self.ds.dataset, "load_l1", True)

        if load_l1:
            states = sample_data.states
            proprio_pos = (
                sample_data.proprio_pos if hasattr(sample_data, "proprio_pos") else None
            )
            proprio_vel = (
                sample_data.proprio_vel if hasattr(sample_data, "proprio_vel") else None
            )
            locations = (
                sample_data.locations if hasattr(sample_data, "locations") else None
            )
        else:
            states = sample_data.l2_states
            proprio_pos = (
                sample_data.l2_proprio_pos
                if hasattr(sample_data, "l2_proprio_pos")
                else None
            )
            proprio_vel = (
                sample_data.l2_proprio_vel
                if hasattr(sample_data, "l2_proprio_vel")
                else None
            )
            locations = (
                sample_data.l2_locations
                if hasattr(sample_data, "l2_locations")
                else None
            )

        input_dim = states.shape[2:]
        print("Inferred input_dim:", input_dim)
        if len(input_dim) == 1:
            input_dim = input_dim[0]

        ppos_dim = proprio_pos.shape[-1] if proprio_pos is not None else 0
        pvel_dim = (
            proprio_vel.shape[-1]
            if proprio_vel is not None and sample_data.proprio_vel is not None
            else 0
        )
        loc_dim = locations.shape[-1] if locations is not None else 0

        # create model
        self.model = HJEPA(
            config.hjepa,
            input_dim=input_dim,
            normalizer=self.ds.normalizer,
            ppos_dim=ppos_dim,
            pvel_dim=pvel_dim,
            loc_dim=loc_dim,
        )

        self.model = self.model.cuda()

        # create objectives
        self.objectives_l1 = self.config.objectives_l1.build_objectives_list(
            name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
        )
        self.objectives_l2 = self.config.objectives_l2.build_objectives_list(
            name_prefix="l2",
            repr_dim=(
                self.model.level2.spatial_repr_dim
                if self.model.level2 is not None
                else None
            ),
        )
        # other stuff...

        load_result = self.maybe_load_model()

        if (
            config.eval_only
            and not config.eval_cfg.probing.full_finetune
            and not load_result
        ):
            print("WARN: probing a random network. Is that intentional?")

        assert not (self.config.hjepa.train_l1 and self.config.hjepa.freeze_l1)

        if self.config.hjepa.freeze_l1:
            print("freezing first level weights")
            for m in self.model.level1.modules():
                for p in m.parameters():
                    p.requires_grad = False

        print(
            "number of params in level 1:",
            utils.count_trainable_parameters(self.model.level1),
        )
        if self.model.level2 is not None:
            print(
                "number of params in level 2:",
                utils.count_trainable_parameters(self.model.level2),
            )

        self.metric_tracker = MetricTracker(window_size=100)

        if self.config.compile_model:
            print("compiling model")
            c_time = time.time()
            self.model = torch.compile(self.model)
            print(f"compilation finished after {time.time() - c_time:.3f}s")

    def maybe_resume(self):
        if not os.path.exists(self.config.output_path):
            return False
        latest_checkpoint = utils.pick_latest_model(self.config.output_path)
        if latest_checkpoint is None:
            return False
        print("resuming from", latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.sample_step = checkpoint["sample_step"]
        print("resumed from epoch", self.epoch, "step", self.step)

    def maybe_load_model(self):
        if self.config.load_checkpoint_path is not None:

            if os.path.isdir(self.config.load_checkpoint_path):
                # load the latest checkpoint from the directory
                load_checkpoint_path = utils.pick_latest_model(
                    self.config.load_checkpoint_path
                )
            else:
                load_checkpoint_path = self.config.load_checkpoint_path

            checkpoint = torch.load(load_checkpoint_path)
            state_dict = checkpoint["model_state_dict"]
            # remove "_orig_mod." prefix from the keys
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            # remove all posterior parameters (incompatible because we don't
            # use it in l1).

            if (
                "backbone.layer1.0.weight" in state_dict
            ):  # this is jepa only model (legacy)
                res = self.model.level1.load_state_dict(state_dict)
            else:
                if self.config.load_l1_only:
                    for k in list(state_dict.keys()):
                        # 1. remove all posterior parameters
                        # (incompatible because we don't use it in l1).
                        # 2. remove everything belonging to l2
                        if (
                            "level2" in k
                            or "posterior" in k
                            or "decoder.converter" in k
                        ):
                            del state_dict[k]
                        elif "decoder" in k:  # this is for loading RSSM
                            del state_dict[k]
                res = self.model.load_state_dict(state_dict, strict=False)
            assert (
                len(res.unexpected_keys) == 0
            ), f"Unexpected keys when loading weights: {res.unexpected_keys}"
            print(f"loaded model from {load_checkpoint_path}")

            # load ensemble weights if theres any
            models = {
                "level1": self.model.level1,
                "level2": self.model.level2,
            }

            for level, model in models.items():
                if model is None:
                    continue

                if f"{level}_ensemble_params" in checkpoint:
                    model.predictor.ensemble_params = checkpoint[
                        f"{level}_ensemble_params"
                    ]
                    model.predictor.buffers = checkpoint[f"{level}_ensemble_buffer"]

            return True
        return False

    def train(self):
        self.optimizer = OptimizerFactory(
            model=self.model,
            optimizer_type=self.config.optimizer_type,
            base_lr=self.config.base_lr,
            l1_to_l2_lr_ratio=self.config.l1_to_l2_lr_ratio,
        ).create_optimizer()

        if self.config.resume_if_possible:
            if self.maybe_resume():
                print("resuming training")

        scheduler = Scheduler(
            schedule=self.config.optimizer_schedule,
            base_lr=self.config.base_lr,
            data_loader=self.ds,
            epochs=self.config.epochs,
            optimizer=self.optimizer,
        )

        first_step = None

        if self.config.eval_at_beginning and not self.config.quick_debug:
            self.validate()

        for epoch in tqdm(range(self.epoch, self.config.epochs + 1), desc="Epoch"):
            self.epoch = epoch
            end_time = time.time()
            for step, batch in (
                pbar := tqdm(
                    enumerate(self.ds, start=epoch * len(self.ds)),
                    desc="Batch",
                    total=len(self.ds),
                    maxinterval=10,
                )
            ):
                if first_step is None:
                    first_step = step
                start_time = time.time()
                if end_time is not None:
                    # data time is the time it took to load the data
                    # (which is the time between the end of the previous
                    # batch and the start of this batch)
                    data_time = start_time - end_time
                else:
                    data_time = None

                # move to cuda and swap batch and time
                s = batch.states.cuda(non_blocking=True).transpose(0, 1)
                a = batch.actions.cuda(non_blocking=True).transpose(0, 1)

                # utils.plot_from_batch(s)

                lr = scheduler.adjust_learning_rate(step)

                self.sample_step += s.shape[1]
                self.step = step

                self.optimizer.zero_grad()

                optional_fields = get_optional_fields(
                    batch, device=s.device, include_l2=not self.config.hjepa.disable_l2
                )

                forward_result = self.model.forward_posterior(s, a, **optional_fields)

                loss_infos = []

                if self.config.hjepa.train_l1:
                    loss_infos += [
                        objective(batch, [forward_result.level1])
                        for objective in self.objectives_l1
                    ]

                if not self.config.hjepa.disable_l2:
                    loss_infos += [
                        objective(batch, [forward_result.level1, forward_result.level2])
                        for objective in self.objectives_l2
                    ]

                total_loss = sum([loss_info.total_loss for loss_info in loss_infos])
                if total_loss.isnan():
                    raise RuntimeError("NaN loss")
                total_loss.backward()
                self.optimizer.step()

                train_time = time.time() - start_time
                log_start_time = time.time()

                self.metric_tracker.update("train_time", train_time)
                self.metric_tracker.update("data_time", data_time)

                if self.config.quick_debug or (step % 100 == 0):
                    metric_log = self.metric_tracker.build_log_dict()
                    pbar.set_description(
                        f"Loss: {total_loss.item():.4f}, "
                        f"train: {metric_log['train_time/mean']:.3f}s, "
                        f"data: {metric_log['data_time/mean']:.3f}s, "
                        f"log: {metric_log['log_time/mean'] if 'log_time/mean' in metric_log else 0:.3f}s"  # noqa
                    )
                    log_dict = {}

                    for loss_info in loss_infos:
                        if hasattr(loss_info, "build_log_dict"):
                            log_dict.update(loss_info.build_log_dict())

                    if data_time is not None:
                        log_dict["data_time"] = data_time

                    Logger.run().log(
                        {
                            "sample_step": self.sample_step,
                            "loss": total_loss.item(),
                            "learning_rate": lr,
                            "custom_step": step,
                            "epoch": epoch,
                            **log_dict,
                            **metric_log,
                        },
                        commit=False,
                    )
                    Logger.run().commit()

                    if step - first_step == 5:
                        return

                self.metric_tracker.update("log_time", time.time() - log_start_time)
                end_time = time.time()

            if (
                self.epoch % self.config.save_every_n_epochs == 0 and self.epoch > 0
            ) or self.epoch >= self.config.epochs:
                self.save_model()

            if (
                self.epoch % self.config.eval_every_n_epochs == 0
                and self.config.eval_during_training
            ) or self.epoch >= self.config.epochs:
                self.validate()

    @torch.no_grad()
    def eval_on_objectives(self):
        if self.val_ds is None:
            return

        losses = {}

        for step, batch in tqdm(enumerate(self.val_ds)):
            # move to cuda and swap batch and time
            s = batch.states.cuda(non_blocking=True).transpose(0, 1)
            a = batch.actions.cuda(non_blocking=True).transpose(0, 1)

            optional_fields = get_optional_fields(
                batch, device=s.device, include_l2=not self.config.hjepa.disable_l2
            )

            forward_result = self.model.forward_posterior(s, a, **optional_fields)

            loss_infos = []

            if self.config.hjepa.train_l1:
                loss_infos += [
                    objective(batch, [forward_result.level1])
                    for objective in self.objectives_l1
                ]

            if not self.config.hjepa.disable_l2:
                loss_infos += [
                    objective(batch, [forward_result.level1, forward_result.level2])
                    for objective in self.objectives_l2
                ]

            for loss_info in loss_infos:
                for attr in loss_info._fields:
                    val = getattr(loss_info, attr)
                    if isinstance(val, str):
                        continue

                    assert isinstance(val, torch.Tensor)
                    assert len(val.shape) == 0

                    key = f"val_epoch_{self.epoch}_{loss_info.name_prefix}/{loss_info.loss_name}_{attr}"

                    if key in losses:
                        losses[key].append(val.item())
                    else:
                        losses[key] = [val.item()]

            if step > 2:
                break

        # take mean over batches
        for key, val in losses.items():
            losses[key] = sum(val) / len(val)

        Logger.run().log(losses)
        Logger.run().commit()

    def validate(self):
        # Keep evaluation deterministic and isolated from training RNG progress.
        eval_seed = int(self.config.seed + self.epoch * 1009 + 17)
        with fixed_rng(eval_seed):
            training = self.model.training
            self.model.eval()

            # compute L2 latent action bounds
            if (
                (not self.config.hjepa.disable_l2)
                and (self.model.level2 is not None)
                and (self.config.eval_cfg.l2_latent_bounds_percentile > 0)
            ):
                max_batches = 5 if self.config.quick_debug else 100
                bounds = compute_l2_latent_bounds(
                    model=self.model,
                    dataset=self.ds,
                    normalizer=self.ds.normalizer,
                    config=self.config,
                    percentile=self.config.eval_cfg.l2_latent_bounds_percentile,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    max_batches=max_batches,
                )

            # evals on the same objectives used for training
            self.eval_on_objectives()

            # create evaluator (for both probing and planning)
            self.evaluator = Evaluator(
                config=self.config.eval_cfg,
                model=self.model,
                quick_debug=self.config.quick_debug,
                normalizer=self.ds.normalizer,
                epoch=self.epoch,
                probing_datasets=self.datasets.probing_datasets,
                l2_probing_datasets=self.datasets.l2_probing_datasets,
                load_checkpoint_path=self.config.load_checkpoint_path,
                output_path=self.config.output_path,
            )

            log_dict = self.evaluator.evaluate()
            log_dict["custom_step"] = self.step

            Logger.run().log(log_dict)
            Logger.run().log_summary(log_dict)
            Logger.run().save_summary(
                f"summary_epoch={self.epoch}_sample_step={self.sample_step}.json"
            )

            for v in log_dict.values():
                if isinstance(v, matplotlib.figure.Figure):
                    plt.close(v)

            if training:
                # if model is previously in training
                self.model.train()

        return

    def save_model(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)

            save_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
                "sample_step": self.sample_step,
            }

            models = {
                "level1": self.model.level1,
                "level2": self.model.level2,
            }

            for level, model in models.items():
                if model is None:
                    continue
                if model.predictor.ensemble_params is not None:
                    save_dict[f"{level}_ensemble_params"] = (
                        model.predictor.ensemble_params
                    )
                    save_dict[f"{level}_ensemble_buffer"] = model.predictor.buffers

            torch.save(
                save_dict,
                os.path.join(
                    self.config.output_path,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )


def main(config: TrainConfig):
    torch.set_num_threads(1)
    trainer = Trainer(config)

    if config.eval_only and not config.quick_debug:
        trainer.validate()
    else:
        trainer.train()

    if config.quick_debug:
        trainer.validate()


if __name__ == "__main__":
    cfg = TrainConfig.parse_from_command_line()
    main(cfg)
