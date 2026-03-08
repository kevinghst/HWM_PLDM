import dataclasses

from pldm_envs.wall.data.offline_wall import OfflineWallDataset
from pldm_envs.wall.data.wall import WallDataset
from pldm_envs.wall.data.single import DotDataset
from pldm_envs.wall.data.wall_expert import (
    WrappedWallExpertDataset,
)
from pldm_envs.wall.data.wall_passing_test import WallPassingTestDataset
from pldm_envs.wall.data.border_passing_test import BorderPassingTestDataset

from pldm.data.utils import make_dataloader, make_dataloader_for_prebatched_ds


# if "AMD" not in torch.cuda.get_device_name(0):
from pldm_envs.diverse_maze.d4rl import D4RLDataset
from pldm_envs.ogbench.dataset import LocoMazeDataset

from pldm.probing.evaluator import ProbingConfig
from pldm_envs.utils.normalizer import Normalizer
from pldm.data.enums import DataConfig, DatasetType, ProbingDatasets, Datasets

from pldm_envs.ogbench_manispace.dataset import ManispaceDataset


class DatasetFactory:
    def __init__(
        self,
        config: DataConfig,
        probing_cfg: ProbingConfig = ProbingConfig(),
        disable_l2: bool = True,
        eval_aae: bool = False,
        aae_chunk_size: int = 10,
        aae_samples: int = 2000,
    ):
        self.config = config
        self.probing_cfg = probing_cfg
        self.disable_l2 = disable_l2
        self.eval_aae = eval_aae
        self.aae_chunk_size = aae_chunk_size
        self.aae_samples = aae_samples

    def create_datasets(self):
        if self.config.dataset_type == DatasetType.Single:
            return self._create_single_datasets()
        elif self.config.dataset_type == DatasetType.Wall:
            return self._create_wall_datasets()
        elif self.config.dataset_type == DatasetType.WallExpert:
            return self._create_wall_expert_datasets()
        elif self.config.dataset_type == DatasetType.D4RL:
            return self._create_d4rl_datasets()
        elif self.config.dataset_type == DatasetType.LocoMaze:
            return self._create_locomaze_datasets()
        elif self.config.dataset_type == DatasetType.OgbenchManispace:
            return self._create_ogbench_manispace_datasets()
        else:
            raise NotImplementedError

    def _create_single_datasets(self):
        ds = DotDataset(self.config.dot_config)
        val_ds = DotDataset(
            dataclasses.replace(self.config.dot_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(ds=ds, val_ds=val_ds)

        return datasets

    def _create_wall_datasets(self):
        if self.config.offline_wall_config.use_offline:
            ds = OfflineWallDataset(config=self.config.offline_wall_config)
            ds = make_dataloader(
                ds=ds, loader_config=self.config, suffix="offline_wall"
            )
        else:
            ds = WallDataset(self.config.wall_config)
            ds = make_dataloader_for_prebatched_ds(
                ds,
                loader_config=self.config,
            )

        probing_datasets = self._create_wall_probing_datasets(ds.normalizer)

        if self.disable_l2:
            datasets = Datasets(
                ds=ds,
                val_ds=None,
                probing_datasets=probing_datasets,
            )
        else:
            l2_probing_datasets = self._create_l2_wall_probing_datasets(ds.normalizer)
            datasets = Datasets(
                ds=ds,
                val_ds=None,
                probing_datasets=probing_datasets,
                l2_probing_datasets=l2_probing_datasets,
            )

        return datasets

    def _create_wall_probing_datasets(self, normalizer: Normalizer):
        probe_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                train=False,
                n_steps=self.probing_cfg.l1_depth,
                traj_n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                expert_cross_wall_rate=0,
            )
        )
        probe_ds = make_dataloader_for_prebatched_ds(
            probe_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        probe_val_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                n_steps=self.probing_cfg.l1_depth,
                traj_n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                train=False,
                expert_cross_wall_rate=0,
            )
        )
        probe_val_ds = make_dataloader_for_prebatched_ds(
            probe_val_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        extra_datasets = {}

        if self.probing_cfg.probe_wall:
            wall_test_ds = WallPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    traj_n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["wall_test"] = make_dataloader_for_prebatched_ds(
                wall_test_ds, loader_config=self.config, normalizer=normalizer
            )
        if self.probing_cfg.probe_border:
            border_test_ds = BorderPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    traj_n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["border_test"] = make_dataloader_for_prebatched_ds(
                border_test_ds, loader_config=self.config, normalizer=normalizer
            )

        probing_datasets = ProbingDatasets(
            ds=probe_ds, val_ds=probe_val_ds, extra_datasets=extra_datasets
        )

        return probing_datasets

    def _create_l2_wall_probing_datasets(self, normalizer: Normalizer):
        probe_l2_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                train=False,
                n_steps=self.probing_cfg.l2_depth,
                traj_n_steps=self.probing_cfg.l2_depth,
                fix_wall_batch_k=None,
                return_l2=True,
            )
        )
        probe_l2_ds = make_dataloader_for_prebatched_ds(
            probe_l2_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        probe_l2_val_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                train=False,
                n_steps=self.probing_cfg.l2_depth,
                traj_n_steps=self.probing_cfg.l2_depth,
                fix_wall_batch_k=None,
                return_l2=True,
            )
        )
        probe_l2_val_ds = make_dataloader_for_prebatched_ds(
            probe_l2_val_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        extra_datasets = {}

        if self.probing_cfg.probe_wall:
            l2_wall_test = WallPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    train=False,
                    n_steps=self.probing_cfg.l2_depth,
                    traj_n_steps=self.probing_cfg.l2_depth,
                    fix_wall_batch_k=None,
                    return_l2=True,
                )
            )
            extra_datasets["l2_wall_test"] = make_dataloader_for_prebatched_ds(
                l2_wall_test,
                loader_config=self.config,
                normalizer=normalizer,
            )

        if self.probing_cfg.probe_border:
            l2_border_test = BorderPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    train=False,
                    n_steps=self.config.l2_depth,
                    traj_n_steps=self.probing_cfg.l2_depth,
                    fix_wall_batch_k=None,
                    return_l2=True,
                )
            )
            extra_datasets["l2_border_test"] = make_dataloader_for_prebatched_ds(
                l2_border_test,
                loader_config=self.config,
                normalizer=normalizer,
            )
        if self.probing_cfg.probe_expert:
            extra_datasets["probe_l2_expert"] = WallDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    train=False,
                    expert_cross_wall_rate=1.0,
                    n_steps=115,  # hard code for now
                    traj_n_steps=115,
                    fix_wall_batch_k=None,
                    return_l2=True,
                ),
            )
            extra_datasets["probe_l2_expert"] = make_dataloader_for_prebatched_ds(
                extra_datasets["probe_l2_expert"],
                loader_config=self.config,
                normalizer=normalizer,
            )

        l2_probing_datasets = ProbingDatasets(
            ds=probe_l2_ds,
            val_ds=probe_l2_val_ds,
            extra_datasets=extra_datasets,
        )

        return l2_probing_datasets

    def _create_wall_expert_datasets(self):
        # We get normalizer from this random first dataset because
        # the first level in hierarchy is trained on that normalization.
        ds = WallDataset(dataclasses.replace(self.config.wall_config, train=False))
        ds = WrappedWallExpertDataset(
            self.config.wall_expert_config, normalizer=ds.normalizer
        )
        val_ds = WrappedWallExpertDataset(
            dataclasses.replace(self.config.wall_expert_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(
            ds=ds,
            val_ds=None,
        )

        return datasets

    def _create_d4rl_datasets(self):
        ds = D4RLDataset(self.config.d4rl_config, load_l1=self.config.d4rl_config.train_l1)
        ds = make_dataloader(ds=ds, loader_config=self.config)

        probe_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.train_path,
                images_path=self.probing_cfg.train_images_path,
                n_steps=self.probing_cfg.l1_depth,
            ),
        )
        probe_ds = make_dataloader(
            ds=probe_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_train",
        )

        probe_val_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.val_path,
                images_path=self.probing_cfg.val_images_path,
                n_steps=self.probing_cfg.l1_depth,
                train=False,
                crop_length=50000,
                batch_size=64,
            ),
        )

        probe_val_ds = make_dataloader(
            ds=probe_val_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_val",
        )

        if self.disable_l2:
            datasets = Datasets(
                ds=ds,
                val_ds=None,
                probing_datasets=ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds),
            )
        else:
            l2_probe_ds = D4RLDataset(
                dataclasses.replace(
                    self.config.d4rl_config,
                    path=self.probing_cfg.train_path,
                    images_path=self.probing_cfg.train_images_path,
                    n_steps=self.probing_cfg.l2_depth,
                ),
                load_l1=self.config.d4rl_config.train_l1,
            )
            l2_probe_ds = make_dataloader(
                ds=l2_probe_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="l2_probe_train",
            )
            l2_probe_val_ds = D4RLDataset(
                dataclasses.replace(
                    self.config.d4rl_config,
                    path=self.probing_cfg.val_path,
                    images_path=self.probing_cfg.val_images_path,
                    n_steps=self.probing_cfg.l2_depth,
                    train=False,
                    crop_length=50000,
                    batch_size=64,
                ),
                load_l1=self.config.d4rl_config.train_l1,
            )
            l2_probe_val_ds = make_dataloader(
                ds=l2_probe_val_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="l2_probe_val",
            )
            datasets = Datasets(
                ds=ds,
                val_ds=None,
                probing_datasets=ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds),
                l2_probing_datasets=ProbingDatasets(
                    ds=l2_probe_ds, val_ds=l2_probe_val_ds
                ),
            )

        return datasets

    def _create_locomaze_datasets(self):
        # TODO unify with _create_d4rl_datasets?
        ds = LocoMazeDataset(self.config.d4rl_config)
        ds = make_dataloader(ds=ds, loader_config=self.config)

        val_ds = None
        if self.config.d4rl_config.val_path is not None:
            val_ds = LocoMazeDataset(
                dataclasses.replace(
                    self.config.d4rl_config,
                    path=self.config.d4rl_config.val_path,
                    train=False,
                )
            )

            val_ds = make_dataloader(
                ds=val_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="val",
            )

        probe_ds = LocoMazeDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.train_path,
                n_steps=self.probing_cfg.l1_depth,
            ),
        )
        probe_ds = make_dataloader(
            ds=probe_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_train",
        )

        probe_val_ds = LocoMazeDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.val_path,
                n_steps=self.probing_cfg.l1_depth,
                train=False,
                crop_length=50000,
                batch_size=64,
                load_top_down_view=self.probing_cfg.visualize_probing,
            ),
        )
        probe_val_ds = make_dataloader(
            ds=probe_val_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_val",
        )

        aae_ds = None
        if self.eval_aae:
            aae_ds = LocoMazeDataset(
                dataclasses.replace(
                    self.config.d4rl_config,
                    path=self.config.d4rl_config.val_path,
                    n_steps=self.aae_chunk_size + 1,
                    train=False,
                    crop_length=self.aae_samples,
                    batch_size=64,
                    load_top_down_view=True,
                ),
            )
            aae_ds = make_dataloader(
                ds=aae_ds,
                loader_config=self.config,
                normalizer=ds.normalizer,
                suffix="aae",
            )

        datasets = Datasets(
            ds=ds,
            val_ds=val_ds,
            probing_datasets=ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds),
            aae_dataset=aae_ds,
        )

        return datasets

    def _create_ogbench_manispace_datasets(self):
        ds = ManispaceDataset(self.config.ogbench_manispace_config)
        ds = make_dataloader(ds=ds, loader_config=self.config)

        val_ds = ManispaceDataset(self.config.ogbench_manispace_config, train=False)
        val_ds = make_dataloader(
            ds=val_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="val",
        )

        return Datasets(
            ds=ds,
            val_ds=None,
            probing_datasets=ProbingDatasets(ds=ds, val_ds=val_ds),
        )
