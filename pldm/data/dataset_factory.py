import dataclasses

from pldm.data.utils import make_dataloader


# if "AMD" not in torch.cuda.get_device_name(0):
from pldm_envs.diverse_maze.d4rl import D4RLDataset

from pldm.probing.evaluator import ProbingConfig
from pldm.data.enums import DataConfig, DatasetType, ProbingDatasets, Datasets



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
        elif self.config.dataset_type == DatasetType.D4RL:
            return self._create_d4rl_datasets()
        elif self.config.dataset_type == DatasetType.LocoMaze:
            return self._create_locomaze_datasets()
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
