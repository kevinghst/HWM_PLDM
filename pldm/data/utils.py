import torch
import numpy as np
import random
from torch.utils.data._utils.collate import default_collate
from pldm_envs.utils.normalizer import Normalizer
from pldm.data.enums import DataConfig


def _worker_init_fn(_worker_id):
    # Prevent each worker from spawning intra-op threads and oversubscribing CPU.
    torch.set_num_threads(1)
    # Keep numpy/python RNG in each worker aligned with PyTorch's worker seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_optional_fields(data, device="cuda", transpose_TB=True, include_l2=True):
    fields = [
        "proprio_vel",
        "proprio_pos",
        "chunked_locations",
        "chunked_proprio_pos",
        "chunked_proprio_vel",
        "goal",
        "locations",
    ]

    if include_l2:
        fields += [
            "l2_states",
            "l2_proprio_vel",
            "l2_proprio_pos",
            "l2_actions",
        ]

    return_dict = {}

    for field in fields:
        if hasattr(data, field):
            field_data = getattr(data, field)
            if transpose_TB and field != "goal":
                field_data = field_data.transpose(0, 1)
            return_dict[field] = field_data.to(device, non_blocking=True)
        else:
            return_dict[field] = None

    return return_dict


class PrioritizedSampler(torch.utils.data.Sampler):
    def __init__(
        self, data_size: int, batch_size: int, alpha: float = 0.7, beta: float = 0.9
    ):
        self.data_size = data_size
        self.batch_size = batch_size
        self.rb = PrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            storage=ListStorage(data_size),
            batch_size=batch_size * 2,
        )
        data = torch.arange(0, self.data_size, 1)
        self.rb.extend(data)

    def update_priority(self, indices, priority):
        self.rb.update_priority(indices, priority)

    def __iter__(self):
        window = Queue(maxsize=self.batch_size)
        window_set = set()
        total = 0
        while total < self.data_size:
            next_elements = self.rb.sample()
            # Only add next element if it hasn't been seen in the last
            # batch_size elements
            for next_element in next_elements:
                if next_element not in window_set:
                    window.put(next_element)
                    window_set.add(next_element)
                    total += 1

                    yield next_element

                    if window.full():  # free up for the next element
                        window_set.remove(window.get())

    def __len__(self):
        return self.data_size


def normalize_collate_fn(normalizer):
    """Returns a collate function that batches and normalizes in workers."""

    def collate_fn(batch):
        collated = default_collate(batch)
        return normalizer.normalize_sample(collated)

    return collate_fn  # Return the collate function for later use


class NormalizedDataLoader:
    """A wrapper around a dataloader that applies normalization."""

    def __init__(self, dataloader, normalizer):
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.config = dataloader.config

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        """Iterate over the dataset, applying normalization to each batch."""
        for batch in self.dataloader:
            # Apply normalization to the required fields
            new_batch = self.normalizer.normalize_sample(batch)
            yield new_batch


def make_dataloader(ds, loader_config, normalizer=None, suffix="", train=True):
    config = ds.config

    # Some datasets (e.g., offline wall) already emit CUDA tensors.
    # pin_memory only works for CPU tensors and will crash otherwise.
    ds_device = getattr(ds, "device", None)
    if ds_device is not None:
        ds_device = torch.device(ds_device)
    pin_memory = not (ds_device is not None and ds_device.type == "cuda")

    print(f"{len(ds)} samples in {suffix} dataset")
    num_workers = loader_config.num_workers if not loader_config.quick_debug else 0
    loader_kwargs = {
        "dataset": ds,
        "batch_size": config.batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "drop_last": True,
        "persistent_workers": (
            not loader_config.quick_debug and num_workers > 0
        ),
        "prefetch_factor": (
            loader_config.prefetch_factor
            if not loader_config.quick_debug and num_workers > 0
            else None
        ),
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
    }

    # Unnormalized loader is used only to estimate normalization statistics.
    loader = torch.utils.data.DataLoader(**loader_kwargs)
    loader.config = config

    if loader_config.normalize:
        if normalizer is None:
            normalizer = Normalizer.build_normalizer(
                loader,
                l2_step_skip=config.l2_step_skip,
                n_samples=1 if loader_config.quick_debug else 10000 // config.batch_size,
                min_max_state=loader_config.min_max_normalize_state,
                normalizer_hardset=loader_config.normalizer_hardset,
                image_based=config.image_based,
            )
    else:
        normalizer = Normalizer.build_id_normalizer()

    loader = torch.utils.data.DataLoader(
        **loader_kwargs,
        collate_fn=normalize_collate_fn(normalizer),
    )
    loader.config = config
    loader.normalizer = normalizer
    ds.normalizer = normalizer

    return loader


def make_dataloader_for_prebatched_ds(
    ds,
    loader_config: DataConfig,
    normalizer=None,
):
    config = ds.config

    if loader_config.normalize:
        if normalizer is None:
            normalizer = Normalizer.build_normalizer(
                ds,
                l2_step_skip=config.l2_step_skip,
                n_samples=1 if loader_config.quick_debug else 10000 // config.batch_size,
                min_max_state=loader_config.min_max_normalize_state,
                normalizer_hardset=loader_config.normalizer_hardset,
            )
    else:
        normalizer = Normalizer.build_id_normalizer()

    loader = NormalizedDataLoader(ds, normalizer)

    ds.normalizer = normalizer

    return loader
