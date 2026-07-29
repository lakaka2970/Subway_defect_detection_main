# Re-export from ultralytics.data.build for subway_yolo compatibility
from ultralytics.data.build import *  # noqa: F403
from ultralytics.data.build import ContiguousDistributedSampler, InfiniteDataLoader, seed_worker

import math
import os

import torch
from torch.utils.data import distributed

from ultralytics.utils import RANK


def build_dataloader(
    dataset,
    batch,
    workers,
    shuffle=True,
    rank=-1,
    drop_last=False,
    pin_memory=True,
    multiprocessing_context=None,
    persistent_workers=None,
):
    """Create and return an InfiniteDataLoader for training or validation.

    This is a subway_yolo override of ``ultralytics.data.build.build_dataloader`` that adds two optional,
    Windows-friendly worker controls. Both are only applied when workers are actually used (``num_workers > 0``),
    so ``workers=0`` mode behaves exactly like the upstream implementation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.
        multiprocessing_context (str, optional): Multiprocessing start method for workers ('spawn'/'fork'/'forkserver').
            'spawn' avoids fork-based deadlocks on Windows. None or empty/'none' keeps the torch default.
        persistent_workers (bool, optional): Keep worker processes alive between epochs to avoid re-spawn overhead.
            Requires ``num_workers > 0``; ignored otherwise.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.
    """
    dataset_len = len(dataset)
    batch = min(batch, dataset_len)
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    samples = len(sampler) if sampler is not None else dataset_len
    drop_last = drop_last and bool(batch) and dataset_len % batch != 0
    batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
    nd = torch.cuda.device_count()  # number of CUDA devices
    # Do not create more worker processes than final loader batches. Single-batch loaders run in-process to avoid
    # persistent DataLoader worker pools that add overhead and can stall tiny datasets while holding CUDA context.
    nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)  # number of workers
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)

    # Normalize the multiprocessing start method; treat empty/'none' as "use torch default".
    if isinstance(multiprocessing_context, str):
        multiprocessing_context = multiprocessing_context.strip().lower()
        if multiprocessing_context in {"", "none", "null", "default"}:
            multiprocessing_context = None

    # Worker options are only valid when workers are actually spawned; passing persistent_workers=True with
    # num_workers=0 raises a ValueError in torch, so gate everything on nw > 0 for backward compatibility.
    worker_kwargs = {}
    if nw > 0:
        if multiprocessing_context is not None:
            worker_kwargs["multiprocessing_context"] = multiprocessing_context
        if persistent_workers is not None:
            worker_kwargs["persistent_workers"] = bool(persistent_workers)

    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
        **worker_kwargs,
    )
