"""
CIFAR-10 data pipeline built on torchvision.datasets.CIFAR10 +
torch.utils.data.DataLoader.

Provides:
  - normalization using precomputed CIFAR-10 channel statistics
  - standard augmentations for training (random crop + horizontal flip)
  - an efficient DataLoader setup (pin_memory, persistent_workers, num_workers)
  - a train/val split carved out of the 50k training images
"""
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms

from config import CIFAR10_MEAN, CIFAR10_STD, TrainConfig


def build_transforms(augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns (train_transform, eval_transform)."""
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    else:
        train_tf = eval_tf

    return train_tf, eval_tf


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / val / test DataLoaders.

    The 50,000-image CIFAR-10 training set is split into train/val according
    to cfg.val_split (held out deterministically using cfg.seed). The 10,000
    image test set is used only for final evaluation.
    """
    train_tf, eval_tf = build_transforms(cfg.augment)

    full_train_raw = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=eval_tf)

    n_total = len(full_train_raw)
    n_val = int(n_total * cfg.val_split)

    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(full_train_raw, train_idx)   # augmented transform
    val_set = Subset(full_train_eval, val_idx)       # clean transform, no augmentation leakage

    common_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False,
                             **common_kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size * 2, shuffle=False,
                              **common_kwargs)

    return train_loader, val_loader, test_loader


def get_dataloaders_ddp(
    cfg: TrainConfig, rank: int, world_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, DistributedSampler]:
    """
    Distributed-training variant of get_dataloaders(): shards the training
    set across `world_size` processes via DistributedSampler (each rank sees
    a disjoint 1/world_size slice per epoch). Validation/test loaders are
    also sharded (each rank evaluates a slice) purely as a throughput
    optimization -- metrics must be all-reduced across ranks by the caller
    to get the true dataset-wide loss/accuracy.

    Returns (train_loader, val_loader, test_loader, train_sampler).
    train_sampler.set_epoch(epoch) must be called before each epoch so the
    shuffling differs per epoch (otherwise every rank replays the same order
    every epoch, which hurts convergence).
    """
    train_tf, eval_tf = build_transforms(cfg.augment)

    full_train_raw = datasets.CIFAR10(root=cfg.data_dir, train=True, download=(rank == 0), transform=train_tf)
    full_train_eval = datasets.CIFAR10(root=cfg.data_dir, train=True, download=(rank == 0), transform=eval_tf)
    test_set = datasets.CIFAR10(root=cfg.data_dir, train=False, download=(rank == 0), transform=eval_tf)

    n_total = len(full_train_raw)
    n_val = int(n_total * cfg.val_split)
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(full_train_raw, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=cfg.seed, drop_last=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    common_kwargs = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=train_sampler,
                               drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, sampler=val_sampler, **common_kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size * 2, sampler=test_sampler, **common_kwargs)

    return train_loader, val_loader, test_loader, train_sampler


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Undo normalization for visualization. Expects a (C,H,W) tensor."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)
