import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from typing import Tuple
from .augmentations import get_transforms

DATA_ROOT = "/Users/callum/Desktop/homework/data"


def get_loaders(
    dataset: str,
    augment: str,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_transforms(augment, dataset)

    if dataset == "cifar10":
        cls = datasets.CIFAR10
        n_classes = 10
    elif dataset == "cifar100":
        cls = datasets.CIFAR100
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_ds = cls(DATA_ROOT, train=True,  transform=train_tf, download=True)
    test_ds  = cls(DATA_ROOT, train=False, transform=test_tf,  download=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader, n_classes
