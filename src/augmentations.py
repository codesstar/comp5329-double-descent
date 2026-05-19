import torch
import torchvision.transforms as T
from typing import Tuple


# ── 标准预处理（所有策略共用）──────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_transforms(augment: str, dataset: str = "cifar10") -> Tuple[T.Compose, T.Compose]:
    """返回 (train_transform, test_transform)"""
    mean = CIFAR10_MEAN if dataset == "cifar10" else CIFAR100_MEAN
    std  = CIFAR10_STD  if dataset == "cifar10" else CIFAR100_STD

    normalize = T.Normalize(mean, std)
    test_tf = T.Compose([T.ToTensor(), normalize])

    if augment == "none":
        train_tf = T.Compose([T.ToTensor(), normalize])

    elif augment == "flip_crop":
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            normalize,
        ])

    elif augment == "cutout":
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            normalize,
            Cutout(n_holes=1, length=16),
        ])

    elif augment == "mixup":
        # MixUp 在 train.py 的 batch 层面处理，这里只做基础变换
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            normalize,
        ])

    elif augment == "color_jitter":
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            normalize,
        ])

    else:
        raise ValueError(f"Unknown augment: {augment}")

    return train_tf, test_tf


class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
            x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        return img * mask.unsqueeze(0)


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对一个 batch 做 MixUp，返回混合后的 (x, soft_labels)"""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    # soft one-hot labels
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mix, y_mix


AUGMENT_NAMES = ["none", "flip_crop", "cutout", "mixup", "color_jitter"]
