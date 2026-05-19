import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, width: int, num_classes: int = 10, input_dim: int = 3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 8档宽度，覆盖欠参数化→过参数化区间
MLP_WIDTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
