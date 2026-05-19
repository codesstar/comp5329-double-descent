import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, base_width: int, num_classes: int = 10):
        super().__init__()
        w = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(w, w, 2, stride=1)
        self.layer2 = self._make_layer(w, w * 2, 2, stride=2)
        self.layer3 = self._make_layer(w * 2, w * 4, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 4, num_classes)

    def _make_layer(self, in_c: int, out_c: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 8档 base_width，覆盖欠参数化→过参数化区间
RESNET_WIDTHS = [4, 8, 16, 24, 32, 48, 64, 96]
