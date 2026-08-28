import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BaselineCNN(nn.Module):

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 32x32 -> 16x16
            _conv_bn_relu(3, 32),
            _conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Stage 2: 16x16 -> 8x8
            _conv_bn_relu(32, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Stage 3: 8x8 -> 4x4
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = out + identity
        return self.relu(out)


class ResNetCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = _conv_bn_relu(3, 32)

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1, dropout=dropout * 0.3),
            ResidualBlock(32, 32, stride=1, dropout=dropout * 0.3),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, dropout=dropout * 0.3),
            ResidualBlock(64, 64, stride=1, dropout=dropout * 0.3),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout * 0.3),
            ResidualBlock(128, 128, stride=1, dropout=dropout * 0.3),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 10, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    if name == "resnet":
        return ResNetCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name!r}. Choose 'baseline' or 'resnet'.")


if __name__ == "__main__":
    # Quick shape sanity check.
    for name in ("baseline", "resnet"):
        m = build_model(name)
        y = m(torch.randn(2, 3, 32, 32))
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name:10s} output={tuple(y.shape)} params={n_params:,}")
