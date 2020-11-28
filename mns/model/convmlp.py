import torch
import torch.nn as nn

from mns.model.layers import ConvBNReLU


class ConvMLP(nn.Module):
    def __init__(self, image_size: int = 160):
        super(ConvMLP, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1)
        )
        dim = (image_size // 80) * 5
        self.mlp = nn.Sequential(
            nn.Linear(32 * dim * dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 99)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x
