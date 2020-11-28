import torch
from torch import nn

from mns.model.layers import ConvBNReLU
from mns.model.nalu import NALUCell


class ConvNALU(nn.Module):
    def __init__(self, image_size: int = 160):
        super(ConvNALU, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1)
        )
        dim = 32 * (5 * (image_size // 80)) ** 2
        self.nalu = NALUCell(dim, 99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1, 3)
        x = self.nalu(x)
        return x
