import torch
from torch import nn

from mns.model.layers import ConvBNReLU, FFResidual, Scattering


class SCL(nn.Module):
    def __init__(self, image_size: int = 80):
        super(SCL, self).__init__()
        self.scattering = Scattering()
        self.conv = nn.Sequential(
            ConvBNReLU(1, 16, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(16, 16, kernel_size=3, padding=1),
            ConvBNReLU(16, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1)
        )
        conv_dimension = 40 * (image_size // 80) * 40 * (image_size // 80)
        self.conv_projection = nn.Sequential(
            nn.Linear(conv_dimension, 80),
            nn.ReLU(inplace=True)
        )
        self.ff_object = FFResidual(80)
        self.attribute_network = nn.Sequential(
            nn.Linear(32 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8)
        )
        self.ff_attribute = FFResidual(80)
        self.relation_network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        self.ff_relation = FFResidual(5 * 80)
        self.score = nn.Linear(5 * 80, 99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size()

        x = x.view(batch_size * num_panels, 1, height, width)
        x = self.conv(x)
        x = x.view(batch_size, num_panels, 32, -1)
        x = self.conv_projection(x)
        x = self.ff_object(x)

        x = self.scattering(x, num_groups=10)
        x = self.attribute_network(x)
        x = x.view(batch_size, num_panels, 10 * 8)
        x = self.ff_attribute(x)

        x = self.scattering(x, num_groups=80)
        x = self.relation_network(x)
        x = x.view(batch_size, 80 * 5)
        x = self.ff_relation(x)
        x = self.score(x).squeeze()
        return x
