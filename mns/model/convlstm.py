import torch
import torch.nn as nn

from mns.model.layers import ConvBNReLU


class ConvLSTM(nn.Module):
    def __init__(self, image_size: int = 160):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(1, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1)
        )
        dim = (image_size // 80) * 5
        self.lstm = nn.LSTM(input_size=32 * dim * dim, hidden_size=256, num_layers=1, batch_first=True)
        self.score = nn.Linear(256, 99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, width, height = x.size()
        x = x.view(batch_size * num_images, 1, width, height)
        x = self.conv(x)
        x = x.view(batch_size, num_images, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.score(x)
        return x
