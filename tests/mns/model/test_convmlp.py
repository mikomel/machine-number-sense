import pytest
import torch

from mns.model import ConvMLP


@pytest.mark.parametrize('image_size', [80, 160])
def test_forward(image_size):
    x = torch.rand(4, 3, image_size, image_size)
    mlp = ConvMLP(image_size=image_size)
    logits = mlp(x)
    assert logits.shape == (4, 99)
