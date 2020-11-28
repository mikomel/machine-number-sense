import pytest
import torch

from mns.model import ConvNALU


@pytest.mark.parametrize('image_size', [80, 160])
def test_forward(image_size):
    x = torch.rand(4, 3, image_size, image_size)
    nalu = ConvNALU(image_size=image_size)
    logits = nalu(x)
    assert logits.shape == (4, 99)
