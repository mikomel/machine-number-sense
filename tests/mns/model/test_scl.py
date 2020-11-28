import pytest
import torch

from mns.model import SCL


@pytest.mark.parametrize('image_size', [80, 160])
def test_forward(image_size):
    x = torch.rand(4, 3, image_size, image_size)
    scl = SCL(image_size=image_size)
    logits = scl(x)
    assert logits.shape == (4, 99)
