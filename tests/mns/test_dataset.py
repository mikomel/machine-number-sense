import pytest
import torch

from mns.dataset import MNSDataset


def test_len():
    dataset = MNSDataset(data_dir='tests/mns')
    assert len(dataset) == 1


@pytest.mark.parametrize('image_size', [80, 160])
def test_getitem(image_size):
    dataset = MNSDataset(data_dir='tests/mns', image_size=image_size)
    iterator = iter(dataset)
    image, target = next(iterator)
    assert image.shape == (3, image_size, image_size)
    assert target.shape == ()
    assert image.dtype == torch.float
    assert target.dtype == torch.long
