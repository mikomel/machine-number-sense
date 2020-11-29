import pytest
import torch
import torch.nn.functional as F
import torchtest

from mns.model import ConvLSTM, ConvMLP, ConvNALU, SCL


@pytest.mark.parametrize('image_size', [80, 160])
@pytest.mark.parametrize('model_builder', [ConvLSTM, ConvMLP, ConvNALU, SCL])
def test_forward(image_size, model_builder):
    x = torch.rand(4, 3, image_size, image_size)
    y = torch.randint(99, (4,), dtype=torch.long)
    model = model_builder(image_size=image_size)
    optimiser = torch.optim.Adam(model.parameters())
    torchtest.test_suite(
        model=model,
        loss_fn=F.cross_entropy,
        optim=optimiser,
        batch=[x, y],
        test_inf_vals=True,
        test_nan_vals=True,
        test_vars_change=True
    )
