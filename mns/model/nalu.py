import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

""" Source: https://github.com/kevinzakka/NALU-pytorch """


class NeuralAccumulatorCell(nn.Module):
    """A Neural Accumulator (NAC) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )


class NALUCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input: torch.Tensor):
        a = self.nac(input)
        g = F.linear(input, self.G, self.bias).sigmoid()
        add_sub = g * a
        log_input = (input.abs() + self.eps).log()
        m = self.nac(log_input).exp()
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )
