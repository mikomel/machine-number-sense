![image](mns.png)

# Machine Number Sense
PyTorch implementation of neural networks for solving problems from the Machine Number Sense dataset [1].
Dataset and official implementation of baseline models can be found in [this repo](https://github.com/zwh1999anne/Machine-Number-Sense-Dataset), created by paper authors.

## Setup
```bash
$ pip install machine_number_sense
```

## Usage

### Baseline models

MLP:
```python
import torch

from mns.model import ConvMLP

x = torch.rand(4, 3, 80, 80)
mlp = ConvMLP(image_size=80)
logits = mlp(x)
logits  # torch.Tensor with shape (4, 99)
```

LSTM:
```python
import torch

from mns.model import ConvLSTM

x = torch.rand(4, 3, 80, 80)
lstm = ConvLSTM(image_size=80)
logits = lstm(x)
logits  # torch.Tensor with shape (4, 99)
```

## Unit tests
```bash
$ python -m pytest tests
```

## Bibliography
[1] Zhang, Wenhe, et al. "Machine number sense: A dataset of visual arithmetic problems for abstract and relational reasoning." Proceedings of the AAAI Conference on Artificial Intelligence. 2020.

## Citations
```bibtex
@inproceedings{zhang2020machine,
  title={Machine number sense: A dataset of visual arithmetic problems for abstract and relational reasoning},
  author={Zhang, Wenhe and Zhang, Chi and Zhu, Yixin and Zhu, Song-Chun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={02},
  pages={1332--1340},
  year={2020}
}
```
