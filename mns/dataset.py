import glob
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MNSDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 160):
        self.data_dir = data_dir
        self.filenames = [f for f in glob.glob(os.path.join(data_dir, '*.npz'))]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = self.filenames[idx]
        data = np.load(filepath)
        image = data['image']
        target = data['target'] - 1
        if self.image_size == 80:
            image = image[:, ::2, ::2]
        image = image.astype('float32') / 255.0
        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)
        return image, target
