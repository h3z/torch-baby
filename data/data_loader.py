import random
from typing import List

import numpy as np
import pandas as pd
import torch
import wandb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.data[index]
        # 这里注意类型不能是 object
        return x, y

    def __len__(self):
        return self.len


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data: np.ndarray, shuffle: bool) -> None:
        super().__init__(data)
        self.len = len(data)
        self.shuffle = shuffle

    def __iter__(self) -> List[int]:
        lst = list(range(self.len))
        if self.shuffle:
            random.shuffle(lst)
        for i in lst:
            yield i

    def __len__(self) -> int:
        return self.len


class DataLoader:
    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.data = df.values
        self.is_train = is_train

    def get(self) -> torch.utils.data.DataLoader:
        dataset = Dataset(self.data)
        sampler = Sampler(self.data, shuffle=self.is_train)
        batch_size = wandb.config["~batch_size"] if self.is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=self.is_train,
        )
