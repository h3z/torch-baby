import random
from typing import List

import numpy as np
import pandas as pd
import torch

from config import config


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
    def __init__(self, data: np.ndarray, is_train: bool) -> None:
        super().__init__(data)
        self.len = len(data)
        self.is_train = is_train

        if config.distributed:
            self.rank = torch.distributed.get_rank()
            self.num_replicas = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

    def __iter__(self) -> List[int]:
        if self.is_train:
            lst = torch.randperm(self.len).tolist()
        else:
            lst = list(range(self.len))

        return iter(self.distributed(lst) if self.is_train else lst)

    def distributed(self, lst):
        return lst[self.rank : len(lst) : self.num_replicas]

    def __len__(self) -> int:
        return self.len


class DataLoader:
    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.data = df.values
        self.is_train = is_train

    def get(self) -> torch.utils.data.DataLoader:
        dataset = Dataset(self.data)
        sampler = Sampler(self.data, is_train=self.is_train)
        batch_size = config["~batch_size"] if self.is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=self.is_train,
        )
