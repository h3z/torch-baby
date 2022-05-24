import pandas as pd
import numpy as np
from typing import List
import torch
import random
import wandb


class DataLoader:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: np.ndarray):
            self.data = data
            self.len = len(self.data)

        def __getitem__(self, index):
            x = self.data[index]
            y = self.data[index]
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

    def __init__(self, df: pd.DataFrame, is_train=False) -> None:
        self.data = df.values
        self.is_train = is_train

    def get(self) -> torch.utils.data.DataLoader:
        dataset = self.Dataset(self.data)
        sampler = self.Sampler(self.data, shuffle=self.is_train)
        batch_size = wandb.config["~batch_size"] if self.is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=self.is_train,
        )
