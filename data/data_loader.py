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

        def __getitem__(self, index):
            x = self.data[index]
            y = self.data[index]
            return x, y

        def __len__(self):
            return len(self.data)

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

    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df.values

    def get(self, is_train=False) -> torch.utils.data.DataLoader:
        dataset = self.Dataset(self.data)
        sampler = self.Sampler(self.data, shuffle=is_train)
        batch_size = wandb.config._batch_size if is_train else len(dataset)

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=is_train,
        )