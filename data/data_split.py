import pandas as pd
from typing import List


def split1(df: pd.DataFrame) -> List[pd.DataFrame]:
    p = len(df) // 10
    border1 = p * 7
    border2 = p * 9

    train, val, test = df[:border1], df[border1:border2], df[border2:]
    return train, val, test


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    return split1(df)
