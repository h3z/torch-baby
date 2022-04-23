import pandas as pd
from typing import List


def split1(df: pd.DataFrame) -> List[pd.DataFrame]:
    p = len(df) // 10
    return df[: p * 7], df[p * 7 : p * 9], df[p * 9 :]


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    return split1(df)
