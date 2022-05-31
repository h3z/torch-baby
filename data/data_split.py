from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    train, val = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
    return train, val, None
