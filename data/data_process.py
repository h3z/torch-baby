import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        df = df.fillna(0)
        self.scaler.fit(df.values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(0)
        df.iloc[:, :] = self.scaler.transform(df.values)
        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(preds)
