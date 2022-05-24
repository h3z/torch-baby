import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.numerical_cols = [...]
        self.scaler.fit(df[self.numerical_cols].fillna(0).values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_cols = self.numerical_cols
        df[numerical_cols] = self.scaler.transform(df[numerical_cols].fillna(0).values)

        col = "..."
        t = pd.get_dummies(df[col]).rename(columns={0: f"{col}_0", 1: f"{col}_1"})
        df = df.drop(columns=col).join(t)

        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(preds)
