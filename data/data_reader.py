import pandas as pd


class DataReader:
    def __init__(self):
        DATA_ROOT = "..."
        datapath = f"{DATA_ROOT}/..."
        self.train = pd.read_pickle(datapath)
