import torch

from config import config


def get():
    if config["~loss"] == "mse":
        return torch.nn.MSELoss()
    elif config["~loss"] == "bce":
        return torch.nn.BCELoss()
