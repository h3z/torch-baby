import torch

from config import config


def get(model: torch.nn.Module):
    if config["~optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["~lr"])
