import torch
import wandb


def get():
    if wandb.config._loss == "mse":
        return torch.nn.MSELoss()
