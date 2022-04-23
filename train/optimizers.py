import torch
import wandb


def get(model: torch.nn.Module):
    if wandb.config._optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=wandb.config._lr)
