import torch
from model import a_model


def get() -> torch.nn.Module:
    return a_model.A_MODEL().to("cuda")
