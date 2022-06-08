import torch

from config import config
from model import a_model


def get() -> torch.nn.Module:
    model = a_model.A_MODEL().to("cuda")
    if config.distributed:
        return torch.nn.parallel.DistributedDataParallel(model)
