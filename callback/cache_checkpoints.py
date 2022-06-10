import torch

from callback.callback import Callback
from config import config


class CacheCheckpoints(Callback):
    def __init__(self) -> None:
        self.n_iter = 0

    def on_epoch_end(self, loss, val_loss, model) -> bool:
        name = f"some_name_"
        cuda = f"cuda_{config.cuda_rank}_" if config.distributed else ""
        torch.save(
            model.checkpoint(),
            f"{config.checkpoints_dir}/{name}{cuda}{self.n_iter}.pt",
        )
        self.n_iter += 1
        return True
