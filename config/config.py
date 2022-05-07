PROJECT_NAME = "..."
ONLINE = True

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    # mkdir /wandb/PROJECT_NAME
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}


RANDOM_STATE = 42
