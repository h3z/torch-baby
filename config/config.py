PROJECT_NAME = "..."
ONLINE = True

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}
