import wandb, utils, os
from model import models
from data import data_split, data_process, data_loader, data_reader
from train import train, early_stopping, wandb_callback, losses, optimizers
from config import config

utils.fix_random()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parameters():
    return {
        "_lr": 0.001,
        "_batch_size": 128,
        "_epochs": 200,
        "_early_stopping_patience": 3,
        "_optimizer": "adam",
        "_loss": "mse",
    }


def main():
    wandb.init(config=get_parameters(), **config.__wandb__)
    print(wandb.config)

    # read csv
    df = data_reader.DataReader().train

    # split
    train_df, val_df, test_df = data_split.split(df)

    # preprocess
    processor = data_process.DataProcess(train_df)
    train_df = processor.preprocess(train_df)
    val_df = processor.preprocess(val_df)
    test_df = processor.preprocess(test_df)

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df).get(is_train=True)
    val_ds = data_loader.DataLoader(val_df).get()
    test_ds = data_loader.DataLoader(test_df).get()

    model = models.get()

    # train
    criterion = losses.get()
    optimizer = optimizers.get()
    callbacks = [early_stopping.EarlyStopping(), wandb_callback.WandbCallback()]

    for epoch in range(wandb.config._epochs):
        loss = train.epoch_train(model, optimizer, train_ds, criterion, callbacks)
        val_loss = train.epoch_val(model, val_ds, criterion, callbacks)
        print(epoch, ": train_loss", loss, "val_loss", val_loss)

        res = [c.on_epoch_end(loss, val_loss, model) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model) for c in callbacks]

    # predict
    preds, gts = train.predict(model, test_ds)

    # post process
    preds, gts = processor.postprocess(preds), processor.postprocess(gts)

    wandb.finish()


if __name__ == "__main__":
    main()