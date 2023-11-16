from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


def configure_callbacks(model_name):
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename=model_name + "_{epoch}_{val_loss:.6f}",
        dirpath="./checkpoint/" + str(datetime.now()),
        save_top_k=10,
        save_weights_only=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    return [lr_monitor, checkpoint]


def init_trainer(model_name, tensorboard_path):
    num_epoch = 10
    trainer = L.Trainer(
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir=tensorboard_path),
        max_epochs=num_epoch,
        accelerator="gpu",
        check_val_every_n_epoch=1,
        callbacks=configure_callbacks(model_name),
    )
    return trainer, num_epoch
