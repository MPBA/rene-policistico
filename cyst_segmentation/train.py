import argparse
import os
from pathlib import Path
import yaml
from utils import object_from_dict
from experiment import SegmentCyst
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# wandb.login()

# wandb.init(project="cyst_segmentation", entity="bussolacompass")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


args = get_args()

with open(args.config_path) as f:
    hparams = yaml.load(f, Loader=yaml.SafeLoader)

image_path = hparams["image_path"]
mask_path = hparams["mask_path"]
model = SegmentCyst(hparams)

Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

checkpoint_callback = ModelCheckpoint(
    filepath="/thunderdisk/cyst_checkpoints",
    monitor="val_iou",
    verbose=True,
    mode="max",
    save_top_k=-1,
)

trainer = pl.Trainer(
    gpus=4,
    max_epochs=100,
    distributed_backend="ddp",  # DistributedDataParallel
    progress_bar_refresh_rate=1,
    benchmark=True,
    callbacks=[checkpoint_callback],
    precision=16,
    gradient_clip_val=5.0,
    num_sanity_val_steps=5,
    sync_batchnorm=True,
    logger=WandbLogger(hparams["experiment_name"]),
    #  resume_from_checkpoint: /epoch=67.ckpt
)

trainer.fit(model)
