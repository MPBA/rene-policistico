# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.domain_templates.unet import UNet
from pytorch_lightning.loggers import WandbLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Cisti(Dataset):
    IMAGE_PATH = "all_images"
    MASK_PATH = "all_masks"

    def __init__(
        self, data_path: str, split: str, img_size: tuple = (1024, 1024), transform=None
    ):
        self.img_size = img_size
        self.ignore_index = None
        self.transform = transform

        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_path, self.MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set (80/20)
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == "train":
            idxs = [idx for idx in range(n_items) if idx not in idxs]
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx]).convert("L")
        mask = mask.resize(self.img_size)
        mask = np.array(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class SegModel(pl.LightningModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 4,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            num_classes=1,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.35675976, 0.37380189, 0.3764753],
                    std=[0.32064945, 0.32098866, 0.32325324],
                ),
            ]
        )
        self.trainset = Cisti(self.data_path, split="train", transform=self.transform)
        self.validset = Cisti(self.data_path, split="valid", transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        print(mask)
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask)
        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {
            "log": log_dict,
            "val_loss": log_dict["val_loss"],
            "progress_bar": log_dict,
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)


model = SegModel(data_path="/thunderdisk/data_rene_policistico_2")
trainer = pl.Trainer(gpus=1)
trainer.fit(model)


# if __name__ == "__main__":
# parser = ArgumentParser(add_help=False)
# parser = SegModel.add_model_specific_args(parser)
# hparams = parser.parse_args()

# main(hparams)
