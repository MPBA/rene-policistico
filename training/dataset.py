import os
from typing import List

from catalyst import utils
from skimage.io import imread
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images, masks=None, mask_path=None, transforms=None) -> None:
        self.images = images
        self.masks = masks
        self.mask_path = mask_path
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        name = image_path.name
        image = utils.imread(image_path)
        mask = None
        if self.transforms is not None:
            if self.masks is not None:
                mask = imread(self.mask_path / f"{os.path.splitext(name)[0]}.png")
                transformed = self.transforms(image=image, mask=mask)
                image_tf = transformed["image"]
                mask_tf = transformed["mask"]
                result = {"image": image_tf, "mask": mask_tf}

            else:
                transformed = self.transforms(
                    image=image,
                )
                image_tf = transformed["image"]
                result = {"image": image_tf}
        else:
            result = {"image": image, "mask": mask}
        result["filename_img"] = image_path.name
        result["filename_mask"] = f"{os.path.splitext(name)[0]}.png"

        return result


import collections
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import numpy as np


def get_loaders(
    images,
    masks,
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 4,
    train_transforms_fn=None,
    valid_transforms_fn=None,
    train_mask_path=None,
    valid_mask_path=None,
) -> dict:

    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)
    np_masks = np.array(masks)

    train_dataset = SegmentationDataset(
        images=np_images[train_indices].tolist(),
        masks=np_masks[train_indices].tolist(),
        transforms=train_transforms_fn,
        mask_path=train_mask_path,
    )

    valid_dataset = SegmentationDataset(
        images=np_images[valid_indices].tolist(),
        masks=np_masks[valid_indices].tolist(),
        transforms=valid_transforms_fn,
        mask_path=valid_mask_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
