from pathlib import Path
from typing import List, Dict, Any, Tuple

import albumentations as albu
import numpy as np
import torch

# from iglovikov_helper_functions.utils.image_utils import load_rgb, load_grayscale
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from torch.utils.data import Dataset
from PIL import Image
from skimage.io import imread


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        length: int = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]
        # print("-------------------------------------------------")
        # print("image_PATH", image_path)
        # print("mask_PATH", mask_path)
        # print("-------------------------------------------------")

        image = np.array(Image.open(image_path))
        mask = imread(mask_path)
        # print("-------------------------------------------------")
        # print("IMG", image)
        # print("MASK", mask)
        # print("-------------------------------------------------")

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": image_path.stem,
            "features": image_to_tensor(image),
            "masks": torch.unsqueeze(mask, 0).float(),
        }
