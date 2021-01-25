import os
import random

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensor
from catalyst import utils
from PIL import Image
from skimage.io import imread
from pathlib import Path


def show_examples(
    name: str,
    image: np.ndarray,
    mask: np.ndarray,
    gt: np.ndarray,
    overlay=False,
    save=False,
    fig_path=None,
    fig_name=None,
):
    plt.figure(figsize=(20, 28))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 3, 2)
    alpha=0.5 if overlay else 1
    if overlay:
        plt.imshow(image, alpha=alpha)
    plt.imshow(mask, alpha=alpha)
    plt.title(f"Mask: {name}")

    plt.subplot(1, 3, 3)
    if overlay:
        plt.imshow(image, alpha=alpha)
    plt.imshow(gt, alpha=alpha)
    plt.title(f"GT: {name}")

    if save:
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, fig_name))


def show(index: int, images, masks, gts, mask_path, transforms=None) -> None:
    image_path = images[index]
    name = image_path.name
    image = utils.imread(image_path)

    gt = imread(mask_path / f"{os.path.splitext(name)[0]}.png")
    mask = imread(mask_path / f"{os.path.splitext(name)[0]}.png")
    #     mask = utils.imread(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask, gt)


def show_random(images, masks, gts, mask_path, transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, gts, mask_path, transforms=None)


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        # albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # albu.GridDistortion(p=0.3),
        # albu.HueSaturationValue(p=0.3),
    ]

    return result


def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose(
        [
            albu.SmallestMaxSize(pre_size, p=1),
            albu.RandomCrop(image_size, image_size, p=1),
        ]
    )

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose(
        [
            albu.LongestMaxSize(pre_size, p=1),
            albu.RandomCrop(image_size, image_size, p=1),
        ]
    )

    # Converts the image to a square of size image_size x image_size
    result = [albu.OneOf([random_crop, rescale, random_crop_big], p=1)]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


def show_random_transform(transforms, image_list):
    # transforms = compose([resize_transforms(), hard_transforms(), post_transforms()])
    length = len(image_list)
    index = random.randint(0, length - 1)
    IMG = image_list[index]

    plt.figure(figsize=(20, 28))
    plt.subplot(1, 2, 1)
    plt.imshow(utils.imread(IMG))
    plt.title(f"Image: {IMG}")

    transformed = transforms(image=utils.imread(IMG))
    transformed_image = transformed["image"]
    TRANSFORMED = np.moveaxis((np.array(transformed_image) * 255).astype("uint8"), 0, 2)

    plt.subplot(1, 2, 2)
    plt.imshow(Image.fromarray(TRANSFORMED))
    plt.title(f"Transformed")

    Image.fromarray(TRANSFORMED).save("prova.png")


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title("Original mask", fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title("Transformed image", fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed mask", fontsize=fontsize)
