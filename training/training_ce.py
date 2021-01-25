# %%
from typing import Callable, List, Tuple

import os
import torch
import catalyst
from catalyst import utils
from dataset import SegmentationDataset
from utils_custom import compose, pre_transforms, post_transforms, show_examples
from torch.utils.data import DataLoader
import numpy as np
from skimage.io import imread
from PIL import Image
from dataset import get_loaders
from torch import nn
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensor

from catalyst.contrib.nn import DiceLoss, IoULoss, MaskCrossEntropyLoss
from torch import optim

from catalyst.contrib.nn import RAdam, Lookahead

from catalyst.dl import (
    DiceCallback,
    IouCallback,
    CriterionCallback,
    MetricAggregationCallback,
    SupervisedRunner,
)
from catalyst.contrib.callbacks import DrawMasksCallback


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(torch.cuda.is_available())

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)


# %%
from pathlib import Path

ROOT = Path("/thunderdisk") / "data_rene_policistico_2"

image_path = ROOT / "all_images"
mask_path = ROOT / "all_masks"
# test_image_path = ROOT / "test"
# test_mask_path = ROOT / "test_masks"


# %%
ALL_IMAGES = sorted(
    [image_path / img for img in os.listdir(image_path) if img.endswith("jpg")]
)

for fname in ALL_IMAGES:
    fname = os.path.basename(fname)
    image = image_path / fname
    if not image.is_file():
        raise FileNotFoundError(image)
    fname = os.path.splitext(fname)[0]
    image_mask = mask_path / f"{fname}.png"
    if not image_mask.is_file():
        print(f"From image: {image}")
        raise FileNotFoundError(image_mask)

print(f"Images:{len(ALL_IMAGES)}")


# %%
ALL_MASKS = sorted(
    [mask_path / mask for mask in os.listdir(mask_path) if mask.endswith("png")]
)

print(f"Masks:{len(ALL_MASKS)}")


# %%
train_transforms = albu.Compose(
    [
        # albu.RandomCrop(width=256, height=256),
        albu.Resize(512, 512),
        albu.HorizontalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        albu.GridDistortion(p=1),
        # albu.RGBShift(p=1),
        ToTensor(),
    ]
)

valid_transforms = compose([pre_transforms(), post_transforms()])

batch_size = 4  # 16

loaders = get_loaders(
    images=ALL_IMAGES,
    masks=ALL_MASKS,
    random_state=SEED,
    train_transforms_fn=train_transforms,
    valid_transforms_fn=valid_transforms,
    batch_size=batch_size,
    valid_size=0.2,
    train_mask_path=mask_path,
    valid_mask_path=mask_path,
    # num_workers=2,
)


# %%
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)


criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss(),
}


# %%

learning_rate = 0.0005
encoder_learning_rate = 0.0005

layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

num_epochs = 1
# logdir = "./logs/segmentation"
logdir = "logs"
device = utils.get_device()
print(f"device: {device}")

# by default SupervisedRunner uses "features" and "targets",
runner = SupervisedRunner(
    device=device, input_key="image", output_key="logits", input_target_key="mask"
)

callbacks = [
    CriterionCallback(input_key="mask", prefix="loss_dice", criterion_key="dice"),
    CriterionCallback(input_key="mask", prefix="loss_iou", criterion_key="iou"),
    CriterionCallback(input_key="mask", prefix="loss_bce", criterion_key="bce"),
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),
    # metrics
    DiceCallback(input_key="mask"),
    IouCallback(input_key="mask"),
    # visualization
    DrawMasksCallback(
        output_key="logits",
        input_image_key="image",
        input_mask_key="mask",
        summary_step=50,
    ),
]

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    # path to save logs
    logdir=logdir,
    num_epochs=num_epochs,
    main_metric="iou",
    minimize_metric=False,
    verbose=True,
)


TRAIN_IMAGES = sorted(image_path.glob("*.jpg"))
# TEST_IMAGES = sorted(test_image_path.glob("*.jpg"))

# create test dataset
train_dataset = SegmentationDataset(TRAIN_IMAGES, transforms=valid_transforms)
# test_dataset = SegmentationDataset(TEST_IMAGES, transforms=valid_transforms)

num_workers: int = 64

infer_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

logdir_root = Path(os.getcwd())

# print(f"{logdir_root}/checkpoints/best.pth")
# # this get predictions for the whole loader
predictions = np.vstack(
    list(
        map(
            lambda x: x["logits"].cpu().numpy(),
            runner.predict_loader(
                loader=infer_loader, resume=f"{logdir_root}/logs/checkpoints/best.pth"
            ),
        )
    )
)

print(type(predictions))
print(predictions.shape)

threshold = 0.5
max_count = 5

for i, (features, logits) in enumerate(zip(train_dataset, predictions)):
    image = utils.tensor_to_ndimage(features["image"], denormalize=False)

    # filename_mask = os.path.splitext(features["filename"])[0]
    filename_mask = os.path.splitext(features["filename_mask"])[0]
    gt = imread(mask_path / f"{filename_mask}.png")
    gt_res = gt.copy()
    gt_res.resize((224, 224))
    gt_im = Image.fromarray(gt_res * 255)

    mask_ = torch.from_numpy(logits[0]).sigmoid()
    mask = utils.detach(mask_ > threshold).astype("uint8")

    #     # Replace mask with real image
    #     # filename_mask = os.path.splitext(features["filename_img"])[0]
    #     # mask = imread(test_image_path / f"{filename_mask}.jpg")

    show_examples(
        name="",
        image=image,
        mask=mask,
        gt=gt_res,
        save=True,
        fig_path=f"predictions/{model.name}/",
        fig_name=f"prediction_{i}.png",
        overlay=True,
    )

    if i >= max_count:
        break
