from collections import namedtuple
from torch import nn
from torch.utils import model_zoo

from segmentation_models_pytorch import Unet, UnetPlusPlus

model = namedtuple("model", ["url", "model"])

models = {
    "Unetplusplus_2021feb": model(
        model=UnetPlusPlus(encoder_name="resnet50", classes=1, encoder_weights="imagenet"),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model
