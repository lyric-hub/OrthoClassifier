# model.py
import timm
from torch import nn

def get_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
