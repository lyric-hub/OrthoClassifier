import timm
from torch import nn

def getmodel(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True
) -> nn.Module:
    model = timm.create_model(
        model_name  = model_name,
        pretrained  = pretrained,
        num_classes = num_classes,
    )
    return model
