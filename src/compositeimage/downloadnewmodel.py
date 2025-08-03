from compositeimage import load_config
from compositeimage import getmodel
import torch
import os


cfg = load_config()

model = getmodel(
    num_classes = cfg.model.num_classes,
    model_name  = cfg.model.name,
    pretrained  = cfg.model.pretrained
)

os.makedirs("models", exist_ok=True)
save_path = os.path.join("models", f"{cfg.model.name}.pt")
torch.save(model, save_path)
print(f"Model weights saved to {save_path}")