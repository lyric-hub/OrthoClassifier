from torchinfo import summary
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("models/resnet18.pt", map_location="cpu")
model.to(device)

summary(model, input_size=(1, 3, 224, 224), device=device.type)
print("Model summary printed successfully.")

for param in model.parameters():
    param.requires_grad = False

