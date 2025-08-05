import torch
import torch.nn as nn
import torch.optim as optim

class ModelWithTemperature(nn.Module):
    def __init__(self, model: nn.Module, init_temp: float = 1.0, valid_loader: torch.utils.data.DataLoader = None):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)    # Move the model to the device
        self.val_loader = valid_loader
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x) / self.temperature

    def Temperature(self) -> float:
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits_list.append(self.model(inputs))
                labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_temp = self.temperature.item()
        print(f"Optimal temperature: {optimal_temp:.4f}")
        return optimal_temp

# ─── USAGE ──────────────────────────────────────────────────────────────────────
# wrapper = ModelWithTemperature(trained_model)
# best_T = wrapper.set_temperature(val_loader)
# ────────────────────────────────────────────────────────────────────────────────

