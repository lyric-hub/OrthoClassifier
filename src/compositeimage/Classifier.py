import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dvclive import Live
from compositeimage import load_config

class Classifier:
    def __init__(self, live_dir: str = None):
        self.cfg = load_config() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.live_dir = live_dir or "dvclive/resnet18"
        self.BuildModel()
        self.BuildLoaders()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.cfg.train.lr)

    def BuildModel(self):
        cfg = self.cfg
        init_path = cfg.init_model_path
        unfreeze_modules = cfg.get("unfreeze_modules", ["fc"])  # default to unfreezing 'fc'
        model = torch.load(init_path, map_location=self.device)
        for param in model.parameters():
            param.requires_grad = False

        for module_name in unfreeze_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                for param in module.parameters():
                    param.requires_grad = True
            else:
                raise ValueError(f"Module '{module_name}' not found in model.")
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_params}")

        self.model = model.to(self.device)


    def BuildLoaders(self):
        cfg = self.cfg
        data_root = os.path.abspath(cfg.data_dir)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.train_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "train"), tf),
            batch_size=cfg.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "val"), tf),
            batch_size=cfg.batch_size
        )
        self.test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "test"), tf),
            batch_size=cfg.batch_size, shuffle=False
        )

    def fit(self, epochs: int = None, save_path: str = None):
        cfg = self.cfg
        epochs = epochs or cfg.epochs
        live = Live(self.live_dir)
        best_acc, counter = 0.0, 0
        final_save = save_path or "models/resnet18.pt"
        os.makedirs(os.path.dirname(final_save), exist_ok=True)

        with live:
            for epoch in range(epochs):
                self.model.train()
                train_losses = []
                for imgs, labels in self.train_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(loss.item())

                avg_train_loss = sum(train_losses) / len(train_losses)
                live.log_metric("train_loss", avg_train_loss)


                self.model.eval()
                val_losses = []
                preds, trues = [], []
                with torch.no_grad():
                    for imgs, labels in self.val_loader:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        out = self.model(imgs)
                        val_loss = self.criterion(out, labels)
                        val_losses.append(val_loss.item())
                        preds.extend(out.argmax(1).cpu().tolist())
                        trues.extend(labels.cpu().tolist())

                avg_val_loss = sum(val_losses) / len(val_losses)
                live.log_metric("val_loss", avg_val_loss)
                val_acc = accuracy_score(trues, preds)
                live.log_metric("val_accuracy", val_acc)
                live.next_step()

                if val_acc > best_acc:
                    best_acc, counter = val_acc, 0
                    torch.save(self.model, final_save)
                else:
                    counter += 1
                    if counter >= cfg.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return {"best_val_accuracy": best_acc}



