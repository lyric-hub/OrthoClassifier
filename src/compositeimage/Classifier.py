import os
import json
from io import BytesIO
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, classification_report
from dvclive import Live
from compositeimage import load_config
import matplotlib.pyplot as plt
#from compositeimage import ModelWithTemperature

class Classifier:
    def __init__(self):
        self.cfg    = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.live   = Live(self.cfg.dvclive.dir or "dvclive/resnet18")
        self._build_model()
        self._build_loaders()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.train.lr
        )

    def _build_model(self):
        model = torch.load(self.cfg.train.init_model_path, map_location=self.device)
        for p in model.parameters():
            p.requires_grad = False
        for m in self.cfg.model.get("unfreeze_modules", ["fc"]):
            if not hasattr(model, m):
                raise ValueError(f"Module '{m}' not in model")
            for p in getattr(model, m).parameters():
                p.requires_grad = True
        print("Trainable parameters:", [n for n,p in model.named_parameters() if p.requires_grad])
        self.model = model.to(self.device)

    def _build_loaders(self):
        data_root = os.path.abspath(self.cfg.data.data_dir)
        self.train_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.predict_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.train_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "train"), self.train_tf),
            batch_size=self.cfg.train.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "val"), self.predict_tf),
            batch_size=self.cfg.train.batch_size
        )
        self.test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(data_root, "test"), self.predict_tf),
            batch_size=self.cfg.train.batch_size, shuffle=False
        )

    def fit(self):
        epochs    = self.cfg.train.epochs
        best_acc  = 0.0
        os.makedirs(os.path.dirname(self.cfg.save.path), exist_ok=True)

        with self.live:
            for epoch in range(epochs):
                self.model.train()
                train_losses = []
                for imgs, labels in self.train_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    out   = self.model(imgs)
                    loss  = self.criterion(out, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(loss.item())
                avg_train_loss = sum(train_losses)/len(train_losses)
                self.live.log_metric("train_loss", avg_train_loss)

                self.model.eval()
                val_losses, preds, trues = [], [], []
                with torch.no_grad():
                    for imgs, labels in self.val_loader:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        out = self.model(imgs)
                        val_losses.append(self.criterion(out, labels).item())
                        preds.extend(out.argmax(1).cpu().tolist())
                        trues.extend(labels.cpu().tolist())
                avg_val_loss = sum(val_losses)/len(val_losses)
                val_acc      = accuracy_score(trues, preds)
                self.live.log_metric("val_loss", avg_val_loss)
                self.live.log_metric("val_accuracy", val_acc)
                self.live.next_step()

                if val_acc > best_acc:
                    best_acc, counter = val_acc, 0
                    torch.save(self.model, self.cfg.save.path)
                else:
                    counter += 1
                    if counter >= self.cfg.train.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return self.model
    
    
    def predict(self, model_wrapper, data, transform=None):
        model = model_wrapper
        device = model.device
        model.eval()

        if transform is None:
            transform = self.predict_tf

        if isinstance(data, DataLoader):
            all_probs = []
            for image in data:
                inputs = image[0] if isinstance(image, (list, tuple)) else image
                inputs = inputs.to(device)
                probs = model.predict_proba(inputs)
                all_probs.append(probs)
            return torch.cat(all_probs, dim=0)

        if isinstance(data, torch.Tensor):
            data = data.to(device)
            return model.predict_proba(data)

        if isinstance(data, str):
            if data.startswith('http'):
                response = requests.get(data)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                img = Image.open(data).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            return model.predict_proba(input_tensor)

        if isinstance(data, Image.Image):
            input_tensor = transform(data).unsqueeze(0).to(device)
            return model.predict_proba(input_tensor)

        raise TypeError(f"Unsupported data type: {type(data)}")



    def metrics(self):

        self.live.log_metric("precision", precision_score(self.trues, self.preds))
        self.live.log_metric("recall",    recall_score(self.trues, self.preds))
        self.live.log_metric("f1_score",  f1_score(self.trues, self.preds))

        fpr, tpr, _ = roc_curve(self.trues, self.probas)
        auc         = roc_auc_score(self.trues, self.probas)
        self.live.log_metric("roc_auc", auc)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr); ax.plot([0,1],[0,1], "--")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
        fig.savefig("roc_curve.png")
        self.live.log_image("roc_curve", "roc_curve.png")
        self.live.next_step()

    