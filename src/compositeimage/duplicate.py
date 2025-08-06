import os
import torch
import imagehash
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from compositeimage import load_config

class DuplicatePredictor:
    def __init__(self):
        self.cfg=load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_models()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.label_map = {
            0: 'frontal_rest',
            1: 'frontal_smile',
            2: 'intraoral_front',
            3: 'intraoral_left',
            4: 'intraoral_right',
            5: 'lower_jaw',
            6: 'upper_jaw',
        }

    def _load_model(self):
        model = torch.load(self.cfg.path, map_location='cpu')
        model.to(self.device)
        model.eval()
        return model

    def predict_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
        return pred.item(), confidence.item()

    def remove_duplicates(self, images_with_preds):
        hash_dict = defaultdict(list)
        for img_path, pred_label, prob in images_with_preds:
            img = Image.open(img_path).resize((128, 128)).convert('L')
            h = imagehash.phash(img)
            hash_dict[str(h)].append((img_path, pred_label, prob))

        unique = []
        print("\n Duplicate Removal Log:")
        for duplicates in hash_dict.values():
            if len(duplicates) > 1:
                print("Duplicate group:")
                for img_path, label, prob in duplicates:
                    print(f" - {os.path.basename(img_path)} ({label}, {prob:.2f})")
                print(f" Keeping: {os.path.basename(duplicates[0][0])} (highest confidence)\n")
            duplicates.sort(key=lambda x: -x[2])  # highest confidence first
            unique.append(duplicates[0])
        return unique
    
    def UniqueImages(self, img_dir):
        image_dir = image_dir
        predictions = []

        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            pred_idx, conf = self.predict_image(img_path)
            label = self.label_map[pred_idx]
            predictions.append((img_path, label, conf))

        unique_images = self.remove_duplicates(predictions)

