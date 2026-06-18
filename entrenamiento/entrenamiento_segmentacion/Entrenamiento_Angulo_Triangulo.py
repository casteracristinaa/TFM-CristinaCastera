import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =================================================
# CONFIG
# =================================================
IMAGES_DIR = "frames_dataset"
LABELS_JSON = "project-1-at-2026-01-07-10-57-ecb81914.json"

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "unet_angulo.pth"

# =================================================
# DATASET
# =================================================
class AngleDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.images_dir = images_dir
        self.transform = transform

        # Filtrar solo tareas con anotaciones
        self.data = [
            item for item in self.data
            if item.get("annotations") and len(item["annotations"]) > 0
        ]

        print(f"📊 Imágenes con anotación: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # NOMBRE REAL DEL ARCHIVO (con hash)
        image_name = os.path.basename(item["data"]["image"])
        img_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No existe: {img_path}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Crear máscara desde el polígono
        for ann in item["annotations"]:
            for res in ann["result"]:
                if res["type"] == "polygonlabels":
                    pts = np.array(res["value"]["points"], dtype=np.float32)
                    pts[:, 0] = pts[:, 0] * w / 100
                    pts[:, 1] = pts[:, 1] * h / 100
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()


# =================================================
# TRANSFORMS
# =================================================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = AngleDataset(LABELS_JSON, IMAGES_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =================================================
# MODEL
# =================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from sklearn.model_selection import train_test_split

# =================================================
# TRAIN / VAL SPLIT
# =================================================
train_data, val_data = train_test_split(
    dataset.data,
    test_size=0.2,
    random_state=42
)

train_dataset = AngleDataset(LABELS_JSON, IMAGES_DIR, transform)
val_dataset = AngleDataset(LABELS_JSON, IMAGES_DIR, transform)

train_dataset.data = train_data
val_dataset.data = val_data

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =================================================
# METRICS
# =================================================
def dice_score(preds, targets, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(preds, targets, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()

    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# =================================================
# TRAINING LOOP
# =================================================
print(f"▶ Entrenando en {DEVICE}")

for epoch in range(EPOCHS):

    # ================= TRAIN =================
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:

        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ================= VALIDATION =================
    model.eval()

    val_dice = 0
    val_iou = 0

    with torch.no_grad():

        for imgs, masks in val_loader:

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)

            val_dice += dice_score(preds, masks)
            val_iou += iou_score(preds, masks)

    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f}"
    )

# =================================================
# SAVE MODEL
# =================================================
torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Modelo guardado en {MODEL_OUT}")
