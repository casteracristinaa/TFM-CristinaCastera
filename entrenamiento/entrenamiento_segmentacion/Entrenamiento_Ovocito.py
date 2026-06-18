import os
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------
# CONFIGURACIÓN
# ------------------------
DATASET_DIR = "frames_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4            # ovocito ocupa más área
LR = 1e-4
EPOCHS = 40
IMG_SIZE = 512
VAL_SPLIT = 0.2
BACKBONE = "resnet34"

LABEL_NAME = "ovocito"

SEED = 42
random.seed(SEED)


def dice_coef(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    return (2 * intersection + eps) / (union + eps)

def iou_score(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection

    return (intersection + eps) / (union + eps)

# ------------------------
# DATASET
# ------------------------
class OvocitoLabelMeDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def json_to_mask(self, json_path, shape):
        mask = np.zeros(shape, dtype=np.uint8)

        with open(json_path, "r") as f:
            data = json.load(f)

        for shape_data in data["shapes"]:
            if shape_data["label"].lower() != LABEL_NAME:
                continue

            points = np.array(shape_data["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        return mask

    def __getitem__(self, idx):
        img_path = self.files[idx]
        json_path = img_path.replace(".jpg", ".json")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.json_to_mask(json_path, image.shape[:2])
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.files)

# ------------------------
# CARGAR ARCHIVOS
# ------------------------
all_images = sorted([
    os.path.join(DATASET_DIR, f)
    for f in os.listdir(DATASET_DIR)
    if f.endswith(".jpg")
])

random.shuffle(all_images)
split_idx = int(len(all_images) * (1 - VAL_SPLIT))

train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

print(f"Train: {len(train_files)} | Val: {len(val_files)}")

# ------------------------
# TRANSFORMACIONES
# ------------------------
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ------------------------
# DATA LOADERS
# ------------------------
train_dataset = OvocitoLabelMeDataset(train_files, train_transform)
val_dataset = OvocitoLabelMeDataset(val_files, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ------------------------
# MODELO
# ------------------------
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

# ------------------------
# LOSS (ajustada a objetos grandes)
# ------------------------
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(preds, targets):
    return 0.5 * dice_loss(preds, targets) + 0.5 * bce_loss(preds, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------
# ENTRENAMIENTO
# ------------------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDACIÓN
    model.eval()
    val_loss = 0

    val_dice = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            val_loss += loss_fn(preds, masks).item()
            val_dice += dice_coef(preds, masks).item()
            val_iou += iou_score(preds, masks).item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

print(f"Val loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "unet_ovocito_best.pth")
        print("✅ Modelo ovocito guardado")

print("🎉 Entrenamiento de ovocito finalizado")
