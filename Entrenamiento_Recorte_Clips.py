import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models # Importación necesaria para modelos pre-entrenados

# ------------------------
# CONFIGURACIÓN
# ------------------------
BASE_DIR = os.path.join("frames_dataset") 
BATCH_SIZE = 32
IMG_SIZE = 224 # <-- IMPORTANTE: ResNet18 requiere una entrada mínima de 224x224
EPOCHS = 10
MODEL_PATH = os.path.join(BASE_DIR, "clasificador_frames_resnet1_1.pth") # Cambiamos el nombre del archivo

# Usamos CUDA si está disponible, si no, CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ Usando dispositivo:", device)

# ------------------------
# TRANSFORMACIONES
# ------------------------
# Se recomienda usar normalización para modelos pre-entrenados
# con los valores estándar de ImageNet
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # <-- Normalización añadida
])

# ------------------------
# DATASETS
# ------------------------
train_dataset = datasets.ImageFolder(BASE_DIR, transform=transform)

# IMPORTANTE: Verifica que tienes dos subcarpetas dentro de BASE_DIR:
# '0_NoValido' y '1_Valido'

num_train = int(0.8 * len(train_dataset))
num_val = len(train_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------------
# MODELO USANDO TRANSFER LEARNING CON ResNet18
# ----------------------------------------------------

# 1. Cargar el modelo pre-entrenado (ResNet18)
# 'weights=True' asegura que se carguen los pesos entrenados en ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) 

# 2. Congelar los parámetros de la red base (feature extractor)
# Esto hace que el modelo entrene solo las nuevas capas que añadiremos
for param in model.parameters():
    param.requires_grad = False

# 3. Reemplazar la capa final de clasificación (originalmente 1000 clases)
num_ftrs = model.fc.in_features # Obtener el número de entradas de la capa FC original

# Definir nuestro nuevo clasificador binario
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 64), # Capa densa intermedia para complejidad
    nn.ReLU(),
    nn.Dropout(0.5),         # Regularización para evitar el sobreajuste (overfitting)
    nn.Linear(64, 1),        # Salida a 1 neurona (clasificación binaria)
    nn.Sigmoid()             # Función de activación para la pérdida BCELoss
)

model = model.to(device)
criterion = nn.BCELoss()

# 4. Ajustar el optimizador: ¡SOLO optimizar los parámetros de las nuevas capas!
# Esto hace que el entrenamiento sea más rápido y estable.
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------------
# ENTRENAMIENTO (Mismo código, pero ahora usando la potente ResNet18)
# ------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validación
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ------------------------
# GUARDAR MODELO
# ------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")