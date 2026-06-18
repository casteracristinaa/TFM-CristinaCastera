"""
SEGMENTATION MODULE
-------------------
Se encarga de:
- Leer vídeo
- Ejecutar inferencia con ambos modelos
- Generar máscaras binarias limpias
"""

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# =================================================
# TRANSFORMACIÓN DE IMAGEN
# =================================================

transform = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# =================================================
# SEGMENTACIÓN DE UN FRAME
# =================================================

def segment_frame(frame, model):
    """
    Ejecuta inferencia sobre un frame y devuelve máscara binaria.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(image=rgb)["image"].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img))[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255

    return mask


# =================================================
# SEGMENTACIÓN DE VIDEO COMPLETO
# =================================================

def segment_video(video_path, model_p, model_o):
    """
    Segmenta un vídeo completo.

    Devuelve:
        masks_p (list)
        masks_o (list)
        fps (float)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    masks_p = []
    masks_o = []

    kernel = np.ones((3, 3), np.uint8)

    frame_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp = segment_frame(frame, model_p)
        mo = segment_frame(frame, model_o)

        # Limpieza morfológica
        mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, kernel)
        mo = cv2.morphologyEx(mo, cv2.MORPH_OPEN, kernel)

        masks_p.append(mp)
        masks_o.append(mo)

        frame_count += 1

    cap.release()

    if config.DEBUG:
        print(f"[INFO] Video procesado: {video_path}")
        print(f"[INFO] Frames totales: {frame_count}")
        print(f"[INFO] FPS: {fps:.2f}")

    return masks_p, masks_o, fps
