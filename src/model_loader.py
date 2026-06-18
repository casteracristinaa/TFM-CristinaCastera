"""
MODEL LOADER
------------
Carga los modelos de segmentación (pipeta y ovocito).
No contiene lógica temporal.
"""

import torch
import segmentation_models_pytorch as smp
import config


def build_unet():
    """
    Construye la arquitectura U-Net.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    return model


def load_model(model_path):
    """
    Carga un modelo desde disco y lo prepara para inferencia.
    """
    model = build_unet()

    state_dict = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    model.eval()

    if config.DEBUG:
        print(f"[INFO] Modelo cargado: {model_path}")
        print(f"[INFO] Dispositivo: {config.DEVICE}")

    return model


def load_models():
    """
    Carga ambos modelos del proyecto.
    Devuelve:
        model_pipeta, model_ovocito
    """
    model_p = load_model(config.MODEL_PIPETA)
    model_o = load_model(config.MODEL_OVOCITO)
    model_a = load_model(config.MODEL_ANGULO)

    return model_p, model_o, model_a
