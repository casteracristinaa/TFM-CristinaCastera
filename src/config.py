"""
CONFIGURACIÓN GLOBAL DEL PIPELINE ICSI
---------------------------------------
Contiene todas las constantes y rutas del proyecto.
No debe contener lógica.
"""

from datetime import date
import os
import torch

# =================================================
# RUTAS
# =================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR_1   = os.path.join(BASE_DIR, "../BBDD/videos/processed") #input
INPUT_DIR  = os.path.join(DATA_DIR_1, "videos_nuevos") 
DATA_DIR   = os.path.join(BASE_DIR, "../../logs") #output
OUTPUT_DIR = os.path.join(DATA_DIR, "7_nuevos-videos") 


# Vídeos largos sin recortar (input del paso 0: recorte de clips)
RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "../BBDD/videos/raw/videos_nuevos")

# Carpeta de depuración del recorte (CSVs de probabilidad, frames óptimos, resumen)
RECORTE_DEBUG_DIR = os.path.join(DATA_DIR, "0_recorte_debug")


MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PIPETA  = os.path.join(MODEL_DIR, "unet_pipeta.pth")
MODEL_OVOCITO = os.path.join(MODEL_DIR, "unet_ovocito.pth")
MODEL_ANGULO= os.path.join(MODEL_DIR, "unet_angulo.pth")

# Dataset global (.xlsx) con una fila por clip: ID, CLIP, velocidad_inyeccion, angulo_triangulo
# Nombrado con la fecha de la ejecución, p.ej. "DATASET-2026-06-18.xlsx"
DATASET_PATH = os.path.join(DATA_DIR, f"DATASET-{date.today().isoformat()}.xlsx")

# Clasificador de frames usado para detectar el momento óptimo de inyección
# (paso 0: recorte de vídeos largos en clips)
CLASSIFIER_FRAMES_PATH = os.path.join(MODEL_DIR, "recorte_clips.pth")

# DEBUG
SAVE_DIFF_MASKS = True
THRESH = 1
MIN_AREA = 5
MAX_AREA = 2000
RECT_PAD = 15
THRESH_PROFILE = 5
MIN_MONOTONIC_FRAMES = 5
MIN_AXIS_INCREMENT = 2.0


# Crear carpeta output si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RECORTE_DEBUG_DIR, exist_ok=True)

# =================================================
# HARDWARE
# =================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================
# IMAGEN
# =================================================

IMG_SIZE = 512

# =================================================
# PARÁMETROS TEMPORALES (segundos)
# =================================================

MIN_GAP_INJ_ASP_SEC = 2.0   # separación mínima real entre inyección y aspiración
ASP_STABLE_SEC      = 1.5   # tiempo mínimo estable dentro para aspiración
APPROACH_MIN_SEC    = 1.0
EXIT_MIN_SEC        = 0.5
MAX_TOTAL_SEC       = 6.0

RETRO_MARGIN_SEC    = 0.2   # margen tras aspiración para buscar retroceso
RETRO_WINDOW_SEC    = 0.3   # ventana de análisis para retroceso

# =================================================
# UMBRALES ESPACIALES
# =================================================

MOVE_THRESHOLD = 1.0
MIN_INTERSECTION_AREA = 10

# =================================================
# DEBUG
# =================================================

DEBUG = False
MAX_DEBUG_PRINTS = 20

SAVE_EVENT_DEBUG_VIDEO = False
SAVE_ASPIRATION_SOFT_DEBUG = False
SAVE_ASPIRATION_HEATMAP_DEBUG = False
SAVE_ASPIRATION_SEGMENTED_DEBUG = False
SAVE_CSV_DATA = False


# =================================================
# PASO 0: RECORTE DE VÍDEOS LARGOS EN CLIPS
# =================================================

CLASSIFIER_IMG_SIZE = 224       # tamaño de entrada del clasificador de frames
PROB_THRESHOLD = 0.99           # probabilidad mínima para considerar un frame "óptimo"
FPS_SAMPLE_RAW = 1              # frames por segundo a muestrear del vídeo largo
MIN_DURATION_SEC = 10           # duración mínima de un intervalo válido
DELTA_ESTABLE = 0.001           # tolerancia de variación para considerar un tramo estable
WINDOW_ESTABLE = 5              # nº de frames consecutivos para considerar estable
