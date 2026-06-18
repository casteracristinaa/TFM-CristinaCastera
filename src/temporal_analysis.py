"""
TEMPORAL ANALYSIS MODULE
------------------------
Calcula métricas temporales a partir de máscaras segmentadas.

Devuelve:
    - dmin  (distancia mínima pipeta-centro ovocito)
    - inside (booleano de intersección)
    - speed (velocidad del centroide de la pipeta)
"""

import numpy as np
import cv2
import config


# =================================================
# UTILIDADES GEOMÉTRICAS
# =================================================

def centroid(mask):
    """
    Calcula el centroide del mayor contorno de la máscara.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return np.array([cx, cy])


def min_dist_to_center(mask_p, center_o):
    """
    Calcula la distancia mínima desde cualquier píxel de la pipeta
    al centro del ovocito.
    """
    ys, xs = np.where(mask_p > 0)

    if len(xs) == 0:
        return np.nan

    distances = np.sqrt((xs - center_o[0])**2 + (ys - center_o[1])**2)

    return np.min(distances)


def pipette_inside_oocyte(mask_p, mask_o, threshold=0.03):
    """
    Devuelve True si al menos 'threshold' porcentaje
    de la pipeta está dentro del ovocito.
    """

    intersection = (mask_p > 0) & (mask_o > 0)

    pip_pixels = np.sum(mask_p > 0)
    inter_pixels = np.sum(intersection)

    if pip_pixels == 0:
        return False

    overlap_ratio = inter_pixels / pip_pixels

    return overlap_ratio > threshold

# =================================================
# MÉTRICAS TEMPORALES
# =================================================

def compute_temporal_metrics(masks_p, masks_o):
    """
    Calcula métricas temporales frame a frame.

    Parámetros:
        masks_p : lista de máscaras pipeta
        masks_o : lista de máscaras ovocito

    Devuelve:
        dmin   : array float
        inside : array bool
        speed  : array float
    """

    n = len(masks_p)

    dmin = np.full(n, np.nan)
    inside = np.zeros(n, dtype=bool)
    speed = np.zeros(n)

    prev_centroid = None

    for i in range(n):

        cp = centroid(masks_p[i])
        co = centroid(masks_o[i])

        # -------------------------------
        # Distancia mínima
        # -------------------------------
        if cp is not None and co is not None:
            dmin[i] = min_dist_to_center(masks_p[i], co)

        # -------------------------------
        # Intersección real
        # -------------------------------
        intersection_area = np.logical_and(
            masks_p[i] > 0,
            masks_o[i] > 0
        ).sum()

        inside[i] = intersection_area >= config.MIN_INTERSECTION_AREA

        # -------------------------------
        # Velocidad del centroide
        # -------------------------------
        if prev_centroid is not None and cp is not None:
            speed[i] = np.linalg.norm(cp - prev_centroid)

        prev_centroid = cp

    if config.DEBUG:
        print("[INFO] Métricas temporales calculadas")
        print(f"[INFO] Frames inside=True: {inside.sum()} / {n}")
        print(f"[INFO] Speed min/mean/max: "
              f"{np.nanmin(speed):.2f} / "
              f"{np.nanmean(speed):.2f} / "
              f"{np.nanmax(speed):.2f}")

    return dmin, inside, speed


def extract_positions(masks_p, masks_o):
    """
    Extrae las posiciones (x, y) del centroide de pipeta y ovocito
    en cada frame.

    Parámetros:
        masks_p : lista de máscaras pipeta
        masks_o : lista de máscaras ovocito

    Devuelve:
        frame_ids    : array de índices de frame
        pipeta_x     : array de posiciones x de pipeta
        pipeta_y     : array de posiciones y de pipeta
        ovocito_x    : array de posiciones x de ovocito
        ovocito_y    : array de posiciones y de ovocito
    """

    n = len(masks_p)

    frame_ids = []
    pipeta_x = []
    pipeta_y = []
    ovocito_x = []
    ovocito_y = []

    for i in range(n):

        cp = centroid(masks_p[i])
        co = centroid(masks_o[i])

        frame_ids.append(i)

        if cp is not None:
            pipeta_x.append(cp[0])
            pipeta_y.append(cp[1])
        else:
            pipeta_x.append(np.nan)
            pipeta_y.append(np.nan)

        if co is not None:
            ovocito_x.append(co[0])
            ovocito_y.append(co[1])
        else:
            ovocito_x.append(np.nan)
            ovocito_y.append(np.nan)

    return (
        np.array(frame_ids),
        np.array(pipeta_x),
        np.array(pipeta_y),
        np.array(ovocito_x),
        np.array(ovocito_y)
    )
