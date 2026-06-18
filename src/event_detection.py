"""
EVENT DETECTION MODULE
----------------------
Contiene la lógica para detectar:

- Aspiración
- Inyección
- Retroceso
- Salida

Recibe métricas temporales y devuelve índices de frame.
"""

import numpy as np
import config
import os
import cv2


# =================================================
# ASPIRACIÓN
# =================================================

def detect_interno(dmin, inside, speed, fps, masks_p, masks_o):
    """
    Detecta aspiración profunda y estable.
    """

    stable_frames = int(config.ASP_STABLE_SEC * fps)
    min_gap_frames = int(config.MIN_GAP_INJ_ASP_SEC * fps)

    n = len(inside)
    candidates = []

    for i in range(0, n - stable_frames):

        inside_ratio = inside[i:i+stable_frames].mean()
        mean_speed   = np.nanmean(speed[i:i+stable_frames])
        mean_dmin    = np.nanmean(dmin[i:i+stable_frames])

        # la pipeta debe estar dentro al menos en el inicio del periodo estable
        if inside[i] and inside_ratio > 0.6 and mean_speed < config.MOVE_THRESHOLD * 1.5:
            candidates.append({
                "frame": i ,
                "dmin": mean_dmin
            })

    if len(candidates) == 0:
        if config.DEBUG:
            print("[WARN] Aspiración no detectada")
        return None

    # función local para verificar que un frame tenga pipeta dentro
    def pipeta_inside(frame_idx):
        mask_p = masks_p[frame_idx]
        mask_o = masks_o[frame_idx]

        cnts_pip, _ = cv2.findContours(mask_p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts_pip:
            return False

        cnt_pip = max(cnts_pip, key=cv2.contourArea)
        pts_pip = cnt_pip.reshape(-1, 2).astype(np.float32)

        mean, eigenvectors = cv2.PCACompute(pts_pip, mean=None)
        direction = eigenvectors[0]
        direction = direction / np.linalg.norm(direction)
        center_pip = mean[0]

        vectors = pts_pip - center_pip
        projections = np.dot(vectors, direction)

        min_proj = np.min(projections)
        max_proj = np.max(projections)

        tip_proj = min_proj
        base_proj = max_proj
        length = base_proj - tip_proj
        third_proj = tip_proj + length / 3.0
        third_point = center_pip + third_proj * direction

        cx, cy = int(third_point[0]), int(third_point[1])

        if 0 <= cy < mask_o.shape[0] and 0 <= cx < mask_o.shape[1]:
            return mask_o[cy, cx] > 0
        return False

    # Elegir el primero después del gap mínimo que también pase la comprobación
    interno = None
    for cand in sorted(candidates, key=lambda x: x["frame"]):
        if cand["frame"] > min_gap_frames and pipeta_inside(cand["frame"]):
            interno = cand["frame"]
            break

    # si ninguno válido, intentar cualquiera de los candidatos en orden
    if interno is None:
        for cand in sorted(candidates, key=lambda x: x["frame"]):
            if pipeta_inside(cand["frame"]):
                interno = cand["frame"]
                break

    if interno is None and config.DEBUG:
        print("[WARN] Ningún candidato interno sensato con pipeta dentro")

    return interno


# =================================================
# INYECCIÓN
# =================================================

def detect_injection(inside, interno, fps):
    """
    Detecta el último frame fuera antes de entrar.
    """

    if interno is None:
        return None

    min_gap_frames = int(config.MIN_GAP_INJ_ASP_SEC * fps)

    injection = None

    for i in range(interno - min_gap_frames, 1, -1):
        if not inside[i] and inside[i+1]:
            injection = i
            break

    if injection is None:
        injection = interno - min_gap_frames

    if config.DEBUG:
        print(f"[OK] Inyección detectada en frame {injection}")

    return injection


# =================================================
# RETROCESO
# =================================================

def detect_retroceso(inside, speed, interno, fps):
    """
    Detecta el inicio del movimiento de salida tras aspiración.
    """

    if interno is None:
        return None

    n = len(inside)

    search_start = interno + int(config.RETRO_MARGIN_SEC * fps)
    retro_window = int(config.RETRO_WINDOW_SEC * fps)

    retroceso = None

    # Velocidad media en zona estable previa
    prev_speed_mean = np.nanmean(
        speed[max(0, interno-5):interno]
    )

    for i in range(search_start, n - retro_window):

        moving = np.nanmean(speed[i:i+retro_window]) > (
            prev_speed_mean + config.MOVE_THRESHOLD
        )

        inside_drop = inside[i-1] and not inside[i]

        if moving or inside_drop:
            retroceso = i
            break

    if retroceso is None:
        if config.DEBUG:
            print("[WARN] Retroceso no detectado")
    else:
        if config.DEBUG:
            print(f"[OK] Retroceso detectado en frame {retroceso}")

    return retroceso


# =================================================
# SALIDA
# =================================================

def detect_salida(inside, retroceso, min_frames_fuera=3):

    if retroceso is None:
        return None

    count = 0

    for i in range(retroceso, len(inside)):
        if not inside[i]:
            count += 1
            if count >= min_frames_fuera:
                return i - min_frames_fuera + 1
        else:
            count = 0

    return None



def save_frame(video_path, frame_idx, output_path):
    """
    Guarda un frame específico de un vídeo.
    """

    if frame_idx is None:
        print("[WARN] Frame None, no se guarda imagen.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el vídeo.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"[ERROR] No se pudo leer el frame {frame_idx}")
        cap.release()
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)

    cap.release()

# =================================================
# DETECCIÓN COMPLETA
# =================================================

def detect_all_events(dmin, inside, speed, fps, masks_p, masks_o, video_path, output_folder):
    """
    Ejecuta la detección completa del patrón ICSI
    y guarda los frames clave.
    """

    interno = detect_interno(dmin, inside, speed, fps, masks_p, masks_o)
    if config.DEBUG:
        print(f"[DEBUG] interno detectado: {interno}")

    injection  = detect_injection(inside, interno, fps)
    retroceso  = detect_retroceso(inside, speed, interno, fps)
    salida     = detect_salida(inside, retroceso)

    # Guardar frames
    if injection is not None:
        save_frame(
            video_path,
            injection,
            os.path.join(output_folder, "frame_injection.png")
        )

    if interno is not None:
        save_frame(
            video_path,
            interno,
            os.path.join(output_folder, "frame_interno.png")
        )

    #if salida is not None:
    #    save_frame(
    #        video_path,
    #        salida,
    #        os.path.join(output_folder, "frame_salida.png")
    #    )

    return {
        "injection": injection,
        "interno": interno,
        "retroceso": retroceso,
        "salida": salida
    }
