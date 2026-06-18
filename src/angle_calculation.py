"""
ANGLE CALCULATION MODULE
------------------------
Calcula el ángulo del triángulo (pipeta) en el frame de aspiración.
"""

from multiprocessing.util import debug

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
import os


def build_model():
    """Construye la arquitectura U-Net para segmentación."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )


def load_angle_models(device):
    """
    Carga los 3 modelos necesarios para cálculo de ángulo.
    Devuelve: (model_ang, model_pip, model_ovo)
    """
    model_ang = build_model()
    model_pip = build_model()
    model_ovo = build_model()

    model_ang.load_state_dict(
        torch.load(config.MODEL_ANGULO, map_location=device)
    )
    model_pip.load_state_dict(
        torch.load(config.MODEL_PIPETA, map_location=device)
    )
    model_ovo.load_state_dict(
        torch.load(config.MODEL_OVOCITO, map_location=device)
    )

    model_ang.to(device).eval()
    model_pip.to(device).eval()
    model_ovo.to(device).eval()

    return model_ang, model_pip, model_ovo


def calcular_angulo_frame(frame_path, model_ang, model_pip, model_ovo, device, debug_dir=None):
    """
    Calcula el ángulo del triángulo en un frame específico.

    Parámetros:
        frame_path : ruta a la imagen del frame
        model_ang  : modelo de segmentación del triángulo
        model_pip  : modelo de segmentación de pipeta
        model_ovo  : modelo de segmentación de ovocito
        device     : dispositivo (cuda o cpu)
        debug_dir  : directorio para guardar máscaras si falla (opcional)

    Devuelve:
        (angle_deg, debug_image, status) : tupla (ángulo en grados, imagen con ángulo dibujado, estado)
                                           angle_deg y debug_image son None si status != "OK"
                                           status: "OK", "READ_ERROR", "NO_PIPETA_MASK", "NO_OVOCITO_MASK", "SIN_SALTO", "FRAME_MAL_EXTRAIDO"
    """

    # =============================
    # LEER IMAGEN
    # =============================
    img_bgr = cv2.imread(frame_path)
    if img_bgr is None:
        print(f"[WARN] No se pudo leer imagen: {frame_path}")
        return None, None, "READ_ERROR"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]

    # =============================
    # TRANSFORM
    # =============================
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    aug = transform(image=img_rgb)
    x = aug["image"].unsqueeze(0).to(device)

    # =============================
    # INFERENCIA
    # =============================
    with torch.no_grad():
        logits_ang = model_ang(x)[0, 0].cpu().numpy()
        logits_pip = model_pip(x)[0, 0].cpu().numpy()
        logits_ovo = model_ovo(x)[0, 0].cpu().numpy()

    prob_ang = 1 / (1 + np.exp(-logits_ang))
    prob_pip = 1 / (1 + np.exp(-logits_pip))
    prob_ovo = 1 / (1 + np.exp(-logits_ovo))

    mask_ang_512 = (prob_ang > 0.2).astype(np.uint8) * 255
    mask_pip_512 = (prob_pip > 0.2).astype(np.uint8) * 255
    mask_ovo_512 = (prob_ovo > 0.2).astype(np.uint8) * 255

    # Redimensionar a tamaño original
    mask_ang = cv2.resize(mask_ang_512, (w0, h0), interpolation=cv2.INTER_NEAREST)
    mask_pip = cv2.resize(mask_pip_512, (w0, h0), interpolation=cv2.INTER_NEAREST)
    mask_ovo = cv2.resize(mask_ovo_512, (w0, h0), interpolation=cv2.INTER_NEAREST)

    # =============================
    # CONTORNO TRIÁNGULO
    # =============================
    cnts_ang, _ = cv2.findContours(mask_ang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    triangle_ok = True
    axis_ok = True
    is_circular = False

    if not cnts_ang:
        print("[WARN] No se detectó máscara de triángulo")
        triangle_ok = False
    else:
        cnt_ang = max(cnts_ang, key=cv2.contourArea)
        pts_ang = cnt_ang.reshape(-1, 2)

        # CHECK CIRCULARIDAD
        area = cv2.contourArea(cnt_ang)
        perimeter = cv2.arcLength(cnt_ang, True)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.50:
                is_circular = True
                print(f"[WARN] Triángulo muy circular (circularity={circularity:.3f})")

    # =============================
    # CONTORNO PIPETA
    # =============================
    cnts_pip, _ = cv2.findContours(mask_pip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts_pip:
        print("[WARN] No se detectó máscara de pipeta")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ang_{os.path.basename(frame_path)}.png"), mask_ang)
            cv2.imwrite(os.path.join(debug_dir, f"mask_pip_{os.path.basename(frame_path)}.png"), mask_pip)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ovo_{os.path.basename(frame_path)}.png"), mask_ovo)
            cv2.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(frame_path)}.png"), img_bgr)
        return None, None, "NO_PIPETA_MASK"

    cnt_pip = max(cnts_pip, key=cv2.contourArea)
    pts_pip = cnt_pip.reshape(-1, 2).astype(np.float32)

    # =============================
    # CONTORNO OVOCITO
    # =============================
    cnts_ovo, _ = cv2.findContours(mask_ovo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts_ovo:
        print("[WARN] No se detectó máscara de ovocito")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ang_{os.path.basename(frame_path)}.png"), mask_ang)
            cv2.imwrite(os.path.join(debug_dir, f"mask_pip_{os.path.basename(frame_path)}.png"), mask_pip)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ovo_{os.path.basename(frame_path)}.png"), mask_ovo)
            cv2.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(frame_path)}.png"), img_bgr)
        return None, None, "NO_OVOCITO_MASK"

    cnt_ovo = max(cnts_ovo, key=cv2.contourArea)
    pts_ovo = cnt_ovo.reshape(-1, 2).astype(np.float32)

    # =============================
    # PCA → EJE CENTRAL PIPETA
    # =============================
    mean, eigenvectors = cv2.PCACompute(pts_pip, mean=None)
    direction = eigenvectors[0]
    direction = direction / np.linalg.norm(direction)
    center_pip = mean[0]

    # =============================
    # VALIDACIÓN DE EJE
    # =============================
    if not triangle_ok:
        print("[WARN] Validación falla: triángulo no OK")
        axis_ok = False
    else:
        # INTERSECCIÓN EJE-PIPETA CON TRIÁNGULO
        vectors = pts_ang - center_pip
        proj_axis = np.dot(vectors, direction)
        perp_vectors = vectors - np.outer(proj_axis, direction)
        perp_dist = np.linalg.norm(perp_vectors, axis=1)

        tolerance = 5
        candidates = perp_dist < tolerance

        if not np.any(candidates):
            print("[WARN] Validación falla: eje no intersecta triángulo")
            axis_ok = False

    # =============================
    # SI FALLA → DETERMINAR RAZÓN
    # =============================
    if not triangle_ok or not axis_ok or is_circular:
        print(f"[WARN] Cálculo de ángulo falló: triangle_ok={triangle_ok}, axis_ok={axis_ok}, circular={is_circular}")
        
        # Proyectar todos los puntos sobre el eje
        vectors = pts_pip - center_pip
        projections = np.dot(vectors, direction)

        # Encontrar extremos de la pipeta
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # La punta suele ser el extremo más negativo (depende orientación)
        tip_proj = min_proj
        base_proj = max_proj

        # Longitud total proyectada
        length = base_proj - tip_proj

        # Punto a 1/3 desde la punta
        third_proj = tip_proj + length / 3.0

        # Coordenada real del primer tercio
        third_point = center_pip + third_proj * direction

        cx, cy = int(third_point[0]), int(third_point[1])

        # comprobar si centro de pipeta está dentro del ovocito
        if 0 <= cy < mask_ovo.shape[0] and 0 <= cx < mask_ovo.shape[1]:
            inside = mask_ovo[cy, cx] > 0
        else:
            inside = False

        if inside:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"mask_ang_{os.path.basename(frame_path)}.png"), mask_ang)
                cv2.imwrite(os.path.join(debug_dir, f"mask_pip_{os.path.basename(frame_path)}.png"), mask_pip)
                cv2.imwrite(os.path.join(debug_dir, f"mask_ovo_{os.path.basename(frame_path)}.png"), mask_ovo)
                cv2.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(frame_path)}.png"), img_bgr)
            return None, None, "SIN_SALTO"
        else:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"mask_ang_{os.path.basename(frame_path)}.png"), mask_ang)
                cv2.imwrite(os.path.join(debug_dir, f"mask_pip_{os.path.basename(frame_path)}.png"), mask_pip)
                cv2.imwrite(os.path.join(debug_dir, f"mask_ovo_{os.path.basename(frame_path)}.png"), mask_ovo)
                cv2.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(frame_path)}.png"), img_bgr)
            return None, None, "FRAME_MAL_EXTRAIDO"

    # =============================
    # CÁLCULO DE ÁNGULO
    # =============================
    proj_candidates = proj_axis[candidates]
    pts_candidates = pts_ang[candidates]
    tip = pts_candidates[np.argmin(proj_candidates)]

    perp = np.array([-direction[1], direction[0]])
    vectors_tip = pts_ang - tip
    proj_perp = np.dot(vectors_tip, perp)
    dists = np.linalg.norm(vectors_tip, axis=1)

    valid = dists > 10
    pts_valid = pts_ang[valid]
    proj_valid = proj_perp[valid]
    dists_valid = dists[valid]

    side1 = proj_valid > 0
    side2 = proj_valid < 0

    if not np.any(side1) or not np.any(side2):
        print("[WARN] No hay puntos en ambos lados del eje")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ang_{os.path.basename(frame_path)}.png"), mask_ang)
            cv2.imwrite(os.path.join(debug_dir, f"mask_pip_{os.path.basename(frame_path)}.png"), mask_pip)
            cv2.imwrite(os.path.join(debug_dir, f"mask_ovo_{os.path.basename(frame_path)}.png"), mask_ovo)
            cv2.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(frame_path)}.png"), img_bgr)
        return None, None, "FRAME_MAL_EXTRAIDO"

    p_sup = pts_valid[side1][np.argmax(dists_valid[side1])]
    p_inf = pts_valid[side2][np.argmax(dists_valid[side2])]

    v1 = p_sup - tip
    v2 = p_inf - tip

    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1, 1)
    angle = np.degrees(np.arccos(cosang))

    print(f"[OK] Ángulo calculado: {angle:.2f}°")

    # =============================
    # CREAR IMAGEN CON ÁNGULO
    # =============================
    debug = img_bgr.copy()
    debug[mask_ang > 0] = (0.6 * debug[mask_ang > 0] + 0.4 * 255).astype(np.uint8)
    


    pt_tip = tuple(tip.astype(int))
    pt_sup = tuple(p_sup.astype(int))
    pt_inf = tuple(p_inf.astype(int))

    # puntos en blanco con borde negro
    cv2.circle(debug, pt_tip, 5, (0, 0, 0), -1)
    cv2.circle(debug, pt_sup, 5, (0, 0, 0), -1)
    cv2.circle(debug, pt_inf, 5, (0, 0, 0), -1)

    # Borde negro (líneas)
    cv2.line(debug, pt_tip, pt_sup, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(debug, pt_tip, pt_inf, (0, 0, 0), 2, cv2.LINE_AA)

    # Dirección de la bisectriz
    bisector = (v1 / np.linalg.norm(v1)) + (v2 / np.linalg.norm(v2))
    bisector = bisector / np.linalg.norm(bisector)

    # Mover el texto alejándolo del triángulo
    offset = 40  # prueba 30–60 según tamaño imagen
    text_x = int(tip[0] + bisector[0] * offset)
    text_y = int(tip[1] + bisector[1] * offset)
    text = f"{angle:.2f} deg"

    pos = (text_x, text_y)

    cv2.putText(
        debug, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 0, 0), 
        3,
        cv2.LINE_AA
    )

    return angle, debug, "OK"
