import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# PATHS
# =========================
INPUT_DIR  = "../logs/4"
OUTPUT_DIR = "../logs/4.4_Angulo"
MODEL_ANGULO = "../models/unet_angulo.pth"
MODEL_PIPETA = "../models/unet_pipeta.pth"
MODEL_OVOCITO = "../models/unet_ovocito.pth"
TXT_PATH   = os.path.join(OUTPUT_DIR, "resultados.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("▶ Usando dispositivo:", device)

# =========================
# MODELOS
# =========================
def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

model_ang = build_model()
model_pip = build_model()
model_ovo = build_model()

model_ang.load_state_dict(torch.load(MODEL_ANGULO, map_location=device))
model_pip.load_state_dict(torch.load(MODEL_PIPETA, map_location=device))
model_ovo.load_state_dict(torch.load(MODEL_OVOCITO, map_location=device))

model_ang.to(device).eval()
model_pip.to(device).eval()
model_ovo.to(device).eval()

# =========================
# TRANSFORM
# =========================
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# =========================
# PROCESAMIENTO
# =========================
with open(TXT_PATH, "w") as ftxt:
    ftxt.write("folder clip angle_deg status\n")

    for folder in sorted(os.listdir(INPUT_DIR)):
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        for clip in sorted(os.listdir(folder_path)):
            clip_path = os.path.join(folder_path, clip)
            if not os.path.isdir(clip_path):
                continue

            print(f"▶ Procesando {folder}/{clip}")
            out_dir = os.path.join(OUTPUT_DIR, folder, clip)
            os.makedirs(out_dir, exist_ok=True)

            img_path = os.path.join(clip_path, "frame_aspiracion.png")
            if not os.path.exists(img_path):
                ftxt.write(f"{folder} {clip} NaN NO_IMAGE\n")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                ftxt.write(f"{folder} {clip} NaN READ_ERROR\n")
                continue

            cv2.imwrite(os.path.join(out_dir, "original.png"), img_bgr)

            try:
                # =========================
                # PREPROCESS
                # =========================
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h0, w0 = img_rgb.shape[:2]

                aug = transform(image=img_rgb)
                x = aug["image"].unsqueeze(0).to(device)

                # =========================
                # INFERENCIA
                # =========================
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

                mask_ang = cv2.resize(mask_ang_512, (w0, h0), interpolation=cv2.INTER_NEAREST)
                mask_pip = cv2.resize(mask_pip_512, (w0, h0), interpolation=cv2.INTER_NEAREST)
                mask_ovo = cv2.resize(mask_ovo_512, (w0, h0), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(os.path.join(out_dir, "mask_angulo.png"), mask_ang)
                cv2.imwrite(os.path.join(out_dir, "mask_pipeta.png"), mask_pip)
                cv2.imwrite(os.path.join(out_dir, "mask_ovo.png"), mask_ovo)

                # =========================
                # CONTORNO TRIÁNGULO
                # =========================
                cnts_ang, _ = cv2.findContours(mask_ang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                triangle_ok = True
                axis_ok = True

                if not cnts_ang:
                    triangle_ok = False
                else:
                    cnt_ang = max(cnts_ang, key=cv2.contourArea)
                    pts_ang = cnt_ang.reshape(-1, 2)

                    # =========================
                    # CHECK CIRCULARIDAD
                    # =========================
                    is_circular = False

                    area = cv2.contourArea(cnt_ang)
                    perimeter = cv2.arcLength(cnt_ang, True)

                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.50:
                            is_circular = True

                # =========================
                # CONTORNO PIPETA
                # =========================
                cnts_pip, _ = cv2.findContours(mask_pip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cnts_pip:
                    ftxt.write(f"{folder} {clip} NaN NO_PIPETA_MASK\n")
                    continue

                cnt_pip = max(cnts_pip, key=cv2.contourArea)
                pts_pip = cnt_pip.reshape(-1, 2).astype(np.float32)

                # =========================
                # CONTORNO OVOCITO
                # =========================
                cnts_ovo, _ = cv2.findContours(mask_ovo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cnts_ovo:
                    ftxt.write(f"{folder} {clip} NaN NO_OVOCITO_MASK\n")
                    continue

                cnt_ovo = max(cnts_ovo, key=cv2.contourArea)
                pts_ovo = cnt_ovo.reshape(-1, 2).astype(np.float32)

                # =========================
                # PCA → EJE CENTRAL PIPETA
                # =========================
                mean, eigenvectors = cv2.PCACompute(pts_pip, mean=None)
                direction = eigenvectors[0]
                direction = direction / np.linalg.norm(direction)
                center_pip = mean[0]

                # =========================
                # SI FALLA TRIÁNGULO → IR A CHECK PIPETA
                # =========================
                if not triangle_ok:
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
                        axis_ok = False

                # =========================
                # SI FALLA TRIÁNGULO O EJE → CLASIFICACIÓN ESPECIAL
                # =========================
                if not triangle_ok or not axis_ok or is_circular:

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
                        ftxt.write(f"{folder} {clip} NaN SIN_SALTO\n")
                    else:
                        ftxt.write(f"{folder} {clip} NaN FRAME_MAL_EXTRAIDO\n")

                    continue

                # =========================
                # SI TODO OK → SEGUIR NORMAL
                # =========================

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
                    ftxt.write(f"{folder} {clip} NaN FRAME_MAL_EXTRAIDO\n")
                    continue

                p_sup = pts_valid[side1][np.argmax(dists_valid[side1])]
                p_inf = pts_valid[side2][np.argmax(dists_valid[side2])]

                v1 = p_sup - tip
                v2 = p_inf - tip

                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cosang = np.clip(cosang, -1, 1)
                angle = np.degrees(np.arccos(cosang))

                ftxt.write(f"{folder} {clip} {angle:.4f} OK\n")

                # =========================
                # DEBUG VISUAL
                # =========================
                debug = img_bgr.copy()
                debug[mask_ang > 0] = (0.6 * debug[mask_ang > 0] + 0.4 * 255).astype(np.uint8)

                cv2.circle(debug, tuple(tip), 5, (0,255,0), -1)
                cv2.circle(debug, tuple(p_sup), 5, (255,0,0), -1)
                cv2.circle(debug, tuple(p_inf), 5, (0,0,255), -1)

                cv2.line(debug, tuple(tip), tuple(p_sup), (255,0,255), 2)
                cv2.line(debug, tuple(tip), tuple(p_inf), (255,0,255), 2)

                cv2.putText(
                    debug, f"{angle:.2f} deg",
                    (int(tip[0]+10), int(tip[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                )

                cv2.imwrite(os.path.join(out_dir, "debug.png"), debug)

                ftxt.write(f"{folder} {clip} {angle:.4f} OK\n")

            except Exception as e:
                ftxt.write(f"{folder} {clip} NaN {str(e)}\n")

print("✅ PROCESO COMPLETO")