import cv2
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import config
from angle_calculation import calcular_angulo_frame, load_angle_models


# ===============================
# LIMPIEZA DE SEÑAL
# ===============================
def quedarse_con_senal_buena(df, max_gap_frames=5):

    df = df.sort_values("frame").reset_index(drop=True)

    # -------------------------------
    # 1. Detectar cortes por salto de frame
    # -------------------------------
    df["frame_diff"] = df["frame"].diff()

    cortes = df.index[df["frame_diff"] > max_gap_frames].tolist()

    # -------------------------------
    # 2. Separar en bloques
    # -------------------------------
    bloques = np.split(df, cortes)

    # -------------------------------
    # 3. Elegir el mejor bloque
    # criterio: mayor longitud
    # -------------------------------
    mejor_bloque = max(bloques, key=len)

    # -------------------------------
    # 4. Limpiar
    # -------------------------------
    mejor_bloque = mejor_bloque.reset_index(drop=True)
    mejor_bloque = mejor_bloque.drop(columns=["frame_diff"])

    return mejor_bloque

def limpiar_e_interpolar_senal(df, fps, umbral_salto=25, min_seg_saturacion=3):

    # -------------------------------
    # 0. Ordenar
    # -------------------------------
    df = df.sort_values("frame").reset_index(drop=True)

    # -------------------------------
    # 1. Rellenar frames perdidos
    # -------------------------------
    full_frames = pd.DataFrame({
        "frame": range(df["frame"].min(), df["frame"].max() + 1)
    })
    df = full_frames.merge(df, on="frame", how="left")

    # -------------------------------
    # 2. Marcar valores inválidos
    # -------------------------------
    df.loc[(df["angle"] <= 0) | (df["angle"] >= 180), "angle"] = np.nan

    # -------------------------------
    # 3. Detectar saltos bruscos
    # -------------------------------
    df["diff"] = df["angle"].diff().abs()
    df.loc[df["angle"].isna(), "diff"] = np.nan

    df["salto_brusco"] = df["diff"] > umbral_salto

    df["salto_aislado"] = (
        (df["salto_brusco"]) &
        (~df["salto_brusco"].shift(1).fillna(False)) &
        (~df["salto_brusco"].shift(-1).fillna(False))
    )

    df.loc[df["salto_aislado"], "angle"] = np.nan

    # -------------------------------
    # 4. Detectar bloques de NaN
    # -------------------------------
    is_nan = df["angle"].isna()

    groups = is_nan.ne(is_nan.shift()).cumsum()
    group_sizes = is_nan.groupby(groups).transform("sum")

    # nº de frames equivalente a X segundos
    min_frames_saturacion = int(min_seg_saturacion * fps)

    # -------------------------------
    # 5. Saturar bloques largos
    # -------------------------------
    mask_saturar = (is_nan) & (group_sizes >= min_frames_saturacion)

    df.loc[mask_saturar, "angle"] = 180

    # -------------------------------
    # 6. Interpolar bloques cortos
    # -------------------------------
    df["angle"] = df["angle"].interpolate(limit_direction="both", limit=10)

    # -------------------------------
    # 7. Limpiar columnas auxiliares
    # -------------------------------
    df = df.drop(columns=["diff", "salto_brusco", "salto_aislado"])

    return df

# ===============================
# SUAVIZADO
# ===============================
def suavizar_senal_angulos(df):

    angles = df["angle"].values

    # -------------------------------
    # 1. FILTRO MEDIANA
    # -------------------------------
    angles_med = pd.Series(angles).rolling(5, center=True).median()

    angles_med = angles_med.fillna(method="bfill").fillna(method="ffill")

    # -------------------------------
    # 2. SAVITZKY-GOLAY
    # -------------------------------
    if len(angles_med) > 11:
        smooth = savgol_filter(
            angles_med,
            window_length=11,
            polyorder=2
        )
    else:
        smooth = angles_med

    df["angle_smooth"] = smooth

    return df

def calcular_angulo_video(video_path, frame_interno, frame_injection, frame_aspiracion, frame_retroceso,output_dir, output_debug):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ang, model_pip, model_ovo = load_angle_models(device)

    cap = cv2.VideoCapture(video_path)

    prev_angle = None
    max_jump = 10       # salto máximo permitido entre frames
    max_drift = 20      # límite duro
    window_size = 5     # ventana de estabilidad

    if not cap.isOpened():
        print("Error abriendo video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===============================
    # CREAR VIDEO DEBUG SOLO SI SE PIDE
    # ===============================
    writer = None

    if output_debug is not None:
        writer = cv2.VideoWriter(
            output_debug,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

    frame_id = 0
    angles = []
    frames = []

    debug_frames = {}

    # directorio de salida
    if output_debug is not None:
        output_dir = os.path.dirname(output_debug)
        base_name = os.path.splitext(os.path.basename(output_debug))[0]
    else:
        # directorio de salida (SIEMPRE el del pipeline)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id < frame_interno:
            if writer is not None:
                writer.write(frame)
            continue

        if frame_id > frame_retroceso:
            if writer is not None:
                writer.write(frame)
            continue

        temp_path = "temp_frame.png"
        cv2.imwrite(temp_path, frame)
      

        angle, debug, status = calcular_angulo_frame(
            temp_path,
            model_ang,
            model_pip,
            model_ovo,
            device
        )

        

        if status == "OK" and 10 < angle < 170:

            # -------------------------------
            # FILTRO TEMPORAL (salto)
            # -------------------------------
            if prev_angle is not None:

                # salto fuerte → ignorar
                if abs(angle - prev_angle) > max_jump:
                    angle = prev_angle

                # drift extremo → bloquear
                if abs(angle - prev_angle) > max_drift:
                    angle = prev_angle

            # -------------------------------
            # FILTRO DE CONSISTENCIA (ventana)
            # -------------------------------
            if len(angles) >= window_size:
                recent = angles[-window_size:]

                if np.nanstd(recent) > 8:
                    angle = prev_angle if prev_angle is not None else angle

            # actualizar referencia
            prev_angle = angle

            angles.append(angle)

            if writer is not None:
                writer.write(debug)

            debug_frames[frame_id] = debug

        else:

            # si falla → mantener continuidad
            if prev_angle is not None:
                angles.append(prev_angle)
            else:
                angles.append(np.nan)

            if writer is not None:
                writer.write(frame)
        frames.append(frame_id)

    cap.release()

    if writer is not None:
        writer.release()

    if len(angles) == 0:
        print("No se calcularon ángulos")
        return

    # ===============================
    # CREAR SEÑAL
    # ===============================
    df = pd.DataFrame({
        "frame": frames,
        "angle": angles
    })
    df = quedarse_con_senal_buena(df)
    df = limpiar_e_interpolar_senal(df,fps)
    df = suavizar_senal_angulos(df)

    # ===============================
    # FRAME CON MENOR ÁNGULO
    # ===============================

    if frame_aspiracion is not None:
        start_frame = max(frame_aspiracion - 30, frame_interno)
    else:
        # fallback: usar todo el tramo de inyección
        start_frame = frame_interno

    end_frame = frame_retroceso

    df_valid = df[
        (df["frame"] >= start_frame) &
        (df["frame"] <= end_frame) &
        (df["angle"] > 0)
    ]

    if len(df_valid) > 0:

        min_angle = df_valid["angle"].min()
        min_row = df_valid.loc[df_valid["angle"].idxmin()]
        frame_min_angle = int(min_row["frame"])

    else:

        frame_min_angle = None
        min_angle = None

    # ===============================
    # GUARDAR IMAGEN DEL MÍNIMO
    # ===============================
    if frame_min_angle is not None and frame_min_angle in debug_frames:

        min_angle_image = os.path.join(output_dir, "frame_min_angle.png")

        cv2.imwrite(
            min_angle_image,
            debug_frames[frame_min_angle]
        )

        print("Imagen del mínimo ángulo guardada:", min_angle_image)

    # ===============================
    # GUARDAR CSV
    # ===============================
    signal_csv = os.path.join(output_dir, base_name + "_signal.csv")

    df.to_csv(signal_csv, index=False)

    print("Señal guardada:", signal_csv)

    # ===============================
    # GUARDAR GRÁFICA
    # ===============================
    plt.figure(figsize=(10,4))

    plt.plot(df["frame"], df["angle"], alpha=0.4, label="raw")
    plt.plot(df["frame"], df["angle_smooth"], linewidth=2, label="smooth")

    plt.xlabel("Frame")
    plt.ylabel("Ángulo (deg)")
    plt.title("Señal del ángulo durante la inyección")

    plt.ylim(0, 130)

    plt.legend()
    plt.grid()

    signal_plot = os.path.join(output_dir, base_name + "_signal.png")

    plt.savefig(signal_plot)
    plt.close()

    print("Gráfica guardada:", signal_plot)

    return df, frame_min_angle, min_angle