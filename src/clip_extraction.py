"""
CLIP EXTRACTION MODULE
-----------------------
Paso 0 del pipeline ICSI.

Toma los vídeos largos (sin recortar) de config.RAW_VIDEOS_DIR, detecta
mediante un clasificador de frames (ResNet18) el instante en que la
inyección está en la posición óptima/estable, y genera los clips
recortados dentro de config.INPUT_DIR/<id_paciente>/, listos para ser
procesados por el resto del pipeline (segmentación, métricas temporales,
detección de eventos, etc.).

No contiene lógica de segmentación ni de detección de eventos.
"""

import os
import re
import csv
import glob
import subprocess
import traceback

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import config


# =================================================
# CLASIFICADOR DE FRAMES
# =================================================

def build_frame_classifier():
    """
    Construye la arquitectura del clasificador binario de frames
    (ResNet18 + cabezal con Sigmoid) usado para estimar la probabilidad
    de que un frame corresponda al momento óptimo de inyección.
    """
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model


def load_frame_classifier():
    """
    Carga el clasificador de frames desde disco y lo prepara para
    inferencia.
    """
    model = build_frame_classifier()

    state_dict = torch.load(config.CLASSIFIER_FRAMES_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(config.DEVICE)
    model.eval()

    if config.DEBUG:
        print(f"[INFO] Clasificador de frames cargado: {config.CLASSIFIER_FRAMES_PATH}")
        print(f"[INFO] Dispositivo: {config.DEVICE}")

    return model


FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])


# =================================================
# UTILIDADES
# =================================================

def extraer_id_paciente(filename):
    """
    Extrae el ID de paciente/procedimiento a partir del nombre de
    fichero del vídeo largo.

    Pensado para nombres del tipo:
        "2024-09-30 07242797 (parte II).mkv"
        "2024-09-30 07242797.mkv"

    Busca el primer grupo de >= 6 dígitos en el nombre y le quita los
    ceros a la izquierda (el ID real de paciente nunca empieza por 0;
    el "0" inicial es solo relleno/formato del nombre de fichero).

    Si no encuentra ningún grupo de dígitos, devuelve el nombre de
    fichero (sin extensión) como fallback, para no perder el vídeo.
    """
    nombre = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"\d{6,}", nombre)
    if not match:
        return nombre

    id_paciente = match.group(0).lstrip("0")

    if not id_paciente:
        # Caso extremo: el bloque de dígitos era todo ceros
        if config.DEBUG:
            print(f"[WARN] ID compuesto solo por ceros en '{filename}', se usa sin recortar")
        return match.group(0)

    return id_paciente


def siguiente_indice_clip(output_dir):
    """
    Calcula el siguiente índice de clip disponible dentro de una
    carpeta de ID, para no sobrescribir clips ya generados en
    ejecuciones anteriores (o por otra "parte" del mismo vídeo largo).
    """
    existentes = glob.glob(os.path.join(output_dir, "clip_*.mkv"))
    indices = []
    for path in existentes:
        nombre = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"clip_(\d+)", nombre)
        if m:
            indices.append(int(m.group(1)))
    return (max(indices) + 1) if indices else 0


def encontrar_inicio_estable(frame_probs, window=None, delta=None):
    """
    Busca, dentro de un tramo de alta probabilidad, la primera ventana
    de frames consecutivos cuya probabilidad se mantiene estable
    (variación <= delta). Devuelve el frame_idx de inicio de esa
    ventana, o None si no se encuentra ninguna.
    """
    window = config.WINDOW_ESTABLE if window is None else window
    delta = config.DELTA_ESTABLE if delta is None else delta

    for i in range(len(frame_probs) - window + 1):
        window_probs = [p for _, p in frame_probs[i:i + window]]
        if max(window_probs) - min(window_probs) <= delta:
            return frame_probs[i][0]
    return None


def calcular_probabilidades_frame(video_path, model):
    """
    Recorre el vídeo largo y calcula, muestreando a config.FPS_SAMPLE_RAW
    frames/segundo, la probabilidad de que cada frame muestreado
    corresponda al momento óptimo de inyección.

    Devuelve:
        frame_probs (list[(frame_idx, prob)])
        fps (float)
        frame_step (int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(fps / config.FPS_SAMPLE_RAW))

    frame_probs = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = FRAME_TRANSFORM(frame_pil).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                prob = model(frame_tensor).squeeze().cpu().item()
            frame_probs.append((frame_count, prob))

        frame_count += 1

    cap.release()

    if config.DEBUG:
        print(f"[INFO] {len(frame_probs)} frames muestreados (fps={fps:.2f}, step={frame_step})")

    return frame_probs, fps, frame_step


def detectar_intervalos(frame_probs, frame_step, threshold=None):
    """
    Agrupa los frames con probabilidad >= threshold en intervalos
    contiguos [inicio, fin] (en índice de frame).
    """
    threshold = config.PROB_THRESHOLD if threshold is None else threshold

    frames_alta_prob = [idx for idx, prob in frame_probs if prob >= threshold]

    intervals = []
    if not frames_alta_prob:
        return intervals

    start = frames_alta_prob[0]
    end = frames_alta_prob[0]
    for f in frames_alta_prob[1:]:
        if f == end + frame_step:
            end = f
        else:
            intervals.append((start, end))
            start = end = f
    intervals.append((start, end))

    return intervals


def guardar_csv_probabilidades(frame_probs, csv_path):
    """
    Guarda las probabilidades por frame en un CSV de depuración.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "probabilidad"])
        writer.writerows(frame_probs)


# =================================================
# RECORTE DE UN VÍDEO LARGO
# =================================================

def procesar_video_largo(video_path, model, output_dir, debug_dir):
    """
    Procesa un único vídeo largo: calcula probabilidades por frame,
    detecta los intervalos válidos de inyección y guarda los clips
    recortados dentro de `output_dir`.

    Devuelve la lista de clips generados (dicts con metadatos), para
    poder construir un resumen global.
    """
    nombre_video = os.path.basename(video_path)

    if config.DEBUG:
        print(f"\n========== Recortando: {nombre_video} ==========")

    frame_probs, fps, frame_step = calcular_probabilidades_frame(video_path, model)

    os.makedirs(debug_dir, exist_ok=True)
    csv_path = os.path.join(
        debug_dir, f"frames_probabilidades_{os.path.splitext(nombre_video)[0]}.csv"
    )
    guardar_csv_probabilidades(frame_probs, csv_path)

    intervals = detectar_intervalos(frame_probs, frame_step)

    if config.DEBUG:
        print(f"▶ {len(intervals)} intervalos detectados con prob >= {config.PROB_THRESHOLD}")

    os.makedirs(output_dir, exist_ok=True)
    clip_idx = siguiente_indice_clip(output_dir)

    cap = cv2.VideoCapture(video_path)
    clips_info = []

    for start, end in intervals:
        duration = (end - start + 1) / fps
        if duration < config.MIN_DURATION_SEC:
            if config.DEBUG:
                print(f"⛔ Intervalo descartado ({duration:.2f}s): frames [{start}, {end}]")
            continue

        sub_probs = [(f, p) for f, p in frame_probs if start <= f <= end]
        stable_start = encontrar_inicio_estable(sub_probs)

        if stable_start is not None:
            start_time = stable_start / fps
            clip_name = f"clip_{clip_idx:03d}.mkv"
        else:
            start_time = start / fps
            stable_start = start
            clip_name = f"clip_{clip_idx:03d}_no_recortado.mkv"

        duration_clip = (end - stable_start + 1) / fps
        output_path = os.path.join(output_dir, clip_name)

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration_clip),
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Frame representativo, para inspección visual rápida
        cap.set(cv2.CAP_PROP_POS_FRAMES, stable_start)
        ret, frame_opt = cap.read()
        if ret:
            frame_debug_path = os.path.join(
                debug_dir, f"clip_{clip_idx:03d}_frame_{stable_start}.jpg"
            )
            cv2.imwrite(frame_debug_path, frame_opt)

        clips_info.append({
            "video_origen": nombre_video,
            "clip": clip_name,
            "frame_inicio": stable_start,
            "frame_fin": end,
            "duracion_sec": duration_clip,
            "recortado": "_no_recortado" not in clip_name
        })

        if config.DEBUG:
            print(
                f"✅ Clip guardado: {clip_name} | Frames [{stable_start} → {end}] | "
                f"Duración: {duration_clip:.2f}s"
            )

        clip_idx += 1

    cap.release()
    return clips_info


# =================================================
# ENTRADA PRINCIPAL DEL PASO 0
# =================================================

def generar_clips_desde_videos_largos():
    """
    Recorre todos los vídeos largos en config.RAW_VIDEOS_DIR, los agrupa
    por ID de paciente/procedimiento (extraído del nombre de fichero) y
    genera los clips recortados dentro de config.INPUT_DIR/<id>/.

    Pensado para ejecutarse una sola vez al inicio del pipeline, antes
    del bucle principal de main.py. Si un vídeo largo falla, se registra
    el error y se continúa con el siguiente (igual que process_clip en
    main.py), para no detener todo el pipeline por un único vídeo.
    """
    if not os.path.isdir(config.RAW_VIDEOS_DIR):
        print(f"[WARN] No existe el directorio de vídeos largos: {config.RAW_VIDEOS_DIR}")
        return

    videos = sorted(
        f for f in os.listdir(config.RAW_VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mkv"))
    )

    if not videos:
        print(f"[WARN] No se encontraron vídeos largos en {config.RAW_VIDEOS_DIR}")
        return

    print(f"🎬 Recortando {len(videos)} vídeo(s) largo(s)...\n")

    model = load_frame_classifier()
    resumen = []

    for nombre_video in videos:
        video_path = os.path.join(config.RAW_VIDEOS_DIR, nombre_video)
        id_paciente = extraer_id_paciente(nombre_video)

        output_dir = os.path.join(config.INPUT_DIR, id_paciente)
        debug_dir = os.path.join(config.RECORTE_DEBUG_DIR, id_paciente)

        try:
            clips_info = procesar_video_largo(video_path, model, output_dir, debug_dir)
            resumen.extend(clips_info)

        except Exception:
            print("❌ ERROR recortando:", video_path)
            print(traceback.format_exc())

            with open("errores_pipeline.txt", "a") as f:
                f.write(f"\n[RECORTE] Video: {video_path}\n")
                f.write(traceback.format_exc())

            print("⏭️ Continuando con el siguiente vídeo largo...\n")

    # Liberar memoria GPU del clasificador antes de cargar el resto de modelos
    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Resumen global de los clips generados en esta ejecución
    if resumen:
        resumen_path = os.path.join(config.RECORTE_DEBUG_DIR, "clips_generados.txt")
        with open(resumen_path, "a") as f:
            for c in resumen:
                f.write(
                    f"[{c['video_origen']}] {c['clip']} | "
                    f"Frames [{c['frame_inicio']} → {c['frame_fin']}] | "
                    f"Duración {c['duracion_sec']:.2f}s | "
                    f"{'Recortado' if c['recortado'] else 'No recortado'}\n"
                )

    print(f"\n✅ Recorte finalizado: {len(resumen)} clip(s) generados en total.\n")