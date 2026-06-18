import os
import cv2
import pandas as pd
import numpy as np
import config


def calcular_velocidad_clip(csv_path, fps):
    """
    Calcula la velocidad de inyección para UN clip específico.
    
    Parámetros:
        csv_path : ruta al archivo posiciones.csv
        fps      : frames por segundo del video
    
    Devuelve:
        velocidad_px_s : velocidad media en píxeles/segundo (float o None)
    """
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Detectar columnas de posición
        if "pipeta_x" not in df.columns or "pipeta_y" not in df.columns:
            return None
        
        # Calcular distancia total recorrida
        dx = df["pipeta_x"].diff()
        dy = df["pipeta_y"].diff()
        distancia_total = np.sqrt(dx**2 + dy**2).sum(skipna=True)
        
        # Tiempo total
        num_frames = len(df)
        tiempo_total = (num_frames - 1) / fps if fps > 0 else 0
        
        if tiempo_total <= 0:
            return None
        
        velocidad = distancia_total / tiempo_total
        return velocidad
        
    except Exception as e:
        print(f"[WARN] Error calculando velocidad: {e}")
        return None


def calcular_velocidad_inyeccion():

    """
    Calcula la velocidad de inyección desde los CSV de posiciones
    generados por el pipeline.
    Lee desde OUTPUT_DIR/carpeta/clip_name/posiciones.csv
    """

    global_rows = []

    # =================================================
    # LOOP PRINCIPAL
    # =================================================
    for carpeta in sorted(os.listdir(config.OUTPUT_DIR)):
        carpeta_path = os.path.join(config.OUTPUT_DIR, carpeta)
        if not os.path.isdir(carpeta_path):
            continue

        print(f"▶ Procesando carpeta: {carpeta}")

        carpeta_rows = []

        for clip_name in sorted(os.listdir(carpeta_path)):
            
            clip_output_path = os.path.join(carpeta_path, clip_name)
            if not os.path.isdir(clip_output_path):
                continue

            # Buscar CSV de posiciones
            csv_path = os.path.join(clip_output_path, "posiciones.csv")

            if not os.path.exists(csv_path):
                print(f"  ⚠️ No existe CSV para {clip_name}")
                continue

            # Obtener FPS desde el video original
            input_video_path = None
            for folder in os.listdir(config.INPUT_DIR):
                folder_path = os.path.join(config.INPUT_DIR, folder)
                if not os.path.isdir(folder_path):
                    continue
                for file in os.listdir(folder_path):
                    if file.startswith(clip_name) and file.endswith((".mp4", ".avi", ".mkv")):
                        input_video_path = os.path.join(folder_path, file)
                        break
                if input_video_path:
                    break

            if input_video_path is None:
                print(f"  ⚠️ No se encontró video para {clip_name}")
                continue

            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps = fps if fps > 0 else 25

            # Cargar CSV de posiciones
            df = pd.read_csv(csv_path)

            # Detectar columnas de posición
            if {"pipeta_x", "pipeta_y"}.issubset(df.columns):
                xcol, ycol = "pipeta_x", "pipeta_y"
            else:
                print(f"  ⚠️ No se encontraron columnas de posición en {clip_name}: {list(df.columns)}")
                continue

            result_path = os.path.join(clip_output_path, "resultados.txt")

            injection = None
            interno = None

            if os.path.exists(result_path):

                with open(result_path) as f:
                    for line in f:
                        if line.startswith("frame_inyeccion"):
                            injection = int(line.split(":")[1])
                        if line.startswith("frame_interno"):
                            interno = int(line.split(":")[1])

            df_segment = df.iloc[injection:interno]
            if injection is None or interno is None:
                print(f"  ⚠️ Frames de evento no encontrados en {clip_name}")
                continue
            # Calcular distancia total recorrida
            dx = df_segment[xcol].diff()
            dy = df_segment[ycol].diff()

            distancia = np.sqrt(dx**2 + dy**2).sum(skipna=True)

            # Tiempo total
            num_frames = len(df_segment)
            tiempo = (num_frames - 1) / fps

            velocidad = distancia / tiempo if tiempo > 0 else np.nan

            # Guardar TXT por clip
            txt_clip = os.path.join(clip_output_path, "velocidad_inyeccion.txt")
            with open(txt_clip, "w") as f:
                f.write(f"clip: {clip_name}\n")
                f.write(f"frames: {num_frames}\n")
                f.write(f"fps: {fps:.2f}\n")
                f.write(f"distancia_total_px: {distancia:.3f}\n")
                f.write(f"tiempo_total_s: {tiempo:.3f}\n")
                f.write(f"velocidad_media_px_s: {velocidad:.3f}\n")

            print(f"  ✅ Velocidad calculada: {clip_name}")

            row = {
                "carpeta": carpeta,
                "clip": clip_name,
                "frames": num_frames,
                "fps": round(fps, 2),
                "distancia_px": round(distancia, 3),
                "tiempo_s": round(tiempo, 3),
                "velocidad_px_s": round(velocidad, 3)
            }

            carpeta_rows.append(row)
            global_rows.append(row)

        # TXT resumen por carpeta
        if carpeta_rows:
            txt_carpeta = os.path.join(carpeta_path, "velocidades_resumen.txt")
            with open(txt_carpeta, "w") as f:
                for r in carpeta_rows:
                    f.write(
                        f"{r['clip']} | "
                        f"velocidad={r['velocidad_px_s']} px/s | "
                        f"tiempo={r['tiempo_s']} s\n"
                    )

    # Guardar tabla global
    if global_rows:
        df_global = pd.DataFrame(global_rows)
        csv_global = os.path.join(config.OUTPUT_DIR, "velocidades_todas.csv")
        df_global.to_csv(csv_global, index=False)
        print(f"\n✅ Tabla global guardada: {csv_global}")



