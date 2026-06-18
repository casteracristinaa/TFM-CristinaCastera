"""
MAIN PIPELINE ICSI
------------------
Orquesta el flujo completo:

1. Carga modelos
2. Segmentación
3. Cálculo de métricas temporales
4. Detección de eventos
5. Guardado de resultados
"""

import os

import config
from clip_extraction import generar_clips_desde_videos_largos
from model_loader import load_models
from segmentation import segment_video
from temporal_analysis import compute_temporal_metrics, extract_positions
from event_detection import detect_all_events
from io_utils import save_results, ensure_dir, save_debug_video, create_debug_video_with_overlays, guardar_en_dataset
from pipeline_heatmap import run_heatmap_analysis
from velocidad_inyeccion import calcular_velocidad_clip, calcular_velocidad_inyeccion
from angle_calculation import load_angle_models, calcular_angulo_frame
import traceback
import torch
import pandas as pd
import cv2
from angulo_video import calcular_angulo_video
from angulo_metricas import calcular_metricas_angulo



def process_clip(video_path, output_dir, model_p, model_o, model_ang, model_pip, model_ovo, id_paciente, clip_name):
    """
    Ejecuta el pipeline completo para un vídeo.
    """

    if config.DEBUG:
        print(f"\n========== Procesando: {video_path} ==========")

    try:

        # 1️. Segmentación
        masks_p, masks_o, fps = segment_video(
            video_path,
            model_p,
            model_o
        )

        # 1b. Extraer posiciones de pipeta y ovocito
        frame_ids, pipeta_x, pipeta_y, ovocito_x, ovocito_y = extract_positions(
            masks_p,
            masks_o
        )

        # Guardar CSV de posiciones
        positions_df = pd.DataFrame({
            "frame": frame_ids,
            "pipeta_x": pipeta_x,
            "pipeta_y": pipeta_y,
            "ovocito_x": ovocito_x,
            "ovocito_y": ovocito_y
        })
        positions_csv = os.path.join(output_dir, "posiciones.csv")
        positions_df.to_csv(positions_csv, index=False)

        # Calcular velocidad de inyección
        velocidad_inyeccion = calcular_velocidad_clip(positions_csv, fps)

        # 2️. Métricas temporales
        dmin, inside, speed = compute_temporal_metrics(
            masks_p,
            masks_o
        )

        # 3️. Detección de eventos
        events = detect_all_events(
            dmin,
            inside,
            speed,
            fps,
            masks_p,
            masks_o,
            video_path,
            output_dir
        )

        # 4️. Heatmap
        artifact_frame = run_heatmap_analysis(
            video_path,
            output_dir,
            events["interno"],
            events["retroceso"],
            events["injection"],
            model_p,
            model_o,
        )
        # 4b. Calcular ángulo del frame de aspiración
        angulo_deg = None
        angulo_status = None
        if artifact_frame is not None:
            frame_aspiracion_path = os.path.join(output_dir, "frame_aspiracion.png")
            if os.path.exists(frame_aspiracion_path):
                angulo_deg, debug_image, angulo_status = calcular_angulo_frame(
                    frame_aspiracion_path,
                    model_ang,
                    model_pip,
                    model_ovo,
                    config.DEVICE
                )
                if angulo_status == "OK" and debug_image is not None:
                    # Guardar imagen con ángulo dibujado
                    angulo_image_path = os.path.join(output_dir, "frame_aspiracion_angulo.png")
                    cv2.imwrite(angulo_image_path, debug_image)
                    if config.DEBUG:
                        print(f"[INFO] Ángulo: {angulo_deg:.2f}° - imagen guardada")
                elif angulo_status != "OK":
                    if config.DEBUG:
                        print(f"[INFO] Ángulo falló: {angulo_status}")

        # =================================================
        # 4c. Señal de ángulo durante inyección
        # =================================================

        angulo_signal_df = None

        if events["interno"] is not None and events["retroceso"] is not None:

            debug_angle_video = os.path.join(output_dir, "angulo_debug.mp4")

            if config.DEBUG:
                print(
                    f"[INFO] Calculando señal de ángulo "
                    f"(frames {events['interno']} → {events['retroceso']})"
                )

            angulo_signal_df, frame_min_angle, min_angle = calcular_angulo_video(
                video_path=video_path,
                frame_interno=events["interno"],
                frame_injection=events["injection"],
                frame_aspiracion=artifact_frame,
                frame_retroceso=events["retroceso"],
                output_dir=output_dir,
                output_debug=debug_angle_video
            )


            duracion_total = None
            tiempo_triangulo = None
            std_angulo = None

            if angulo_signal_df is not None:

                duracion_total, tiempo_triangulo, std_angulo = calcular_metricas_angulo(
                    angulo_signal_df,
                    events["interno"],
                    events["retroceso"],
                    fps
                )

        else:
            if config.DEBUG:
                print("[WARN] No se pudo calcular señal de ángulo: eventos no detectados")


        # 5️. Guardar frames clave
        save_results(
            output_dir,
            artifact_frame,
            fps,
            events,
            velocidad_inyeccion,
            angulo_deg,
            angulo_status,
            frame_min_angle,
            min_angle,
            duracion_total,
            tiempo_triangulo,
            std_angulo
        )

        # 5b. Guardar fila en el dataset global (.xlsx)
        guardar_en_dataset(id_paciente, clip_name, velocidad_inyeccion, angulo_deg)

        # 6️. Crear video debug con overlays
        #create_debug_video_with_overlays(
        #    video_path,
        #    output_dir,
        #    events,
        #    artifact_frame,
        #    fps
        #)



        print("✅ Video procesado correctamente")

    except Exception as e:

        print("❌ ERROR procesando:", video_path)
        print(traceback.format_exc())

        with open("errores_pipeline.txt", "a") as f:
            f.write(f"\nVideo: {video_path}\n")
            f.write(traceback.format_exc())

        # limpiar GPU si estás usando CUDA (solo si está disponible)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        print("⏭️ Continuando con el siguiente vídeo...\n")



# =================================================
# MAIN
# =================================================

if __name__ == "__main__":

    print("🚀 Iniciando PIPELINE ICSI...\n")

    # 0️. Recortar vídeos largos en clips, organizados por ID de paciente
    generar_clips_desde_videos_largos()

    # Cargar modelos una sola vez
    model_p, model_o, model_a = load_models()
    model_ang, model_pip, model_ovo = load_angle_models(config.DEVICE)

    # Recorrer carpetas de input
    for folder in sorted(os.listdir(config.INPUT_DIR)):

        folder_path = os.path.join(config.INPUT_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        for clip in os.listdir(folder_path):

            if not clip.endswith((".mp4", ".avi", ".mkv")):
                continue

            video_path = os.path.join(folder_path, clip)

            clip_name = os.path.splitext(clip)[0]

            output_dir = os.path.join(
                config.OUTPUT_DIR,
                folder,
                clip_name
            )

            ensure_dir(output_dir)

            process_clip(video_path, output_dir, model_p, model_o, model_ang, model_pip, model_ovo,
                         id_paciente=folder, clip_name=clip_name)

        

    print("\n✅ PIPELINE COMPLETO FINALIZADO")

    # Calcular velocidades de inyección desde los CSVs generados
    #print("\n▶ Calculando velocidades de inyección...")
    #calcular_velocidad_inyeccion()