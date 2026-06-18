import numpy as np

def calcular_metricas_angulo(
        angulo_signal_df,
        frame_interno,
        frame_retroceso,
        fps):

    if angulo_signal_df is None or len(angulo_signal_df) == 0:
        return None, None, None

    # -----------------------------
    # DURACIÓN TOTAL
    # -----------------------------
    duracion_total = (frame_retroceso - frame_interno) / fps

    # -----------------------------
    # TIEMPO DE TRIÁNGULO
    # -----------------------------
    frames_con_triangulo = angulo_signal_df[
        angulo_signal_df["angle"] > 0
    ]["frame"].values

    tiempo_triangulo = None

    if len(frames_con_triangulo) > 0:

        primer_frame = frames_con_triangulo[0]
        ultimo_frame = frames_con_triangulo[-1]

        tiempo_triangulo = (ultimo_frame - primer_frame) / fps

    # -----------------------------
    # DESVIACIÓN TÍPICA (COMO ANTES)
    # -----------------------------
    std_angle = angulo_signal_df["angle"].std()

    return duracion_total, tiempo_triangulo, std_angle