import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# -------------------------------
# CONFIG
# -------------------------------
BASE_PATH = "../../logs/6_nuevos"
SAT_VALUE = 180
MAX_JUMP = 20  # saltos no físicos
RECOVERY_WINDOW = 20  # frames para comprobar si se recupera

def ajustar_tendencia(signal):

    y = signal.copy()

    # rellenar temporalmente para poder filtrar
    y_interp = pd.Series(y).interpolate(limit_direction="both")

    # suavizado fuerte (clave)
    y_smooth = savgol_filter(y_interp, window_length=51, polyorder=2)

    return y_smooth
# -------------------------------
# PREPROCESADO INTELIGENTE
# -------------------------------
def preprocess_signal(df):

    # -------------------------------
    # 0. coger señal base
    # -------------------------------
    raw_signal = pd.to_numeric(df["angle"], errors="coerce")
    #print(file_path, "MAX:", np.nanmax(raw_signal), "MIN:", np.nanmin(raw_signal))
    

    if "angle_smooth" in df.columns:
        signal = df["angle_smooth"].copy()
    else:
        signal = raw_signal.copy()

    signal = pd.to_numeric(signal, errors="coerce")

    # -------------------------------
    # detectar zonas saturadas largas
    # -------------------------------
    is_sat = signal >= 170  # margen

    groups = (is_sat != is_sat.shift()).cumsum()
    group_sizes = is_sat.groupby(groups).transform("sum")

    # mínimo de frames para considerar saturación real
    min_sat_len = 10

    sat_blocks = (is_sat) & (group_sizes >= min_sat_len)

    #print("Picos detectados:", signal.isna().sum())

    # -------------------------------
    # 1. eliminar picos (ruido puntual)
    # -------------------------------
    signal_temp = signal.copy()
    diff = signal_temp.diff().abs()
    signal[diff > MAX_JUMP] = np.nan

  
    # -------------------------------
    # 1.5 eliminar segmentos completos
    # -------------------------------
    window = 5
    expand = 2  # cuantos vecinos borrar alrededor

    for i in range(window, len(raw_signal) - window):

        local = raw_signal.iloc[i-window:i+window+1]

        if local.isna().any():
            continue

        center = raw_signal.iloc[i]
        median_local = np.median(local)

        if abs(center - median_local) > MAX_JUMP * 0.4:

            # borrar no solo el punto, sino el bloque
            start = max(0, i - expand)
            end = min(len(signal), i + expand + 1)

            signal.iloc[start:end] = np.nan
            
    # -------------------------------
    # 2. detectar bloques de NaN
    # -------------------------------
    is_nan = signal.isna()
    groups = (is_nan != is_nan.shift()).cumsum()

    for g in groups[is_nan].unique():

        idx = groups[groups == g].index
        start = idx[0]
        end = idx[-1]

        # -------------------------------
        # comprobar si se recupera
        # -------------------------------
        next_valid = signal.loc[end+1:end+RECOVERY_WINDOW].dropna()

        if len(next_valid) > 0:
            # ✔ se recupera → NO tocar (se interpolará después)
            continue
        else:
            # ❌ no se recupera → saturar
            signal.loc[idx] = SAT_VALUE

    

    # -------------------------------
    # 3. interpolar TODO lo recuperable
    # -------------------------------
    # interpolar solo donde NO es saturación
    signal_interp = signal.copy()
    signal_interp[sat_blocks] = np.nan
    signal_interp = signal_interp.interpolate(limit_direction="both")

    # restaurar saturación
    signal_interp[sat_blocks] = SAT_VALUE
    signal = signal_interp


    # -------------------------------
    # eliminar caídas desde saturación
    # -------------------------------
    for i in range(1, len(signal)):
        if signal.iloc[i-1] >= 170 and signal.iloc[i] < 150:
            signal.iloc[i] = SAT_VALUE
    
  
    # -------------------------------
    # 4. ajustar a parábola
    # -------------------------------
    signal_fit = ajustar_tendencia(signal.copy())

    # no modificar zonas saturadas
    signal_fit[sat_blocks] = SAT_VALUE

    # eliminar puntos que rompen la forma
    diff = np.abs(signal - signal_fit)

    mask = (diff > 10) & (~sat_blocks)
    signal[mask] = np.nan

    # interpolar otra vez
    signal = signal.interpolate(limit_direction="both")

    # -------------------------------
    # 5. fallback final
    # -------------------------------
    signal = signal.fillna(SAT_VALUE)

    return signal

# -------------------------------
# NORMALIZACIÓN (MEJORADA)
# -------------------------------
def normalize_signal(signal):

    # separar valores válidos de saturación
    signal_valid = signal.copy()
    signal_valid[signal_valid == SAT_VALUE] = np.nan

    min_val = signal_valid.min()
    max_val = signal_valid.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val - min_val == 0:
        return signal * 0

    norm = (signal_valid - min_val) / (max_val - min_val)

    # mantener saturación como -1 (o puedes usar 1 si prefieres)
    norm = norm.fillna(-1)

    return norm

# -------------------------------
# PROCESADO GLOBAL
# -------------------------------
total = 0

for id_folder in os.listdir(BASE_PATH):
    id_path = os.path.join(BASE_PATH, id_folder)

    if not os.path.isdir(id_path):
        continue

    for clip_folder in os.listdir(id_path):
        clip_path = os.path.join(id_path, clip_folder)

        if not os.path.isdir(clip_path):
            continue

        for file in os.listdir(clip_path):
            if file.endswith(".csv") and "signal" in file and "processed" not in file:

                file_path = os.path.join(clip_path, file)


                try:
                    df = pd.read_csv(file_path)

             
                    # -------------------------------
                    # PROCESAR
                    # -------------------------------
                    df["angle_clean"] = preprocess_signal(df)
                    df["angle_norm"] = normalize_signal(df["angle_clean"])

                    # -------------------------------
                    # GUARDAR
                    # -------------------------------
                    new_path = file_path.replace(".csv", "_processed.csv")
                    df.to_csv(new_path, index=False)

                    total += 1
                    print(f"OK: {id_folder}/{clip_folder}")


                    # -------------------------------
                    # GUARDAR GRÁFICA
                    # -------------------------------
                    plt.figure(figsize=(10,4))

                    # raw (si existe)
                    if "angle" in df.columns:
                        plt.plot(df["frame"], df["angle"], alpha=0.3, label="raw")

                    # clean
                    plt.plot(df["frame"], df["angle_clean"],linewidth=2, label="smooth")

                    plt.xlabel("Frame")
                    plt.ylabel("Ángulo (deg)")
                    plt.ylim(0, 130)
                    plt.title("Señal procesada")

                    plt.legend()
                    plt.grid()

                    plot_path = file_path.replace(".csv", "_processed.png")

                    plt.savefig(plot_path)
                    plt.close()

                except Exception as e:
                    print(f"ERROR en {file_path}: {e}")


print(f"\nTOTAL PROCESADOS: {total}")
print("FIN")