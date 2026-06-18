import os
import shutil
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
BASE_PATH = "../../logs/3_señal_temporal"
DATASET_PATH = "../../../BBDD/Labels/processed/kk/Dataset_full.xlsx"

DRY_RUN = False  # ⚠️ True = no borra, solo muestra

# -------------------------------
# NORMALIZADORES
# -------------------------------
def normalize_id(x):
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x.zfill(8)

def normalize_clip(x):
    x = str(x).strip()
    num = int(x.split("_")[1])
    return f"clip_{num:03d}"

# -------------------------------
# 1. Leer dataset
# -------------------------------
df = pd.read_excel(DATASET_PATH)

df["ID"] = df["ID"].apply(normalize_id)
df["CLIP"] = df["CLIP"].apply(normalize_clip)

valid_pairs = set(zip(df["ID"], df["CLIP"]))

print(f"Pares válidos en Excel: {len(valid_pairs)}")

# -------------------------------
# 2. Recorrer carpetas
# -------------------------------
eliminados = 0
mantenidos = 0
archivos_borrados = 0

for id_folder in os.listdir(BASE_PATH):
    id_path = os.path.join(BASE_PATH, id_folder)

    if not os.path.isdir(id_path):
        continue

    id_norm = normalize_id(id_folder)

    for clip_folder in os.listdir(id_path):
        clip_path = os.path.join(id_path, clip_folder)

        if not os.path.isdir(clip_path):
            continue

        # -------------------------------
        # Normalizar clip
        # -------------------------------
        try:
            clip_norm = normalize_clip(clip_folder)
        except:
            print(f"⚠️ Nombre raro, eliminando carpeta: {clip_path}")
            if not DRY_RUN:
                shutil.rmtree(clip_path)
            eliminados += 1
            continue

        # -------------------------------
        # 1. BORRAR CARPETAS NO VÁLIDAS
        # -------------------------------
        if (id_norm, clip_norm) not in valid_pairs:
            print(f"🗑️ Eliminando carpeta: {id_norm}/{clip_norm}")
            if not DRY_RUN:
                shutil.rmtree(clip_path)
            eliminados += 1
            continue

        mantenidos += 1

        # -------------------------------
        # 2. LIMPIAR ARCHIVOS INTERNOS
        # -------------------------------
        expected_prefix = f"{clip_folder}_signal"

        for f in os.listdir(clip_path):
            file_path = os.path.join(clip_path, f)

            if not os.path.isfile(file_path):
                continue

            # mantener solo archivos signal
            if not f.startswith(expected_prefix):
                print(f"🧹 Borrando archivo: {file_path}")
                if not DRY_RUN:
                    os.remove(file_path)
                archivos_borrados += 1

# -------------------------------
# 3. LIMPIAR IDs VACÍOS (opcional)
# -------------------------------
for id_folder in os.listdir(BASE_PATH):
    id_path = os.path.join(BASE_PATH, id_folder)

    if os.path.isdir(id_path) and not os.listdir(id_path):
        print(f"🧹 Eliminando carpeta vacía: {id_path}")
        if not DRY_RUN:
            os.rmdir(id_path)

# -------------------------------
# RESUMEN
# -------------------------------
print("\n--- RESUMEN ---")
print(f"Clips mantenidos: {mantenidos}")
print(f"Carpetas eliminadas: {eliminados}")
print(f"Archivos borrados: {archivos_borrados}")
print("FIN")