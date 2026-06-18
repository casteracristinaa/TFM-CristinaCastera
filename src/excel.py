import pandas as pd

# =========================
# RUTAS DE LOS EXCELS
# =========================
excel_origen = "../../BBDD/labels/raw/Export_ProyectoICSI_mayo2026.xlsx"
excel_destino = "../../BBDD/labels/processed/DATASET-Nuevo-Enfoque.xlsx"

# =========================
# CARGAR ARCHIVOS
# =========================
# Hoja2 del excel origen
df_origen = pd.read_excel(excel_origen, sheet_name="Hoja2")

# Primer sheet del dataset destino
df_destino = pd.read_excel(excel_destino)

# =========================
# COLUMNAS A COPIAR
# =========================
columnas_copiar = [
    "fec_cp.-",
    "fec_pn.-",
    "FEC1",
    "NUEVA FEC",
    "TIPO DE FECUNDACIÓN",
    "LlegadaBlasto",
    "CalidadAB",
    "embrionFIN.-",
    "BLUT_NOPGT",
    "BlastoUtil",
    "BLEUPLOIDE",
    "BlastoTransferible",
    "resultadoEmbrionArrays",
    "t2",
    "t3",
    "t4",
    "t5",
    "tB",
    "Ticm",
    "Tte"
]

# =========================
# LIMPIEZA DE IDs
# =========================
# En el excel origen el ID tiene un 0 delante que hay que quitar
df_origen["ID_limpio"] = (
    df_origen["N_PROT_FIV_VC.-"]
    .astype(str)
    .str.lstrip("0")
)

# En el destino convertimos a string también
df_destino["ID_str"] = df_destino["ID"].astype(str)

# =========================
# EMPAREJAR FILAS POR ORDEN
# =========================
# Guardamos qué filas del origen ya hemos usado
indices_usados = set()

for idx_destino, fila_destino in df_destino.iterrows():

    id_destino = fila_destino["ID_str"]

    # Buscar filas coincidentes en origen que aún no se hayan usado
    coincidencias = df_origen[
        (df_origen["ID_limpio"] == id_destino) &
        (~df_origen.index.isin(indices_usados))
    ]

    # Si existe al menos una coincidencia
    if len(coincidencias) > 0:

        # Cogemos la primera disponible
        idx_origen = coincidencias.index[0]

        # Marcar como usada
        indices_usados.add(idx_origen)

        # Copiar columnas
        for col in columnas_copiar:

            if col in df_origen.columns:
                df_destino.loc[idx_destino, col] = df_origen.loc[idx_origen, col]

# =========================
# ELIMINAR COLUMNAS AUXILIARES
# =========================
df_destino.drop(columns=["ID_str"], inplace=True)

# =========================
# GUARDAR RESULTADO
# =========================
salida = "DATASET-nuevo-enfoque_actualizado.xlsx"

df_destino.to_excel(salida, index=False)

print(f"Archivo guardado correctamente: {salida}")