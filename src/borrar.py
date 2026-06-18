import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

archivo = "../../BBDD/labels/raw/Export_ProyectoICSI_mayo2026.xlsx"
hoja = "Hoja2"

# Columna donde están los números
columna_id = "N_PROT_FIV_VC.-"

# IDs a eliminar
ids_eliminar = [
    "7240636",
    "7240676",
    "7240683",
    "7240709",
    "7240794",
    "7240864",
    "7240975",
    "7241241",
    "7241490",
    "7241554",
    "7241666",
    "7241826",
    "7241987",
    "7242240",
    "7242248",
    "7242282",
    "7242761",
    "7242769",
    "7242797",
    "7242837",
    "7242890",
    "7242924",
    "7242930",
    "7243074",
    "7243085",
    "7243134",
    "7243223",
    "7243303",
    "7243481",
    "7243630",
    "7250077",
    "7250125",
    "7250205",
    "7250244",
    "7250255",
    "7250323",
    "7250334",
    "7250394",
    "7250478",
    "7250815",
    "7250935",
    "7251027",
    "7251112",
    "7251134",
    "7251219",
    "7251432",
    "7251518",
    "7251625"
]

# =========================
# LEER EXCEL
# =========================

df = pd.read_excel(archivo, sheet_name=hoja)

# =========================
# NORMALIZAR COLUMNA
# =========================

df[columna_id] = (
    df[columna_id]
    .astype(str)
    .str.strip()
    .str.lstrip("0")
)

# =========================
# ELIMINAR FILAS
# =========================

filas_antes = len(df)

df_filtrado = df[~df[columna_id].isin(ids_eliminar)]

filas_despues = len(df_filtrado)

print(f"Filas antes: {filas_antes}")
print(f"Filas después: {filas_despues}")
print(f"Filas eliminadas: {filas_antes - filas_despues}")

# =========================
# GUARDAR RESULTADO
# =========================

archivo_salida = "Export_ProyectoICSI_mayo2026_filtrado.xlsx"

with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:
    df_filtrado.to_excel(writer, sheet_name=hoja, index=False)

print(f"\nArchivo guardado como: {archivo_salida}")