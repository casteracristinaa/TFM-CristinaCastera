from fileinput import filename
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from lime.lime_tabular import LimeTabularExplainer
import shap
import warnings
warnings.filterwarnings("ignore")
# =========================
# CONFIG
# =========================
DATASET_PATH = "../../../BBDD/labels/processed/DATASET.xlsx"
SIGNAL_BASE_PATH = "../../logs/3_señal_temporal"

y_true_all = []
y_pred_all = []

# =========================
# FUNCIONES SEÑAL
# =========================
def resumir_signal(signal, n_bins=6):
    bins = np.array_split(signal, n_bins)
    return np.array([np.median(b) for b in bins])

def extraer_features_signal(ruta_csv):
    try:
        df_signal = pd.read_csv(ruta_csv)
        angulo = df_signal["angle_clean"].values

        if len(angulo) < 10:
            return "SHORT", None

        features = {
            "rango": np.max(angulo) - np.min(angulo),
            "std_signal": np.std(angulo),
            "energia": np.sum(np.diff(angulo)**2),
            "velocidad_angular": np.mean(np.abs(np.diff(angulo))),
            "min_val": np.min(angulo),
            "std_total": np.std(angulo),
            "std_last": np.std(angulo[-30:]),
            "stability_ratio": np.std(angulo[-30:]) / (np.std(angulo) + 1e-6),
            "peaks": np.sum((np.diff(np.sign(np.diff(angulo))) < 0)),
            "stable_zone": np.sum(np.abs(np.diff(angulo)) < 0.5) / len(angulo),
        }

        bins = resumir_signal(angulo, n_bins=6)
        for i, val in enumerate(bins):
            features[f"bin_{i}"] = val

        return "OK", features

    except:
        return "ERROR", None

# =========================
# CREAR DATASET
# =========================
df = pd.read_excel(DATASET_PATH)
df = df.dropna(subset=["fecundado"])

filas = []

for _, row in df.iterrows():

    id_raw = str(row["ID"])

    if "_" in id_raw:
        base, sufijo = id_raw.split("_")
        id_ = base.zfill(8) + "_" + sufijo
    else:
        id_ = id_raw.zfill(8)
    clip = row["CLIP"]

    ruta_signal = os.path.join(
        SIGNAL_BASE_PATH,
        id_,
        clip,
        f"{clip}_signal_processed.csv" # f"{clip}_signal_processed.csv"
    )

    estado, features_signal = extraer_features_signal(ruta_signal)

    if estado != "OK":
        continue

    fila = features_signal.copy()

    # 🔥 AÑADE ESTO
    fila["ID"] = id_
    fila["CLIP"] = clip

    fila["MAG"] = row["MAG"]
    fila["EDAD_OVO"] = row["EDAD OVO"]
    fila["OVO_MII_INSEMIN_IN"] = row["OVO_MII_INSEMIN_IN"]
    fila["velocidad_inyeccion"] = row["velocidad_inyeccion"]
    fila["fecundado"] = row["fecundado"]

    filas.append(fila)

df_final = pd.DataFrame(filas)

print("Shape of final dataset:", df_final.shape)
print("Value counts of 'fecundado':")
print(df_final["fecundado"].value_counts())


# =========================
# COMPARACIÓN CON DATASET ORIGINAL
# =========================

df_original = pd.read_excel(DATASET_PATH)
df_original = df_original.dropna(subset=["fecundado"])

# Crear IDs consistentes
ids_original = set(
    df_original["ID"].astype(str).str.zfill(8) + "_" + df_original["CLIP"].astype(str)
)

ids_filtrado = set(
    df_final["ID"].astype(str) + "_" + df_final["CLIP"].astype(str)
)

# Diferencias
ids_eliminados = ids_original - ids_filtrado


print("Total original:", len(ids_original))
print("Total filtrado:", len(ids_filtrado))
print("Eliminados:", len(ids_eliminados))

# =========================
# FEATURES
# =========================
features_base = ["MAG", "EDAD_OVO", "OVO_MII_INSEMIN_IN"]

features_signal = [col for col in df_final.columns if col.startswith("bin_") or col in [
    "rango", "std_signal", "energia", "velocidad_angular",
    "min_val", "std_total", "std_last", "stability_ratio",
    "peaks", "stable_zone"
]]


# =========================
# FUNCIÓN EVALUACIÓN
# =========================
def evaluar_modelo(features,nombre_modelo):

    X = df_final[features]
    y = df_final["fecundado"]

    y_true_all = []
    y_pred_all = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aucs, f1s = [], [], []
    importancias_folds = []

    fp_all = []

    ids = df_final["ID"].astype(str) + "_" + df_final["CLIP"].astype(str)

    for train_idx, test_idx in skf.split(X, y):
        ids_test = ids.iloc[test_idx]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
           
        # 🔥 FALSOS POSITIVOS
        fp_mask = (y_pred == 1) & (y_test.values == 0)
        fp_ids = ids_test[fp_mask]

        fp_all.extend(fp_ids.tolist())
        y_prob = model.predict_proba(X_test)[:, 1]

        # 🔥 AÑADIR ESTO
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))

        importancias_folds.append(model.feature_importances_)
    
    # 🔥 GUARDAR
    filename = f"falsos_positivos_{nombre_modelo}.txt"

    with open(filename, "w") as f:
        for item in fp_all:
            f.write(item + "\n")

    importancias_media = np.mean(importancias_folds, axis=0)

    # 🔥 MATRIZ CORRECTA
    cm = confusion_matrix(y_true_all, y_pred_all)

    TN, FP, FN, TP = cm.ravel()

    # Sensibilidad
    sensibilidad = TP / (TP + FN)

    # Especificidad
    especificidad = TN / (TN + FP)

    print("\nMatriz de confusión global:")
    print(cm)

    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")

    return (
        np.mean(accs),
        np.mean(aucs),
        np.mean(f1s),
        sensibilidad,
        especificidad,
        importancias_media
    )

# =========================
# MODELO BASE
# =========================
acc1, auc1, f11, sens1, esp1, imp1 = evaluar_modelo(features_base, "base")

print("\nENFOQUE 2 - BASE")
print("Accuracy:", acc1)
print("AUC:", auc1)
print("F1:", f11)
print("Sensibilidad:", sens1)
print("Especificidad:", esp1)

# =========================
# MODELO EXTENDIDO
# =========================
features_ext = features_base + ["velocidad_inyeccion"] + features_signal



acc2, auc2, f12, sens2, esp2, imp2 = evaluar_modelo(features_ext, "extendido")

print("\nENFOQUE 2 - BASE + VELOCIDAD + SEÑAL")
print("Accuracy:", acc2)
print("AUC:", auc2)
print("F1:", f12)
print("Sensibilidad:", sens2)
print("Especificidad:", esp2)

# =========================
# IMPORTANCIAS
# =========================
df_imp_base = pd.DataFrame({
    "feature": features_base,
    "importance": imp1
}).sort_values(by="importance", ascending=False)

df_imp_ext = pd.DataFrame({
    "feature": features_ext,
    "importance": imp2
}).sort_values(by="importance", ascending=False)

print("\nIMPORTANCIAS BASE:")
print(df_imp_base)

print("\nIMPORTANCIAS EXTENDIDO (TOP 15):")
print(df_imp_ext.head(15))
import shap

# =========================
# MODELO FINAL (CONFIG 3)
# =========================
X_raw = df_final[features_ext].copy()
y = df_final["fecundado"]

# imputación
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

# balanceo
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# modelo final
model_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model_final.fit(X_res, y_res)


# =========================
# SHAP (FORMA ROBUSTA)
# =========================
explainer = shap.TreeExplainer(model_final)

shap_values = explainer(X)   # 👈 IMPORTANTE (NO shap_values())

# coger clase positiva
shap_values_class1 = shap_values[:, :, 1]

# =========================
# RENOMBRAR FEATURES PARA PLOTS
# =========================
rename_map = {
    "MAG": "Puntuación Magenta™",                      # "Magenta™ Score"
    "EDAD_OVO": "Edad del ovocito",                   # "Oocyte Age"
    "OVO_MII_INSEMIN_IN": "Nº de ovocitos maduros",  # "Number of Mature Oocytes"
    "velocidad_inyeccion": "Velocidad de inyección",  # "Injection Velocity"
    "rango": "Rango",                                 # "Range"
    "std_signal": "Desviación estándar",              # "Standard Deviation"
    "energia": "Energía",                             # "Energy"
    "velocidad_angular": "Velocidad angular",         # "Angular Velocity"
    "min_val": "Valor mínimo",                        # "Minimum Value"
    "std_total": "Desviación estándar total",         # "Total Standard Deviation"
    "std_last": "Desviación estándar (últimos 30)",   # "Standard Deviation (Last 30)"
    "stability_ratio": "Ratio de estabilidad",        # "Stability Ratio"
    "peaks": "Picos",                                 # "Peaks"
    "stable_zone": "Zona estable",                    # "Stable Zone"
    "bin_0": "Intervalo 0",                           # "Bin 0"
    "bin_1": "Intervalo 1",                           # "Bin 1"
    "bin_2": "Intervalo 2",                           # "Bin 2"
    "bin_3": "Intervalo 3",                           # "Bin 3"
    "bin_4": "Intervalo 4",                           # "Bin 4"
    "bin_5": "Intervalo 5",                           # "Bin 5"
    "bin_6": "Intervalo 6",                           # "Bin 6"
    "bin_7": "Intervalo 7",                           # "Bin 7"
}

features_ext_renamed = [rename_map.get(f, f) for f in features_ext]

import matplotlib.pyplot as plt
import numpy as np

shap_vals = shap_values_class1.values

# Separar contribuciones positivas y negativas
mean_pos = np.where(shap_vals > 0, shap_vals, 0).mean(axis=0)
mean_neg = np.where(shap_vals < 0, shap_vals, 0).mean(axis=0)

# Ordenar por importancia absoluta total
order = np.argsort(mean_pos + np.abs(mean_neg))
features_sorted = np.array(features_ext)[order]
mean_pos_sorted = mean_pos[order]
mean_neg_sorted = mean_neg[order]

# Bar chart — cambia features_sorted por la versión renombrada
features_sorted_renamed = [rename_map.get(f, f) for f in features_sorted]

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(features_sorted_renamed, mean_pos_sorted, color="#d73027", label="↑ fecundación")
ax.barh(features_sorted_renamed, mean_neg_sorted, color="#4575b4", label="↓ fecundación")
ax.axvline(0, color="black", linewidth=0.8)
ax.legend(loc="lower right")
ax.set_xlabel("Valor SHAP medio")
ax.set_title("Impacto promedio de cada característica en la predicción")
plt.tight_layout()
plt.savefig("shap_bar_direccion.png", bbox_inches="tight", dpi=150)
plt.close()

# Waterfall — pasar feature_names directamente
shap.plots.waterfall(shap_values_class1[0], show=False)
plt.gca().set_yticklabels([rename_map.get(t.get_text(), t.get_text()) 
                           for t in plt.gca().get_yticklabels()])
plt.savefig("shap_waterfall.png", bbox_inches="tight", dpi=150)
plt.close()

# =========================
# BEESWARM
# =========================

shap_values_class1.feature_names = features_ext_renamed

shap.plots.beeswarm(shap_values_class1, max_display=len(features_ext_renamed), show=False)

# Traducir etiquetas automáticas
fig = plt.gcf()

# Barra de color
cbar_ax = fig.axes[-1]
cbar_ax.set_ylabel("Valor de la variable")

# Cambiar High / Low
cbar_ax.set_yticklabels(["Bajo", "Alto"])

# Eje X
plt.xlabel("Valor SHAP (impacto en la predicción)")

plt.title("Gráfico SHAP Beeswarm")
plt.tight_layout()
plt.savefig("shap_beeswarm_español.png", bbox_inches="tight", dpi=150)
plt.close()


rename_map = {
    "MAG": "Puntuación Magenta™",
    "EDAD_OVO": "Edad del ovocito",
    "OVO_MII_INSEMIN_IN": "Nº de ovocitos maduros",
    "velocidad_inyeccion": "Velocidad de inyección",
    "rango": "Rango",
    "std_signal": "Desv. estándar",
    "energia": "Energía",
    "velocidad_angular": "Velocidad angular",
    "min_val": "Valor mínimo",
    "std_total": "Desv. estándar total",
    "std_last": "Desv. estándar (últ. 30)",
    "stability_ratio": "Ratio de estabilidad",
    "peaks": "Picos",
    "stable_zone": "Zona estable",
    **{f"bin_{i}": f"Intervalo {i}" for i in range(8)},
}
 
# =========================
# PREPARAR DATOS
# =========================
X_raw = df_final[features_ext].copy()
y = df_final["fecundado"]
 
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)
 
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_imp, y)
 
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_res, y_res)
 
features_renamed = [rename_map.get(f, f) for f in features_ext]
n_features = len(features_ext)
 
# =========================
# LIME — recoger peso Y valor original por muestra
# Usamos discretize_continuous=False para evitar que los nombres
# de las features lleven rangos y así el matching es exacto.
# =========================
print("Calculando LIME (beeswarm)...")
 
explainer_lime = LimeTabularExplainer(
    training_data=X_imp,
    feature_names=features_renamed,
    class_names=["No fecundado", "Fecundado"],
    mode="classification",
    discretize_continuous=False,   # ← clave: sin discretizar
    random_state=42,
)
 
N_SAMPLES = min(200, len(X_imp))
rng = np.random.default_rng(42)
indices = rng.choice(len(X_imp), size=N_SAMPLES, replace=False)
 
# matrices: filas=muestras, columnas=features
lime_weights = np.zeros((N_SAMPLES, n_features))   # peso LIME (con signo)
feature_values = np.zeros((N_SAMPLES, n_features)) # valor original normalizado [0,1]
 
for i, idx in enumerate(indices):
    exp = explainer_lime.explain_instance(
        data_row=X_imp[idx],
        predict_fn=model.predict_proba,
        num_features=n_features,
        labels=(1,),
    )
    lime_dict = dict(exp.as_list(label=1))
    for j, fname in enumerate(features_renamed):
        if fname in lime_dict:
            lime_weights[i, j] = lime_dict[fname]
    # guardar valores originales de esta muestra
    feature_values[i, :] = X_imp[idx]
 
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} muestras...")
 
print("LIME calculado.\n")
 
# Normalizar feature_values a [0,1] por columna (para colorear igual que SHAP)
feat_min = feature_values.min(axis=0)
feat_max = feature_values.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1  # evitar división por cero
feature_values_norm = (feature_values - feat_min) / feat_range
 
# =========================
# SHAP — mismo modelo, mismos datos
# =========================
print("Calculando SHAP...")
explainer_shap = shap.TreeExplainer(model)
shap_obj = explainer_shap(X_imp)
shap_vals = shap_obj[:, :, 1].values          # clase positiva, todas las muestras
shap_mean_abs = np.abs(shap_vals).mean(axis=0)
print("SHAP calculado.\n")
 
# =========================
# ORDEN COMPARTIDO: por importancia SHAP (de mayor a menor, de arriba a abajo)
# =========================
order = np.argsort(shap_mean_abs)[::-1]       # mayor importancia primero
feats_ordered = [features_renamed[i] for i in order]
 
# =========================
# FIGURA — dos beeswarms lado a lado
# =========================
cmap = plt.get_cmap("RdBu_r")   # rojo=alto, azul=bajo (igual que SHAP por defecto)
 
fig, axes = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
fig.subplots_adjust(wspace=0.05)
 
def draw_beeswarm(ax, weights_matrix, feat_values_norm, order, feats_ordered, title, xlabel):
    """Dibuja un beeswarm con jitter vertical para separar puntos solapados."""
    n_feats = len(order)
    rng_jitter = np.random.default_rng(0)
 
    for row_idx, feat_idx in enumerate(order):
        w = weights_matrix[:, feat_idx]
        v = feat_values_norm[:, feat_idx]
        colors_pts = cmap(v)
 
        # jitter vertical proporcional a la densidad
        jitter = rng_jitter.uniform(-0.35, 0.35, size=len(w))
        y_pos = (n_feats - 1 - row_idx) + jitter   # invertir para que el top quede arriba
 
        ax.scatter(w, y_pos, c=colors_pts, s=10, alpha=0.6, linewidths=0)
 
    ax.axvline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feats_ordered[::-1], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # dentro de la función, al final
    max_val = np.abs(weights_matrix).max() * 1.1
    ax.set_xlim(-max_val, max_val)
 
# Panel SHAP
draw_beeswarm(
    axes[0], shap_vals, 
    # normalizar shap feature values igual
    (X_imp - X_imp.min(axis=0)) / np.where((X_imp.max(axis=0)-X_imp.min(axis=0))==0, 1, X_imp.max(axis=0)-X_imp.min(axis=0)),
    order, feats_ordered,
    title="SHAP Beeswarm",
    xlabel="Valor SHAP (impacto en predicción)"
)
 
# Panel LIME
draw_beeswarm(
    axes[1], lime_weights, feature_values_norm,
    order, feats_ordered,
    title="LIME Beeswarm",
    xlabel="Peso LIME (impacto en predicción)"
)
axes[1].set_yticklabels([])  # ocultar etiquetas duplicadas
 
# Barra de color compartida
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Valor de la variable", fontsize=9)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Bajo", "Alto"])
 
fig.suptitle("Comparación SHAP vs LIME — Importancia y dirección de cada feature\n"
             "(ordenadas por importancia SHAP)", fontsize=12, y=1.01)
 
plt.savefig("lime_shap_beeswarm_comparacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("Guardado: lime_shap_beeswarm_comparacion.png")
 
# =========================
# RANKING TABLE
# =========================
lime_mean_abs = np.abs(lime_weights).mean(axis=0)
 
df_shap_rank = pd.DataFrame({"Feature": features_renamed, "SHAP_abs": shap_mean_abs})
df_shap_rank = df_shap_rank.sort_values("SHAP_abs", ascending=False).reset_index(drop=True)
df_shap_rank["Ranking_SHAP"] = df_shap_rank.index + 1
 
df_lime_rank = pd.DataFrame({"Feature": features_renamed, "LIME_abs": lime_mean_abs})
df_lime_rank = df_lime_rank.sort_values("LIME_abs", ascending=False).reset_index(drop=True)
df_lime_rank["Ranking_LIME"] = df_lime_rank.index + 1
 
df_cmp = df_shap_rank[["Feature","Ranking_SHAP"]].merge(df_lime_rank[["Feature","Ranking_LIME"]], on="Feature")
df_cmp["Diff"] = (df_cmp["Ranking_SHAP"] - df_cmp["Ranking_LIME"]).abs()
df_cmp = df_cmp.sort_values("Ranking_SHAP")
 
spearman_r = df_cmp["Ranking_SHAP"].corr(df_cmp["Ranking_LIME"], method="spearman")
 
print("=" * 60)
print("COMPARACIÓN DE RANKINGS: SHAP vs LIME")
print("=" * 60)
print(df_cmp.to_string(index=False))
print(f"\nCorrelación de Spearman: {spearman_r:.3f}")
print("\n✅ Archivo generado: lime_shap_beeswarm_comparacion.png")

# -------------------------------------------------------------------------------
"""
Análisis de importancia de features con LIME — Beeswarm style
Comparación directa con SHAP (dirección + importancia)

IMPORTANTE: ejecutar después del script principal, con df_final,
features_ext y rename_map ya en memoria.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from lime.lime_tabular import LimeTabularExplainer
import shap
import warnings
warnings.filterwarnings("ignore")

rename_map = {
    "MAG": "Puntuación Magenta™",
    "EDAD_OVO": "Edad del ovocito",
    "OVO_MII_INSEMIN_IN": "Nº de ovocitos maduros",
    "velocidad_inyeccion": "Velocidad de inyección",
    "rango": "Rango",
    "std_signal": "Desv. estándar",
    "energia": "Energía",
    "velocidad_angular": "Velocidad angular",
    "min_val": "Valor mínimo",
    "std_total": "Desv. estándar total",
    "std_last": "Desv. estándar (últ. 30)",
    "stability_ratio": "Ratio de estabilidad",
    "peaks": "Picos",
    "stable_zone": "Zona estable",
    **{f"bin_{i}": f"Intervalo {i}" for i in range(8)},
}

# =========================
# PREPARAR DATOS
# =========================
X_raw = df_final[features_ext].copy()
y = df_final["fecundado"]

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_imp, y)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_res, y_res)

features_renamed = [rename_map.get(f, f) for f in features_ext]
n_features = len(features_ext)

# =========================
# LIME — recoger peso Y valor original por muestra
# Usamos discretize_continuous=False para evitar que los nombres
# de las features lleven rangos y así el matching es exacto.
# =========================
print("Calculando LIME (beeswarm)...")

explainer_lime = LimeTabularExplainer(
    training_data=X_imp,
    feature_names=features_renamed,
    class_names=["No fecundado", "Fecundado"],
    mode="classification",
    discretize_continuous=False,   # ← clave: sin discretizar
    random_state=42,
)

N_SAMPLES = min(200, len(X_imp))
rng = np.random.default_rng(42)
indices = rng.choice(len(X_imp), size=N_SAMPLES, replace=False)

# matrices: filas=muestras, columnas=features
lime_weights = np.zeros((N_SAMPLES, n_features))   # peso LIME (con signo)
feature_values = np.zeros((N_SAMPLES, n_features)) # valor original normalizado [0,1]

for i, idx in enumerate(indices):
    exp = explainer_lime.explain_instance(
        data_row=X_imp[idx],
        predict_fn=model.predict_proba,
        num_features=n_features,
        labels=(1,),
    )
    lime_dict = dict(exp.as_list(label=1))
    for j, fname in enumerate(features_renamed):
        if fname in lime_dict:
            lime_weights[i, j] = lime_dict[fname]
    # guardar valores originales de esta muestra
    feature_values[i, :] = X_imp[idx]

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} muestras...")

print("LIME calculado.\n")

# Normalizar feature_values a [0,1] por columna (para colorear igual que SHAP)
feat_min = feature_values.min(axis=0)
feat_max = feature_values.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1  # evitar división por cero
feature_values_norm = (feature_values - feat_min) / feat_range

# =========================
# SHAP — mismo modelo, mismos datos
# =========================
print("Calculando SHAP...")
explainer_shap = shap.TreeExplainer(model)
shap_obj = explainer_shap(X_imp)
shap_vals = shap_obj[:, :, 1].values          # clase positiva, todas las muestras
shap_mean_abs = np.abs(shap_vals).mean(axis=0)
print("SHAP calculado.\n")

# =========================
# ORDEN COMPARTIDO: por importancia SHAP (de mayor a menor, de arriba a abajo)
# =========================
order = np.argsort(shap_mean_abs)[::-1]       # mayor importancia primero
feats_ordered = [features_renamed[i] for i in order]

# =========================
# FIGURA — dos beeswarms lado a lado
# =========================
cmap = plt.get_cmap("RdBu_r")   # rojo=alto, azul=bajo (igual que SHAP por defecto)

fig, axes = plt.subplots(1, 2, figsize=(20, 11), sharey=False)
fig.subplots_adjust(wspace=0.05)

def draw_beeswarm(ax, weights_matrix, feat_values_norm, order, feats_ordered, title, xlabel):
    """Dibuja un beeswarm con jitter vertical para separar puntos solapados."""
    n_feats = len(order)
    rng_jitter = np.random.default_rng(0)

    for row_idx, feat_idx in enumerate(order):
        w = weights_matrix[:, feat_idx]
        v = feat_values_norm[:, feat_idx]
        colors_pts = cmap(v)

        # jitter vertical proporcional a la densidad
        jitter = rng_jitter.uniform(-0.35, 0.35, size=len(w))
        y_pos = (n_feats - 1 - row_idx) + jitter   # invertir para que el top quede arriba

        ax.scatter(w, y_pos, c=colors_pts, s=10, alpha=0.6, linewidths=0)

    ax.axvline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feats_ordered[::-1], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    # dentro de la función, al final
    max_val = np.abs(weights_matrix).max() * 1.1
    ax.set_xlim(-max_val, max_val)

# Panel SHAP
draw_beeswarm(
    axes[0], shap_vals, 
    # normalizar shap feature values igual
    (X_imp - X_imp.min(axis=0)) / np.where((X_imp.max(axis=0)-X_imp.min(axis=0))==0, 1, X_imp.max(axis=0)-X_imp.min(axis=0)),
    order, feats_ordered,
    title="SHAP Beeswarm",
    xlabel="Valor SHAP (impacto en predicción)"
)

# Panel LIME
draw_beeswarm(
    axes[1], lime_weights, feature_values_norm,
    order, feats_ordered,
    title="LIME Beeswarm",
    xlabel="Peso LIME (impacto en predicción)"
)
axes[1].set_yticklabels([])  # ocultar etiquetas duplicadas

# Barra de color compartida
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Valor de la variable", fontsize=9)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Bajo", "Alto"])

fig.suptitle("Comparación SHAP vs LIME — Importancia y dirección de cada feature\n"
             "(ordenadas por importancia SHAP)", fontsize=12, y=1.01)

plt.savefig("lime_shap_beeswarm_comparacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("Guardado: lime_shap_beeswarm_comparacion.png")

# =========================
# RANKING TABLE
# =========================
lime_mean_abs = np.abs(lime_weights).mean(axis=0)

df_shap_rank = pd.DataFrame({"Feature": features_renamed, "SHAP_abs": shap_mean_abs})
df_shap_rank = df_shap_rank.sort_values("SHAP_abs", ascending=False).reset_index(drop=True)
df_shap_rank["Ranking_SHAP"] = df_shap_rank.index + 1

df_lime_rank = pd.DataFrame({"Feature": features_renamed, "LIME_abs": lime_mean_abs})
df_lime_rank = df_lime_rank.sort_values("LIME_abs", ascending=False).reset_index(drop=True)
df_lime_rank["Ranking_LIME"] = df_lime_rank.index + 1

df_cmp = df_shap_rank[["Feature","Ranking_SHAP"]].merge(df_lime_rank[["Feature","Ranking_LIME"]], on="Feature")
df_cmp["Diff"] = (df_cmp["Ranking_SHAP"] - df_cmp["Ranking_LIME"]).abs()
df_cmp = df_cmp.sort_values("Ranking_SHAP")

spearman_r = df_cmp["Ranking_SHAP"].corr(df_cmp["Ranking_LIME"], method="spearman")

print("=" * 60)
print("COMPARACIÓN DE RANKINGS: SHAP vs LIME")
print("=" * 60)
print(df_cmp.to_string(index=False))
print(f"\nCorrelación de Spearman: {spearman_r:.3f}")
print("\n✅ Archivo generado: lime_shap_beeswarm_comparacion.png")


# =========================
# LIME — EXPLICACIONES INDIVIDUALES
# =========================

def plot_lime_individual(idx_muestra, etiqueta=None, guardar_como=None):
    """
    Plot LIME individual para una muestra concreta.
    idx_muestra : índice en X_imp (0-based)
    """
    exp = explainer_lime.explain_instance(
        data_row=X_imp[idx_muestra],
        predict_fn=model.predict_proba,
        num_features=n_features,
        labels=(0, 1),
    )

    prob_no = model.predict_proba(X_imp[idx_muestra].reshape(1, -1))[0][0]
    prob_si = model.predict_proba(X_imp[idx_muestra].reshape(1, -1))[0][1]
    pred_cls = "Fecundado" if prob_si >= 0.5 else "No fecundado"

    lime_list = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=False)
    feat_names = [x[0] for x in lime_list]
    weights    = [x[1] for x in lime_list]
    colors     = ["#e05c4b" if w > 0 else "#5b8db8" for w in weights]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(feat_names) * 0.45 + 2)),
                             gridspec_kw={"width_ratios": [1, 3]})

    # Panel izquierdo: probabilidades
    ax_prob = axes[0]
    ax_prob.barh(["No fecundado", "Fecundado"], [prob_no, prob_si],
                 color=["#5b8db8", "#e05c4b"], height=0.5)
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Probabilidad")
    ax_prob.set_title("Predicción", fontsize=10, fontweight="bold")
    for i, v in enumerate([prob_no, prob_si]):
        ax_prob.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax_prob.spines[["top", "right"]].set_visible(False)

    # Panel derecho: contribuciones
    ax_bar = axes[1]
    y_pos = range(len(feat_names))
    ax_bar.barh(list(y_pos), weights, color=colors, height=0.6, alpha=0.85)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(feat_names, fontsize=8.5)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    max_abs = max(abs(w) for w in weights)
    ax_bar.set_xlim(-max_abs * 1.2, max_abs * 1.2)
    ax_bar.set_xlabel("Peso LIME  (rojo = ↑ fecundación  /  azul = ↓ fecundación)", fontsize=9)
    ax_bar.set_title("Contribución de cada feature", fontsize=10, fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)

    titulo = etiqueta or f"Muestra {idx_muestra} — Predicción: {pred_cls}"
    fig.suptitle(titulo, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Guardado: {guardar_como}")
    else:
        plt.show()


# Ejemplos automáticos: un TP, un TN y un FP
idx_tp = np.where((y.values == 1) & (model.predict(X_imp) == 1))[0]
idx_tn = np.where((y.values == 0) & (model.predict(X_imp) == 0))[0]
idx_fp = np.where((y.values == 0) & (model.predict(X_imp) == 1))[0]
idx_fn = np.where((y.values == 1) & (model.predict(X_imp) == 0))[0]  

if len(idx_tp): plot_lime_individual(idx_tp[0], "Verdadero Positivo (TP) — fecundado predicho correctamente", "lime_individual_TP.png")
if len(idx_tn): plot_lime_individual(idx_tn[0], "Verdadero Negativo (TN) — no fecundado predicho correctamente", "lime_individual_TN.png")
if len(idx_fp): plot_lime_individual(idx_fp[0], "Falso Positivo (FP) — predijo fecundado pero era no fecundado", "lime_individual_FP.png")
if len(idx_fn): plot_lime_individual(idx_fn[0], "Falso Negativo (FN) — predijo no fecundado pero era fecundado", "lime_individual_FN.png")  
# Para cualquier muestra por índice:
# plot_lime_individual(42, guardar_como="lime_muestra_42.png")

print("\n✅ Plots individuales: lime_individual_TP/TN/FP.png")