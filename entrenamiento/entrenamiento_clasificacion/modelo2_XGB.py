import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================
DATASET_PATH = "DATASET.xlsx"
SIGNAL_BASE_PATH = "../../logs/3_señal_temporal"

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
            return None

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

        bins = resumir_signal(angulo, n_bins=8)
        for i, val in enumerate(bins):
            features[f"bin_{i}"] = val

        return features

    except:
        return None

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
        f"{clip}_signal_processed.csv"
    )

    features_signal = extraer_features_signal(ruta_signal)

    if features_signal is None:
        continue

    fila = features_signal.copy()

    fila["ID"] = id_
    fila["CLIP"] = clip
    fila["MAG"] = row["MAG"]
    fila["EDAD_OVO"] = row["EDAD OVO"]
    fila["OVO_MII_INSEMIN_IN"] = row["OVO_MII_INSEMIN_IN"]
    fila["velocidad_inyeccion"] = row["velocidad_inyeccion"]
    fila["fecundado"] = row["fecundado"]

    filas.append(fila)

df_final = pd.DataFrame(filas)

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
def evaluar_modelo(features):

    X = df_final[features]
    y = df_final["fecundado"]

    print("N muestras:", len(X))
    print("Distribución clases:\n", y.value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aucs, f1s = [], [], []

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # quita estas dos líneas:
        # ros = RandomOverSampler(random_state=42)
        # X_train, y_train = ros.fit_resample(X_train, y_train)

        # añade esto:
        ratio = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,   # <- añade esto
            random_state=42,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # =========================
    # MATRIZ DE CONFUSIÓN
    # =========================
    cm = confusion_matrix(y_true_all, y_pred_all)

    TN, FP, FN, TP = cm.ravel()

    # Sensibilidad
    sensibilidad = TP / (TP + FN)

    # Especificidad
    especificidad = TN / (TN + FP)

    print("\nMatriz de confusión:")
    print(cm)

    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")

    return (
        np.mean(accs),
        np.mean(aucs),
        np.mean(f1s),
        sensibilidad,
        especificidad
    )

# =========================
# MODELO BASE
# =========================
acc1, auc1, f11, sens1, esp1 = evaluar_modelo(features_base)

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

acc2, auc2, f12, sens2, esp2 = evaluar_modelo(features_ext)

print("\nENFOQUE 2 - BASE + VELOCIDAD + SEÑAL")
print("Accuracy:", acc2)
print("AUC:", auc2)
print("F1:", f12)
print("Sensibilidad:", sens2)
print("Especificidad:", esp2)