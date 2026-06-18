import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

# =========================
# CARGAR DATASET
# =========================
df = pd.read_excel("../../../BBDD/labels/processed/DATASET.xlsx")

# =========================
# VARIABLES
# =========================
features_base = ["MAG", "EDAD OVO", "OVO_MII_INSEMIN_IN"]
features_extra = ["angulo_triangulo", "velocidad_inyeccion"]

target = "fecundado"

# =========================
# LIMPIEZA
# =========================
df = df.dropna(subset=[target])

# =========================
# FUNCIÓN EVALUACIÓN
# =========================
def evaluar_modelo(features):

    X = df[features]
    y = df[target]

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

        # calcula el ratio en cada fold (importante: sobre y_train del fold)
        ratio = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,   # <- aquí
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

print("\nENFOQUE 1 - BASE")
print("Accuracy:", acc1)
print("AUC:", auc1)
print("F1:", f11)
print("Sensibilidad:", sens1)
print("Especificidad:", esp1)

# =========================
# MODELO EXTENDIDO
# =========================
features_ext = features_base + features_extra

acc2, auc2, f12, sens2, esp2 = evaluar_modelo(features_ext)

print("\nENFOQUE 1 - BASE + ANGULO + VELOCIDAD")
print("Accuracy:", acc2)
print("AUC:", auc2)
print("F1:", f12)
print("Sensibilidad:", sens2)
print("Especificidad:", esp2)