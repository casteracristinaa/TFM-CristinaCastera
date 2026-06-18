import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aucs, f1s = [], [], []
    importancias_folds = []
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        importancias_folds.append(model.feature_importances_)

    importancias_media = np.mean(importancias_folds, axis=0)

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
        np.mean(accs), np.std(accs),
        np.mean(aucs), np.std(aucs),
        np.mean(f1s), np.std(f1s),
        sensibilidad,
        especificidad,
        importancias_media
    )
# =========================
# MODELO BASE
# =========================
acc1, acc1_std, auc1, auc1_std, f11, f11_std, sens1, esp1, imp1 = evaluar_modelo(features_base)

print("\nENFOQUE 1 - BASE")
print(f"Accuracy: {acc1:} ± {acc1_std:.3f}")
print(f"AUC:      {auc1:} ± {auc1_std:.3f}")
print(f"F1:       {f11:} ± {f11_std:.3f}")
print(f"Sensibilidad: {sens1:.4f}")
print(f"Especificidad: {esp1:.4f}")

# =========================
# MODELO EXTENDIDO
# =========================
features_ext = features_base + features_extra

acc2, acc2_std, auc2, auc2_std, f12, f12_std, sens2, esp2, imp2 = evaluar_modelo(features_ext)

print("\nENFOQUE 1 - BASE + ANGULO + VELOCIDAD")
print(f"Accuracy: {acc2:} ± {acc2_std:.3f}")
print(f"AUC:      {auc2:} ± {auc2_std:.3f}")
print(f"F1:       {f12:} ± {f12_std:.3f}")
print(f"Sensibilidad: {sens2:.4f}")
print(f"Especificidad: {esp2:.4f}")


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

print("\nIMPORTANCIAS EXTENDIDO:")
print(df_imp_ext)



