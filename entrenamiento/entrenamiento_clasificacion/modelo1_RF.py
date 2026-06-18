import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
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

    y_true_all = []
    y_pred_all = []

    print("N muestras:", len(X))
    print("Distribución clases:\n", y.value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aucs, f1s = [], [], []
    importancias_folds = []

    for train_idx, test_idx in skf.split(X, y):

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
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))

        importancias_folds.append(model.feature_importances_)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    importancias_media = np.mean(importancias_folds, axis=0)

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
acc1, auc1, f11, sens1, esp1, imp1 = evaluar_modelo(features_base)

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

acc2, auc2, f12, sens2, esp2, imp2 = evaluar_modelo(features_ext)

print("\nENFOQUE 1 - BASE + ANGULO + VELOCIDAD")
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

print("\nIMPORTANCIAS EXTENDIDO:")
print(df_imp_ext)




