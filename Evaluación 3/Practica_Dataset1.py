# Evaluación y Validación de Modelos de Machine Learning
# Aplicación práctica Dataset 1: Telco Customer Churn - Kaggle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# 1.1 Carga del dataset  
df_churn = pd.read_csv("Dataset 1 Telco Customer Churn - Kaggle.csv")  
df_churn.head(), df_churn.shape, df_churn.info()

# ### 1.2 Exploración y limpieza inicial  
# - Revisar valores nulos  
# - Revisar duplicados  
# - Transformar algunas columnas (por ejemplo, `TotalCharges` que viene como string)  
# - Observaciones generales: proporción de churn, distribución de variables


# Convertir TotalCharges a numérico (coercing errores)  
df_churn["TotalCharges"] = pd.to_numeric(df_churn["TotalCharges"], errors="coerce")  
# Contar nulos  
print("Valores nulos por columna:\n", df_churn.isna().sum().sort_values(ascending=False).head(10))  

# Imputar con la mediana los valores faltantes en TotalCharges  
median_total = df_churn["TotalCharges"].median()  
df_churn["TotalCharges"].fillna(median_total, inplace=True)

# Revisar duplicados  
print("Duplicados:", df_churn.duplicated().sum())

# ### 1.3 Codificación / transformación de variables categóricas  
# - Variables binarias “Yes/No” → 1 / 0  
# - Variables categóricas con más de dos niveles → one-hot encoding  
# - Escalamiento de variables numéricas


# Mapear columnas binarias  
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]  
for c in binary_cols:
    df_churn[c] = df_churn[c].map({"Yes": 1, "No": 0})

# Algunas columnas tienen “No internet service” o “No phone service” — podemos reemplazar esas cadenas por “No”  
# (esto simplifica el encoding).  
cols_replace = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]  
for c in cols_replace:
    df_churn[c] = df_churn[c].replace({"No internet service": "No", "No phone service": "No"})

# Ahora aplica one-hot encoding a variables categóricas restantes  
cat_cols = [c for c in df_churn.columns if df_churn[c].dtype == "object" and c not in ["customerID"]]
df_churn_enc = pd.get_dummies(df_churn, columns=cat_cols, drop_first=True)

# Escalamiento de variables numéricas  
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = MinMaxScaler()
df_churn_enc[num_cols] = scaler.fit_transform(df_churn_enc[num_cols])


# ### 1.4 División en entrenamiento y prueba  


X = df_churn_enc.drop(columns=["customerID", "Churn"])  
y = df_churn_enc["Churn"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

print("Tamaño train:", X_train.shape, "Tamaño test:", X_test.shape)
print("Proporciones de clase en test:", np.bincount(y_test) / len(y_test))


# ### 1.5 Validación cruzada + selección de modelo  
# Probemos dos modelos: regresión logística y Random Forest.


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Regresión logística
param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr,
                       cv=cv, scoring="roc_auc", n_jobs=-1)
grid_lr.fit(X_train, y_train)
print("Mejor C para LR:", grid_lr.best_params_, "Mejor AUC (CV):", grid_lr.best_score_)

# Random Forest
param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf,
                       cv=cv, scoring="roc_auc", n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Mejores params RF:", grid_rf.best_params_, "Mejor AUC (CV):", grid_rf.best_score_)


# ### 1.6 Evaluación en el conjunto de prueba  

 
def evaluar_modelo(model, X_t, y_t, nombre="Modelo"):
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:,1]
    print(f"--- Evaluación {nombre} ---")
    print("Accuracy:", accuracy_score(y_t, y_pred))
    print("Precision:", precision_score(y_t, y_pred))
    print("Recall:", recall_score(y_t, y_pred))
    print("F1:", f1_score(y_t, y_pred))
    print("ROC AUC:", roc_auc_score(y_t, y_proba))
    print("Matriz de confusión:\n", confusion_matrix(y_t, y_pred))
    print(classification_report(y_t, y_pred))
    # Gráficas ROC y PR
    RocCurveDisplay.from_estimator(model, X_t, y_t)
    plt.title(f"ROC Curve — {nombre}")
    plt.show()
    PrecisionRecallDisplay.from_estimator(model, X_t, y_t)
    plt.title(f"Precision-Recall Curve — {nombre}")
    plt.show()

# Evaluar los modelos optimizados  
best_lr = grid_lr.best_estimator_
best_rf = grid_rf.best_estimator_

evaluar_modelo(best_lr, X_test, y_test, "Logistic Regression")
evaluar_modelo(best_rf, X_test, y_test, "Random Forest")


# ### 1.7 Diagnóstico de overfitting / underfitting  
# - Comparar métricas en entrenamiento vs validación/prueba  
# - Curvas de aprendizaje


# Ver desempeño en los sets de entrenamiento (usando validación interna)  
print("LR rendimiento en entrenamiento:", roc_auc_score(y_train, best_lr.predict_proba(X_train)[:,1]))
print("RF rendimiento en entrenamiento:", roc_auc_score(y_train, best_rf.predict_proba(X_train)[:,1]))

# Curva de aprendizaje (pipe para, por ejemplo, RandomForest)
train_sizes, train_scores, val_scores = learning_curve(best_rf, X_train, y_train,
                                                       cv=cv, scoring="roc_auc",
                                                       train_sizes=[0.1,0.3,0.5,0.7,1.0], n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label="Train AUC")
plt.plot(train_sizes, val_scores_mean, label="CV AUC")
plt.xlabel("Tamaño del set de entrenamiento")
plt.ylabel("ROC AUC")
plt.title("Curva de aprendizaje – Random Forest")
plt.legend()
plt.show()


# ### 1.8 Interpretabilidad / importancia de variables  

  
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_feats = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_feats.values, y=top_feats.index)
plt.title("Top 10 features (importancia) — Random Forest")
plt.show()

# También puedes ver coeficientes de la regresión logística  
coefs = pd.Series(best_lr.coef_[0], index=X.columns).sort_values()
print("Coeficientes LR (ordenados):\n", coefs.head(10), coefs.tail(10))