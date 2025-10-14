# Evaluación y Validación de Modelos de Machine Learning
# Aplicación práctica Dataset 2: Credit Card Default - UCI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


# 1.1 Se Carga el dataset desde archivo Excel (.xls)

df_credit = pd.read_excel("Dataset 2 Credit Card Default - UCI.xls", header=1, engine='xlrd')
print("Archivo .xls cargado correctamente")

print("="*70)
print("PRIMERAS OBSERVACIONES DEL DATASET")
print("="*70)
print(df_credit.head())

print(f"\nDimensiones del dataset: {df_credit.shape[0]} filas x {df_credit.shape[1]} columnas")

print("\nTipos de datos por columna:")
df_credit.info()

# Se Muestran los nombres de todas las columnas

print("\n" + "="*70)
print("NOMBRES DE COLUMNAS EN EL DATASET:")
print("="*70)
for i, col in enumerate(df_credit.columns):
    print(f"{i:2d}. {col}")
print("="*70)


# 2.2 Se hace Exploración y Limpieza de los datos

print("\n" + "="*70)
print("REVISIÓN DE CALIDAD DE DATOS")
print("="*70)
print("\nValores faltantes por columna:")
nulos = df_credit.isna().sum()
if nulos.sum() > 0:
    print(nulos[nulos > 0].sort_values(ascending=False))
else:
    print("No hay valores faltantes")

duplicados = df_credit.duplicated().sum()
print(f"\n Registros duplicados encontrados: {duplicados}")
if duplicados > 0:
    df_credit = df_credit.drop_duplicates()
    print(f"\n Duplicados eliminados. Nueva forma: {df_credit.shape}")

# Se Identifica variable objetivo

candidatas = [col for col in df_credit.columns if 'default' in col.lower()]
if len(candidatas) > 0:
    target_col = candidatas[0]
    print(f"\n Variable objetivo identificada: '{target_col}'")
else:
    target_col = df_credit.columns[-1]
    print(f"\n No se encontró columna con 'default'. Usando la ultima: '{target_col}'")

print(f"\n Variable objetivo seleccionada: '{target_col}'")
print(f"\nDistribución de casos:")
print(df_credit[target_col].value_counts())
print(f"\nTasa de default en el dataset: {df_credit[target_col].mean():.2%}")

print("\nEstadísticas descriptivas de variables numéricas:")
print(df_credit.describe())

## 2.3 Se preparan y se transforman las variables

print("\n" + "="*70)
print("PREPROCESAMIENTO DE DATOS")
print("="*70)

df_procesado = df_credit.copy()

vars_categoricas_posibles = ['SEX', 'EDUCATION', 'MARRIAGE']
vars_categoricas = [col for col in vars_categoricas_posibles if col in df_procesado.columns]

print(f"\nVariables categóricas encontradas: {vars_categoricas}")
if len(vars_categoricas) > 0:
    df_procesado = pd.get_dummies(df_procesado, columns=vars_categoricas, drop_first=True)
    print(f" One-hot encoding aplicado a: {vars_categoricas}")

col_id = None
for col in df_procesado.columns:
    if col.upper() == 'ID':
        col_id = col
        break

print(f"\n Columna ID identificada: {col_id if col_id else 'No encontrada'}")

vars_numericas = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
vars_a_excluir = [target_col]
if col_id:
    vars_a_excluir.append(col_id)
vars_numericas = [col for col in vars_numericas if col not in vars_a_excluir]

print(f"\n Variables numéricas a normalizar: {len(vars_numericas)}")
normalizador = MinMaxScaler()
df_procesado[vars_numericas] = normalizador.fit_transform(df_procesado[vars_numericas])
print(" Normalización aplicada (Min-Max Scaler)")

print(f"\nTotal de variables después del preprocesamiento: {len(df_procesado.columns)}")


# 4. Se realiza separación en conjuntos de entrenamiento y prueba

print("\n" + "="*70)
print("DIVISIÓN DEL DATASET")
print("="*70)

# Definimos variables predictoras (X) y variable objetivo (y)

columnas_excluir = [target_col]
if col_id:
    columnas_excluir.append(col_id)

X = df_procesado.drop(columns=columnas_excluir)
y = df_procesado[target_col]

print(f"\n Forma de X: {X.shape}")
print(f" Forma de y: {y.shape}")

# hacemos división estratificada 80% entrenamiento - 20% prueba

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y, 
    random_state=42
)

print(f"\n División completada:")
print(f"   Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X):.1%})")
print(f"   Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X):.1%})")

print(f"\n Distribución de clases:")
print(f"   Train - Clase 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train):.1%})")
print(f"   Train - Clase 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train):.1%})")
print(f"   Test  - Clase 0: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test):.1%})")
print(f"   Test  - Clase 1: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test):.1%})")


