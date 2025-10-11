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

# Balancear clases:
from imblearn.over_sampling import SMOTE

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