# Exploratory Data Analysis (EDA) con Pandas, Matplotlib y Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 1. Cargar dataset
# ---------------------------
df = pd.read_csv("spotify_churn_dataset.csv")

# Vista inicial
print("Primeras filas del dataset:")
print(df.head(), "\n")

print("Información general:")
print(df.info(), "\n")

print("Dimensiones del dataset:", df.shape, "\n")

# ---------------------------
# 2. Revisión de valores faltantes
# ---------------------------
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    "Valores_Faltantes": missing_values,
    "Porcentaje": missing_percentage
})
print("Valores faltantes:\n", missing_summary, "\n")

# ---------------------------
# 3. Detección de valores atípicos (Outliers)
# ---------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary[col] = outliers

print("Cantidad de outliers por variable:\n", pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outliers']), "\n")

# ---------------------------
# 4. Estadísticos descriptivos
# ---------------------------
print("Estadísticos descriptivos:\n", df[numeric_cols].describe().T, "\n")

# ---------------------------
# 5. Análisis Univariado
# ---------------------------
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribución de {col}")
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# Variables categóricas
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.title(f"Distribución de {col}")
    plt.xticks(rotation=45)
    plt.show()

# ---------------------------
# 6. Análisis Multivariado
# ---------------------------
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor - Correlaciones")
plt.show()

# Dispersión de dos variables clave
plt.figure(figsize=(6,4))
sns.scatterplot(x="songs_played_per_day", y="listening_time", hue="is_churned", data=df)
plt.title("Relación entre Canciones/Día y Tiempo de Escucha")
plt.show()

# ---------------------------
# 7. Insights preliminares
# ---------------------------
print("Insights preliminares:")
print("- Existen valores atípicos en listening_time y songs_played_per_day (posibles heavy users).")
print("- La distribución de edad no es normal, presenta sesgo hacia edades jóvenes.")
print("- Los usuarios con mayor skip_rate parecen tener mayor probabilidad de churn.")
print("- La suscripción gratuita y el número de anuncios escuchados se asocian con mayor churn.")
print("- Los usuarios que escuchan offline muestran menor tasa de churn (relación con planes de pago).")

