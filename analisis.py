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

#----------------------------
# FASE 3 PREPROCESAMIENTO Y REDUCCIÓN 
#----------------------------

# Hago una copia de la data para no modificar el original
datos = df.copy()

# ---------------------------
# Paso 1: Codificar variables categoricas
# ---------------------------
# Primero identifico cuales son categoricas
categoricas = datos.select_dtypes(include=["object"]).columns.tolist()
print(f"\nVariables categoricas encontradas: {categoricas}")
print(f"Total: {len(categoricas)} variables\n")

# ---------------------------
# Paso 2: Las convierto a numeros con LabelEncoder
# ---------------------------

encoders_guardados = {}

for columna in categoricas:
    print(f"Codificando: {columna}")
    valores_unicos = datos[columna].unique()
    print(f"  Valores: {valores_unicos}")
    
    encoder = LabelEncoder()
    datos[columna] = encoder.fit_transform(datos[columna])
    encoders_guardados[columna] = encoder

    # Muestro como quedo la codificacion

    print(f"  Codificacion: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}\n")

print("Datos despues de codificar:")
print(datos.head())
print(f"\nVerificando tipos de datos:\n{datos.dtypes}\n")

# Se verifica si hay valores nulos antes de continuar  

print("Valores nulos por columna:")
print(datos.isnull().sum())
if datos.isnull().sum().sum() > 0:
    print("\nRellenando valores nulos con la mediana...")
    datos = datos.fillna(datos.median())
else:
    print("No hay valores nulos, podemos continuar\n")
    
    # ---------------------------
# Paso 3: Escalar los datos
# ---------------------------
# Uso StandardScaler para PCA
print("Aplicando StandardScaler a todas las variables...")

escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datos)

# Convierto de nuevo a DataFrame para facilitar el trabajo

datos_escalados = pd.DataFrame(datos_escalados, columns=datos.columns)

print("\nEstadisticas despues del escalado:")
print(datos_escalados.describe())

# Visualizo el efecto del escalado en una variable

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Antes del escalado
datos.iloc[:, 2].hist(bins=25, ax=ax1, color='Purple', edgecolor='black')
ax1.set_title(f'Antes del escalado\n{datos.columns[2]}')
ax1.set_ylabel('Frecuencia')

# Despues del escalado
datos_escalados.iloc[:, 2].hist(bins=25, ax=ax2, color='green', edgecolor='black')
ax2.set_title(f'Despues del escalado\n{datos.columns[2]}')
ax2.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
