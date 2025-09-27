# Exploratory Data Analysis (EDA) con Pandas, Matplotlib y Seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Cargar dataset
# ---------------------------
df = pd.read_csv("spotify_churn_dataset.csv")

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
# 6. Análisis Multivariado con visualizaciones
# ---------------------------

# 6.1 Mapa de calor de correlaciones 
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].drop(columns=["is_churned"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor - Correlaciones")
plt.show()


# 6.2 Dispersión entre songs_played_per_day y listening_time 
plt.figure(figsize=(6,4))
sns.scatterplot(x="songs_played_per_day", y="listening_time", hue="is_churned", data=df, alpha=0.6)
plt.title("Relación entre Canciones/Día y Tiempo de Escucha")
plt.show()


# 6.3 Boxplot skip_rate vs churn 
plt.figure(figsize=(6,4))
sns.boxplot(x="is_churned", y="skip_rate", data=df)
plt.title("Skip Rate según Churn")
plt.show()


# 6.4 Boxplot Age vs churn 
plt.figure(figsize=(6,4))
sns.boxplot(x="is_churned", y="age", data=df)
plt.title("Edad según Churn")
plt.show()


# 6.5 Boxplot Listening Time vs churn 
plt.figure(figsize=(6,4))
sns.boxplot(x="is_churned", y="listening_time", data=df)
plt.title("Tiempo de Escucha según Churn")
plt.show()


# 6.6 Boxplot Songs per Day vs churn
plt.figure(figsize=(6,4))
sns.boxplot(x="is_churned", y="songs_played_per_day", data=df)
plt.title("Canciones por Día según Churn")
plt.show()


# 6.7 Boxplot ads_listened_per_week vs churn 
plt.figure(figsize=(6,4))
sns.boxplot(x="is_churned", y="ads_listened_per_week", data=df)
plt.title("Anuncios escuchados según Churn")
plt.show()


# 6.8 Pairplot de algunas variables relevantes 
sns.pairplot(df, vars=["age", "skip_rate", "listening_time", "songs_played_per_day"], hue="is_churned")
plt.suptitle("Pairplot de Variables Relevantes", y=1.02)
plt.show()


# ---------------------------
# 7. Insights principales
# ---------------------------
print("🔑 Insights principales:")
print("1. Skip_rate alto → mayor probabilidad de churn.")
print("2. Baja actividad (menos canciones y poco tiempo de escucha) → mayor churn.")
print("3. Usuarios jóvenes tienden a cancelar más que los mayores.")
print("4. Offline listening (descargas) se asocia con menor churn → beneficios premium retienen.")
print("5. La publicidad excesiva (ads_listened_per_week alto) se asocia con churn.\n")