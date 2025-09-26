# Exploratory Data Analysis (EDA) con Pandas, Matplotlib y Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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

# ---------------------------
# Paso 4: Se Aplica PCA
# ---------------------------
print("\n--- Analisis de Componentes Principales (PCA) ---\n")

# Se aplica PCA completo para ver cuanta varianza explica cada componente

pca_completo = PCA()
pca_completo.fit(datos_escalados)

varianza_explicada = pca_completo.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# Resultados

print("Varianza explicada por cada componente:")
for i in range(min(8, len(varianza_explicada))):
    print(f"  Componente {i+1}: {varianza_explicada[i]*100:.2f}%  |  Acumulado: {varianza_acumulada[i]*100:.2f}%")

# Se Calculan cuantos componentes son necesarios para explicar el 95% de la varianza

componentes_necesarios = np.argmax(varianza_acumulada >= 0.95) + 1
print(f"\nComponentes necesarios para 95% de varianza: {componentes_necesarios}")
print(f"Reduccion: de {len(datos.columns)} variables a {componentes_necesarios} componentes\n")

# Grafico de varianza 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Grafico 1: barras por componente

ax1.bar(range(1, len(varianza_explicada)+1), varianza_explicada*100, 
        color='teal', alpha=0.7)
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Varianza Explicada (%)')
ax1.set_title('Varianza por Componente')
ax1.grid(axis='y', alpha=0.4)

# Grafico 2: linea acumulada

ax2.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada*100, 
         marker='o', linewidth=2, color='darkred')
ax2.axhline(y=95, color='pink', linestyle='--', linewidth=1.5, label='Umbral 95%')
ax2.axvline(x=componentes_necesarios, color='blue', linestyle='--', 
            linewidth=1.5, label=f'{componentes_necesarios} componentes')
ax2.set_xlabel('Numero de Componentes')
ax2.set_ylabel('Varianza Acumulada (%)')
ax2.set_title('Varianza Explicada Acumulada')
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# Se aplica PCA con el numero optimo de componentes

pca_final = PCA(n_components=componentes_necesarios)
datos_reducidos = pca_final.fit_transform(datos_escalados)

print(f"Forma original: {datos_escalados.shape}")
print(f"Forma reducida: {datos_reducidos.shape}")
print(f"Varianza total retenida: {varianza_acumulada[componentes_necesarios-1]*100:.2f}%\n")

# ---------------------------
# Paso 5: Visualizaciones EN 2D Y 3D
# ---------------------------

print("Visualizaciones...\n")

# Visualizacion en 2D primeras 2 componentes

pca_2d = PCA(n_components=2)
componentes_2d = pca_2d.fit_transform(datos_escalados)

plt.figure(figsize=(10, 7))
if 'is_churned' in df.columns:
    colores = df['is_churned']
    scatter = plt.scatter(componentes_2d[:, 0], componentes_2d[:, 1], 
                         c=colores, cmap='plasma', alpha=0.6, s=40)
    plt.colorbar(scatter, label='Churn (0=No, 1=Si)')
else:
    plt.scatter(componentes_2d[:, 0], componentes_2d[:, 1], alpha=0.6, s=40)

plt.xlabel(f'Componente 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'Componente 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title('Proyeccion PCA en 2 Dimensiones')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualizacion en 3D primeras 3 componentes

pca_3d = PCA(n_components=3)
componentes_3d = pca_3d.fit_transform(datos_escalados)

figura = plt.figure(figsize=(10, 7))
ax = figura.add_subplot(111, projection='3d')

if 'is_churned' in df.columns:
    colores = df['is_churned']
    scatter = ax.scatter(componentes_3d[:, 0], componentes_3d[:, 1], componentes_3d[:, 2],
                        c=colores, cmap='viridis', alpha=0.5, s=35)
    figura.colorbar(scatter, ax=ax, label='Churn', pad=0.1)
else:
    ax.scatter(componentes_3d[:, 0], componentes_3d[:, 1], componentes_3d[:, 2], 
              alpha=0.5, s=35)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Proyeccion PCA en 3 Dimensiones')
plt.tight_layout()
plt.show()

