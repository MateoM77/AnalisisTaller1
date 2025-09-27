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

# Analisis de que variables contribuyen mas a cada componente
# Tomamos las primeras 3 componentes para analizar

contribuciones = pca_final.components_[:3].T
df_contribuciones = pd.DataFrame(
    contribuciones,
    columns=['PC1', 'PC2', 'PC3'],
    index=datos.columns
)

print("Variables con mayor contribucion al Componente 1:")
print(df_contribuciones['PC1'].abs().sort_values(ascending=False).head(5))

# Mapa de Calor de contribuciones

plt.figure(figsize=(11, 7))
sns.heatmap(df_contribuciones.T, annot=True, cmap='RdBu_r', 
            center=0, fmt='.2f', linewidths=0.5)
plt.title('Contribucion de Variables a Componentes Principales')
plt.xlabel('Variables Originales')
plt.ylabel('Componentes')
plt.tight_layout()
plt.show()

# ---------------------------
# Se Guardan Resultados
# ---------------------------
print("\nGuardando archivos procesados...")

# Se Guardan los datos escalados
datos_escalados.to_csv("spotify_datos_escalados.csv", index=False)

# Se Guardan los datos con PCA aplicado

df_pca = pd.DataFrame(datos_reducidos, 
                      columns=[f'PC{i+1}' for i in range(componentes_necesarios)])
df_pca.to_csv("spotify_datos_pca.csv", index=False)

print("Archivos guardados:")
print("  - spotify_datos_escalados.csv")
print("  - spotify_datos_pca.csv")


# ---------------------------
# Resumen final
# ---------------------------
print("\n" + "="*60)
print("RESUMEN DEL PREPROCESAMIENTO")
print("="*60)
print(f"Variables categoricas codificadas: {len(categoricas)}")
print(f"Metodo de escalado: StandardScaler")
print(f"Reduccion dimensional: {len(datos.columns)} -> {componentes_necesarios} componentes")
print(f"Varianza retenida: {varianza_acumulada[componentes_necesarios-1]*100:.2f}%")
print(f"Los datos estan listos para modelado")
print("="*60)