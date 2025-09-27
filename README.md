# Proyecto de Análisis de Datos – Spotify Churn

## 📌 Descripción

El objetivo fue explorar distintas bases de datos, seleccionar la más adecuada, realizar un **Análisis Exploratorio de Datos (EDA)** y aplicar técnicas de **preprocesamiento y reducción de dimensionalidad (PCA)**.

---

## 🚀 Instrucciones de Ejecución

### 1. Requisitos previos

Asegúrate de tener instalado:

* Python 3.8 o superior

### 2. Instalación de librerías necesarias

Ejecuta en tu entorno de Python:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Ejecución de los notebooks

1. Clona este repositorio:

   ```bash
   git clone https://github.com/MateoM77/AnalisisTaller1.git
   ```
2. Abre los notebooks:

   * `Fase2_EDA.py`
   * `Fase3_Preprocesamiento.py`

### 4. Archivos generados

Al finalizar la ejecución, se crean automáticamente:

* `spotify_datos_escalados.csv`
* `spotify_datos_pca.csv`

---

## 🗂️ Fase 1 – Exploración de Bases de Datos

Se exploraron tres bases de datos de diferentes tipos:

1. **Conjunto de Datos de Señales de Tránsito** – tipo imágenes.

   * Datos visuales de señales viales.
   * Aplicación: clasificación de imágenes en el contexto de seguridad vial.

2. **Dataset Iris** – tipo tabular.

   * Dataset clásico con características de flores.
   * Aplicación: pruebas rápidas de clasificación y reducción de dimensionalidad.

3. **Spotify Churn Dataset** – tipo texto/tabular.

   * Datos de usuarios de Spotify con información de uso y cancelación de suscripción.
   * Aplicación: análisis de churn (abandono) de usuarios.

### ✅ Justificación de la elección de Spotify Churn

Se seleccionó esta base de datos porque:

* **Tiene muchas variables numéricas y categóricas**, lo que permite aplicar un análisis exploratorio completo.
* Presenta un **problema real de negocio (churn)** altamente relevante en la industria de streaming.
* Ofrece **posibilidades de hipótesis interesantes**, como la relación entre el tiempo de escucha, la tasa de skips o el tipo de suscripción con la probabilidad de abandono.
* Su tamaño es **manejable para aplicar EDA, preprocesamiento y PCA** sin limitaciones técnicas.

---

## 🔎 Fase 2 – Análisis Exploratorio de Datos (EDA)

Con el dataset de Spotify se realizaron los siguientes pasos:

* **Revisión de valores faltantes**: identificación y cuantificación de nulos.
* **Detección de outliers**: mediante IQR y boxplots.
* **Análisis univariado**: distribución de variables numéricas y categóricas.
* **Análisis multivariado**: correlaciones, scatterplots y mapas de calor.
* **Insights preliminares**:

  * Los usuarios con mayor *skip rate* presentan mayor probabilidad de churn.
  * La suscripción gratuita y escuchar anuncios se asocian con mayor churn.
  * Escuchar música offline se relaciona con menor tasa de churn.

---

## ⚙️ Fase 3 – Preprocesamiento y Reducción de Dimensionalidad

1. **Codificación de variables categóricas** con LabelEncoder.
2. **Escalado de datos** con StandardScaler.
3. **Reducción de dimensionalidad con PCA**:

   * Se redujo el dataset de *n* variables originales a un número óptimo de componentes principales.
   * Se retuvo más del **95% de la varianza**.
4. **Visualizaciones** en 2D y 3D para observar la separación de usuarios churn y no churn.
5. **Contribución de variables** a los componentes principales mediante un heatmap.

---

## 📂 Entregables en el Repositorio

* `Fase1_BasesdeDatos.pdf` → Exploración bases de datos de diferente tipo.
* `Fase2_EDA.py` → Exploración y análisis de datos.
* `Fase3_Preprocesamiento.py` → Codificación, escalado y PCA.
* `spotify_churn_dataset.csv` → Base de datos Spotify churn
* `spotify_datos_escalados.csv` → Datos procesados y escalados.
* `spotify_datos_pca.csv` → Datos reducidos con PCA.
* `README.md` → Documento descriptivo del proyecto.

---

## 🎥 Video Explicativo

Se incluye un video donde se presentan:

* Las bases exploradas y justificación de la elegida.
* Proceso de EDA y preprocesamiento.
* Resultados principales e insights.

---

## ✅ Conclusiones

* La base de datos de **Spotify Churn** permitió un análisis más completo y aplicable al mundo real.
* El EDA permitió identificar variables críticas asociadas con el abandono de usuarios.
* El preprocesamiento y PCA facilitaron la reducción de complejidad conservando la varianza.
* Los datos quedaron listos para fases futuras de **modelado predictivo**.
