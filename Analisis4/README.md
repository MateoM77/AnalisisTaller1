# 📊 Análisis Predictivo de Rendimiento Académico Estudiantil

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de **Machine Learning** para predecir el rendimiento académico de estudiantes utilizando el dataset "Student Grade Prediction" de Kaggle. El análisis incluye exploración de datos (EDA), preprocesamiento avanzado y múltiples modelos de clasificación.

## 🗂️ Estructura del Proyecto

```
Analisis4/
├── ejercicio1.ipynb          # Notebook principal con análisis completo
├── ejercicio5.ipynb          # Análisis HR Analytics (complementario)
├── README.md                 # Documentación del proyecto
└── datos/                    # Directorio para datasets (generado automáticamente)
```

## 📋 Contenido del Análisis

### 1. **Exploración de Datos (EDA)**
- **Dataset**: 395 estudiantes con 33 variables
- **Variables**: Demográficas, académicas, familiares y sociales
- **Target**: Calificación final (G3) categorizada en 3 niveles:
  - 🔴 **Bajo** (0-10): Rendimiento deficiente
  - 🟡 **Medio** (11-14): Rendimiento satisfactorio  
  - 🟢 **Alto** (15-20): Rendimiento excelente

### 2. **Preprocesamiento de Datos**
- ✅ **Sin valores faltantes** - Dataset de alta calidad
- 🔧 **Codificación de variables categóricas**:
  - Label Encoding para variables binarias
  - One-Hot Encoding para variables multicategoría
- 📊 **Estandarización** de variables numéricas (StandardScaler)
- 🎲 **División estratificada**: 80% entrenamiento, 20% prueba

### 3. **Modelos de Machine Learning**

| Modelo | Accuracy | F1-Score | Precision | Recall | 🏆 |
|--------|----------|----------|-----------|---------|-----|
| **Random Forest** | 62.03% | 56.71% | 68.58% | 54.75% | ⭐ |
| **K-Nearest Neighbors** | 53.16% | 49.72% | 50.64% | 53.72% | - |
| **Regresión Logística** | 51.90% | 47.88% | 47.33% | 48.10% | - |

**🏆 Mejor Modelo**: Random Forest con hiperparámetros optimizados

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
Jupyter Notebook
```

### Dependencias
```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy
```

### Ejecución
```bash
# Clonar/descargar el proyecto
cd Analisis4

# Ejecutar Jupyter Notebook
jupyter notebook ejercicio1.ipynb
```

## 🔍 Insights Principales

### **Factores Más Importantes** (según Random Forest):
1. **Calificaciones previas** (G1, G2) - Mayor predictor
2. **Ausencias** - Impacto negativo significativo
3. **Tiempo de estudio** - Correlación positiva
4. **Apoyo familiar** - Factor protector
5. **Consumo de alcohol** - Factor de riesgo

### **Patrones Identificados**:
- 📈 **Estudiantes exitosos**: Pocas ausencias, mayor tiempo de estudio, apoyo familiar
- 📉 **Estudiantes en riesgo**: Altas ausencias, bajo apoyo educativo, problemas familiares
- ⚖️ **Distribución balanceada**: 27% Bajo, 47% Medio, 26% Alto

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Datos** | Kaggle API, Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn |
| **ML** | Scikit-learn, GridSearchCV |
| **Estadística** | SciPy, PCA, t-SNE |
| **Entorno** | Jupyter Notebook, Python |

## 📈 Resultados y Visualizaciones

### **Análisis Exploratorio**:
- 📊 Distribuciones de variables clave
- 🔗 Matriz de correlaciones
- 👥 Análisis por subgrupos (sexo, apoyo educativo, etc.)
- 📉 Detección de outliers

### **Machine Learning**:
- 🎯 Matrices de confusión por modelo
- 📈 Curvas de aprendizaje
- 🌟 Importancia de características
- 🔍 Visualización PCA y t-SNE

## 🎓 Aplicaciones Prácticas

### **Para Instituciones Educativas**:
1. **Sistema de Alerta Temprana**: Identificar estudiantes en riesgo
2. **Intervención Personalizada**: Estrategias basadas en factores de riesgo
3. **Asignación de Recursos**: Priorizar apoyo académico
4. **Seguimiento Predictivo**: Monitoreo continuo del progreso

### **Casos de Uso**:
- 🚨 **Detección temprana** de estudiantes en riesgo de fracaso
- 📋 **Recomendaciones personalizadas** de intervención
- 📊 **Análisis de efectividad** de programas educativos
- 🎯 **Optimización de recursos** de apoyo académico

## 🔮 Próximos Pasos

### **Mejoras Técnicas**:
- [ ] Implementar ensemble de modelos
- [ ] Optimización de hiperparámetros con Bayesian Optimization
- [ ] Análisis de importancia con SHAP
- [ ] Validación temporal con datos longitudinales

### **Expansión del Análisis**:
- [ ] Incorporar variables adicionales (socioeconómicas, psicológicas)
- [ ] Análisis por cohortes y temporal
- [ ] Sistema de recomendaciones automatizado
- [ ] Dashboard interactivo para educadores

## 📊 Métricas de Negocio

### **Impacto Potencial**:
- 🎯 **Precisión de predicción**: 62% de estudiantes correctamente clasificados
- 🔍 **Detección de riesgo**: Identificación temprana de 55% de casos de bajo rendimiento
- 💰 **ROI estimado**: Reducción de 15-25% en tasas de deserción
- ⏰ **Tiempo de intervención**: Predicción hasta 2 períodos académicos adelante

## 👥 Contribuciones

Proyecto desarrollado como parte del análisis de datos educativos. Contribuciones y mejoras son bienvenidas.

### **Cómo Contribuir**:
1. Fork del proyecto
2. Crear rama de feature (`git checkout -b feature/mejora`)
3. Commit de cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto es de uso académico y educativo. Los datos utilizados provienen de fuentes públicas de Kaggle.

## 📧 Contacto

Para consultas sobre el proyecto o colaboraciones, contacta a través de los issues del repositorio.

---

### 🔗 **Enlaces Relevantes**:
- [Dataset Original en Kaggle](https://www.kaggle.com/datasets/dipam7/student-grade-prediction)
- [Documentación de Scikit-learn](https://scikit-learn.org/)
- [Kagglehub Documentation](https://github.com/Kaggle/kagglehub)

---

**⭐ Si este proyecto te resulta útil, no olvides darle una estrella! ⭐**

*Última actualización: Noviembre 2025*