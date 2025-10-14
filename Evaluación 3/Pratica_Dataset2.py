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


## 2.1 Se Carga el dataset desde archivo Excel (.xls)

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


## 2.2 Se hace Exploración y Limpieza de los datos

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


## 2.4 Se realiza separación en conjuntos de entrenamiento y prueba

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

## 2.5 Entrenamiento y optimización de Modelos

print("\n" + "="*70)
print("BUSQUEDA DE HIPERPARAMETROS OPTIMOS")
print("="*70)
print(" Este proceso puede tardar varios minutos...\n")

# Configuramos la validación cruzada estratificada con 3 particiones

validacion_cruzada = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Modelo 1: Regresión Logística

print(" Entrenando Regresion Logistica con GridSearchCV...")
parametros_lr = {"C": [0.01, 0.1, 1, 10, 100]}

busqueda_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    parametros_lr,
    cv=validacion_cruzada,
    scoring="roc_auc",
    n_jobs=-1
)
busqueda_lr.fit(X_train, y_train)

print(f"   Mejor parametro C: {busqueda_lr.best_params_['C']}")
print(f"  ROC-AUC en validacion cruzada: {busqueda_lr.best_score_:.4f}")

# Modelo 2: Random Forest

print("\n Entrenando Random Forest con GridSearchCV...")
parametros_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 15]
}

busqueda_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    parametros_rf,
    cv=validacion_cruzada,
    scoring="roc_auc",
    n_jobs=-1
)
busqueda_rf.fit(X_train, y_train)

print(f"  Mejores parametros: {busqueda_rf.best_params_}")
print(f"  ROC-AUC en validacion cruzada: {busqueda_rf.best_score_:.4f}")

# Modelo 3: Support Vector Machine

print("\n Entrenando SVM con GridSearchCV...")
parametros_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"]
}

busqueda_svm = GridSearchCV(
    SVC(probability=True, random_state=42),
    parametros_svm,
    cv=validacion_cruzada,
    scoring="roc_auc",
    n_jobs=-1
)
busqueda_svm.fit(X_train, y_train)

print(f"  Mejores parámetros: {busqueda_svm.best_params_}")
print(f"  ROC-AUC en validación cruzada: {busqueda_svm.best_score_:.4f}")

## 2.6. EVALUACIÓN DE RENDIMIENTO EN CONJUNTO DE PRUEBA

def calcular_metricas(modelo, X_eval, y_eval, nombre_modelo="Modelo"):
    """
    Calcula y visualiza métricas de desempeño del modelo
    """
    # Se Generan predicciones
    predicciones = modelo.predict(X_eval)
    probabilidades = modelo.predict_proba(X_eval)[:, 1]
    
    # Se Calculan métricas
    print(f"\n{'='*70}")
    print(f"RESULTADOS - {nombre_modelo}")
    print(f"{'='*70}")
    print(f"Exactitud (Accuracy):  {accuracy_score(y_eval, predicciones):.4f}")
    print(f"Precisión (Precision): {precision_score(y_eval, predicciones):.4f}")
    print(f"Sensibilidad (Recall): {recall_score(y_eval, predicciones):.4f}")
    print(f"F1-Score:              {f1_score(y_eval, predicciones):.4f}")
    print(f"ROC-AUC:               {roc_auc_score(y_eval, probabilidades):.4f}")
    
    # Matriz de confusión
    print("\nMatriz de Confusión:")
    matriz_confusion = confusion_matrix(y_eval, predicciones)
    print(f"   TN: {matriz_confusion[0,0]:5d}  |  FP: {matriz_confusion[0,1]:5d}")
    print(f"   FN: {matriz_confusion[1,0]:5d}  |  TP: {matriz_confusion[1,1]:5d}")
    
    # Reporte detallado
    print("\nReporte Detallado:")
    print(classification_report(y_eval, predicciones, target_names=['No Default', 'Default']))
    
    # Gráficos de rendimiento
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Curva ROC
    RocCurveDisplay.from_estimator(modelo, X_eval, y_eval, ax=ejes[0])
    ejes[0].set_title(f"Curva ROC - {nombre_modelo}", fontsize=12, weight='bold')
    ejes[0].grid(True, alpha=0.3)
    
    # Curva Precision-Recall
    PrecisionRecallDisplay.from_estimator(modelo, X_eval, y_eval, ax=ejes[1])
    ejes[1].set_title(f"Curva Precision-Recall - {nombre_modelo}", fontsize=12, weight='bold')
    ejes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Se Extraen los mejores modelos entrenados
modelo_lr = busqueda_lr.best_estimator_
modelo_rf = busqueda_rf.best_estimator_
modelo_svm = busqueda_svm.best_estimator_

# Se evalua cada modelo en el conjunto de prueba
print("\n" + "="*70)
print("EVALUACIÓN EN DATOS DE PRUEBA")
print("="*70)

calcular_metricas(modelo_lr, X_test, y_test, "Regresión Logística")
calcular_metricas(modelo_rf, X_test, y_test, "Random Forest")
calcular_metricas(modelo_svm, X_test, y_test, "SVM")

## 2.7 Analisis de sobreajuste y curvas de aprendizaje

print("\n" + "="*70)
print("DIAGNÓSTICO DE SOBREAJUSTE")
print("="*70)

# Comparamos desempeño en entrenamiento vs prueba
print("\nComparación de ROC-AUC entre conjuntos:")
print(f"{'Modelo':<25} {'Train':>10} {'Test':>10} {'Diferencia':>12}")
print("-"*60)

modelos_dict = {
    "Regresión Logística": modelo_lr,
    "Random Forest": modelo_rf,
    "SVM": modelo_svm
}

for nombre, modelo in modelos_dict.items():
    auc_train = roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    diferencia = auc_train - auc_test
    print(f"{nombre:<25} {auc_train:>10.4f} {auc_test:>10.4f} {diferencia:>12.4f}")

# Se Generan curvas de aprendizaje para cada modelo
print("\n Generando curvas de aprendizaje...")

fig, ejes = plt.subplots(1, 3, figsize=(18, 5))

lista_modelos = [
    (modelo_lr, "Regresión Logística", ejes[0]),
    (modelo_rf, "Random Forest", ejes[1]),
    (modelo_svm, "SVM", ejes[2])
]

for modelo, nombre, ax in lista_modelos:
    print(f"   Calculando para {nombre}...")
    tamanios, scores_train, scores_val = learning_curve(
        modelo, X_train, y_train,
        cv=validacion_cruzada,
        scoring="roc_auc",
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
        n_jobs=-1
    )
    
    # Se Calculan promedios y desviaciones
    media_train = np.mean(scores_train, axis=1)
    std_train = np.std(scores_train, axis=1)
    media_val = np.mean(scores_val, axis=1)
    std_val = np.std(scores_val, axis=1)
    
    # Grafica de las curvas ROC junto con sus bandas de confianza
    ax.plot(tamanios, media_train, 'o-', label="Entrenamiento", linewidth=2, markersize=6)
    ax.fill_between(tamanios, media_train - std_train, media_train + std_train, alpha=0.15)
    
    ax.plot(tamanios, media_val, 's-', label="Validación", linewidth=2, markersize=6)
    ax.fill_between(tamanios, media_val - std_val, media_val + std_val, alpha=0.15)
    
    ax.set_xlabel("Número de muestras de entrenamiento", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title(f"{nombre}", fontsize=12, weight='bold')
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.suptitle("Curvas de Aprendizaje por Modelo", fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.show()

## 2.8 Se interpretan los Modelos

print("\n" + "="*70)
print("ANALISIS DE IMPORTANCIA DE VARIABLES")
print("="*70)

# Random Forest: Feature Importance

print("\nVariables más relevantes según Random Forest:")
importancias = pd.Series(modelo_rf.feature_importances_, index=X.columns)
top_features = importancias.sort_values(ascending=False).head(15)
print(top_features)

plt.figure(figsize=(10, 6))
colores = sns.color_palette("viridis", len(top_features))
sns.barplot(x=top_features.values, y=top_features.index, palette=colores)
plt.title("Variables con Mayor Importancia - Random Forest", fontsize=13, weight='bold')
plt.xlabel("Importancia Relativa", fontsize=11)
plt.ylabel("Variable", fontsize=11)
plt.tight_layout()
plt.show()

# Regresión Logística: Coeficientes

print("\nCoeficientes de Regresión Logística:")
coeficientes = pd.Series(modelo_lr.coef_[0], index=X.columns).sort_values()

print("\n→ Variables que disminuyen la probabilidad de default (coef. negativos):")
print(coeficientes.head(10))

print("\n→ Variables que aumentan la probabilidad de default (coef. positivos):")
print(coeficientes.tail(10))

# Visualización de coeficientes más influyentes

plt.figure(figsize=(10, 8))
top_coefs = pd.concat([coeficientes.head(10), coeficientes.tail(10)])
colores_coefs = ['#3498db' if valor < 0 else '#e74c3c' for valor in top_coefs.values]

sns.barplot(x=top_coefs.values, y=top_coefs.index, palette=colores_coefs)
plt.title("Coeficientes más Influyentes - Regresión Logística\nAzul: reduce default | Rojo: aumenta default",
          fontsize=12, weight='bold')
plt.xlabel("Coeficiente", fontsize=11)
plt.ylabel("Variable", fontsize=11)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.show()

## 2.9 Realizamos comparaciones e identificamod el mejor Modelo

print("\n" + "="*70)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*70)

# Se Construye tabla comparativa
tabla_resultados = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Random Forest', 'SVM'],
    'AUC_CV': [
        busqueda_lr.best_score_,
        busqueda_rf.best_score_,
        busqueda_svm.best_score_
    ],
    'AUC_Train': [
        roc_auc_score(y_train, modelo_lr.predict_proba(X_train)[:, 1]),
        roc_auc_score(y_train, modelo_rf.predict_proba(X_train)[:, 1]),
        roc_auc_score(y_train, modelo_svm.predict_proba(X_train)[:, 1])
    ],
    'AUC_Test': [
        roc_auc_score(y_test, modelo_lr.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, modelo_rf.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, modelo_svm.predict_proba(X_test)[:, 1])
    ],
    'F1_Score': [
        f1_score(y_test, modelo_lr.predict(X_test)),
        f1_score(y_test, modelo_rf.predict(X_test)),
        f1_score(y_test, modelo_svm.predict(X_test))
    ],
    'Accuracy': [
        accuracy_score(y_test, modelo_lr.predict(X_test)),
        accuracy_score(y_test, modelo_rf.predict(X_test)),
        accuracy_score(y_test, modelo_svm.predict(X_test))
    ]
})

print("\n")
print(tabla_resultados.to_string(index=False))

# Identificamos el mejor Modelo
mejor_idx = tabla_resultados['AUC_Test'].idxmax()
mejor_modelo = tabla_resultados.loc[mejor_idx, 'Modelo']
mejor_auc = tabla_resultados.loc[mejor_idx, 'AUC_Test']

print("\n" + "="*70)
print(f" MODELO RECOMENDADO: {mejor_modelo}")
print(f"   ROC-AUC en conjunto de prueba: {mejor_auc:.4f}")
print("="*70)

print("\n Análisis completo")
