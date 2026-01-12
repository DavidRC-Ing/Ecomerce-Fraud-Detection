# 🔍 E-commerce Fraud Detection - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Fraud%20Detection-red.svg)](https://github.com)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com)

## 📋 Descripción del Proyecto

Sistema avanzado de **detección de fraude en transacciones de e-commerce** utilizando técnicas de Machine Learning. Este proyecto implementa un modelo predictivo que identifica transacciones fraudulentas en tiempo real, optimizando el balance entre la captura de fraudes y la minimización de falsos positivos mediante una matriz de costos personalizada.

El modelo alcanza un **Recall del 79.4%** capturando 1,050 de 1,322 fraudes en el conjunto de prueba, con un umbral óptimo de decisión de **0.485** que minimiza el costo operacional total.

---

## 🎯 Objetivos del Proyecto

- **Detectar transacciones fraudulentas** con alta precisión en un entorno de e-commerce
- **Optimizar el umbral de decisión** mediante una matriz de costos que refleja el impacto real del negocio
- **Minimizar el costo operacional** balanceando falsos positivos (revisiones innecesarias) y falsos negativos (fraudes no detectados)
- **Implementar feature engineering avanzado** para capturar patrones complejos de comportamiento fraudulento
- **Proporcionar un sistema escalable** y adaptable a diferentes contextos de negocio

---

## 🚀 Características Principales

### 🔬 Machine Learning Avanzado
- **Algoritmos implementados**: LightGBM, XGBoost, Random Forest, Logistic Regression
- **Optimización de hiperparámetros** con Optuna para búsqueda bayesiana
- **Validación cruzada estratificada** para garantizar robustez del modelo
- **Manejo de desbalanceo de clases** mediante técnicas de ponderación

### 📊 Feature Engineering Sofisticado
- **Transformaciones logarítmicas** para variables con distribuciones asimétricas
- **Features de comportamiento del usuario**: promedio histórico, frecuencia de transacciones
- **Señales temporales**: hora del día, día de la semana
- **Indicadores geográficos**: discrepancia entre país de facturación y envío
- **Features derivadas**: ratios, diferencias y banderas de outliers

### 💰 Optimización Basada en Costos
- **Matriz de costos personalizada**: C_FN=50, C_FP=1
- **Umbral óptimo calculado**: 0.4850
- **Reducción de costos**: ~67% vs estrategia de "no alertar nada"
- **Tasa de alertas controlada**: 15.50% del total de transacciones

### 📈 Métricas de Rendimiento
- **Precision**: 11.3% (1 de cada 9 alertas es fraude real)
- **Recall**: 79.4% (captura 1,050 de 1,322 fraudes)
- **F1-Score**: 0.198
- **ROC-AUC**: Evaluación de capacidad discriminativa del modelo
- **Precision-Recall AUC**: Métrica clave para datasets desbalanceados

---

## 📁 Estructura del Proyecto

```
fraud-detection/
│
├── Project_fraud_Detection.ipynb    # Notebook principal con análisis completo
├── transactions.csv                  # Dataset de transacciones (no incluido)
├── README.md                         # Este archivo
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── preprocessing.py              # Funciones de preprocesamiento
│   ├── feature_engineering.py        # Creación de features
│   ├── model_training.py             # Entrenamiento de modelos
│   └── evaluation.py                 # Métricas y evaluación
│
└── models/
    └── best_model.pkl                # Modelo entrenado guardado
```

---

## 🛠️ Tecnologías y Librerías

### Core ML/Data Science
```python
- pandas >= 1.3.0          # Manipulación de datos
- numpy >= 1.21.0          # Operaciones numéricas
- scikit-learn >= 1.0.0    # Algoritmos ML y métricas
- lightgbm >= 3.3.0        # Gradient Boosting optimizado
- xgboost >= 1.5.0         # Extreme Gradient Boosting
- optuna >= 2.10.0         # Optimización de hiperparámetros
```

### Visualización
```python
- matplotlib >= 3.4.0      # Gráficos estáticos
- seaborn >= 0.11.0        # Visualizaciones estadísticas
- plotly >= 5.3.0          # Gráficos interactivos
```

### Utilidades
```python
- imbalanced-learn >= 0.8.0  # Manejo de desbalanceo
- joblib >= 1.1.0            # Serialización de modelos
```

---

## 📊 Resultados Principales

### Matriz de Confusión (Test Set)
```
                    Predicción
                 Negativo  Positivo
Real Negativo     50,376     8,241  (TN / FP)
Real Positivo        272     1,050  (FN / TP)
```

### Métricas Clave
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Umbral Óptimo** | 0.4850 | Punto de decisión que minimiza costo total |
| **True Positives (TP)** | 1,050 | Fraudes correctamente identificados |
| **False Positives (FP)** | 8,241 | Transacciones legítimas marcadas como fraude |
| **False Negatives (FN)** | 272 | Fraudes no detectados |
| **True Negatives (TN)** | 50,376 | Transacciones legítimas correctamente clasificadas |
| **Precision** | 11.3% | De las alertas generadas, 11.3% son fraudes reales |
| **Recall** | 79.4% | Se capturan 79.4% de todos los fraudes |
| **F1-Score** | 0.198 | Media armónica entre Precision y Recall |
| **Tasa de Alertas** | 15.50% | Porcentaje de transacciones que requieren revisión |
| **Costo Total** | 21,841 | Costo operacional con umbral óptimo |

### Análisis de Costos
- **Costo de "No alertar nada"**: ~66,100 (solo FN)
- **Costo de "Alertar todo"**: ~58,617 (solo FP)
- **Costo con modelo optimizado**: 21,841
- **Reducción de costo**: ~67% vs no alertar, ~63% vs alertar todo

---

## 🔍 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones de variables numéricas y categóricas
- Identificación de outliers mediante IQR y percentiles
- Estudio de correlaciones y relaciones con la variable objetivo
- Análisis de desbalanceo de clases (fraudes vs transacciones legítimas)

### 2. Feature Engineering
```python
# Transformaciones logarítmicas
log_amount = log1p(amount)
log_shipping_distance = log1p(shipping_distance_km)

# Features de comportamiento del usuario
amount_minus_user_avg = amount - avg_amount_user
amount_over_user_avg = amount / avg_amount_user
tx_per_day = total_transactions_user / account_age_days

# Banderas de outliers
is_high_amount = (amount > percentile_99)
is_high_distance = (shipping_distance_km > percentile_99)

# Features temporales
hour_of_day = extract_hour(transaction_date)
day_of_week = extract_dow(transaction_date)

# Features geográficas
country_mismatch = (billing_country != shipping_country)
```

### 3. Entrenamiento de Modelos
- **División de datos**: 80% entrenamiento, 20% prueba (estratificada)
- **Validación cruzada**: 5-fold stratified cross-validation
- **Optimización de hiperparámetros**: Optuna con 100 trials
- **Métricas de evaluación**: ROC-AUC, PR-AUC, F1-Score, Recall

### 4. Optimización del Umbral de Decisión
```python
# Matriz de costos
C_FN = 50  # Costo de no detectar un fraude
C_FP = 1   # Costo de una falsa alarma

# Búsqueda del umbral óptimo
for threshold in np.linspace(0, 1, 201):
    predictions = (probabilities >= threshold).astype(int)
    cost = C_FP * FP + C_FN * FN

optimal_threshold = argmin(costs)  # 0.4850
```

### 5. Evaluación y Validación
- Análisis de curvas ROC y Precision-Recall
- Matriz de confusión y métricas derivadas
- Análisis de feature importance
- Validación de estabilidad del modelo

---

## 💡 Insights y Hallazgos Clave

### Patrones de Fraude Identificados
1. **Montos atípicos**: Transacciones con montos significativamente superiores al promedio del usuario
2. **Distancias de envío extremas**: Envíos a ubicaciones inusualmente lejanas
3. **Discrepancia geográfica**: País de facturación diferente al de envío
4. **Cuentas nuevas**: Mayor riesgo en cuentas con menos de 30 días de antigüedad
5. **Patrones temporales**: Ciertos horarios y días de la semana con mayor incidencia

### Variables Más Importantes
1. `amount` y `log_amount` - Monto de la transacción
2. `shipping_distance_km` - Distancia de envío
3. `avg_amount_user` - Promedio histórico del usuario
4. `country_mismatch` - Discrepancia entre países
5. `account_age_days` - Antigüedad de la cuenta

---

## 🎓 Aplicaciones y Casos de Uso

- **E-commerce**: Detección en tiempo real de transacciones fraudulentas
- **Fintech**: Prevención de fraude en pagos digitales
- **Banca**: Identificación de operaciones sospechosas
- **Seguros**: Detección de reclamaciones fraudulentas
- **Telecomunicaciones**: Prevención de fraude en suscripciones

---



## 📝 Notas del Autor

> **⚠️ Áreas de Mejora Identificadas**
> 
> Este proyecto representa un sólido punto de partida para un sistema de detección de fraude en producción. Sin embargo, para alcanzar un nivel enterprise-ready, se recomienda priorizar las siguientes mejoras:
> 
> 1. **Validación Temporal**: Implementar split temporal y backtesting para evitar data leakage y evaluar el modelo de forma más realista.
> 
> 2. **Causalidad en Features**: Asegurar que todas las features agregadas (especialmente las de usuario) se calculen solo con información histórica disponible al momento de la transacción.
> 
> 3. **Métricas Operacionales**: Incorporar Precision@K, Recall@K y curvas de captura para alinear mejor con restricciones operativas reales (ej: capacidad del equipo de revisión).
> 
> 4. **Interpretabilidad**: Añadir análisis SHAP para explicar predicciones individuales y generar confianza en stakeholders no técnicos.
> 
> 5. **Monitoreo Continuo**: Implementar sistema de detección de drift (features y performance) para mantener el modelo actualizado en producción.
> 
> 6. **Optimización Multi-Objetivo**: Considerar múltiples restricciones simultáneas (costo, capacidad operativa, recall mínimo) en lugar de solo minimizar costo.
> 
> Estas mejoras elevarían el proyecto de un excelente ejercicio académico a un sistema production-grade capaz de operar en un entorno empresarial real con millones de transacciones diarias.

---

## 🙏 Agradecimientos

- Dataset inspirado en casos reales de e-commerce
- Comunidad de Kaggle por recursos y discusiones
- Librerías open-source que hicieron posible este proyecto

---

## 📚 Referencias

1. **Fraud Detection**: Dal Pozzolo, A., et al. (2015). "Calibrating Probability with Undersampling for Unbalanced Classification"
2. **Cost-Sensitive Learning**: Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning"
3. **Imbalanced Learning**: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data"
4. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
5. **Gradient Boosting**: Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub ⭐**

</div>
