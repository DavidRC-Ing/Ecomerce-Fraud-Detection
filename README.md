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

## 📈 Próximos Pasos y Mejoras Futuras

### Mejoras Técnicas Prioritarias

#### 1. ⏰ Validación Temporal
**Problema actual**: El split train/test es aleatorio, lo que puede causar data leakage temporal y sobreestimar el rendimiento.

**Solución propuesta**:
```python
# Implementar split temporal
train = data[data['transaction_date'] < '2023-10-01']
test = data[data['transaction_date'] >= '2023-10-01']

# Backtesting con ventanas deslizantes
for train_window, test_window in time_series_split(data):
    model.fit(train_window)
    evaluate(model, test_window)
```

**Impacto esperado**: Evaluación más realista del rendimiento en producción, detección de concept drift.

---

#### 2. 🔒 Prevención de Data Leakage
**Problema actual**: Features como `avg_amount_user` pueden estar calculadas usando información del futuro.

**Solución propuesta**:
```python
# Calcular features solo con información pasada
def create_user_features_causal(df, date_col):
    df_sorted = df.sort_values(date_col)
    df['avg_amount_user_past'] = df.groupby('user_id')['amount'].expanding().mean().shift(1)
    return df

# Asegurar que train y test se procesan por separado
X_train = create_features(train_data)
X_test = create_features(test_data, fit_on_train=True)
```

**Impacto esperado**: Métricas más conservadoras pero más confiables para producción.

---

#### 3. 📊 Métricas Operacionales Avanzadas
**Problema actual**: Solo se reporta Precision/Recall global, sin considerar capacidad operativa.

**Solución propuesta**:
```python
# Precision@K y Recall@K
def precision_at_k(y_true, y_proba, k):
    top_k_indices = np.argsort(y_proba)[-k:]
    return y_true[top_k_indices].mean()

# Curva de captura (Capture Rate)
def capture_curve(y_true, y_proba, max_review_rate=0.20):
    # Cuántos fraudes capturo si reviso el top X% de transacciones
    pass

# Dos umbrales operacionales
threshold_cost_optimal = 0.485      # Minimiza costo
threshold_capacity_constrained = 0.65  # Limita alertas al 5% (capacidad operativa)
```

**Impacto esperado**: Mejor alineación con restricciones operativas reales (ej: equipo de revisión limitado).

---

#### 4. 🔍 Interpretabilidad del Modelo
**Problema actual**: Modelo de caja negra sin explicaciones de predicciones individuales.

**Solución propuesta**:
```python
import shap

# SHAP values para explicar predicciones
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizaciones
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Feature importance global
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Impacto esperado**: Mayor confianza del equipo de negocio, cumplimiento regulatorio, debugging más fácil.

---

#### 5. 🎯 Optimización Multi-Objetivo
**Problema actual**: Solo se optimiza costo, sin considerar otras restricciones.

**Solución propuesta**:
```python
# Optimización con múltiples objetivos
def multi_objective_optimization(threshold):
    cost = calculate_cost(threshold)
    alert_rate = calculate_alert_rate(threshold)
    recall = calculate_recall(threshold)

    # Restricciones
    if alert_rate > 0.10:  # Máximo 10% de alertas
        return float('inf')
    if recall < 0.70:      # Mínimo 70% de recall
        return float('inf')

    return cost

# Frontera de Pareto: costo vs recall
pareto_front = []
for t in thresholds:
    pareto_front.append((cost(t), recall(t), alert_rate(t)))
```

**Impacto esperado**: Flexibilidad para ajustar el modelo según cambios en capacidad operativa o prioridades de negocio.

---

#### 6. 🔄 Monitoreo y Re-entrenamiento
**Problema actual**: Modelo estático sin sistema de monitoreo de degradación.

**Solución propuesta**:
```python
# Sistema de monitoreo
class ModelMonitor:
    def __init__(self, model, baseline_metrics):
        self.model = model
        self.baseline_metrics = baseline_metrics

    def check_drift(self, new_data):
        # Drift en distribución de features
        feature_drift = self.detect_feature_drift(new_data)

        # Drift en performance
        current_metrics = self.evaluate(new_data)
        performance_drift = self.compare_metrics(current_metrics, self.baseline_metrics)

        if feature_drift or performance_drift:
            self.trigger_retraining_alert()

    def detect_feature_drift(self, new_data):
        # KS test, PSI (Population Stability Index), etc.
        pass

# Reentrenamiento automático
if drift_detected:
    retrain_model(new_data)
    validate_new_model()
    deploy_if_better()
```

**Impacto esperado**: Modelo siempre actualizado, detección temprana de degradación.

---

### Mejoras de Features

#### 7. 🌐 Features de Red y Grafos
```python
# Análisis de red de usuarios/dispositivos/IPs
- Grado de conectividad (cuántos usuarios comparten mismo dispositivo/IP)
- Clustering coefficient
- PageRank score
- Detección de comunidades sospechosas
```

#### 8. 📱 Features de Dispositivo y Sesión
```python
- device_fingerprint_hash
- session_duration
- number_of_page_views
- time_since_last_transaction
- velocity_features (transacciones por hora/día)
```

#### 9. 🧠 Features de Secuencia Temporal
```python
# LSTM o Transformer para capturar patrones secuenciales
- Secuencia de montos de últimas N transacciones
- Secuencia de ubicaciones
- Patrones de navegación
```

---

### Mejoras de Arquitectura

#### 10. 🚀 Despliegue en Producción
```python
# API REST con FastAPI
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('best_model.pkl')

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    features = preprocess(transaction)
    probability = model.predict_proba([features])[0][1]
    is_fraud = probability >= OPTIMAL_THRESHOLD
    return {"fraud_probability": probability, "is_fraud": is_fraud}
```

#### 11. 📦 Containerización
```dockerfile
# Dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 12. ☁️ Infraestructura Cloud
- **AWS SageMaker** o **Azure ML** para entrenamiento escalable
- **AWS Lambda** o **Cloud Functions** para inferencia serverless
- **Kafka** o **Kinesis** para streaming de transacciones en tiempo real
- **MLflow** para tracking de experimentos y versionado de modelos

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

## 👤 Autor

**Tu Nombre**
- LinkedIn: [tu-perfil](https://linkedin.com/in/tu-perfil)
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu.email@ejemplo.com

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

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
