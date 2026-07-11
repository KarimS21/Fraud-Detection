# DOCUMENTO TÉCNICO: ARQUITECTURA Y FLUJOS
## Motor de Ranking de Riesgo de Fraude - Semana 10

---

## 1. Arquitectura del Sistema

### 1.1 Componentes Principales

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTRADA: NUEVA TRANSACCIÓN                       │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│         EXTRACCIÓN DE FEATURES (Semanas 1-7)                        │
├─────────────────────────────────────────────────────────────────────┤
│ • Limpieza de datos                                                 │
│ • Feature engineering (logging, ratios, one-hot)                    │
│ • Features de clustering (K-means, DBSCAN labels)                   │
│ • Reducción dimensional (PCA 24→?, SVD 24→?)                        │
│ • Escalado estándar (StandardScaler)                                │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  VECTOR DE ENTRADA │
        │  (features scaled) │
        └────────┬───────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│  BASELINE        │ │  XGBOOST         │
│  HISTÓRICO       │ │  OPTIMIZADO      │
├──────────────────┤ ├──────────────────┤
│ score_baseline = │ │ score_strong =   │
│ P(fraude|cat)    │ │ predict_proba    │
│                  │ │                  │
│ [0, 0.25]        │ │ [0, 1]           │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  COMBINAR SCORES    │
        │ (ensemble opcional) │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  SCORE FINAL [0,1]  │
        │  (probabilidad)     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  RANK EN POOL       │
        │ (ordenar por score) │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  DECISION RULE      │
        │ (threshold based)   │
        └──────────┬──────────┘
                   │
         ┌─────────┴────────────┐
         ▼                      ▼
    ┌─────────┐            ┌─────────┐
    │ APROBAR │            │ REVISAR │
    │ (score  │            │ (score  │
    │  <0.3)  │            │ ≥0.3)   │
    └─────────┘            └─────────┘
         │                      │
         ▼                      ▼
   [Autorizar]         [Investigar]
```

### 1.2 Flujo de Datos (Training)

```
SEMANA 5: Feature Engineering
  Raw Data (fraudTrain.csv)
         │
         ├─ Limpieza, missing values, outliers
         ├─ Feature engineering (log, ratios, one-hot)
         │
         ▼
  X_features (2D array o sparse matrix)
  Features básicas: ~100-200 variables

SEMANA 5: Reducción Dimensional
  X_features
         │
         ├─ PCA: 100 → 24 componentes (retener 90% varianza)
         ├─ SVD: 100 → 24 componentes
         │
         ▼
  X_pca, X_svd (2D arrays)

SEMANA 7: Clustering
  X_features (scaled)
         │
         ├─ K-Means: asignar cluster labels (0-k)
         ├─ DBSCAN: asignar noise labels (-1, 0-k)
         │
         ▼
  cluster_labels, dbscan_labels (1D arrays)

SEMANA 7/10: Integración
  [X_pca] + [X_svd] + [cluster_labels] + [features_original]
         │
         ├─ Concatenar
         ├─ Escalar con StandardScaler
         │
         ▼
  X_cluster_scaled (final training features)

SEMANA 10: Modelado
  X_cluster_scaled, y_train
         │
         ├─ Grid Search CV con XGBoost
         │  Parámetros: max_depth, learning_rate, etc.
         │
         ▼
  best_xgb (modelo optimizado)
```

### 1.3 Flujo de Datos (Testing/Evaluation)

```
DATASET: df_model_test (21,000 transacciones)
         │
         ├─ Extraer features: X_test_cluster_scaled
         ├─ Extraer target: y_test
         │
         ▼
BASELINE:
  df_model_test['categoria_comercio'] 
         │
         ├─ Lookup en tabla de riesgos históricos
         │  (calculada en df_model training)
         │
         ▼
  y_score_baseline (21,000 valores [0, 0.25])

XGBOOST:
  X_test_cluster_scaled
         │
         ├─ best_xgb.predict_proba()
         │
         ▼
  y_score_strong (21,000 valores [0, 1])

EVALUACIÓN:
  y_score_baseline, y_score_strong, y_test
         │
         ├─ Calcular ROC-AUC, PR-AUC, etc.
         ├─ Calcular Recall@K, Precision@K
         ├─ Generar matrices de confusión
         │
         ▼
  evaluation_report, topk_report, visualizaciones
```

---

## 2. Pipeline de Features

### 2.1 Fuentes de Features

| Tipo | Fuente | Dimensiones | Descripción |
|------|--------|-------------|-------------|
| **Originales** | fraudTrain.csv | ~40 cols | Variables crudas de transacción |
| **Engineered** | Semanas 1-4 | ~100-200 | Derivadas, interacciones, one-hot |
| **Reducidas** | Semana 5 | 24 (PCA) + 24 (SVD) = 48 | Componentes principales |
| **Clustering** | Semana 7 | 2-3 | Labels de clusters |
| **Scaled** | Preprocesador | 48-51 | Todas las anteriores normalizadas |

### 2.2 Transformaciones Específicas

```python
# Pseudocódigo del pipeline

class FraudFeaturePipeline:
    def __init__(self):
        # Cargar modelos preentrenados de semanas anteriores
        self.scaler_numeric = load_model('numeric_scaler.pkl')
        self.encoder_cat = load_model('cat_encoder.pkl')
        self.pca_model = load_model('pca.pkl')
        self.svd_model = load_model('svd.pkl')
        self.kmeans_model = load_model('kmeans.pkl')
        self.dbscan_model = load_model('dbscan.pkl')
        self.xgb_model = load_model('xgb.pkl')
    
    def transform(self, raw_transaction):
        # 1. Extracción y limpieza
        features_dict = extract_raw_features(raw_transaction)
        
        # 2. Feature engineering
        features_dict = apply_feature_engineering(features_dict)
        
        # 3. Conversión a arrays
        X_numeric = features_dict['numeric_features']
        X_cat_high = features_dict['cat_high_cardinality']
        X_cat_low = features_dict['cat_low_cardinality']
        
        # 4. Escalado
        X_numeric_scaled = self.scaler_numeric.transform(X_numeric)
        X_cat_high_encoded = self.encoder_cat.transform(X_cat_high)
        
        # 5. Reducción dimensional
        X_pca = self.pca_model.transform(np.hstack([X_numeric_scaled, X_cat_high_encoded]))
        X_svd = self.svd_model.transform(np.hstack([X_numeric_scaled, X_cat_high_encoded]))
        
        # 6. Clustering
        cluster_label = self.kmeans_model.predict(X_numeric_scaled)[0]
        
        # 7. Concatenar
        X_final = np.hstack([X_pca, X_svd, [cluster_label]])
        
        return X_final
    
    def predict_fraud_score(self, raw_transaction):
        X_final = self.transform(raw_transaction)
        score = self.xgb_model.predict_proba([X_final])[0, 1]
        return score
```

---

## 3. Especificación de Entrada/Salida

### 3.1 Entrada al Modelo

**Formato:** Vector numérico de 48-51 dimensiones (float64)

**Origen:** X_test_cluster_scaled

**Componentes:**
- 24 dimensiones: PCA (componentes principales)
- 24 dimensiones: SVD (descomposición en valores singulares)
- 1 dimensión: Cluster label (0-k)
- Posibles dimensiones adicionales: features categóricas one-hot

**Rango:** [-3, +3] (después de standardization)

### 3.2 Salida del Modelo

**Formato:** Escalar float64

**Rango:** [0, 1] (probabilidad)

**Interpretación:**
- score ∈ [0.0, 0.3): Baja probabilidad de fraude (α = 5%)
- score ∈ [0.3, 0.7): Media probabilidad (revisar)
- score ∈ [0.7, 0.9): Alta probabilidad (investigar)
- score ∈ [0.9, 1.0]: Crítica (acción inmediata)

---

## 4. Comparativa Detallada: Baseline vs XGBoost

### 4.1 Algoritmo Baseline

```
BASELINE: Risk by Commercial Category

Input: transacción con 'categoria_comercio'

Step 1: Lookup histórico
  risk_table = df_model.groupby('categoria_comercio')['es_fraude'].mean()
  
Step 2: Asignar score
  score = risk_table[categoria_comercio]
  
Step 3: Fallback para categorías nuevas
  if categoria_comercio not in risk_table:
    score = global_fraud_rate (0.10)
  
Output: score ∈ [0, 0.25]

Ventajas:
  ✓ 100% interpretable
  ✓ O(1) tiempo de ejecución
  ✓ No requiere features complejas
  ✓ Basado en hechos históricos

Desventajas:
  ✗ Muy simplista
  ✗ No captura interacciones
  ✗ Granularidad gruesa (solo por categoría)
  ✗ No usa información individual de transacción
```

### 4.2 Algoritmo XGBoost

```
XGBOOST: Gradient Boosting Optimized

Input: X_cluster_scaled (48-51 dimensiones)

Step 1: Ensemble de árboles
  Cada árbol: shallow decision tree (~3-8 profundidad)
  100-1000 árboles secuenciales
  
Step 2: Predicción
  score_raw = suma ponderada de outputs de árboles
  score_proba = sigmoid(score_raw)  # Normalizar a [0,1]
  
Step 3: Threshold
  if score_proba >= 0.5: predicts FRAUD
  else: predicts NO FRAUD

Output: score ∈ [0, 1]

Ventajas:
  ✓ Captura patrones no-lineales complejos
  ✓ Maneja interacciones automáticamente
  ✓ Granularidad fina por transacción individual
  ✓ Probabilidades bien calibradas
  ✓ Regularización integrada

Desventajas:
  ✗ Menos interpretable (black-box)
  ✗ Requiere features engineered
  ✗ Riesgo de overfitting si no se regulariza
  ✗ Latencia computacional mayor
```

### 4.3 Matriz de Comparación

| Aspecto | Baseline | XGBoost | Ganador |
|---------|----------|---------|---------|
| Precisión (ROC-AUC) | 0.726 | 0.992 | XGBoost ⭐⭐⭐ |
| Discriminación | Pobre | Excelente | XGBoost ⭐⭐⭐ |
| Interpretabilidad | Perfecta | Difícil | Baseline ⭐⭐⭐ |
| Velocidad | Instant | ms | Baseline ⭐⭐⭐ |
| Escalabilidad | Perfecta | Buena | Baseline |
| Robustez | Débil | Fuerte | XGBoost ⭐⭐⭐ |
| Recall@5% | 13.0% | 49.95% | XGBoost ⭐⭐⭐ |
| Precisión@5% | 26.0% | 99.9% | XGBoost ⭐⭐⭐ |

**VEREDICTO:** XGBoost en todos los aspectos operativos críticos

---

## 5. Validación Cruzada y Métricas

### 5.1 Estrategia de Validación

```
Datos originales (N transacciones)
        │
        ├─ 70%: Train (gridSearch.fit)
        │  - Fit XGBoost con CV internal (5 folds)
        │  - Buscar mejores hiperparámetros
        │
        ├─ 20%: Validation (model selection)
        │  - Evaluar cada candidato de GridSearch
        │  - Seleccionar best_xgb por ROC-AUC
        │
        └─ 10%: Test (evaluación final)
           - Generar métricas offline
           - Simular comportamiento en producción
```

### 5.2 Métricas de Rendimiento Offline

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

```
Definición: Integral bajo la curva de TPR vs FPR

ROC-AUC = ∫ TPR(threshold) d(FPR(threshold)) para threshold ∈ [0,1]

Interpretación:
  0.5  = Modelo aleatorio
  0.7+ = Modelo aceptable
  0.9+ = Modelo excelente
  1.0  = Predicción perfecta

XGBoost: 0.9917 → Extraordinario (solo 80 falsos positivos de 18,900)
```

#### PR-AUC (Precision-Recall - Area Under Curve)

```
Definición: Integral bajo la curva de Precision vs Recall

PR-AUC = ∫ Precision(threshold) d(Recall(threshold))

Ventaja sobre ROC-AUC: Mejor para clases desbalanceadas (como fraude)

XGBoost: 0.9578 → Excelente (Precision y Recall ambas altas)
Baseline: 0.2077 → Pobre (TP=0, así que Precision=0)
```

#### Recall@K (Coverage at K%)

```
Definición: Fracción de fraudes capturados al revisar top-K%

Recall@K = (Fraudes en top K%) / (Total de fraudes)

Ejemplo XGBoost:
  K=1%:   Recall@1% = 10.0%  (210 transacciones → 210 fraudes)
  K=5%:   Recall@5% = 49.95% (1,050 transacciones → 1,049 fraudes)
  K=10%:  Recall@10% = 89.0% (2,100 transacciones → 1,870 fraudes)

Interpretación operativa:
  "Revisar el top 5% del ranking = capturar ~50% de todos los fraudes"
```

#### Lift@K

```
Definición: Ganancia sobre modelo aleatorio

Lift@K = Precision@K / Base rate

Ejemplo:
  XGBoost @ K=5%: Lift = 99.9% / 10% = 9.99x
  
Interpretación:
  "Revisando el top 5%, es 10x más probable encontrar fraude
   que si revisara transacciones al azar"
```

---

## 6. Manejo de Errores y Edge Cases

### 6.1 Fraudes No Detectados (False Negatives)

```
Cantidad: 400 fraudes de 2,100 (19.05%)

Características:
  • Scores: Tipicamente [0.3, 0.7) (zona gris)
  • Pattern: Imitan bien a transacciones legales
  • Causa posible: 
    - Features insuficientes para discriminación
    - Variabilidad de patrones de fraude
    - Rare combinations de variables

Acción Recomendada:
  1. Análisis post-hoc de estos casos
  2. Investigación manual adicional
  3. Posible ajuste de features en próximo ciclo
  4. Escalada a equipo de fraude investigativo
```

### 6.2 Falsos Positivos (False Positives)

```
Cantidad: 80 no-fraudes de 18,900 (0.42%)

Características:
  • Scores: Tipicamente [0.7, 1.0] (alto riesgo)
  • Pattern: Transacciones legales con patrones inusuales
  • Ejemplos:
    - Compra en categoría poco frecuente
    - Monto muy alto (pero legítimo)
    - Ubicación geográfica rara
    - Hora inusual

Acción Recomendada:
  1. Revisión rápida (segundos, no minutos)
  2. Validación de contexto (¿cliente conocido?)
  3. Información al cliente si se rechaza
  4. Feedback al modelo para próximos entrenamientos
```

### 6.3 Casos de Riesgo en Producción

| Escenario | Manejo |
|-----------|--------|
| **Modelo offline vs online diverge** | Reentrenar semanal, monitorear drift |
| **Nueva categoría comercial** | Asignar score global (fallback) |
| **Feature missing/corrupted** | Usar imputation median, alertar |
| **Score indeterminado (NaN)** | Redirigir a revisión manual |
| **Latencia >100ms** | Usar caché de scores precalculados |

---

## 7. Monitoreo en Producción

### 7.1 Dashboard de Métricas

```
REAL-TIME METRICS:

┌────────────────────────────────────────────┐
│ Transacciones procesadas (hoy): 150,000    │
│ Fraudes flagged: 12,000 (8%)               │
│ Fraudes confirmados: 1,200 (10% de los)    │
│ FP ratio: 0.41% (500 falsos positivos)     │
│                                            │
│ ROC-AUC (últimas 7 días): 0.9915           │
│ Recall@5%: 49.8%                          │
│ Precisión@5%: 99.8%                       │
│                                            │
│ Latencia promedio: 45ms                    │
│ Error rate: 0.02%                         │
└────────────────────────────────────────────┘
```

### 7.2 Alertas Automáticas

| Condición | Acción |
|-----------|--------|
| ROC-AUC < 0.98 | Revisar datos, posible reentrenamiento |
| Recall@5% < 45% | Ajuste de threshold o features |
| Latencia > 200ms | Optimizar código o aumentar recursos |
| FP ratio > 1% | Revisar parámetros de regularización |
| Distribución de scores anómala | Verificar feature engineering |

---

## 8. Plan de Implementación

### Fase 1: Piloto (Semana 1-2)
```
1. Validación técnica del modelo
2. Testing con 1,000 transacciones
3. Reunión con equipo de fraude
4. Ajuste de thresholds según feedback
```

### Fase 2: Despliegue (Semana 3-4)
```
1. Integración en sistema de producción
2. Monitoreo intensivo (reporte diario)
3. Capacitación de analistas
4. Procedimientos de escalada
```

### Fase 3: Optimización (Mes 2)
```
1. Recopilación de datos reales
2. Análisis de errores en producción
3. Reentrenamiento con datos nuevos
4. Ajuste fino de thresholds por segmento
```

---

## 9. Referencias Técnicas

### Librerías Utilizadas
- **scikit-learn:** Pipeline, GridSearchCV, métricas
- **XGBoost:** Modelo de gradient boosting
- **pandas:** Manipulación de datos
- **numpy:** Cálculos numéricos
- **matplotlib/seaborn:** Visualización

### Hiperparámetros XGBoost (Recomendados)
```python
xgb_params = {
    'n_estimators': 100,        # Número de árboles
    'max_depth': 5,              # Profundidad máxima
    'learning_rate': 0.1,        # Tasa de aprendizaje
    'subsample': 0.8,            # Fracción de samples
    'colsample_bytree': 0.8,     # Fracción de features
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'objective': 'binary:logistic',
    'random_state': 42
}
```

---

**Documento Técnico Completo - Semana 10**  
**Versión:** 1.0  
**Fecha:** Junio 2026
