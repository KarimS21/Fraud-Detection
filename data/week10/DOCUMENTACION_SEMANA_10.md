# Documentación: Motor de Ranking de Riesgo de Fraude
## Semana 10 -

**Fecha:** Junio 2026  
**Asunto:** Sistema de Predicción y Ranking de Fraude en Transacciones  
**Estado:** ✅ Completado

---

## 1. Resumen Ejecutivo

Este documento presenta el desarrollo y evaluación de un **motor de predicción y ranking de riesgo de fraude** diseñado para priorizar la revisión de transacciones sospechosas. El sistema combina un baseline histórico simple con un modelo de machine learning avanzado (XGBoost optimizado), permitiendo a los analistas enfocarse en los casos de mayor riesgo.

### Resultados Clave:
- **Mejora en ROC-AUC:** De 0.7260 (baseline) a **0.9917** (XGBoost) → **36.5% de mejora**
- **Mejora en PR-AUC:** De 0.2077 (baseline) a **0.9578** (XGBoost) → **361% de mejora**
- **Recall@5%:** De 13% (baseline) a **50%** (XGBoost) → **XGBoost captura 4x más fraudes**
- **Precisión@5%:** De 26% (baseline) a **99.9%** (XGBoost)

---

## 2. Objetivos y Alcance

### 2.1 Objetivo General
Desarrollar un sistema de ranking de fraude que:
- Asigne scores de riesgo a cada transacción
- Ordene el conjunto de candidatos por probabilidad de fraude
- Maximice la detección temprana minimizando el costo operativo

### 2.2 Alcance
- **Período:** Semanas 1-10 del proyecto
- **Datos:** 21,000 transacciones de prueba (Dataset: `df_model_test`)
- **Target:** Variable `es_fraude` (binaria: fraude/no-fraude)
- **Clase positiva:** 2,100 fraudes (10% de incidencia)

---

## 3. Metodología

### 3.1 Definición del Candidate Pool

El **candidate pool** es el conjunto completo de transacciones a evaluar (offline):
- **Tamaño:** 21,000 transacciones
- **Fuente:** Conjunto de prueba (`df_model_test`)
- **Fracción de fraude:** 10% (~2,100 fraudes reales)
- **Consistencia:** Ambos sistemas se evalúan sobre el mismo pool

### 3.2 Sistema Baseline: Baseline Histórico

El baseline representa una estrategia simple de negocio:

**Algoritmo:**
1. Calcular la tasa histórica de fraude por categoría comercial
2. Asignar ese score a cada transacción según su categoría
3. En caso de categoría nueva, usar la tasa global

**Fórmula:**
```
score_baseline = tasa_fraude_historica[categoria_comercio]
                 si categoria no existe: score = tasa_global
```

**Parámetros:**
- Variable de agrupamiento: `categoria_comercio`
- Tasa global: 10.0%

**Ventajas del Baseline:**
- ✅ Completamente interpretable
- ✅ Sin requerimiento computacional
- ✅ Útil como punto de referencia
- ✅ Basado en patrones históricos reales

---

### 3.3 Sistema Fuerte: XGBoost Optimizado

El sistema fuerte utiliza un modelo de gradient boosting optimizado mediante búsqueda en grilla.

**Arquitectura:**
1. **Entrada:** Features reducidas y escaladas (X_test_cluster_scaled)
2. **Modelo:** XGBoost Classifier
3. **Salida:** Probabilidad predicha de fraude [0, 1]

**Optimización:**
- Algoritmo: GridSearchCV
- Parámetros explorados: hiperparámetros de profundidad, tasa de aprendizaje, regularización
- Métrica de validación: ROC-AUC

**Características de entrada:**
- Features PCA/SVD reducidas (semana 5)
- Features de clustering (semana 7)
- Features engineered (semanas anteriores)
- Escalado standardizado

**Ventajas del Sistema Fuerte:**
- ✅ Captura patrones no-lineales
- ✅ Maneja interacciones entre features
- ✅ Optimizado para máximo ROC-AUC
- ✅ Probabilidades bien calibradas

---

## 4. Evaluación Offline

### 4.1 Protocolo de Evaluación

**Metrics Utilizadas:**

| Métrica | Descripción | Fórmula | Rango |
|---------|-------------|---------|-------|
| **ROC-AUC** | Área bajo la curva ROC | Integral de TP rate vs FP rate | [0, 1] |
| **PR-AUC** | Área bajo la curva Precision-Recall | Integral de Precision vs Recall | [0, 1] |
| **Precision** | Tasa de acierto en predicciones positivas | TP / (TP + FP) | [0, 1] |
| **Recall** | Cobertura de positivos reales | TP / (TP + FN) | [0, 1] |
| **F1-Score** | Media armónica de Precision y Recall | 2 × (P × R) / (P + R) | [0, 1] |
| **Recall@K** | Fraudes capturados en top-K% del pool | Fraudes en top-K / Total fraudes | [0, 1] |
| **Precision@K** | Tasa de acierto en top-K | Fraudes en top-K / K | [0, 1] |
| **Lift@K** | Ganancia sobre modelo aleatorio | Precision@K / Base rate | [0, ∞) |

**Threshold:** 0.5 (para clasificación binaria)

---

### 4.2 Resultados de Evaluación

#### Tabla 1: Métricas de Rendimiento Completo

| Métrica | Baseline Histórico | XGBoost Optimizado | Mejora |
|---------|-------------------|-------------------|--------|
| **ROC-AUC** | 0.7260 | 0.9917 | +36.5% ✅ |
| **PR-AUC** | 0.2077 | 0.9578 | +361% ✅ |
| **Precision** | 0.00% | 95.51% | - |
| **Recall** | 0.00% | 80.95% | - |
| **F1-Score** | 0.00% | 0.8763 | - |
| **True Negatives** | 18,900 | 18,820 | -80 |
| **False Positives** | 0 | 80 | +80 |
| **False Negatives** | 2,100 | 400 | -1,700 ✅ |
| **True Positives** | 0 | 1,700 | +1,700 ✅ |

**Interpretación:**
- El baseline NO detecta fraudes (TP=0), por lo que es inadecuado como sistema de decisión
- XGBoost captura el **80.95%** de los fraudes con solo **0.42%** de falsas alarmas
- La mejora en PR-AUC es especialmente relevante para problemas desbalanceados

---

#### Tabla 2: Análisis de Ranking (Top-K)

| Sistema | % del Pool | Transacciones | Fraudes Capturados | Precision@K | Recall@K | Lift@K |
|---------|-----------|----------------|-------------------|-------------|----------|---------|
| **Baseline** | 1% | 210 | 70 | 33.3% | 3.3% | 3.33x |
| **Baseline** | 5% | 1,050 | 273 | 26.0% | 13.0% | 2.60x |
| **Baseline** | 10% | 2,100 | 540 | 25.7% | 25.7% | 2.57x |
| **XGBoost** | 1% | 210 | **210** ✅ | **100%** ✅ | **10.0%** | **10.0x** ✅ |
| **XGBoost** | 5% | 1,050 | **1,049** ✅ | **99.9%** ✅ | **49.95%** ✅ | **9.99x** ✅ |
| **XGBoost** | 10% | 2,100 | **1,870** ✅ | **89.0%** | **89.0%** ✅ | **8.90x** ✅ |

**Insights:**
- Al revisar el **top 1%** del pool, XGBoost identifica el **100%** de fraudes en esa porción
- Al revisar el **top 5%**, XGBoost captura casi el **50%** de todos los fraudes (vs 13% del baseline)
- El Lift@K de ~10x en el 1% indica que el modelo es extraordinariamente selectivo

---

### 4.3 Matriz de Confusión (XGBoost)

```
                 Predicción Negativa  |  Predicción Positiva
Real Negativo     18,820 (99.58%)     |  80 (0.42%)
Real Positivo     400 (19.05%)        |  1,700 (80.95%)
```

**Interpretación:**
- **Especificidad:** 99.58% → Mínimas falsas alarmas
- **Sensibilidad:** 80.95% → Excelente captura de fraudes
- **Balance:** El modelo favorece la detección (minimiza FN) sin saturar con falsos positivos

---

## 5. Análisis de Casos

### 5.1 Casos Fuertes: Aciertos del Modelo

#### Fraudes Bien Priorizados (Score Alto)
- **Definición:** Fraudes reales con puntuación XGBoost alta
- **Cantidad:** Top 10 casos mostrados
- **Característica:** Son fraudes que el modelo identifica correctamente
- **Valor:** Permiten orientar equipos de investigación a casos de alto riesgo confirmado

#### No-Fraudes Bien Descartados (Score Bajo)
- **Definición:** Transacciones legales con puntuación XGBoost baja
- **Cantidad:** Top 10 casos mostrados
- **Característica:** Transacciones normales que no generan alarma
- **Valor:** Reducen costo operativo al evitar revisiones innecesarias

---

### 5.2 Casos de Fallo: Limitaciones del Modelo

#### Fraudes No Priorizados (Score Bajo)
- **Definición:** Fraudes reales que el modelo no detecta (FN)
- **Cantidad:** 400 casos (~19% del total de fraudes)
- **Causa:** Fraudes que imitan patrones legales
- **Acción:** Requieren análisis adicional, posible actuación manual

#### No-Fraudes Sobre-Priorizados (Score Alto)
- **Definición:** Transacciones legales con puntuación alta (FP)
- **Cantidad:** 80 casos (~0.42% del total)
- **Causa:** Transacciones legales con patrones poco comunes
- **Acción:** Revisión rápida, bajo impacto operativo

---

### 5.3 Análisis de Distribuciones

#### Baseline Histórico:
- Distribución multimodal (varía por categoría comercial)
- Fraudes distribuidos a lo largo de todo el rango
- Separación pobre entre fraude y no-fraude

#### XGBoost:
- Distribución bimodal clara
- Fraudes concentrados en scores altos (>0.8)
- No-fraudes concentrados en scores bajos (<0.2)
- Separación excelente entre clases

**Conclusión:** XGBoost aprende patrones que permiten discriminación clara.

---

## 6. Tipo de Sistema

### 6.1 Clasificación del Proyecto

Este proyecto se clasifica como un **MOTOR DE PREDICCIÓN Y RANKING DE RIESGO**:

| Criterio | Clasificación |
|----------|---|
| **¿Es recomendación?** | ❌ No. No recomienda productos ni contenidos a usuarios. |
| **¿Es ranking?** | ✅ **SÍ**. Ordena transacciones por riesgo de fraude. |
| **¿Es predicción?** | ✅ **SÍ**. Predice probabilidad de fraude para cada transacción. |
| **¿Es segmentación?** | ⚠️ Parcial. La segmentación (Semana 7) alimenta contexto pero NO es componente principal. |

### 6.2 Justificación Técnica

**Ranking:**
- Output principal: scores ordenados de riesgo
- Propósito: priorizar transacciones para revisión
- Métrica: Recall@K y Precision@K

**Predicción:**
- Modelo supervisado XGBoost
- Output: probabilidad de fraude [0, 1]
- Métrica: ROC-AUC, PR-AUC

**No Recomendación:**
- No hay sistema de recomendación de items
- No hay usuario destinatario
- No hay contenido personalizado

---

## 7. Alineación de Datos y Arquitectura

### 7.1 Pipeline Completo

```
Semana 1-4: Limpieza y Feature Engineering
    ↓
Semana 5: Reducción Dimensional (PCA, SVD)
    ↓
Semana 7: Clustering (K-Means, DBSCAN)
    ↓
Semana 10: Predicción y Ranking
    ├─ Baseline: Score histórico por categoría
    └─ XGBoost: Modelo optimizado con features reducidas
```

### 7.2 Alineación de Datos

| Componente | Fuente | Propósito | Alineación |
|-----------|--------|----------|-----------|
| **Baseline** | `df_model` | Tablas de riesgo histórico | ✅ Misma población training |
| **XGBoost Features** | `X_test_cluster_scaled` | Features de entrada | ✅ Features reducidas y escaladas |
| **Target** | `y_test` | Variable real de fraude | ✅ Etiquetas consistentes |
| **Candidate Pool** | `df_model_test` | Datos de evaluación | ✅ Mismo conjunto de prueba |

---

## 8. Protocolo Operativo

### 8.1 Flujo de Uso en Producción

```
Nueva transacción entra al sistema
    ↓
Extraer features (PCA/SVD, clustering)
    ↓
Pasar por XGBoost
    ↓
Obtener score [0, 1]
    ↓
Ordenar en ranking de riesgo
    ↓
Revisor humano examina top-N transacciones
    ↓
Acción: Aprobar / Rechazar
```

### 8.2 Escenarios de Decisión

| Rango de Score | Acción Recomendada | Prioridad |
|---------------|------------------|-----------|
| **0.0 - 0.3** | Aprobar automático | Baja |
| **0.3 - 0.7** | Revisión ligera | Media |
| **0.7 - 0.9** | Revisión profunda | Alta |
| **0.9 - 1.0** | Revisión especializada | Crítica |

### 8.3 Métricas de Monitoreo

Después del despliegue, monitorear:
- **Drift de Score:** ¿Los scores cambian significativamente?
- **Recall Real:** ¿Cuántos fraudes se detectan en la práctica?
- **False Positive Rate:** ¿Cuántos legales se rechazan?
- **Latencia:** ¿Tiempo de predicción por transacción?

---

## 9. Limitaciones y Consideraciones

### 9.1 Limitaciones del Modelo

1. **Cobertura Incompleta:**
   - 19% de los fraudes NO se detectan (FN=400)
   - Requiere investigación complementaria

2. **Falsos Positivos:**
   - 80 transacciones legales flagged (~0.42%)
   - Bajo pero no nulo; requiere filtrado manual

3. **Dependencia de Features:**
   - El modelo requiere todas las features del pipeline
   - Cambios en el proceso de features afectan rendimiento

4. **Drift Temporal:**
   - Modelo entrenado en datos históricos
   - Nuevos patrones de fraude pueden evadir detección

### 9.2 Consideraciones de Implementación

1. **Reentrenamiento Periódico:**
   - Revisar cada 1-3 meses
   - Incorporar nuevos fraudes descubiertos

2. **Feedback Loop:**
   - Fraudes detectados en la práctica → reentrenamiento
   - Falsos positivos confirmados → ajuste de umbrales

3. **Interpretabilidad:**
   - XGBoost es menos interpretable que baseline
   - Documentar decisiones controversiales

4. **Cumplimiento Normativo:**
   - Verificar cumplimiento de regulaciones locales
   - Auditoría de decisiones rechazadas

---

## 10. Conclusiones

### 10.1 Resumen de Hallazgos

✅ **Sistema exitoso:** XGBoost mejora dramáticamente sobre el baseline  
✅ **ROC-AUC 0.9917:** Excelente discriminación entre fraude y no-fraude  
✅ **PR-AUC 0.9578:** Excelente en el contexto de clase minoritaria  
✅ **Operativo:** Ranking claro y accionable para equipos de investigación  
✅ **Documentado:** Protocolo de evaluación transparente y reproducible  

### 10.2 Recomendaciones

1. **Corto Plazo:**
   - Deployar XGBoost como sistema de ranking primario
   - Mantener baseline como referencia de comparación
   - Monitorear falsos positivos/negativos en producción

2. **Mediano Plazo:**
   - Investigar los 400 fraudes no detectados
   - Crear módulo de investigación manual para casos dudosos
   - Implementar pipeline de feedback

3. **Largo Plazo:**
   - Explorar ensambles (XGBoost + LightGBM + CatBoost)
   - Investigar métodos de explicabilidad (SHAP, LIME)
   - Considerar sistema de aprendizaje continuo

### 10.3 Entregables Finales

| Archivo | Descripción |
|---------|-------------|
| `offline_evaluation_report_week10.csv` | Métricas de evaluación baseline vs XGBoost |
| `topk_ranking_report_week10.csv` | Análisis de ranking (Recall@K, Precision@K, Lift@K) |
| `candidate_pool_ranked_week10.csv` | Pool completo con scores y rankings de ambos sistemas |
| `interpretacion_week10.txt` | Interpretación técnica en formato texto |
| `analisis_rendimiento_completo.png` | Visualización de curvas ROC, matrices y métricas |
| `distribucion_scores_y_metricas_topk.png` | Histogramas de distribuciones y gráficos Top-K |
| `DOCUMENTACION_SEMANA_10.md` | Este documento (documentación completa) |

---

## 11. Apéndice: Diccionario de Variables

### Variables de Entrada

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `X_test_cluster_scaled` | ndarray | Features escaladas de test (PCA + clustering) |
| `y_test` | Series | Etiquetas de fraude reales (0/1) |
| `df_model_test` | DataFrame | Datos de contexto empresarial |
| `best_xgb` | Modelo | XGBoost optimizado (GridSearchCV) |

### Variables de Salida

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `candidate_pool` | DataFrame | Pool de 21,000 transacciones con scores |
| `y_score_baseline` | array | Scores del baseline [0, 1] |
| `y_score_strong` | array | Scores XGBoost [0, 1] |
| `evaluation_report` | DataFrame | Métricas de rendimiento |
| `topk_report` | DataFrame | Análisis de ranking |

---

## 12. Referencias y Documentos Relacionados

- [Week 5: Feature Engineering](../week5/README.md)
- [Week 7: Clustering Analysis](../week7/README.md)
- [Interpretación Técnica](./interpretacion_week10.txt)
- [Dataset de Training](../raw/fraudTrain.csv)
- [Dataset de Testing](../raw/fraudTest.csv)

---

**Documento generado:** Junio 2026  
**Versión:** 1.0 Final  
**Estado:** ✅ Listo para Entrega
