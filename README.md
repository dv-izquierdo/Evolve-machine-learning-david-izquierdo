# 🛍️ Customer Repurchase Prediction — Online Retail II

Modelo de clasificación supervisada para predecir la probabilidad de que un cliente vuelva a comprar en una tienda de retail online en los próximos **3 meses**, a partir de su historial de transacciones.

---

## 📦 Dataset

**Online Retail II** — [Kaggle](https://www.kaggle.com/datasets/ineubytes/online-retail-ecommerce-dataset)

Conjunto de datos con todas las transacciones realizadas por un comercio minorista online con sede en el Reino Unido entre el **1 de diciembre de 2009** y el **9 de diciembre de 2011**. La empresa vende principalmente artículos de regalo únicos para cualquier ocasión, con una base de clientes predominantemente mayorista.

| Campo | Tipo | Descripción |
|---|---|---|
| `Invoice` | Nominal | Número de factura (6 dígitos). Prefijo `C` = cancelación |
| `StockCode` | Nominal | Código único de producto (5 dígitos) |
| `Description` | Nominal | Nombre del artículo |
| `Quantity` | Numérico | Unidades por transacción |
| `InvoiceDate` | Datetime | Fecha y hora de la transacción |
| `Price` | Numérico | Precio unitario en libras esterlinas (£) |
| `Customer ID` | Nominal | Identificador único de cliente (5 dígitos) |
| `Country` | Nominal | País de residencia del cliente |

**Volumen tras limpieza:** ~1.06 M líneas · 4,372 clientes únicos identificados · 24 meses de histórico

---

## 🔄 Pasos realizados

### 1. Análisis Exploratorio (EDA)

- Inspección inicial del dataframe: tipos de datos, dimensiones, muestra aleatoria.
- **Calidad de datos:** detección y eliminación de filas duplicadas, nulos en `Customer ID` y `Description`.
- **Separación de cancelaciones:** facturas con prefijo `C` extraídas para análisis paralelo.
- **Eliminación de outliers** en `Quantity` y `Price` mediante método IQR (multiplicador 1.5).
- Filtrado de precios `≤ 0` (registros erróneos).
- **Agrupación de países** con presencia residual (<0.5% de transacciones) bajo la categoría `Others`.
- Creación de la columna `Revenue = Quantity × Price`.
- Visualizaciones clave:
  - Raincloud plots de `Quantity` y `Price`.
  - Serie temporal de ingresos mensuales (pico estacional Q4: sep–nov).
  - Análisis de cancelaciones por cliente.

### 2. Preparación de datos y Feature Engineering

- **Fecha de corte:** `2011-08-01`. Los datos anteriores se usan para construir las features; los datos posteriores para etiquetar la variable objetivo.
- **Variable objetivo `Purchase`:** `1` si el cliente realizó al menos una compra entre ago–nov 2011, `0` en caso contrario.
- **Agregación por cliente** (excluido el año 2009 por datos parciales):

| Feature | Descripción |
|---|---|
| `Total_tickets` | Número medio anual de facturas distintas |
| `Avg_Ticket_Revenue` | Ticket medio en £ por compra |
| `Avg_Ticket_Quantity` | Cantidad media de artículos por ticket |
| `total_cancellations` | Total de líneas canceladas |
| `Recency` | Días transcurridos desde la última compra hasta la fecha de corte |
| `Is_Wholesale` | Flag mayorista (top 25% en gasto y cantidad) |
| `Country` | País del cliente |

- Unión con estadísticas de cancelaciones (`.merge` left join, nulos → 0).
- **One-hot encoding** de `Country` (drop_first=True, categoría de referencia: primera alfabéticamente).
- Eliminación de `Total_Revenue` y `Customer ID` para evitar data leakage y redundancias.
- **Normalización** de variables numéricas con `StandardScaler`.
- Análisis de correlación entre variables mediante heatmap.

### 3. Modelado

- **Split:** 80% train / 20% test, estratificado por la variable objetivo (`random_state=42`).
- **Validación cruzada:** `StratifiedKFold` con 5 folds.
- Modelos entrenados y comparados:

| Modelo | CV AUC |
|---|---|
| Logistic Regression | ~0.76 |
| Random Forest | ~0.81 |
| Gradient Boosting | **~0.87** |
| XGBoost | ~0.86 |

- Métricas evaluadas: AUC-ROC, Average Precision, Classification Report, Confusion Matrix.

### 4. Optimización de hiperparámetros

- **Optuna** con 50 trials sobre `GradientBoostingClassifier`, optimizando:
  - `n_estimators` (100–500)
  - `max_depth` (3–10)
  - `learning_rate` (0.01–0.1, log scale)
- Dirección de optimización: maximizar AUC en validación cruzada.

### 5. Explicabilidad

- **Dalex Explainer** sobre el modelo final para obtener importancia global de variables (`model_parts()`).
- Variables más relevantes por importancia media en permutación:

| Ranking | Variable | Importancia relativa |
|---|---|---|
| 1 | `Recency` | 34% |
| 2 | `Total_tickets` (Frecuencia) | 27% |
| 3 | `Avg_Ticket_Revenue` (Monetario) | 19% |
| 4 | `total_cancellations` | 12% |
| 5 | `Country_*` / Estacionalidad | 8% |

---

## ✅ Conclusiones

**El modelo de Gradient Boosting optimizado alcanza un AUC de 0.89**, superando significativamente al clasificador aleatorio (0.50) y a los modelos de referencia (Logistic Regression: 0.76, Random Forest: 0.81).

Los principales aprendizajes del proyecto son:

- **La recencia es el predictor dominante.** Un cliente que no ha comprado en más de 90 días tiene una probabilidad ~4× menor de volver. La velocidad de reacción comercial es crítica.

- **Solo el ~30% de los clientes repite en el siguiente trimestre.** Identificar correctamente este segmento permite focalizar recursos comerciales donde hay mayor retorno esperado.

- **Los clientes recurrentes generan 5× más revenue** que los compradores únicos. Proteger esta cohorte tiene un impacto desproporcionado en la cuenta de resultados.

- **El pico estacional de Q4 (sep–nov) concentra el 38% de las transacciones anuales.** El modelo debe reentrenarse con datos actualizados antes de este período para maximizar su utilidad.

- **Los scores son directamente accionables:** exportar la columna de probabilidad al CRM permite segmentar la base en *"clientes a fidelizar"* (prob ≥ 0.50) y *"clientes a retener"* (prob < 0.50) con estrategias diferenciadas.

### Próximos pasos sugeridos

- Piloto A/B con 200 clientes para validar el impacto incremental de las acciones de fidelización.
- Reentrenamiento mensual del modelo con nuevas transacciones para mantener la calidad predictiva.
- Enriquecimiento del dataset con datos de satisfacción (NPS) y devoluciones para mejorar las features.
- Explorar modelos de supervivencia (e.g. BG/NBD) como alternativa probabilística para la predicción de churn.

---

## 🗂️ Estructura del repositorio

```
├── online_retail_II_EDA-basic.ipynb   # Notebook principal (EDA + modelo)
├── README.md                          # Este documento
└── Customer_Repurchase_Model.pdf # Presentación para el equipo de negocio
```

---

## 🛠️ Librerías principales

```
pandas · numpy · matplotlib · seaborn · scikit-learn · xgboost · optuna · dalex · ptitprince
```

---
