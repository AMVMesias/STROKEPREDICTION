# 🧠 Predicción de Accidente Cerebrovascular (ACV)
## Actividad Integradora - Comparación de Modelos Supervisados

**Dataset:** Stroke Prediction Dataset (Kaggle)  
**Modelos:** KNN, Perceptrón, MLP  
**Equipo:** Mesías Mariscal, Denise Rea, Julio Viche

---

## 1. Introducción y Descripción del Problema

### 1.1 Planteamiento del Problema

El **accidente cerebrovascular (ACV)** es una de las principales causas de muerte y discapacidad a nivel mundial. Según la Organización Mundial de la Salud, aproximadamente 15 millones de personas sufren un ACV anualmente, y de ellos, 5 millones mueren y otros 5 millones quedan con discapacidad permanente. **Predecir el riesgo de ACV** es crucial para implementar intervenciones médicas preventivas. Este proyecto busca desarrollar modelos de clasificación supervisada que, utilizando variables clínicas como edad, hipertensión, nivel de glucosa e IMC, puedan identificar pacientes con alto riesgo de sufrir un ACV, permitiendo a los profesionales de la salud tomar acciones preventivas oportunas.

### 1.2 Descripción del Dataset

| Característica | Valor |
|----------------|-------|
| **Nombre** | Stroke Prediction Dataset |
| **Fuente** | Kaggle |
| **Registros** | 5,110 |
| **Variables** | 12 (11 predictoras + 1 objetivo) |
| **Variable objetivo** | `stroke` (0 = No ACV, 1 = ACV) |

### 1.3 Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id` | Numérica | Identificador único del paciente |
| `gender` | Categórica | Género del paciente (Male, Female, Other) |
| `age` | Numérica | Edad del paciente en años |
| `hypertension` | Binaria | 0 = sin hipertensión, 1 = con hipertensión |
| `heart_disease` | Binaria | 0 = sin enfermedad cardíaca, 1 = con enfermedad |
| `ever_married` | Categórica | Estado civil (Yes/No) |
| `work_type` | Categórica | Tipo de empleo (Private, Self-employed, Govt_job, children, Never_worked) |
| `Residence_type` | Categórica | Tipo de residencia (Urban/Rural) |
| `avg_glucose_level` | Numérica | Nivel promedio de glucosa en sangre (mg/dL) |
| `bmi` | Numérica | Índice de masa corporal |
| `smoking_status` | Categórica | Estado de fumador (formerly smoked, never smoked, smokes, Unknown) |
| `stroke` | **Objetivo** | **1 = tuvo ACV, 0 = no tuvo ACV** |

---

## 3. EDA y Preprocesamiento

### 3.1 Carga y Exploración Inicial
- Se cargó el dataset y se revisaron dimensiones, columnas y primeras filas.
- Se analizaron tipos de datos y estadísticas descriptivas.

### 3.2 Análisis de Valores Faltantes
- Se identificó que la variable `bmi` tenía valores faltantes.
- Se imputó `bmi` con la mediana, por ser más robusta ante outliers y adecuada para su distribución sesgada.

### 3.3 Distribución de la Variable Objetivo
- El dataset está altamente desbalanceado: la clase positiva (stroke=1) es minoritaria.

### 3.4 Visualizaciones EDA
- Histogramas para `age`, `avg_glucose_level`, `bmi`.
- Gráficos de barras para la variable objetivo y relaciones simples (`stroke` vs `hypertension`, `stroke` vs `smoking_status`).

### 3.5 Preprocesamiento de Datos
- Eliminación de columna `id`.
- Imputación de `bmi` con la mediana.
- Codificación One-Hot de variables categóricas.
- Partición estratificada 80/20 en train/test.
- Estandarización de variables numéricas (`age`, `avg_glucose_level`, `bmi`) con StandardScaler.
- Balanceo de clases en entrenamiento usando SMOTE.

---

## 4. Modelos y Entrenamiento

Se entrenaron tres modelos con validación cruzada:
1. **KNN** con k = {3, 5, 7, 9, 11}
2. **Perceptrón** con diferentes `max_iter` y `eta0`
3. **MLP** con diferentes arquitecturas de capas ocultas

---

## 5. Evaluación y Comparación

- Se compararon los modelos usando Accuracy, Precision, Recall y F1-Score.
- Se priorizó el Recall por el contexto clínico (minimizar falsos negativos).
- Se presentaron matrices de confusión y gráficos comparativos.

---

## 6. Conclusiones y Modelo Recomendado

- El modelo recomendado fue seleccionado considerando métricas y contexto clínico.
- Se priorizó el Recall para la clase positiva (stroke=1).
- Se justificó la elección considerando la importancia de minimizar falsos negativos.
- Se discutieron ventajas y limitaciones de cada modelo.

---

## Reflexión sobre Preprocesamiento

- El escalado de variables fue fundamental para KNN y MLP.
- La imputación de valores faltantes evitó la pérdida de datos.
- El desbalance de clases afectó las métricas; SMOTE mejoró el Recall.
- Se sugirieron posibles mejoras: feature engineering, probar otros algoritmos, ajustar umbral, validación externa.

---

## Resumen Final

| Aspecto | Descripción |
|---------|-------------|
| **Dataset** | Stroke Prediction Dataset (5,110 registros, 12 variables) |
| **Preprocesamiento** | Imputación BMI (mediana), One-Hot Encoding, StandardScaler, SMOTE |
| **Modelos** | KNN (k óptimo), Perceptrón (hiperparámetros óptimos), MLP (arquitectura óptima) |
| **Métrica Principal** | Recall (contexto clínico - detectar pacientes en riesgo) |
| **Validación** | Train/Test 80/20 estratificado + Cross-Validation 5-fold |

---

## Justificación del Modelo Recomendado

El modelo recomendado fue seleccionado considerando tanto las métricas cuantitativas (recall, F1-score, accuracy, precision) como el contexto clínico del problema. En la predicción de ACV, el recall es especialmente importante, ya que permite identificar la mayor cantidad posible de pacientes en riesgo, minimizando los falsos negativos. Un falso negativo puede tener consecuencias graves en la salud del paciente, mientras que un falso positivo solo implica exámenes adicionales. Además, se consideró la estabilidad del modelo, su interpretabilidad y el balance entre precisión y sensibilidad. Por estas razones, el modelo seleccionado ofrece el mejor compromiso entre desempeño y aplicabilidad clínica.

- **Métricas clave:**
  - Recall y F1-score altos para la clase positiva (stroke=1)
  - Buen balance con precisión y accuracy
- **Contexto clínico:**
  - Prioridad en minimizar falsos negativos
  - Modelo robusto y aplicable en la práctica médica
- **Otros factores:**
  - Interpretabilidad y facilidad de implementación
  - Costo computacional razonable

## Limitaciones y Posibles Mejoras

- **Desbalance de clases:** Aunque se aplicó SMOTE para balancear el conjunto de entrenamiento, el dataset original presenta una fuerte desproporción entre clases. Esto puede afectar la generalización del modelo y la interpretación de las métricas.
- **Tamaño del dataset:** El número de registros es limitado para un problema clínico, lo que puede restringir la capacidad de los modelos para aprender patrones complejos y generalizar a nuevos datos.
- **Variables disponibles:** El dataset solo incluye variables clínicas básicas. Incluir información adicional (historial médico, hábitos, genética) podría mejorar la predicción.
- **Modelos probados:** Solo se evaluaron KNN, Perceptrón y MLP. Probar otros algoritmos como Random Forest, XGBoost o SVM podría aportar mejoras.
- **Ajuste de umbral:** Se utilizó el umbral estándar de 0.5 para clasificación. Ajustar este valor podría optimizar el recall o la precisión según el objetivo clínico.
- **Validación externa:** Los resultados deben validarse con datos de otros hospitales o cohortes para asegurar la robustez del modelo.

**Posibles mejoras:**
- Probar técnicas avanzadas de balanceo de clases
- Realizar feature engineering para crear nuevas variables
- Ajustar hiperparámetros con búsqueda más exhaustiva
- Implementar interpretabilidad de modelos (SHAP, LIME)
- Validar el modelo en datos reales y prospectivos
