
# Actividad de Aprendizaje Integradora

## Tema
Comparación de modelos supervisados con preprocesamiento de datos

**Dataset:** Stroke Prediction Dataset (Kaggle)

---

## 1. Contexto de la Actividad

En clase se han revisado:
- Aprendizaje supervisado (clasificación)
- K-Nearest Neighbors (KNN)
- Perceptrón
- Redes neuronales (MLP)
- Preprocesamiento de datos (limpieza, imputación, codificación, normalización/estandarización)
- Validación cruzada

En esta actividad integradora trabajarán con el **Stroke Prediction Dataset**, un conjunto de datos clínicos con 5,110 registros y 12 variables (edad, género, hipertensión, enfermedad cardíaca, tipo de trabajo, nivel de glucosa, IMC, hábito de fumar y la variable objetivo `stroke`: 1 = tuvo ACV, 0 = no).

**Objetivo:** Construir y comparar modelos supervisados para predecir el riesgo de sufrir un ACV, y recomendar un modelo justificando la decisión con métricas y el contexto del problema.

---

## 2. Dataset

Cada equipo deberá:
1. Crear una cuenta (si es necesario) e ingresar a Kaggle.
2. Buscar el dataset: “Stroke Prediction Dataset”.
3. Descargar el archivo `healthcare-dataset-stroke-data.csv` y cargarlo en su entorno de trabajo (Jupyter/Colab).

---

## 3. Objetivos Específicos

Al finalizar, el equipo deberá ser capaz de:
1. Implementar un pipeline completo de clasificación: **EDA → preprocesamiento → partición → entrenamiento → evaluación**.
2. Entrenar y comparar tres modelos:
	- KNN
	- Perceptrón
	- Red neuronal (MLP)
3. Calcular e interpretar métricas de clasificación (accuracy, precision, recall, F1, matriz de confusión).
4. Proponer y justificar una recomendación de modelo para el problema de predicción de ACV.

---

## 4. Actividades a Realizar

### 4.1 Organización del Equipo
- Formar equipos de 3 estudiantes.

### 4.2 Paso 1: Comprensión del Problema y del Dataset
En el notebook:
1. Redactar un planteamiento breve del problema (4–6 líneas):
	- ¿Qué significa predecir stroke?
	- ¿Por qué es importante detectar pacientes de alto riesgo?
2. Describir el dataset:
	- Número de registros y columnas.
	- Listado de variables con una frase explicativa (ejemplo: age, hypertension, bmi, smoking_status, etc.).
	- Identificar la variable objetivo: `stroke` (0/1).

### 4.3 Paso 2: Análisis Exploratorio y Preprocesamiento
1. **EDA básico:**
	- `head()`, `info()`, `describe()`
	- Conteo de valores faltantes (especialmente en `bmi`)
	- Distribución de la variable `stroke` (ver si el dataset está desbalanceado)
2. **Gráficos sugeridos:**
	- Histograma de `age`, `avg_glucose_level`, `bmi`
	- Gráfico de barras de `stroke` (0 vs 1)
	- Alguna relación simple (ej. `stroke` vs `hypertension`, `stroke` vs `smoking_status`)
3. **Preprocesamiento obligatorio:**
	- **Valores faltantes:** Imputar `bmi` (por ejemplo, con la media o mediana). Justificar brevemente.
	- **Codificación de variables categóricas:** Aplicar One-Hot Encoding o método similar a `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
	- **Escalado de variables numéricas:** Normalización o estandarización de `age`, `avg_glucose_level`, `bmi`. Explicar por qué el escalado es importante para KNN y MLP.
	- **Partición de datos:** Dividir en train/test, por ejemplo 80% / 20% (usando `train_test_split`).

### 4.4 Paso 3: Entrenamiento de Modelos
Entrenar todos los modelos con el mismo conjunto de entrenamiento preprocesado.

#### a) K-Nearest Neighbors (KNN)
- Entrenar un KNN con varios valores de k (por ejemplo: 3, 5, 7, 9, 11)
- Registrar para cada k:
	- Accuracy
	- Precision
	- Recall
	- F1-score

#### b) Perceptrón
- Entrenar un modelo Perceptrón.
- Probar al menos 2 configuraciones de hiperparámetros (diferentes `max_iter`, diferentes `eta0` - tasa de aprendizaje).
- Comentar que el Perceptrón es un modelo lineal y cómo podría afectar eso al problema.

#### c) Red Neuronal (MLPClassifier)
- Entrenar un MLPClassifier con:
	- 1–2 capas ocultas (por ejemplo `(16,)` o `(32,16)`)
	- Probar al menos 2 configuraciones (cambiar `hidden_layer_sizes`, `max_iter` o función de activación)
- Indicar la arquitectura elegida (número de capas y neuronas)
- Usar validación cruzada en el conjunto de entrenamiento para comparar modelos.

### 4.5 Paso 4: Evaluación, Comparación y Recomendación
Para cada modelo (KNN “óptimo”, Perceptrón elegido, MLP elegido):
1. Calcular en el conjunto de prueba:
	- Accuracy
	- Precision
	- Recall
	- F1-score
	- Matriz de confusión
2. Construir una tabla comparativa con filas = modelos y columnas = métricas.
3. Analizar resultados:
	- ¿Qué modelo tiene mejor accuracy?
	- ¿Cuál tiene mejor recall para la clase `stroke = 1`?
	- En un contexto clínico, ¿qué métrica es más importante y por qué?
	- ¿Se observa algún trade-off entre precisión y recall?
4. **Recomendación de modelo:**
	- Elegir un modelo recomendado para que un hospital lo use como apoyo a la decisión.
	- Justificar con:
	  - Métricas (especialmente recall y F1 para la clase positiva)
	  - Argumentos cualitativos (interpretabilidad, costo de cómputo, estabilidad)
5. **Reflexión sobre preprocesamiento:**
	- ¿Qué ocurrió antes y después de escalar?
	- ¿Qué sucedería si no se imputan los valores faltantes?
	- ¿El posible desbalance de clases afectó alguna métrica?

---

## 5. Entregables

Cada equipo debe entregar:

### 1. Notebook Jupyter (.ipynb)
- Código ordenado y comentado
- Secciones:
	1. Introducción y descripción del problema
	2. EDA y preprocesamiento
	3. Modelos y entrenamiento
	4. Evaluación y comparación
	5. Conclusiones y modelo recomendado
- Gráficos y tablas integrados

### 2. Informe corto (2–4 páginas)
Puede ir en celdas Markdown dentro del mismo notebook o como PDF/Word aparte. Debe incluir:
1. Contexto y objetivo del problema
2. Descripción del dataset (Stroke Prediction Dataset de Kaggle)
3. Resumen de preprocesamiento aplicado
4. Descripción de los tres modelos y parámetros principales usados
5. Tabla comparativa de métricas
6. Modelo recomendado y justificación
7. Comentario breve de limitaciones y posibles mejoras

# Actividad de Aprendizaje Integradora

## Tema
Comparación de modelos supervisados con preprocesamiento de datos

**Dataset:** Stroke Prediction Dataset (Kaggle)

---

## 1. Contexto de la Actividad

En clase se han revisado:
- Aprendizaje supervisado (clasificación)
- K-Nearest Neighbors (KNN)
- Perceptrón
- Redes neuronales (MLP)
- Preprocesamiento de datos (limpieza, imputación, codificación, normalización/estandarización)
- Validación cruzada

En esta actividad integradora trabajarán con el **Stroke Prediction Dataset**, un conjunto de datos clínicos con 5,110 registros y 12 variables (edad, género, hipertensión, enfermedad cardíaca, tipo de trabajo, nivel de glucosa, IMC, hábito de fumar y la variable objetivo `stroke`: 1 = tuvo ACV, 0 = no).

**Objetivo:** Construir y comparar modelos supervisados para predecir el riesgo de sufrir un ACV, y recomendar un modelo justificando la decisión con métricas y el contexto del problema.

---

## 2. Dataset

Cada equipo deberá:
1. Crear una cuenta (si es necesario) e ingresar a Kaggle.
2. Buscar el dataset: “Stroke Prediction Dataset”.
3. Descargar el archivo `healthcare-dataset-stroke-data.csv` y cargarlo en su entorno de trabajo (Jupyter/Colab).

---

## 3. Objetivos Específicos

Al finalizar, el equipo deberá ser capaz de:
1. Implementar un pipeline completo de clasificación: **EDA → preprocesamiento → partición → entrenamiento → evaluación**.
2. Entrenar y comparar tres modelos:
	- KNN
	- Perceptrón
	- Red neuronal (MLP)
3. Calcular e interpretar métricas de clasificación (accuracy, precision, recall, F1, matriz de confusión).
4. Proponer y justificar una recomendación de modelo para el problema de predicción de ACV.

---

## 4. Actividades a Realizar

### 4.1 Organización del Equipo
- Formar equipos de 3 estudiantes.

### 4.2 Paso 1: Comprensión del Problema y del Dataset
En el notebook:
1. Redactar un planteamiento breve del problema (4–6 líneas):
	- ¿Qué significa predecir stroke?
	- ¿Por qué es importante detectar pacientes de alto riesgo?
2. Describir el dataset:
	- Número de registros y columnas.
	- Listado de variables con una frase explicativa (ejemplo: age, hypertension, bmi, smoking_status, etc.).
	- Identificar la variable objetivo: `stroke` (0/1).

### 4.3 Paso 2: Análisis Exploratorio y Preprocesamiento
1. **EDA básico:**
	- `head()`, `info()`, `describe()`
	- Conteo de valores faltantes (especialmente en `bmi`)
	- Distribución de la variable `stroke` (ver si el dataset está desbalanceado)
2. **Gráficos sugeridos:**
	- Histograma de `age`, `avg_glucose_level`, `bmi`
	- Gráfico de barras de `stroke` (0 vs 1)
	- Alguna relación simple (ej. `stroke` vs `hypertension`, `stroke` vs `smoking_status`)
3. **Preprocesamiento obligatorio:**
	- **Valores faltantes:** Imputar `bmi` (por ejemplo, con la media o mediana). Justificar brevemente.
	- **Codificación de variables categóricas:** Aplicar One-Hot Encoding o método similar a `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
	- **Escalado de variables numéricas:** Normalización o estandarización de `age`, `avg_glucose_level`, `bmi`. Explicar por qué el escalado es importante para KNN y MLP.
	- **Partición de datos:** Dividir en train/test, por ejemplo 80% / 20% (usando `train_test_split`).

### 4.4 Paso 3: Entrenamiento de Modelos
Entrenar todos los modelos con el mismo conjunto de entrenamiento preprocesado.

#### a) K-Nearest Neighbors (KNN)
- Entrenar un KNN con varios valores de k (por ejemplo: 3, 5, 7, 9, 11)
- Registrar para cada k:
	- Accuracy
	- Precision
	- Recall
	- F1-score

#### b) Perceptrón
- Entrenar un modelo Perceptrón.
- Probar al menos 2 configuraciones de hiperparámetros (diferentes `max_iter`, diferentes `eta0` - tasa de aprendizaje).
- Comentar que el Perceptrón es un modelo lineal y cómo podría afectar eso al problema.

#### c) Red Neuronal (MLPClassifier)
- Entrenar un MLPClassifier con:
	- 1–2 capas ocultas (por ejemplo `(16,)` o `(32,16)`)
	- Probar al menos 2 configuraciones (cambiar `hidden_layer_sizes`, `max_iter` o función de activación)
- Indicar la arquitectura elegida (número de capas y neuronas)
- Usar validación cruzada en el conjunto de entrenamiento para comparar modelos.

### 4.5 Paso 4: Evaluación, Comparación y Recomendación
Para cada modelo (KNN “óptimo”, Perceptrón elegido, MLP elegido):
1. Calcular en el conjunto de prueba:
	- Accuracy
	- Precision
	- Recall
	- F1-score
	- Matriz de confusión
2. Construir una tabla comparativa con filas = modelos y columnas = métricas.
3. Analizar resultados:
	- ¿Qué modelo tiene mejor accuracy?
	- ¿Cuál tiene mejor recall para la clase `stroke = 1`?
	- En un contexto clínico, ¿qué métrica es más importante y por qué?
	- ¿Se observa algún trade-off entre precisión y recall?
4. **Recomendación de modelo:**
	- Elegir un modelo recomendado para que un hospital lo use como apoyo a la decisión.
	- Justificar con:
	  - Métricas (especialmente recall y F1 para la clase positiva)
	  - Argumentos cualitativos (interpretabilidad, costo de cómputo, estabilidad)
5. **Reflexión sobre preprocesamiento:**
	- ¿Qué ocurrió antes y después de escalar?
	- ¿Qué sucedería si no se imputan los valores faltantes?
	- ¿El posible desbalance de clases afectó alguna métrica?

---

## 5. Entregables

Cada equipo debe entregar:

### 1. Notebook Jupyter (.ipynb)
- Código ordenado y comentado
- Secciones:
	1. Introducción y descripción del problema
	2. EDA y preprocesamiento
	3. Modelos y entrenamiento
	4. Evaluación y comparación
	5. Conclusiones y modelo recomendado
- Gráficos y tablas integrados

### 2. Informe corto (2–4 páginas)
Puede ir en celdas Markdown dentro del mismo notebook o como PDF/Word aparte. Debe incluir:
1. Contexto y objetivo del problema
2. Descripción del dataset (Stroke Prediction Dataset de Kaggle)
3. Resumen de preprocesamiento aplicado
4. Descripción de los tres modelos y parámetros principales usados
5. Tabla comparativa de métricas
6. Modelo recomendado y justificación
7. Comentario breve de limitaciones y posibles mejoras