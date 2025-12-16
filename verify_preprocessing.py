# Script de verificación de preprocesamiento
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("VERIFICACIÓN DE PREPROCESAMIENTO - STROKE PREDICTION")
print("="*60)

# 1. Cargar dataset
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
print(f"\n✓ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# 2. Exploración inicial
print("\n" + "="*60)
print("PRIMERAS 5 FILAS")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("TIPOS DE DATOS")
print("="*60)
print(df.dtypes)

print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60)
print(df.describe())

# 3. Valores faltantes
print("\n" + "="*60)
print("ANÁLISIS DE VALORES FALTANTES")
print("="*60)
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Verificar BMI con 'N/A' como string
if df['bmi'].dtype == 'object':
    bmi_na_count = (df['bmi'] == 'N/A').sum()
    print(f"\nBMI con valor 'N/A' (string): {bmi_na_count}")
else:
    bmi_na_count = df['bmi'].isna().sum()
    print(f"\nBMI con NaN: {bmi_na_count}")

# 4. Distribución de stroke
print("\n" + "="*60)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (STROKE)")
print("="*60)
stroke_counts = df['stroke'].value_counts()
print(f"Clase 0 (No ACV): {stroke_counts[0]} ({stroke_counts[0]/len(df)*100:.2f}%)")
print(f"Clase 1 (ACV):    {stroke_counts[1]} ({stroke_counts[1]/len(df)*100:.2f}%)")
print(f"⚠️  Ratio de desbalance: {stroke_counts[0]/stroke_counts[1]:.1f}:1")

# 5. Preprocesamiento
print("\n" + "="*60)
print("APLICANDO PREPROCESAMIENTO")
print("="*60)

df_processed = df.copy()

# 5.1 Eliminar ID
df_processed = df_processed.drop('id', axis=1)
print("✓ Columna 'id' eliminada")

# 5.2 Imputar BMI
df_processed['bmi'] = df_processed['bmi'].replace('N/A', np.nan)
df_processed['bmi'] = pd.to_numeric(df_processed['bmi'])
bmi_median = df_processed['bmi'].median()
df_processed['bmi'] = df_processed['bmi'].fillna(bmi_median)
print(f"✓ BMI imputado con mediana: {bmi_median:.2f}")

# 5.3 One-Hot Encoding
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
print(f"✓ One-Hot Encoding aplicado. Nuevas columnas: {df_processed.shape[1]}")

# 5.4 Separar X e y
X = df_processed.drop('stroke', axis=1)
y = df_processed['stroke']
print(f"✓ Separación X/y: X={X.shape}, y={y.shape}")

# 5.5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Split estratificado: Train={len(X_train)}, Test={len(X_test)}")

# 5.6 Escalado
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("✓ StandardScaler aplicado")

# 6. Verificación final
print("\n" + "="*60)
print("VERIFICACIÓN FINAL")
print("="*60)
print(f"\nDimensiones X_train: {X_train_scaled.shape}")
print(f"Dimensiones X_test:  {X_test_scaled.shape}")

print("\n--- Distribución en Train ---")
print(f"Clase 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"Clase 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")

print("\n--- Distribución en Test ---")
print(f"Clase 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"Clase 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

print("\n--- Estadísticas después del escalado (Train) ---")
print(X_train_scaled[numerical_cols].describe().loc[['mean', 'std']].round(4))

print("\n--- Columnas finales del dataset ---")
print(list(X_train_scaled.columns))

print("\n" + "="*60)
print("✅ PREPROCESAMIENTO COMPLETADO CORRECTAMENTE")
print("="*60)
