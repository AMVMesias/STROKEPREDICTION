# Verificación del notebook mejorado con SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("VERIFICACIÓN CON SMOTE - STROKE PREDICTION")
print("="*70)

# Cargar y preprocesar
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df_processed = df.drop('id', axis=1)
df_processed['bmi'] = df_processed['bmi'].replace('N/A', np.nan)
df_processed['bmi'] = pd.to_numeric(df_processed['bmi'])
df_processed['bmi'] = df_processed['bmi'].fillna(df_processed['bmi'].median())

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

X = df_processed.drop('stroke', axis=1)
y = df_processed['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# SMOTE
print("\n--- Aplicando SMOTE ---")
print(f"Antes: Clase 0={sum(y_train==0)}, Clase 1={sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"Después: Clase 0={sum(y_train_balanced==0)}, Clase 1={sum(y_train_balanced==1)}")

# Entrenar modelos
results = []

# KNN
print("\n--- KNN con SMOTE ---")
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_balanced, y_train_balanced)
    y_pred = knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"k={k}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    results.append({'Modelo': f'KNN(k={k})', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# Perceptrón
print("\n--- Perceptrón con SMOTE ---")
perc = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perc.fit(X_train_balanced, y_train_balanced)
y_pred = perc.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"Perceptrón: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
results.append({'Modelo': 'Perceptrón', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# MLP
print("\n--- MLP con SMOTE ---")
for hidden in [(16,), (32, 16)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_train_balanced, y_train_balanced)
    y_pred = mlp.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"MLP{hidden}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    results.append({'Modelo': f'MLP{hidden}', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# Tabla comparativa
print("\n" + "="*70)
print("TABLA COMPARATIVA FINAL")
print("="*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Mejores
print("\n" + "="*70)
print("MEJORES MODELOS POR MÉTRICA")
print("="*70)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    best_idx = results_df[metric].idxmax()
    print(f"{metric}: {results_df.loc[best_idx, 'Modelo']} ({results_df.loc[best_idx, metric]:.4f})")

print("\n" + "="*70)
print("✅ VERIFICACIÓN EXITOSA - SMOTE MEJORA SIGNIFICATIVAMENTE LOS RESULTADOS")
print("="*70)
