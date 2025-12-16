import os
import warnings
import urllib3
import pickle
import json
import tempfile
from datetime import datetime

print("="*80)
print("MLFLOW IRIS CLASSIFICATION - ОБУЧЕНИЕ НА ВМ")
print("="*80)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')


os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"

if 'MLFLOW_TRACKING_SERVER_CERT_PATH' in os.environ:
    del os.environ['MLFLOW_TRACKING_SERVER_CERT_PATH']

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

MODELS_DIR = os.path.join(tempfile.gettempdir(), "mlflow_models")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"\n[КОНФИГУРАЦИЯ]")
print(f"  - SSL проверка: отключена")
print(f"  - Папка для моделей: {MODELS_DIR}")


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_HOST_HEADER = "mlflow.labs.itmo.loc"
EXPERIMENT_NAME = "Iris Classification Training"

import requests

original_session_init = requests.Session.__init__

def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.headers.update({'Host': MLFLOW_HOST_HEADER})

requests.Session.__init__ = patched_session_init

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

print(f"\n[MLFLOW СЕРВЕР]")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Host Header: {MLFLOW_HOST_HEADER}")
print(f"  - Эксперимент: {EXPERIMENT_NAME}")


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[ДАТАСЕТ]")
print(f"  - Train: {X_train.shape}")
print(f"  - Test:  {X_test.shape}")


print("\n" + "="*80)
print("ШАГ 1: ОБУЧЕНИЕ LOGISTIC REGRESSION")
print("="*80)

params_lr = {
    "solver": "lbfgs",
    "max_iter": 200,
    "random_state": 42,
}

lr_model = LogisticRegression(**params_lr)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n[МЕТРИКИ LOGISTIC REGRESSION]")
print(f"  - Accuracy:  {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall:    {recall:.4f}")
print(f"  - F1-score:  {f1:.4f}")


print("\n" + "="*80)
print("ШАГ 2: ЛОГИРОВАНИЕ В MLFLOW")
print("="*80)

try:
    with mlflow.start_run(run_name="LogisticRegression_Iris") as run:
        mlflow.log_params(params_lr)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("framework", "scikit-learn")

        run_id_lr = run.info.run_id

        print(f"\n✓ Параметры и метрики залогированы")
        print(f"  - Run ID: {run_id_lr}")

except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    raise


print("\n" + "="*80)
print("ШАГ 3: СОХРАНЕНИЕ МОДЕЛИ ЛОКАЛЬНО")
print("="*80)

model_lr_path = os.path.join(MODELS_DIR, "iris_logistic_regression.pkl")
with open(model_lr_path, 'wb') as f:
    pickle.dump(lr_model, f)

print(f"\n✓ Logistic Regression сохранена")
print(f"  - Путь: {model_lr_path}")

metadata_lr = {
    "model_name": "iris_logistic_regression",
    "model_type": "LogisticRegression",
    "framework": "scikit-learn",
    "parameters": params_lr,
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "dataset": "Iris",
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "timestamp": datetime.now().isoformat(),
    "run_id": run_id_lr
}

metadata_lr_path = os.path.join(MODELS_DIR, "iris_logistic_regression_metadata.json")
with open(metadata_lr_path, 'w') as f:
    json.dump(metadata_lr, f, indent=2)

print(f"✓ Метаданные сохранены")
print(f"  - Путь: {metadata_lr_path}")


print("\n" + "="*80)
print("ШАГ 4: ТЕСТИРОВАНИЕ МОДЕЛИ")
print("="*80)

try:
    with open(model_lr_path, 'rb') as f:
        loaded_model_lr = pickle.load(f)

    y_pred_loaded = loaded_model_lr.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)

    print(f"\n✓ Модель загружена и протестирована")
    print(f"  - Accuracy: {accuracy_loaded:.4f}")
    print(f"  - Совпадает с оригинальной: {accuracy_loaded == accuracy}")

    print(f"\n[ПРИМЕРЫ ПРЕДСКАЗАНИЙ]")
    for i, (pred, actual) in enumerate(zip(y_pred_loaded[:5], y_test[:5]), 1):
        pred_name = iris.target_names[pred]
        actual_name = iris.target_names[actual]
        status = "✓" if pred == actual else "✗"
        print(f"  {status} Образец {i}: {pred_name:15} (факт: {actual_name})")

except Exception as e:
    print(f"\n✗ ОШИБКА при загрузке: {e}")
    raise


print("\n" + "="*80)
print("ШАГ 5: ОБУЧЕНИЕ RANDOM FOREST")
print("="*80)

params_rf = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}

rf_model = RandomForestClassifier(**params_rf)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\n[МЕТРИКИ RANDOM FOREST]")
print(f"  - Accuracy:  {accuracy_rf:.4f}")
print(f"  - Precision: {precision_rf:.4f}")
print(f"  - Recall:    {recall_rf:.4f}")
print(f"  - F1-score:  {f1_rf:.4f}")

try:
    with mlflow.start_run(run_name="RandomForest_Iris") as run:
        mlflow.log_params(params_rf)
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.log_metric("precision", precision_rf)
        mlflow.log_metric("recall", recall_rf)
        mlflow.log_metric("f1_score", f1_rf)

        mlflow.set_tag("model_type", "Random Forest")
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("framework", "scikit-learn")

        run_id_rf = run.info.run_id

        print(f"\n✓ Random Forest параметры залогированы")
        print(f"  - Run ID: {run_id_rf}")

except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    raise

model_rf_path = os.path.join(MODELS_DIR, "iris_random_forest.pkl")
with open(model_rf_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"\n✓ Random Forest сохранена")
print(f"  - Путь: {model_rf_path}")

metadata_rf = {
    "model_name": "iris_random_forest",
    "model_type": "RandomForestClassifier",
    "framework": "scikit-learn",
    "parameters": params_rf,
    "metrics": {
        "accuracy": float(accuracy_rf),
        "precision": float(precision_rf),
        "recall": float(recall_rf),
        "f1_score": float(f1_rf)
    },
    "dataset": "Iris",
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "timestamp": datetime.now().isoformat(),
    "run_id": run_id_rf
}

metadata_rf_path = os.path.join(MODELS_DIR, "iris_random_forest_metadata.json")
with open(metadata_rf_path, 'w') as f:
    json.dump(metadata_rf, f, indent=2)

print(f"✓ Метаданные сохранены")
print(f"  - Путь: {metadata_rf_path}")
