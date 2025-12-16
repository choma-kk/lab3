import os
import warnings
import urllib3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("="*80)
print("4. ЗАГРУЗКА И ТЕСТИРОВАНИЕ МОДЕЛИ ИЗ MLFLOW REGISTRY")
print("="*80)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')


os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"

if 'MLFLOW_TRACKING_SERVER_CERT_PATH' in os.environ:
    del os.environ['MLFLOW_TRACKING_SERVER_CERT_PATH']

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_HOST_HEADER = "mlflow.labs.itmo.loc"

print(f"\n[КОНФИГУРАЦИЯ]")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Host Header: {MLFLOW_HOST_HEADER}")


import requests

original_session_init = requests.Session.__init__

def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.headers.update({'Host': MLFLOW_HOST_HEADER})

requests.Session.__init__ = patched_session_init


import mlflow
import mlflow.pyfunc

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)


print("\n" + "="*80)
print("ПОДГОТОВКА ТЕСТОВЫХ ДАННЫХ")
print("="*80)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Данные загружены")
print(f"  - Training set: {X_train.shape[0]} образцов")
print(f"  - Test set: {X_test.shape[0]} образцов")
print(f"  - Features: {X_train.shape[1]}")


print("\n" + "="*80)
print("ЗАГРУЗКА МОДЕЛЕЙ ИЗ REGISTRY")
print("="*80)

models_to_test = [
    "iris-logistic-regression",
    "iris-random-forest"
]

results = {}

for model_name in models_to_test:
    print(f"\n[{model_name}]")

    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"  ✓ Модель загружена")
        print(f"    - URI: {model_uri}")


        print(f"\n  [ТЕСТ 1: Предсказание на тестовом наборе]")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"    - Accuracy:  {accuracy:.4f}")
        print(f"    - Precision: {precision:.4f}")
        print(f"    - Recall:    {recall:.4f}")
        print(f"    - F1-score:  {f1:.4f}")

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


        print(f"\n  [ТЕСТ 2: Матрица ошибок]")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n    Матрица ошибок (3x3 для 3 классов):")
        print(f"    Строки = истинные классы, столбцы = предсказанные")

        for i, row in enumerate(cm):
            print(f"    Класс {i}: {row}")


        print(f"\n  [ТЕСТ 3: Отчет по классам]")

        report = classification_report(y_test, y_pred, target_names=iris.target_names)
        print(f"\n{report}")


        print(f"  [ТЕСТ 4: Примеры предсказаний (первые 10)]")

        for i in range(min(10, len(X_test))):
            pred_class = y_pred[i]
            true_class = y_test[i]
            pred_name = iris.target_names[pred_class]
            true_name = iris.target_names[true_class]

            status = "✓" if pred_class == true_class else "✗"
            print(f"    {status} Образец {i+1}: предсказ = {pred_name:15} | факт = {true_name}")


        print(f"\n  [ТЕСТ 5: Проверка переобучения]")

        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"    - Accuracy на train: {accuracy_train:.4f}")
        print(f"    - Accuracy на test:  {accuracy:.4f}")

        overfit_ratio = (accuracy_train - accuracy) / accuracy * 100
        print(f"    - Переобучение: {overfit_ratio:+.2f}%")

        if overfit_ratio > 10:
            print(f"      ⚠ Возможно переобучение!")
        else:
            print(f"      ✓ Модель в норме")

    except Exception as e:
        print(f"  ✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


print("\n" + "="*80)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*80)

if len(results) > 1:
    print("\n[Сравнение метрик]")

    for model_name, metrics in results.items():
        print(f"\n  {model_name}:")
        for metric, value in metrics.items():
            print(f"    - {metric}: {value:.4f}")

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n  🏆 Лучшая модель: {best_model[0]} (accuracy: {best_model[1]['accuracy']:.4f})")
