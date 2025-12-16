import os
import warnings
import urllib3

print("="*80)
print("5. СРАВНЕНИЕ ЗАПУСКОВ В MLFLOW")
print("="*80)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')


os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"

if 'MLFLOW_TRACKING_SERVER_CERT_PATH' in os.environ:
    del os.environ['MLFLOW_TRACKING_SERVER_CERT_PATH']

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_HOST_HEADER = "mlflow.labs.itmo.loc"
EXPERIMENT_NAME = "Iris Classification Training"

print(f"\n[КОНФИГУРАЦИЯ]")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Эксперимент: {EXPERIMENT_NAME}")


import requests

original_session_init = requests.Session.__init__

def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.headers.update({'Host': MLFLOW_HOST_HEADER})

requests.Session.__init__ = patched_session_init


import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


print("\n" + "="*80)
print("ПОЛУЧЕНИЕ ВСЕХ RUNS")
print("="*80)

try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"✗ Эксперимент не найден: {EXPERIMENT_NAME}")
        exit(1)

    print(f"\n✓ Эксперимент найден: {experiment.name} (ID: {experiment.experiment_id})")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"✓ Найдено runs: {len(runs)}")

except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


print("\n" + "="*80)
print("АНАЛИЗ МЕТРИК")
print("="*80)

run_data = []

for run in runs:

    params = run.data.params
    metrics = run.data.metrics

    run_info = {
        "Run Name": run.info.run_name,
        "Run ID": run.info.run_id[:8] + "...",  # Сокращенный ID
        "Status": run.info.status,
        "Accuracy": metrics.get("accuracy", 0),
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1-Score": metrics.get("f1_score", 0),
    }

    run_data.append(run_info)


if run_data:
    df = pd.DataFrame(run_data)

    print("\n[Таблица сравнения всех runs]")
    print("\n" + df.to_string(index=False))


    print("\n" + "="*80)
    print("СТАТИСТИКА")
    print("="*80)

    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for metric in metrics_cols:
        values = df[metric]
        print(f"\n[{metric}]")
        print(f"  - Минимум: {values.min():.4f}")
        print(f"  - Максимум: {values.max():.4f}")
        print(f"  - Среднее:  {values.mean():.4f}")
        print(f"  - Стд. отклонение: {values.std():.4f}")


    print("\n" + "="*80)
    print("ЛУЧШИЕ RUNS")
    print("="*80)

    best_accuracy = df.loc[df["Accuracy"].idxmax()]
    best_f1 = df.loc[df["F1-Score"].idxmax()]

    print(f"\n🏆 Лучший по Accuracy:")
    print(f"  - Model: {best_accuracy['Run Name']}")
    print(f"  - Accuracy: {best_accuracy['Accuracy']:.4f}")

    print(f"\n🏆 Лучший по F1-Score:")
    print(f"  - Model: {best_f1['Run Name']}")
    print(f"  - F1-Score: {best_f1['F1-Score']:.4f}")


    print("\n" + "="*80)
    print("СРАВНЕНИЕ ПО ТИПАМ МОДЕЛЕЙ")
    print("="*80)

    lr_runs = df[df["Run Name"].str.contains("Logistic", case=False)]
    rf_runs = df[df["Run Name"].str.contains("Random", case=False)]

    if len(lr_runs) > 0:
        print(f"\n[Logistic Regression] ({len(lr_runs)} запусков)")
        print(f"  - Средний Accuracy: {lr_runs['Accuracy'].mean():.4f}")
        print(f"  - Средний F1-Score: {lr_runs['F1-Score'].mean():.4f}")

    if len(rf_runs) > 0:
        print(f"\n[Random Forest] ({len(rf_runs)} запусков)")
        print(f"  - Средний Accuracy: {rf_runs['Accuracy'].mean():.4f}")
        print(f"  - Средний F1-Score: {rf_runs['F1-Score'].mean():.4f}")

    if len(lr_runs) > 0 and len(rf_runs) > 0:
        lr_acc = lr_runs['Accuracy'].mean()
        rf_acc = rf_runs['Accuracy'].mean()

        print(f"\n📊 Сравнение:")
        if lr_acc > rf_acc:
            print(f"  Logistic Regression лучше на {(lr_acc - rf_acc)*100:.2f}%")
        elif rf_acc > lr_acc:
            print(f"  Random Forest лучше на {(rf_acc - lr_acc)*100:.2f}%")
        else:
            print(f"  Модели примерно равны")
