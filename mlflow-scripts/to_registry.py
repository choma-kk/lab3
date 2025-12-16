import os
import warnings
import urllib3
import json
import tempfile

print("="*80)
print("РЕГИСТРАЦИЯ МОДЕЛЕЙ В MLFLOW MODEL REGISTRY")
print("="*80)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')


os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"

if 'MLFLOW_TRACKING_SERVER_CERT_PATH' in os.environ:
    del os.environ['MLFLOW_TRACKING_SERVER_CERT_PATH']

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_HOST_HEADER = "mlflow.labs.itmo.loc"
EXPERIMENT_NAME = "Iris Classification Training"

print(f"\n[КОНФИГУРАЦИЯ]")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Host Header: {MLFLOW_HOST_HEADER}")
print(f"  - Эксперимент: {EXPERIMENT_NAME}")


import requests

original_session_init = requests.Session.__init__

def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.headers.update({'Host': MLFLOW_HOST_HEADER})

requests.Session.__init__ = patched_session_init


import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


print("\n" + "="*80)
print("ПОЛУЧЕНИЕ ПОСЛЕДНИХ RUNS")
print("="*80)

try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"✗ Эксперимент не найден: {EXPERIMENT_NAME}")
        exit(1)

    print(f"\n✓ Эксперимент найден: {experiment.name} (ID: {experiment.experiment_id})")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"✓ Найдено runs: {len(runs)}")

    latest_lr = None
    latest_rf = None

    for run in runs:
        if "logistic" in run.info.run_name.lower() and (latest_lr is None or run.info.start_time > latest_lr.info.start_time):
            latest_lr = run
        elif "random" in run.info.run_name.lower() and (latest_rf is None or run.info.start_time > latest_rf.info.start_time):
            latest_rf = run

    print(f"\n✓ Последние версии:")
    if latest_lr:
        print(f"  - LogisticRegression (ID: {latest_lr.info.run_id})")
    if latest_rf:
        print(f"  - RandomForest (ID: {latest_rf.info.run_id})")

except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


print("\n" + "="*80)
print("РЕГИСТРАЦИЯ МОДЕЛЕЙ В MODEL REGISTRY")
print("="*80)

models_registered = []


if latest_lr:
    print(f"\n[LogisticRegression_Iris]")

    try:
        with mlflow.start_run(run_id=latest_lr.info.run_id):

            MODELS_DIR = os.path.join(tempfile.gettempdir(), "mlflow_models")
            metadata_file = os.path.join(MODELS_DIR, "iris_logistic_regression_metadata.json")

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            model_file = os.path.join(MODELS_DIR, "iris_logistic_regression.pkl")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris-logistic-regression",
                input_example=None
            )

            print(f"  ✓ Модель залогирована")
            print(f"    - Model URI: {model_info.model_uri}")
            print(f"    - Registered name: iris-logistic-regression")

            models_registered.append({
                "name": "iris-logistic-regression",
                "version": model_info.registered_model_version,
                "stage": "None",
                "uri": model_info.model_uri
            })

    except Exception as e:
        print(f"  ✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if latest_rf:
    print(f"\n[RandomForest_Iris]")

    try:
        with mlflow.start_run(run_id=latest_rf.info.run_id):

            MODELS_DIR = os.path.join(tempfile.gettempdir(), "mlflow_models")
            metadata_file = os.path.join(MODELS_DIR, "iris_random_forest_metadata.json")

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            model_file = os.path.join(MODELS_DIR, "iris_random_forest.pkl")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris-random-forest",
                input_example=None
            )

            print(f"  ✓ Модель залогирована")
            print(f"    - Model URI: {model_info.model_uri}")
            print(f"    - Registered name: iris-random-forest")

            models_registered.append({
                "name": "iris-random-forest",
                "version": model_info.registered_model_version,
                "stage": "None",
                "uri": model_info.model_uri
            })

    except Exception as e:
        print(f"  ✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


print("\n" + "="*80)
print("ПЕРЕХОД МОДЕЛЕЙ В PRODUCTION")
print("="*80)

try:
    if latest_lr:
        print(f"\n[LogisticRegression_Iris]")

        lr_model = client.get_latest_versions("iris-logistic-regression", stages=["None"])
        if lr_model:
            latest_version = lr_model[0].version

            client.transition_model_version_stage(
                name="iris-logistic-regression",
                version=latest_version,
                stage="Production"
            )

            print(f"  ✓ Версия {latest_version} → Production")

    if latest_rf:
        print(f"\n[RandomForest_Iris]")

        rf_model = client.get_latest_versions("iris-random-forest", stages=["None"])
        if rf_model:
            latest_version = rf_model[0].version

            client.transition_model_version_stage(
                name="iris-random-forest",
                version=latest_version,
                stage="Production"
            )

            print(f"  ✓ Версия {latest_version} → Production")

except Exception as e:
    print(f"\n✗ ОШИБКА при переводе: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*80)
print("СТАТУС МОДЕЛЕЙ В REGISTRY")
print("="*80)

try:
    all_models = client.search_registered_models()

    for model in all_models:
        if "iris" in model.name.lower():
            print(f"\n[{model.name}]")
            print(f"  - Версий: {len(model.latest_versions)}")

            for version in model.latest_versions:
                print(f"    - v{version.version}: {version.current_stage}")

except Exception as e:
    print(f"\n✗ ОШИБКА при получении статуса: {e}")
