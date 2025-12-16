#!/bin/bash

source /opt/mlflow/mlflow-env/bin/activate

exec mlflow server \
 --backend-store-uri sqlite////opt/mlflow/mlruns/mlflow.db \
 --default-artifact-root /opt/mlflow/mlruns \
 --host 0.0.0.0 \
 --port 5000 \
 --cors-allowed-origins "*" \
 --disable-security-middleware \
