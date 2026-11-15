#!/usr/bin/env bash
# Ejecuta toda la Actividad 8 desde cero (Linux / macOS / Git Bash).
#   bash monitoreo/ejecutar_todo.sh
# Deja el panel de MLflow abierto en http://127.0.0.1:5000
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1
rm -rf mlflow salidas/game_day salidas/semanal
mkdir -p mlflow/artefactos

# Servidor de MLflow con artefactos "proxied": las rutas quedan relativas y el respaldo es portable.
python -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
  --artifacts-destination mlflow/artefactos --host 127.0.0.1 --port 5000 > salidas/mlflow_server.log 2>&1 &
SERVIDOR=$!
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
for i in $(seq 1 30); do curl -s "$MLFLOW_TRACKING_URI/health" > /dev/null && break; sleep 1; done

python -m pytest tests -q
python src/simular_game_day.py     # plano de servicio: INC-02 (rollback) e INC-03 (escalamiento)
python src/monitoreo_semanal.py    # plano de modelo: INC-01 (drift + reentrenamiento)
python src/exportar_evidencias.py  # PNG de los tableros + respaldo portable de MLflow

echo "Listo. Panel de MLflow: $MLFLOW_TRACKING_URI (PID $SERVIDOR; termina con: kill $SERVIDOR)"
