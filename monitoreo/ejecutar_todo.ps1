# Ejecuta toda la Actividad 8 desde cero (Windows / PowerShell).
#   powershell -ExecutionPolicy Bypass -File monitoreo\ejecutar_todo.ps1
# Deja el panel de MLflow abierto en http://127.0.0.1:5000
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
Remove-Item -Recurse -Force mlflow, salidas\game_day, salidas\semanal -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force mlflow\artefactos | Out-Null

$servidor = Start-Process -PassThru -WindowStyle Hidden python -ArgumentList @(
  "-m", "mlflow", "server", "--backend-store-uri", "sqlite:///mlflow/mlflow.db",
  "--artifacts-destination", "mlflow/artefactos", "--host", "127.0.0.1", "--port", "5000")
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
for ($i = 0; $i -lt 30; $i++) {
  try { Invoke-WebRequest "$env:MLFLOW_TRACKING_URI/health" -UseBasicParsing | Out-Null; break } catch { Start-Sleep 1 }
}

python -m pytest tests -q
python src\simular_game_day.py
python src\monitoreo_semanal.py
python src\exportar_evidencias.py
Write-Host "Listo. Panel de MLflow: $env:MLFLOW_TRACKING_URI (cierra el proceso $($servidor.Id) al terminar)"
