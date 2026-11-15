# Monitoreo, alertas y respuesta a incidentes · Actividad 8 (`ops-v1.0.0`)

Monitorización proactiva del chatbot de CAB (`app-v1.0.0` de la Fase 2): métricas, logs y trazas,
alertas basadas en SLO y presupuesto de error, tableros operativos, detección de *data drift*, runbooks y
respuestas automatizadas (rollback, escalamiento de réplicas y reentrenamiento), todo registrado en MLflow.
Se adaptó la estructura del *starter* de la profesora (`actividad8_observabilidad_starter`): los archivos de
salida conservan sus nombres (`monitoring_summary.json`, `alerts.json`, `traces.jsonl`, `monitoring_log`).

## Estructura

```text
monitoreo/
├── config/monitoring_config.yaml   SLI, SLO, presupuesto de error, reglas A1–A6, severidad, ruteo y límites
├── src/
│   ├── simular_game_day.py         plano de servicio: game day 12/11/2026 (INC-02 rollback, INC-03 escalamiento)
│   ├── monitoreo_semanal.py        plano de modelo: oct–dic 2026 (INC-01 drift + reentrenamiento)
│   ├── exportar_evidencias.py      PNG de los tableros y respaldo portable de MLflow
│   └── obs/                        paquete reutilizable
│       ├── alertas.py              motor de alertas: ventanas, quema, severidad, agrupación, deduplicación
│       ├── drift.py                PSI, KS, JSD, χ² y % fuera de rango
│       ├── telemetria.py           logs JSON y trazas (JSONL + MLflow Tracing)
│       ├── registro.py             Model Registry: versiones y alias @champion
│       ├── respuestas.py           rollback, escalar réplicas, reentrenar challenger (con límites)
│       ├── runbooks.py             ejecución de un runbook con evidencia (Markdown + JSON)
│       ├── servicio.py             API real de la Fase 2 + router de reproducción + cola
│       ├── modelos_eta.py          entrenamiento/predicción del ETA con la receta de la Fase 2
│       └── tablero.py              tablero operativo HTML
├── runbooks/                       RB-01 (rollback), RB-02 (drift/reentrenamiento), RB-03 (saturación)
├── tests/test_monitoreo.py         9 pruebas del motor de alertas, drift y escalamiento
├── salidas/game_day/ y semanal/    evidencias generadas (resúmenes, alertas, notificaciones, logs, trazas,
│                                   tableros, ejecuciones de runbook y postmortems)
└── ejecutar_todo.ps1 / .sh         ejecuta todo desde cero y deja MLflow en http://127.0.0.1:5000
```

## Cómo ejecutar

Desde la raíz del repositorio, con Python 3.12 o 3.13:

```powershell
python -m venv .venv-mon
.\.venv-mon\Scripts\Activate.ps1
python -m pip install -r monitoreo\requirements.txt
powershell -ExecutionPolicy Bypass -File monitoreo\ejecutar_todo.ps1
```

En Linux/macOS/Git Bash: `bash monitoreo/ejecutar_todo.sh`. Tarda unos 4 minutos.

**Ver el panel sin volver a ejecutar:** descomprimir `mlflow_respaldo_A8.zip` dentro de `monitoreo/` y correr
`python -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --artifacts-destination mlflow/artefactos --port 5000`.

## Decisiones importantes

- **La API de la Fase 2 no se modificó**: se importa `backend/app/main.py` y se inicializa con un router de
  reproducción que devuelve las decisiones reales del mDeBERTa-v3 (logits y latencia medidos en Docker para los
  160 mensajes de la prueba externa; coincidencia 160/160). Así se puede simular sin Hugging Face.
- **Dos planos**: servicio (ventanas de 5 min) y modelo (semanal, con etiquetas que llegan al entregarse la orden).
- **Umbrales trazables**: cada SLO cita su origen (línea base de la Fase 2 o KPI de la Fase 1) en el YAML.
- **Automatización con límites**: rollback y réplicas son automáticos; la promoción de un modelo reentrenado la
  aprueba una persona. El rollback es real (Model Registry de MLflow); agregar réplicas y el límite de tasa se
  simulan en la cola, y los pasos humanos de los runbooks son roles simulados del game day.
- **Ruido**: en el plano semanal solo se notifica al abrir, al subir de severidad y al cerrar
  (`renotificar_por_tiempo` en el YAML).
- **Infraestructura**: memoria del contenedor y reinicios se documentan como indicadores informativos (línea base de
  la Fase 2: 1.7–1.9 GB); no se simulan en el game day.

## Resultados (resumen)

| Incidente | Alerta | Respuesta | MTTD | MTTR | Resultado |
|---|---|---|---|---|---|
| INC-01 drift de carga de planta | A4 (P3) → A5 (P2) | Reentrenamiento + aprobación | 7 días | — | Mitigado parcialmente, escalado |
| INC-02 despliegue defectuoso del ETA | A3 (P1) | Rollback del alias @champion | 10 min | 10 min | Resuelto |
| INC-03 saturación por el portal de ARM-01 | A1 (P1), A2 agrupada | 1 → 3 réplicas + límite de tasa | 5 min | 10 min | Resuelto |

Pruebas: `python -m pytest monitoreo/tests -q` (9 pruebas).
