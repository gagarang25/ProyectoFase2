# Registro de cambios

## opt-v1.0.0 · Actividad 9 (escalamiento eficiente y responsable)
- Carpeta `optimizacion/`: exportación a ONNX, 5 variantes del router y compuertas de calidad/equidad (Q1–Q4, D1).
- Variante elegida: ONNX Runtime con embeddings INT8 (mismas 560 decisiones; p95 −69 %, memoria −46 %). INT8 completo rechazado.
- ETA y riesgo precalculados por lotes; medición de `/api/chat` de punta a punta (480/480 respuestas idénticas).
- Simulación de autoescalado con el game day de la A8 y auditoría de fairness por idioma y por segmento de cliente.
- La API (`backend/app`) no cambió; los componentes nuevos están en `optimizacion/src/optimizaciones_api.py`.

## ops-v1.0.0 · Actividad 8 (monitoreo y respuesta a incidentes)
- Carpeta `monitoreo/`: SLI/SLO, presupuesto de error y reglas de alerta A1–A6 en `config/monitoring_config.yaml`.
- Motor de alertas con ventanas, quema del presupuesto, severidad P1–P3, agrupación y deduplicación.
- Logs JSON y trazas (JSONL + MLflow Tracing); tableros operativos HTML; drift con PSI, KS, JSD y χ².
- Runbooks RB-01 (rollback), RB-02 (drift y reentrenamiento) y RB-03 (saturación) con respuestas automatizadas.
- Simulación de 3 incidentes registrada en MLflow (Model Registry `cab_eta_entrega` v1–v3). La API no cambió.

## app-v1.0.0 · Proyecto Fase 2
- API REST (FastAPI) con router de intención, SAP simulado de solo lectura, modelo de ETA y riesgo de retraso.
- Chat web (nginx) que consume la API; `docker compose` con dos contenedores.
- `Dockerfile` único para la nube (API + chat estático); despliegue documentado en AWS Elastic Beanstalk y simulado con la imagen única.
- Correcciones de la validación en contenedor (982a6e2): healthcheck de nginx por IPv4 y pruebas que no escriben en la bitácora.
- Manual de despliegue y documento de validación y pruebas en `docs/`.
- Pruebas: pytest (unitarias e integración), 35 casos funcionales y extremos contra el contenedor, estrés y drift.

## model-v1.0.0
- Router mDeBERTa-v3 ajustado con data-v3.0.0, calibrado con temperatura y publicado en Hugging Face Hub.
- Modelo de ETA (HistGradientBoostingRegressor, pérdida absoluta) con rango por cuantiles 10 % y 90 %.
- Clasificador de riesgo de retraso de la A6 reentrenado con los mismos hiperparámetros (umbral 0.17).

## data-v3.0.0
- Mensajes de CAB en español e inglés + Bitext (A4) con marcadores reemplazados y reetiquetado del "ETA".
- Prueba externa de 160 mensajes escritos a mano (incluye los 50 de la A5).

## code-v1.0.0
- Código de la API, del frontend, de entrenamiento y de pruebas.
