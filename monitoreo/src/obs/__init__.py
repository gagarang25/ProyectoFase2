"""Paquete de observabilidad del chatbot de CAB (Actividad 8, ops-v1.0.0).

Módulos:
    config      carga monitoring_config.yaml y define rutas.
    drift       PSI, Kolmogorov-Smirnov, Jensen-Shannon y chi-cuadrada.
    alertas     motor de reglas basadas en SLO: ventanas consecutivas, quema del
                presupuesto de error, severidad, deduplicación y notificaciones.
    telemetria  logs JSON estructurados y trazas (archivo JSONL y MLflow Tracing).
    registro    MLflow Model Registry: versiones del modelo de ETA y alias @champion.
    respuestas  acciones automatizadas: rollback, escalamiento y reentrenamiento.
    runbooks    ejecución paso a paso de un runbook con evidencia.
    tablero     dashboard operativo (HTML + PNG).
    servicio    simulación del plano de servicio (game day con la API real).
    modelo      monitoreo semanal del plano de modelo (ETA con etiquetas).
"""
