# Ejecución del runbook RB-01 · INC-02

**Despliegue defectuoso del modelo de ETA** · severidad P1

| # | Hora | Etapa | Ejecuta | Qué se hizo | Resultado | Evidencia |
|---|---|---|---|---|---|---|
| 1 | 2026-11-12 10:40 | identificacion | motor de alertas | Alerta A3 (P1): desacuerdo con el campeón en sombra y fechas imposibles en 2 ventanas seguidas | desacuerdo mediano 4.0 días; imposibles 0% (esta regla no se activó: v2 no dio fechas menores al tránsito) | notificaciones.jsonl, alerts.json |
| 2 | 2026-11-12 10:40 | diagnostico | automatización | Correlacionar con cambios recientes: registro de despliegues | v2 desplegada a las 10:30 (único cambio); la señal empezó en esa ventana | monitoring_log.jsonl.gz (evento despliegue_modelo) |
| 3 | 2026-11-12 10:40 | accion | automatización | Rollback automático: mover @champion a la versión anterior y recargar | @champion v2 → v1 (ejecutado) | MLflow Model Registry (alias y etiqueta estado=revertida) |
| 4 | 2026-11-12 10:50 | validacion | automatización | Confirmar 2 ventanas sanas con la versión restaurada | desacuerdo con la sombra &lt; 1 día y 0 % de fechas imposibles con v1 desde 10:40 | ventanas.csv, tablero 03 |
| 5 | 2026-11-12 10:50 | validacion | especialista MLOps | Confirmar con etiquetas (backtest con órdenes ya entregadas) | v2 habría tenido MAE 3.64 días y cobertura 74% vs v1 1.05 y 90% | backtest_v2.json |
| 6 | 2026-11-12 10:50 | cierre | especialista MLOps | Bloquear v2 (etiqueta estado=revertida), abrir defecto del ETL SAP v2.3 y agregar prueba de contrato de datos al pipeline | incidente cerrado; postmortem sin culpables | postmortem_INC-02.md |

## Resultado

- **estado:** resuelto
- **mttd min:** 10.0
- **tiempo mitigacion min:** 0.0
- **mttr min:** 10.0
- **escalado:** False
- **respuestas afectadas:** 46
