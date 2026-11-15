# Ejecución del runbook RB-03 · INC-03

**Latencia alta y errores por saturación** · severidad P1

| # | Hora | Etapa | Ejecuta | Qué se hizo | Resultado | Evidencia |
|---|---|---|---|---|---|---|
| 1 | 2026-11-12 11:50 | identificacion | motor de alertas | Alerta A1 (P1): quema del presupuesto de error | disponibilidad 1.6%, p95 3000 ms, quema 1 h 147× | notificaciones.jsonl |
| 2 | 2026-11-12 11:50 | diagnostico | automatización | Revisar saturación (CPU) y tráfico por cliente en los logs | CPU 100%; ARM-01 (98% del tráfico); 16.4 solicitudes/s (normal 0.5) | monitoring_log.jsonl.gz agrupado por cliente |
| 3 | 2026-11-12 11:50 | accion | automatización | Escalamiento automático de réplicas (regla: CPU ≥ 85 %) | réplicas 1 → 3 (listas en 90 s) | ventanas.csv (replicas, trabajadores) |
| 4 | 2026-11-12 11:55 | diagnostico | ingeniero de guardia | Revisar trazas de solicitudes lentas: ¿dónde se pierde el tiempo? | en la ventana del pico, la espera en cola fue el 100% del tiempo; el span del router siguió en 131 ms (mediana) → falla de capacidad, no del modelo | traces.jsonl / MLflow Traces |
| 5 | 2026-11-12 11:55 | escalamiento | ingeniero de guardia | Avisar al ejecutivo de cuenta de ARM-01 (origen del tráfico) | se abre caso con TI de ARM-01 | notificaciones.jsonl |
| 6 | 2026-11-12 12:15 | accion | ingeniero de guardia | Aplicar límite de 2 solicitudes/s por token (configuración del gateway) | ARM-01 recibe 429 en el exceso; los otros 4 clientes no se afectan | monitoring_log.jsonl.gz (status 429) |
| 7 | 2026-11-12 12:25 | validacion | automatización | Confirmar 30 min (6 ventanas) dentro del SLO | p95 &lt; 400 ms y disponibilidad ≥ 99.5 % desde 11:55 | ventanas.csv, tablero 05 |
| 8 | 2026-11-12 12:40 | escalamiento | ejecutivo de cuenta ARM-01 | Ejecutivo de cuenta confirma con TI de ARM-01 la corrección del bucle del portal (consulta cada 15 min en lugar de continua) | tráfico de ARM-01 vuelve a lo normal | ventanas.csv (rps) |
| 9 | 2026-11-12 13:20 | accion | automatización | Regresar a 1 réplica (CPU &lt; 50 % durante 30 min) | réplicas 3 → 1 | ventanas.csv |
| 10 | 2026-11-12 13:25 | cierre | ingeniero de guardia | Documentar, dejar el límite de tasa como permanente y proponer autoescalado por CPU en la nube | incidente cerrado | postmortem_INC-03.md |

## Resultado

- **estado:** resuelto
- **mttd min:** 5.0
- **tiempo mitigacion min:** 5.0
- **mttr min:** 10.0
- **escalado:** True
- **errores 5xx:** 6318
- **presupuesto disp restante:** -251%
