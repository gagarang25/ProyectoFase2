# Ejecución del runbook RB-02 · INC-01

**Data drift y degradación del modelo de ETA** · severidad P2

| # | Hora | Etapa | Ejecuta | Qué se hizo | Resultado | Evidencia |
|---|---|---|---|---|---|---|
| 1 | 2026-11-16 00:00 | identificacion | motor de alertas | Alerta A4 (P3): carga de planta por encima del máximo de entrenamiento 2 semanas seguidas | 100% de las órdenes con carga > 0.97; carga media 1.01 (antes ~0.80); KS 1.00 | alerts.json, drift_por_variable.csv |
| 2 | 2026-11-16 00:00 | diagnostico | especialista MLOps | ¿Es drift real o error de ingestión? Revisar nulos, esquema y contrastar la carga con el plan de producción | 0 nulos en las 22 entradas; esquema sin cambios; Planeación confirma arranque del programa de ARM-03 y temporada alta → drift real | monitoring_log.jsonl (nulos_entrada) |
| 3 | 2026-11-16 00:00 | diagnostico | automatización | ¿Cambió algo más? PSI/KS en numéricas y JSD/χ² en categóricas | PSI máx. numéricas 0.05, JSD máx. categóricas 0.005: el cambio está aislado en la carga de planta | drift_por_variable.csv |
| 4 | 2026-11-16 00:00 | diagnostico | especialista MLOps | Cuantificar impacto con las etiquetas disponibles | MAE semanal 1.07 días (SLO ≤ 1.30): todavía dentro, pero las órdenes con producción tardan más en entregarse → la degradación aparecerá con retraso | semanas.csv |
| 5 | 2026-11-16 00:00 | accion | especialista MLOps | Decisión P3: no reentrenar aún (faltan etiquetas); vigilar A5 y preparar datos para el challenger | monitoreo semanal continúa | incidentes.json |
| 6 | 2026-11-30 00:00 | escalamiento | motor de alertas | Alerta A5: MAE semanal > 1.30 dos semanas seguidas; el incidente sube a P2 | MAE 1.53 días; requiere producción 2.36 vs con inventario 0.93 → la degradación viene de la producción en planta saturada | notificaciones.jsonl, semanas.csv |
| 7 | 2026-11-30 00:00 | accion | automatización (pipeline T18) | Reentrenamiento semiautomático del challenger (intento 1): historia + 1779 órdenes de producción con peso ×10; rango calibrado +1 día(s) | validación (última semana, 253 entregas): MAE 1.41 vs campeón 1.53 (8% mejor); cobertura 88% → compuerta APROBADA | MLflow: run registro_eta_v3 |
| 8 | 2026-12-01 10:00 | accion | Especialista MLOps | Aprobación humana y promoción: @champion → v3 (v1 queda en sombra 7 días; rollback inmediato si la sombra muestra desacuerdo) | órdenes nuevas desde el 02/12 se estiman con v3 | MLflow Model Registry (alias champion) |
| 9 | 2026-12-14 00:00 | validacion | automatización | Comparar v3 contra v1 (sombra) en las mismas 241 órdenes nuevas ya entregadas | MAE v3 0.87 vs v1 0.81 (7% peor); cobertura 100% vs 97% | semanas.csv (mae_v3_acumulado) |
| 10 | 2026-12-28 00:00 | validacion | automatización | Comparar v3 contra v1 (sombra) en las mismas 691 órdenes nuevas ya entregadas | MAE v3 1.21 vs v1 1.26 (4% mejor); cobertura 95% vs 92% | semanas.csv (mae_v3_acumulado) |
| 11 | 2026-12-28 10:00 | escalamiento | especialista MLOps → gerente de servicio | El MAE sigue arriba del SLO con la planta saturada: se escala al dueño del producto (Gerente de servicio al cliente) con opciones | decisión: mantener v3, comunicar siempre rango + motivo ('planta al X % de capacidad'), confirmación humana de pedidos urgentes en planta > 0.97 (R03) y revisar el SLO por nivel de carga en la Fase 3 | postmortem_INC-01.md |
| 12 | 2026-12-28 12:00 | cierre | especialista MLOps | Cerrar el incidente técnico y abrir mejora: variable de carga proyectada y eventos de rastreo del transportista (pendiente de la Fase 2) | incidente cerrado con mitigación parcial documentada | postmortem_INC-01.md |

## Resultado

- **estado:** mitigado parcialmente (escalado)
- **inicio drift:** 2026-11-09
- **deteccion drift P3:** 2026-11-16
- **deteccion degradacion P2:** 2026-11-30
- **mttd dias:** 7
- **dias hasta nuevo modelo:** 1
- **kpi R10 reaccion:** cumplido (≤ 10 días hábiles)
