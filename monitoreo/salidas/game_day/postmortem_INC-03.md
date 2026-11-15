# Postmortem INC-03 · Saturación por consultas en bucle del portal de ARM-01

**Severidad:** P1 (A1) con A2 agrupada · **Runbook:** RB-03 · **Formato:** sin culpables

## Qué pasó
A las 11:45 el portal de proveedores de ARM-01 empezó a consultar todas sus OC en un bucle continuo
(≈16 solicitudes/s contra 0.5 normales). Con 1 réplica (2 trabajadores) la capacidad era de ≈13 solicitudes/s:
la cola creció, las solicitudes pasaron de 3 s y el chat devolvió errores a los cinco clientes.

## Detección y respuesta
- Detección: alerta A1 (quema rápida del presupuesto de disponibilidad) a las 11:50 (MTTD 5 min).
  A2 (latencia) se agrupó en el mismo incidente y no generó otra notificación.
- Diagnóstico: CPU al 100 %, un solo cliente con la mayor parte del tráfico y trazas con la espera en cola como
  causa principal (el router seguía en ≈130 ms): problema de capacidad, no del modelo.
- Respuesta automática: 1 → 3 réplicas. Servicio dentro del SLO a las 11:55 (MTTR 10 min).
- Respuesta manual: límite de 2 solicitudes/s por token (12:15) y corrección del portal por TI de ARM-01 (12:40).
- Presupuesto de error de disponibilidad restante al cierre del día: -251%.

## Mejoras (acciones)
1. Límite de tasa por token como configuración permanente del gateway.
2. Autoescalado por CPU en la nube (Elastic Beanstalk o Cloud Run, según el manual de la Fase 2).
3. Endpoint por lotes para integraciones (una consulta con varias OC) y documentación para clientes.
4. Política de presupuesto de error: si queda < 25 %, se congelan despliegues no urgentes.
