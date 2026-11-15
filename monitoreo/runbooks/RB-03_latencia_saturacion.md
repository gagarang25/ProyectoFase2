# RB-03 · Latencia alta y errores por saturación de la API (escalamiento)

| Campo | Valor |
|---|---|
| Versión del runbook | 1.0 (ops-v1.0.0) |
| Alertas que lo activan | **A1** (P1, disponibilidad) y **A2** (P2, latencia) |
| Tipo | **Automático** (réplicas) + **manual** (límite de tasa, contacto con el cliente) |
| Responsable | Ingeniero de guardia; Gerente de TI si se agota el presupuesto de error |
| Riesgos de la Fase 1 | R11 (disponibilidad ≥ 99 %), R14 (adopción: una mala experiencia aleja a los clientes) |

## 1. Señal que activa el runbook
- **A1 (P1):** quema del presupuesto de disponibilidad ≥ 14.4× en 1 h **y** en los últimos 5 min
  (a ese ritmo se gasta el 2 % del presupuesto de 28 días en una hora).
- **A2 (P2):** quema del presupuesto de latencia ≥ 6× en 30 min y en 5 min, 2 ventanas seguidas.
- A1 y A2 forman un solo incidente (grupo `servicio_api`): si las dos se activan, se notifica una vez.

## 2. Impacto esperado
Los cinco clientes del piloto ven el chat lento o sin respuesta (timeout de 3 s); se consume el presupuesto de error.

## 3. Diagnóstico inicial
1. **Saturación:** CPU ≥ 85 % → problema de capacidad. Si la CPU está normal, escalar no ayuda (ver paso 5).
2. **Tráfico por cliente** en la bitácora: ¿un cliente concentra el tráfico?, ¿es el chat web o una integración?
3. **Trazas** de solicitudes lentas: si el tiempo está en `cola.espera` y el span del router sigue normal, es
   infraestructura, no el modelo.
4. **Memoria y reinicios del contenedor** (`docker stats`, healthcheck): si hay reinicios por falta de memoria
   (la API usa 1.7–1.9 GB), agregar réplicas no basta; subir la memoria asignada o revisar una fuga.
5. Revisar si hubo un despliegue reciente (si sí, considerar RB-01).

## 4. Acción principal
1. **Automática:** `escalar_replicas()` de 1 a 3 réplicas (máximo 4; listas en ≈ 90 s).
   Límites: solo si CPU ≥ 85 %; 15 min de enfriamiento entre cambios.
2. **Manual (con aprobación):** límite de 2 solicitudes/s por token en el gateway: el cliente que causa el
   exceso recibe HTTP 429 y los demás no se afectan.
3. **Al terminar:** regresar a 1 réplica cuando la CPU esté < 50 % durante 30 min.

## 5. Escalamiento
- Ejecutivo de cuenta del cliente que origina el tráfico (corregir la integración).
- Gerente de TI si el presupuesto de error de 28 días se agota: se congelan despliegues no urgentes.
- Si la CPU no está saturada y la latencia sigue alta: revisar dependencias (SAP, red) y RB-01.

## 6. Validación de recuperación
30 min (6 ventanas) con p95 < 400 ms y disponibilidad ≥ 99.5 %.

## 7. Evidencia a conservar
`ventanas.csv`, tableros antes/después, `notificaciones.jsonl`, `monitoring_log.jsonl.gz` (tráfico por
cliente, status 429/504), trazas en `traces.jsonl` y MLflow, ejecución del runbook y postmortem.

## 8. Mejora posterior
Límite de tasa permanente, autoescalado por CPU en la nube, endpoint por lotes para integraciones y
evaluación de reglas cada minuto (en la simulación se evalúa cada 5 min).
