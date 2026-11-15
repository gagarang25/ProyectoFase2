# Postmortem INC-01 · Data drift por carga de planta y degradación del ETA

**Severidad:** P3 → P2 · **Runbook:** RB-02 · **Formato:** sin culpables

## Qué pasó
Desde el 09/11/2026 las tres plantas operan 15 puntos arriba de lo normal (temporada alta + programa nuevo de
ARM-03). El modelo de ETA nunca vio cargas mayores a 0.97: con la planta saturada la producción se alarga y
el modelo, que no puede extrapolar, subestima los días.

## Detección (proactiva, en dos pasos)
- 2026-11-16: A4 (P3) por drift de entrada, **sin esperar etiquetas** (MTTD 7 días). La
  semana del 02/11 ya tenía 23.5 % de órdenes arriba de 0.97 por el pico normal de noviembre (quedó "pendiente");
  con la primera semana de drift (100 %) se cumplieron las dos ventanas.
- 2026-11-30: A5 (P2) cuando el MAE semanal pasó 1.30 dos semanas seguidas. A6 no se activó:
  la cobertura del rango se mantuvo arriba de 85 %.

## Respuesta
- Reentrenamiento semiautomático: 1 intento(s); compuerta aprobada en 1.
- Mejora en validación: MAE 1.41 vs 1.53 del campeón (8%)
- Nuevo modelo en producción 1 días después de la alerta P2 (KPI R10: cumplido (≤ 10 días hábiles)).

## Resultado
- En las órdenes nuevas ya entregadas: MAE v3 1.21 vs v1 1.26 en las mismas
  órdenes; cobertura v3 95%.
- Última semana: MAE servido 1.75; si siguiera v1 1.94; sin drift 1.27.
- Estado: **mitigado parcialmente (escalado)**. Reentrenar redujo el error, pero con la planta saturada la incertidumbre real es mayor y
  el SLO de 1.30 días no se recupera. Se escaló al dueño del producto con opciones (ver RB-02).

## Mejoras (acciones)
1. Variable de carga proyectada (plan de producción) para que el modelo vea la saturación antes de que ocurra.
2. SLO de MAE estratificado por nivel de carga, revisado con el negocio.
3. Eventos de rastreo del transportista (pendiente desde la Fase 2) para reducir la incertidumbre del tránsito.
4. Reentrenamiento programado mensual con ventana deslizante, además del disparado por alertas.
