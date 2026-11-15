# Postmortem INC-02 · Despliegue defectuoso del modelo de ETA (v2)

**Severidad:** P1 · **Runbook:** RB-01 · **Formato:** sin culpables

## Qué pasó
A las 10:30 se desplegó la versión 2 del modelo de ETA, reentrenada con la extracción de SAP v2.3. Ese ETL invirtió
el campo `inventario_disponible` (1 pasó a significar "agotado"). La validación fuera de línea usó los mismos datos
invertidos y dio MAE 1.05 días, por eso la versión pasó la compuerta.
En producción, la API envía el campo correcto: el modelo sumaba días de producción a piezas en inventario y los
quitaba a piezas que había que fabricar.

## Detección y respuesta
- Detección: alerta A3 a las 10:40 (MTTD 10 min): el ETA servido difería del
  campeón en sombra y aparecieron fechas menores que el tiempo de tránsito.
- Respuesta: rollback automático del alias @champion a v1 (segundos). MTTR 10 min.
- Respuestas afectadas: 46 respuestas con ETA durante 10 min.

## Confirmación con etiquetas
Backtest con 784 órdenes del piloto ya entregadas: v2 habría tenido MAE 3.64 días y
cobertura 74%; v1 tiene 1.05 días y 90%.

## Qué funcionó
La sombra del campeón anterior detectó el problema sin esperar etiquetas, y el rollback por alias fue inmediato.

## Mejoras (acciones)
1. Prueba de contrato de datos en el pipeline: la proporción de `inventario_disponible` no puede cambiar más de 10 puntos
   contra la extracción anterior (habría bloqueado v2 antes de entrenar).
2. Despliegue canary: 10 % del tráfico durante 30 min con la sombra activa antes de pasar al 100 %.
3. Validar cada versión nueva también contra el conjunto de prueba congelado de la versión anterior.
