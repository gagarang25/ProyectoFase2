# RB-01 · Despliegue defectuoso del modelo de ETA (rollback)

| Campo | Valor |
|---|---|
| Versión del runbook | 1.0 (ops-v1.0.0) |
| Alertas que lo activan | **A3** (P1) |
| Tipo | **Automático** con validación humana posterior |
| Responsable | Especialista MLOps (guardia) |
| Riesgos de la Fase 1 | R05 (fechas sin respaldo), R10, R11 (cambios en SAP) |

## 1. Señal que activa el runbook
- **Métrica:** `eta_desacuerdo_mediana` (mediana de |ETA servido − ETA del campeón anterior en sombra|) y
  `eta_imposibles` (% de ETA menores que el tránsito estándar).
- **Condición:** desacuerdo ≥ 1 día **o** imposibles ≥ 5 %, durante **2 ventanas de 5 min** seguidas.
- **Severidad:** P1. Notifica a la guardia (pager) y a #cab-incidentes.
- **Por qué sin etiquetas:** la fecha real de entrega se conoce días después; la sombra permite detectar en minutos.

## 2. Impacto esperado
Los clientes reciben fechas estimadas incorrectas (riesgo crítico R05). Un ETA optimista provoca reclamos y
fletes urgentes; uno pesimista, pedidos duplicados.

## 3. Diagnóstico inicial
1. Revisar en la bitácora si hubo un **despliegue reciente** del modelo (evento `despliegue_modelo`).
2. Comparar la distribución de predicciones (PSI de la última hora) contra la referencia de v1.
3. Abrir 2 o 3 trazas de solicitudes con ETA: comparar el span `eta.estimar` con `eta.sombra_champion`.
4. Descartar que el cambio venga de la mezcla de clientes (un cliente puede mover el PSI sin que el modelo falle).

## 4. Acción principal (automatizada)
1. `rollback_modelo_eta()`: mueve el alias **@champion** a la versión anterior en el MLflow Model Registry
   y la API recarga el modelo. No se copian archivos, por eso tarda segundos.
2. Etiquetar la versión fallida con `estado=revertida` y el motivo.

**Límites de seguridad (cuándo NO automatizar):**
- Si no existe una versión anterior registrada → escalar a humano.
- Máximo 1 rollback por hora: si la versión restaurada también falla, se evita un ciclo y se escala.

## 5. Escalamiento
- Si el rollback no se ejecuta o la señal sigue después de 2 ventanas → Especialista MLOps + Gerente de TI.
- Si se confirma un defecto en la extracción de SAP → integrador SAP (póliza de soporte de R11).

## 6. Validación de recuperación
- 2 ventanas seguidas con desacuerdo < 1 día y 0 % de fechas imposibles con la versión restaurada.
- Confirmación con etiquetas (backtest con órdenes ya entregadas) antes de cerrar.

## 7. Evidencia a conservar
`alerts.json`, `notificaciones.jsonl`, `ventanas.csv`, tablero antes/después, etiquetas de versión en el
Model Registry, `backtest_v2.json`, ejecución del runbook (`runbook_RB-01_INC-02.md`).

## 8. Mejora posterior
Prueba de contrato de datos antes de entrenar, despliegue canary al 10 % y validación contra el conjunto de
prueba congelado de la versión anterior.
