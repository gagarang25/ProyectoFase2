# RB-02 · Data drift y degradación del modelo de ETA (reentrenamiento)

| Campo | Valor |
|---|---|
| Versión del runbook | 1.0 (ops-v1.0.0) |
| Alertas que lo activan | **A4** (P3, drift de entrada) → **A5** (P2, MAE) y **A6** (P2, presupuesto de cobertura) |
| Tipo | **Semiautomático**: el pipeline reentrena y valida; una persona aprueba la promoción |
| Responsable | Especialista MLOps; dueño del producto: Gerente de servicio al cliente |
| Riesgos de la Fase 1 | R10 (pérdida de precisión; reacción ≤ 10 días hábiles), R08 (cobertura ≥ 85 %), R03 |

## 1. Señal que activa el runbook
- **A4 (P3, preventiva):** ≥ 20 % de las órdenes de la semana con `utilizacion_planta` arriba del máximo visto
  en el entrenamiento (0.97), 2 semanas seguidas. No necesita etiquetas.
- **A5 (P2):** MAE semanal del ETA (entregas de la semana) > 1.30 días, 2 semanas seguidas.
- **A6 (P2):** la cobertura del rango de 4 semanas consume el presupuesto de error (15 %) al ritmo de agotarlo.
- A4, A5 y A6 forman **un solo incidente** (grupo `drift_eta`): A5 sube la severidad, no abre otro.

## 2. Impacto esperado
El ETA subestima los días de las órdenes que requieren producción; los clientes reciben fechas optimistas.
Mientras la cobertura del rango se mantenga ≥ 85 %, el compromiso de R08 sigue cumpliéndose.

## 3. Diagnóstico inicial
1. **¿Drift real o error de ingestión?** Nulos, esquema y rangos de las 22 entradas; contrastar la carga de
   planta con el plan de producción (Planeación).
2. **¿Qué más cambió?** PSI/KS en numéricas y JSD/χ² en categóricas (`drift_por_variable.csv`).
3. **¿Cuánto afecta?** MAE y cobertura por semana de entrega y por segmento (requiere producción vs inventario).
4. Recordar el **retraso de las etiquetas**: las órdenes con producción tardan más en entregarse.

## 4. Acción principal
- **Con A4 (P3):** no reentrenar todavía; vigilar y preparar los datos.
- **Con A5/A6 (P2):** `reentrenar_challenger()`:
  1. Historia + producción entregada hasta 7 días antes del corte, con peso ×10 a la producción reciente.
  2. Calibrar el rango con la última semana (se ensancha k días hasta cubrir ≥ 87 %).
  3. **Compuerta:** MAE ≤ 95 % del campeón **y** cobertura ≥ 85 % en la semana de validación.
  4. Si pasa: registrar como `@challenger`; **una persona aprueba** y se mueve `@champion`. La versión anterior
     queda en sombra 7 días (si la sombra muestra desacuerdo, aplica RB-01).
  5. Si no pasa: se mantiene el campeón y se reintenta con una semana más de etiquetas (máximo 3 intentos).

**Por qué no es totalmente automático:** el ETA es un compromiso con el cliente; reentrenar con datos de una
situación anormal puede empeorar el modelo cuando la situación termina. La decisión necesita contexto de negocio.

## 5. Escalamiento
Si 2 semanas después de la promoción el MAE sigue fuera del SLO, se escala al **dueño del producto** con
opciones: comunicar siempre rango y motivo, confirmación humana de pedidos urgentes en planta saturada (R03),
o revisar el SLO para condiciones de alta carga.

## 6. Validación de recuperación
- Comparar la versión nueva contra la sombra (v1) **en las mismas órdenes** ya entregadas.
- Cobertura del rango ≥ 85 % y MAE semanal ≤ 1.30 durante 2 semanas para cerrar como resuelto.

## 7. Evidencia a conservar
`semanas.csv`, `drift_por_variable.csv`, `alerts.json`, `notificaciones.jsonl`, tableros por semana,
corrida de registro del challenger en MLflow, ejecución del runbook y postmortem.

## 8. Mejora posterior
Variable de carga proyectada, reentrenamiento mensual con ventana deslizante, SLO por nivel de carga y eventos
de rastreo del transportista.
