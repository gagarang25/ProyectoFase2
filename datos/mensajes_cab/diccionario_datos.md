# Diccionario de datos · data-v3.0.0 (mensajes para el router de intención)

Se generan con `python entrenamiento/generar_mensajes_cab.py` (semilla 42). Las huellas SHA-256 están en `versiones.json`.

| Archivo | Filas | Uso |
|---|---|---|
| `bitext_muestra_a4.csv` | 2,000 | Origen: muestra balanceada de Bitext de la Actividad 4 (data-v1.0.0, CDLA-Sharing-1.0) |
| `entrenamiento.csv` | 3,200 | Ajuste del modelo |
| `validacion.csv` | 400 | Mejor época, calibración (temperatura) y umbral |
| `prueba.csv` | 400 | Evaluación con mensajes del mismo estilo que el entrenamiento |
| `prueba_externa_a5.csv` | 50 | Prueba fuera de plantilla de la Actividad 5 (se incluye en la v3) |
| `prueba_externa_v3.csv` | 160 | Evaluación final: 50 de la A5 + 110 nuevos escritos a mano (no salen de las plantillas) |

## Columnas de entrenamiento / validación / prueba

| Columna | Descripción |
|---|---|
| id | Consecutivo dentro del archivo |
| mensaje | Texto del cliente |
| intencion | `estatus_orden`, `fecha_entrega`, `queja`, `contactar_ejecutivo` u `otro` |
| idioma | `es` o `en` |
| fuente | `bitext_a4` (Bitext con marcadores reemplazados) o `cab_simulado` (plantillas de CAB) |
| plantilla | Plantilla de CAB que generó el mensaje, o la nota de reetiquetado en Bitext |

## Columnas de la prueba externa

| Columna | Descripción |
|---|---|
| intencion | Intención correcta, o `ambigua` si el mensaje admite varias |
| aceptables | Intenciones con las que una respuesta automática sería correcta (separadas por `|`); vacío = ninguna |
| idioma | `es`, `en` o `mix` (spanglish) |
| tipo | `claro`, `spanglish`, `ambiguo` o `fuera_dominio` |
| origen | `A5` (Actividad 5) o `Fase2` |

## Cambios respecto a data-v1.0.0 y data-v2.0.0

1. Los 135 mensajes de estatus de Bitext que preguntan por el "ETA" se reetiquetaron como `fecha_entrega`
   (en CAB el ETA es la fecha estimada de llegada; hallazgo de la A5).
2. Los marcadores `{{Order Number}}` se reemplazaron por números de OC realistas y se agregó un número de OC a 35 %
   de los mensajes de otras intenciones, para que traer un número de OC no delate el estatus (atajo de la A5).
3. Se agregaron 2,000 mensajes simulados de CAB (80 % en español) con vocabulario de la industria y ruido realista.
