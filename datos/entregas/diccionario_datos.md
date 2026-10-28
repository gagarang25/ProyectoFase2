# Diccionario de datos · data-v1.0.0

Datos **simulados** de entregas de Componentes Automotrices del Bajío (CAB, empresa ficticia). Se generan con
`python entrenamiento/generar_datos_entregas.py` (semilla 42). Las huellas SHA-256 están en `versiones.json`.

| Archivo | Periodo | Uso |
|---|---|---|
| `entregas_cab.csv` | 01/10/2024 – 30/09/2026 | Entrenamiento (antes del 01/05/2026) y prueba (desde el 01/05/2026) |
| `piloto_ab_cab.csv` | 01/10/2026 – 31/12/2026 | Prueba A/B simulada: solo los 5 clientes del piloto de la Fase 1 |

Cada fila es una orden de compra. **Objetivo:** `retraso` (1 = la orden llegó después de la fecha comprometida).

## Columnas de entrada (se conocen al confirmar la orden)

| Columna | Tipo | Descripción |
|---|---|---|
| tipo_cliente | categórica | `armadora` o `tier1` (otro proveedor de la industria). |
| planta | categórica | Planta de origen: Querétaro, San Juan del Río o Celaya. |
| region_destino | categórica | Bajío, Centro, Noreste, Norte o EE.UU. |
| cruce_frontera | 0/1 | 1 si el envío pasa por aduana hacia EE.UU. |
| distancia_km | numérica | Distancia aproximada de la planta al cliente. |
| transportista | categórica | TR-A a TR-E (TR-E es el transportista internacional). |
| modalidad | categórica | Dedicado, Consolidado o Paquetería. |
| familia_producto | categórica | Estampados, Arneses, Plásticos inyectados, Maquinados o Ensambles. |
| cantidad_piezas, peso_kg | numéricas | Tamaño del pedido. |
| inventario_disponible | 0/1 | 1 si las piezas ya están en almacén (no requieren producción). |
| utilizacion_planta | numérica | Carga de la planta esa semana (1.0 = 100 % de la capacidad). |
| pedido_urgente | 0/1 | El cliente pidió una fecha más corta que la habitual. |
| cambios_pedido | entero | Veces que el cliente modificó la orden (0 a 4). |
| produccion_std_dias, transito_std_dias | enteros | Tiempos estándar que SAP usa para comprometer la fecha. |
| dias_comprometidos | entero | Días entre el pedido y la fecha comprometida. |
| holgura_dias | entero | `dias_comprometidos − produccion_std − transito_std` (variable derivada). |
| mes, dia_semana | enteros | Calendario del pedido. |
| envios_previos_ruta | entero | Órdenes del mismo cliente con el mismo transportista en los 365 días anteriores. |
| tasa_retraso_transportista_90d | numérica | Proporción de entregas tardías del transportista en los 90 días anteriores (solo entregas ya realizadas; vacía si hay menos de 30). |

## Columnas que NO se usan como entrada

| Columna | Motivo |
|---|---|
| id_orden, cliente, fecha_pedido | Identificadores (el cliente se usa solo para estratificar la prueba A/B). |
| fecha_compromiso | Redundante con `dias_comprometidos`. |
| fecha_entrega_real, dias_reales, dias_retraso | Solo se conocen después de la entrega: usarlas sería fuga de información. |
| retraso | Variable objetivo. |

## Cómo se simula el retraso

El tiempo real de cada orden es la suma de producción (más lenta si la planta pasa de 88 % de carga o si hubo
cambios en el pedido), tránsito (con un retraso propio de cada transportista y de la modalidad), aduana (más lenta en
diciembre), temporada de lluvias en el norte y un 3 % de eventos imprevisibles (bloqueos, clima). El modelo no ve
estos tiempos reales, solo las columnas de entrada, por eso no puede acertar siempre.
