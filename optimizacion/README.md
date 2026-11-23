# Optimización, escalamiento y equidad · Actividad 9 (opt-v1.0.0)

Mediciones que sustentan el rediseño del chatbot de CAB: escalabilidad y costos (Tema 21) y auditoría de
gobernanza y fairness (Tema 22). **La API de la Fase 2 no se modifica**: los componentes optimizados se conectan
con la misma interfaz (`src/optimizaciones_api.py`).

| Paso | Script | Qué mide | Salida en `resultados/` |
|---|---|---|---|
| 1 | `exportar_onnx.py` | Exporta model-v1.0.0 a ONNX y crea 4 variantes cuantizadas | `../modelos_onnx/` (no se sube a Git) + `manifiesto_onnx.json` |
| 2 | `medir_variante.py` | Latencia p50/p95/p99, throughput, memoria y calidad de cada variante (2 hilos) | `crudo_*.json` |
| 3 | `evaluar_variantes.py` | Compuertas Q1–Q4 + D1 y fairness por idioma | `comparacion_variantes.csv`, `equidad_idioma.csv`, `resumen_optimizacion.json` |
| 3b | `evaluar_cascada.py` | Palanca evaluada: router en cascada TF-IDF → mDeBERTa | `cascada_router.csv` |
| 3c | `diagnostico_int8.py` | Por qué falla INT8 (cuantización selectiva de capas) | `diagnostico_int8.csv` |
| 4 | `medir_api.py` | `/api/chat` de punta a punta: actual vs propuesta (480 solicitudes) | `api_*.json` |
| 5 | `simular_autoescalado.py` | Game day de la A8 con 5 arquitecturas × 3 escenarios | `simulacion_autoescalado*.csv/json` |
| 6 | `auditar_equidad_entregas.py` | Fairness de ETA y riesgo por segmento | `equidad_entregas*.csv/json` |

```bash
pip install -r backend/requirements.txt -r optimizacion/requirements.txt
bash optimizacion/ejecutar_todo.sh        # ≈ 25 min en 2 vCPU; requiere salida_router/modelo/ (model-v1.0.0)
```

## Resultados principales (2 vCPU)

| | Actual (PyTorch fp32) | Propuesta (ONNX + embeddings INT8 + ETA por lotes) |
|---|---|---|
| p95 de `/api/chat` | 166.5 ms | 52.1 ms |
| Capacidad por réplica | 7.6 sol/s | 24.2 sol/s |
| Memoria del proceso | 2,239 MB | 1,204 MB |
| Respuestas iguales | — | 480/480 (solo cambia la confianza ≤ 0.04) |
| Disponibilidad en el game day simulado | 65.3 % | 100 % (con límite de tasa) |

- La cuantización INT8 de activaciones (3 variantes) **se rechaza**: accuracy externa 0.96 → ≈ 0.52.
- Fairness: brecha de 5.8 puntos de precisión automática en español; recall de riesgo bajo para TR-A (0.698) y Dedicado (0.725).

Qué es real y qué es simulado: las métricas de modelos y API son mediciones reales; el tráfico, el autoescalado
y la factura de nube son simulaciones (CAB es ficticia, no se contrató infraestructura).
