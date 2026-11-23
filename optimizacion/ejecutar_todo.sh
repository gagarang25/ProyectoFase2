#!/usr/bin/env bash
# Actividad 9 · Reproduce todas las mediciones de optimización, escalamiento y equidad (≈ 25 min en 2 vCPU).
# Requisitos: pip install -r optimizacion/requirements.txt  y el modelo model-v1.0.0 en salida_router/modelo/
set -euo pipefail
cd "$(dirname "$0")/src"
export PYTHONPATH="$(pwd)/../../backend:${PYTHONPATH:-}"

python exportar_onnx.py                          # 1. ONNX + variantes cuantizadas (modelos_onnx/, no se sube a Git)
for v in pytorch_fp32 onnx_fp32 onnx_emb8 onnx_int8_matmul onnx_int8_pc onnx_int8_total; do
  python medir_variante.py "$v"                   # 2. calidad, latencia, throughput y memoria de cada variante
done
python evaluar_variantes.py                      # 3. compuertas de calidad y equidad por idioma
python evaluar_cascada.py                        # 3b. palanca evaluada: router en cascada
python diagnostico_int8.py                       # 3c. por qué falla INT8 (cuantización selectiva)
for m in actual onnx propuesta; do
  python medir_api.py "$m"                        # 4. /api/chat de punta a punta (API real de la Fase 2)
done
python simular_autoescalado.py                   # 5. capacidad y autoescalado con la carga del game day (A8)
python auditar_equidad_entregas.py               # 6. fairness de ETA y riesgo por segmento de cliente
echo "Listo: resultados en optimizacion/resultados/"
