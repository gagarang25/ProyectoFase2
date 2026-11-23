"""Paso 1 · Exporta el router mDeBERTa-v3 (model-v1.0.0) a ONNX y genera las variantes cuantizadas.

Variantes (todas con los MISMOS pesos entrenados; no se reentrena nada):
  onnx_fp32         Mismo modelo en ONNX Runtime (optimización de ejecución, sin perder precisión).
  onnx_emb8         Solo la tabla de embeddings (Gather, 251,000 × 768) en INT8: cuantización de pesos,
                    sin cuantizar activaciones. Es la parte más pesada del modelo multilingüe.
  onnx_int8_matmul  Cuantización dinámica INT8 por defecto de todas las MatMul (pesos y activaciones).
  onnx_int8_pc      INT8 por canal y solo MatMul con pesos (MatMulConstBOnly), la configuración más conservadora.
  onnx_int8_total   onnx_int8_matmul + embeddings en INT8 (la opción "máxima compresión").

Uso:  python optimizacion/src/exportar_onnx.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

from comun import CARPETA_MODELO, CARPETA_ONNX, MAX_LEN


def sha256(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cuantizar(nombre: str) -> None:
    """Se ejecuta en un proceso aparte (sin PyTorch cargado): cuantizar el modelo de 1.1 GB usa varios GB de RAM."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # La variante "total" parte de la int8_matmul y solo agrega la tabla de embeddings (Gather):
    # cuantizar todo de una vez desde fp32 superó los 7 GB de RAM del entorno de medición.
    origen, ops, extra = {
        "onnx_emb8": ("router_fp32.onnx", ["Gather"], {}),
        "onnx_int8_matmul": ("router_fp32.onnx", ["MatMul"], {}),
        "onnx_int8_pc": ("router_fp32.onnx", ["MatMul"], {"per_channel": True,
                                                         "extra_options": {"MatMulConstBOnly": True}}),
        "onnx_int8_total": ("router_int8_matmul.onnx", ["Gather"], {}),
    }[nombre]
    destino = CARPETA_ONNX / f"router_{nombre.split('onnx_')[1]}.onnx"
    quantize_dynamic(CARPETA_ONNX / origen, destino, weight_type=QuantType.QInt8, op_types_to_quantize=ops, **extra)


def exportar() -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CARPETA_MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(CARPETA_MODELO, dtype=torch.float32).eval()
    ejemplo = tok(["¿cuándo llega mi orden 4500123456?", "where is my order"], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
    entradas = ("input_ids", "attention_mask")
    ruta_fp32 = CARPETA_ONNX / "router_fp32.onnx"
    inicio = time.perf_counter()
    torch.onnx.export(modelo, tuple(ejemplo[k] for k in entradas), ruta_fp32, input_names=list(entradas),
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: "lote", 1: "tokens"} for k in entradas} | {"logits": {0: "lote"}})
    return time.perf_counter() - inicio


def main() -> None:
    CARPETA_ONNX.mkdir(parents=True, exist_ok=True)
    ruta_fp32 = CARPETA_ONNX / "router_fp32.onnx"
    # Cada paso pesado corre en un proceso aparte para liberar la memoria del modelo de PyTorch (≈ 2.5 GB)
    t_export = 0.0
    if not ruta_fp32.exists():
        inicio = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--exportar"], check=True)
        t_export = time.perf_counter() - inicio
    variantes = {"onnx_fp32": ruta_fp32}
    for nombre in ("onnx_emb8", "onnx_int8_matmul", "onnx_int8_pc", "onnx_int8_total"):
        destino = CARPETA_ONNX / f"router_{nombre.split('onnx_')[1]}.onnx"
        if not destino.exists():
            subprocess.run([sys.executable, __file__, "--cuantizar", nombre], check=True)
        variantes[nombre] = destino

    manifiesto = {
        "modelo_origen": "model-v1.0.0 (salida_router/modelo/model.safetensors)",
        "sha256_origen": sha256(CARPETA_MODELO / "model.safetensors"),
        "opset": 17, "torch": __import__("importlib.metadata").metadata.version("torch"), "segundos_exportacion": round(t_export, 1) if t_export else "no medido (el ONNX fp32 ya existía)",
        "variantes": {k: {"archivo": v.name, "mb": round(v.stat().st_size / 1e6, 1), "sha256": sha256(v)}
                      for k, v in variantes.items()},
    }
    (CARPETA_ONNX / "manifiesto_onnx.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cuantizar":
        cuantizar(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "--exportar":
        CARPETA_ONNX.mkdir(parents=True, exist_ok=True)
        exportar()
    else:
        main()
