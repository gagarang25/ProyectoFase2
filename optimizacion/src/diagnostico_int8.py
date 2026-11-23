"""Paso 3c · Diagnóstico: ¿por qué falla la cuantización INT8 de las capas lineales?

Se cuantizan solo algunas capas del modelo (cuantización selectiva) para ubicar dónde se pierde la calidad.
Cada variante se evalúa con la prueba externa (160) y la interna (400), sin medir latencia.
Salida: resultados/diagnostico_int8.csv
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from comun import CARPETA_MODELO, CARPETA_ONNX, CARPETA_RES, MAX_LEN, cargar_conjuntos, decisiones, metricas

VARIANTES = {   # nombre: (descripción, filtro de nodos MatMul)
    "ffn_12_capas": ("Solo la red feed-forward (FFN) de las 12 capas", "ffn"),
    "capas_0_5": ("Proyecciones y FFN de las capas 0 a 5", "L0_6"),
    "capas_6_11": ("Proyecciones y FFN de las capas 6 a 11", "L6_12"),
}


def cuantizar(modo: str) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m = onnx.load(str(CARPETA_ONNX / "router_fp32.onnx"), load_external_data=False)
    mm = [n.name for n in m.graph.node if n.op_type == "MatMul"]
    con_pesos = [n for n in mm if "_proj/" in n or "dense" in n]   # excluye MatMul entre activaciones
    if modo == "ffn":
        nodos = [n for n in mm if "/intermediate/dense" in n or ("/output/dense" in n and "attention" not in n)]
    else:
        a, b = map(int, modo[1:].split("_"))
        nodos = [n for n in con_pesos if any(f"/layer.{i}/" in n for i in range(a, b))]
    quantize_dynamic(CARPETA_ONNX / "router_fp32.onnx", CARPETA_ONNX / f"diag_{modo}.onnx",
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], nodes_to_quantize=nodos,
                     per_channel=True, extra_options={"MatMulConstBOnly": True})


def evaluar(ruta) -> dict:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    op = ort.SessionOptions()
    op.intra_op_num_threads = 2
    s = ort.InferenceSession(str(ruta), op, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(CARPETA_MODELO / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    res = {}
    for c, df in cargar_conjuntos().items():
        lg = []
        for t in df["mensaje"]:
            e = tok.encode(t)
            lg.append(s.run(["logits"], {"input_ids": np.array([e.ids]), "attention_mask": np.array([e.attention_mask])})[0][0])
        for k, v in metricas(df, decisiones(np.array(lg))).items():
            res[f"{c}_{k}"] = v
    return res


def main() -> None:
    # Primero se cuantiza todo (cada una en su propio proceso) y después se evalúa: así nunca hay dos modelos
    # de ~1 GB en memoria al mismo tiempo.
    for _, modo in VARIANTES.values():
        if not (CARPETA_ONNX / f"diag_{modo}.onnx").exists():
            subprocess.run([sys.executable, __file__, "--cuantizar", modo], check=True)
    filas = []
    for nombre, (desc, modo) in VARIANTES.items():
        ruta = CARPETA_ONNX / f"diag_{modo}.onnx"
        filas.append({"variante": nombre, "descripcion": desc, **evaluar(ruta)})
        ruta.unlink()   # solo se conservan los resultados
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_RES / "diagnostico_int8.csv", index=False)
    print(tabla[["variante", "externa_accuracy", "externa_cobertura_automatica",
                 "externa_respuestas_automaticas_erroneas", "prueba_accuracy"]])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cuantizar":
        cuantizar(sys.argv[2])
    else:
        main()
