"""Paso 2 · Mide UNA variante del router en un proceso limpio (para medir bien la memoria).

Mide, con 2 hilos de CPU (como la API y como una instancia de 2 vCPU):
  * calidad: logits de la prueba interna (400) y de la prueba externa escrita a mano (160);
  * latencia por mensaje con lote de 1 (así llega un chat): p50, p95, p99;
  * throughput: mensajes por segundo con lote 1 y con micro-lotes de 8;
  * memoria: RSS del proceso tras importar librerías, tras cargar el modelo y pico tras inferir;
  * tiempo de carga (arranque en frío de una réplica).

Las variantes ONNX NO importan PyTorch: así se mide la imagen que se desplegaría sin torch.

Uso:  python optimizacion/src/medir_variante.py <variante>
      variante ∈ {pytorch_fp32, onnx_fp32, onnx_int8_matmul, onnx_int8_total}
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np
import psutil

VARIANTE = sys.argv[1]
PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


class RouterTorch:
    def __init__(self, carpeta, max_len, hilos):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(hilos)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(carpeta)
        self.m = AutoModelForSequenceClassification.from_pretrained(carpeta, dtype=torch.float32).eval()
        self.max_len = max_len

    def ids(self, textos):
        e = self.tok(list(textos), truncation=True, max_length=self.max_len, padding=True, return_tensors="np")
        return e["input_ids"].astype(np.int64), e["attention_mask"].astype(np.int64)

    def logits(self, ids, mask):
        with self.torch.no_grad():
            return self.m(input_ids=self.torch.from_numpy(ids),
                          attention_mask=self.torch.from_numpy(mask)).logits.float().numpy()


class RouterOnnx:
    def __init__(self, ruta_onnx, carpeta, max_len, hilos):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = hilos
        opciones.inter_op_num_threads = 1
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(str(ruta_onnx), opciones, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(carpeta / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def ids(self, textos):
        enc = self.tok.encode_batch(list(textos))
        return (np.array([e.ids for e in enc], dtype=np.int64),
                np.array([e.attention_mask for e in enc], dtype=np.int64))

    def logits(self, ids, mask):
        return self.s.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]


def main() -> None:
    from comun import CARPETA_MODELO, CARPETA_ONNX, CARPETA_RES, HILOS, MAX_LEN, cargar_conjuntos

    datos = cargar_conjuntos()
    rss_importar = rss_mb()
    inicio = time.perf_counter()
    if VARIANTE == "pytorch_fp32":
        r = RouterTorch(CARPETA_MODELO, MAX_LEN, HILOS)
    else:
        r = RouterOnnx(CARPETA_ONNX / f"router_{VARIANTE.removeprefix('onnx_')}.onnx", CARPETA_MODELO, MAX_LEN, HILOS)
    seg_carga = time.perf_counter() - inicio
    rss_carga = rss_mb()

    # Calidad: un mensaje a la vez, igual que la API
    salida = {}
    for nombre, df in datos.items():
        filas, ids_guardados = [], []
        for texto in df["mensaje"]:
            ids, mask = r.ids([texto])
            filas.append(r.logits(ids, mask)[0])
            ids_guardados.append(ids[0].tolist())
        salida[nombre] = {"logits": np.array(filas).tolist(), "ids": ids_guardados}

    # Latencia con lote 1: calentamiento + 2 pasadas por los 560 mensajes
    todos = list(datos["prueba"]["mensaje"]) + list(datos["externa"]["mensaje"])
    for t in todos[:30]:
        r.logits(*r.ids([t]))
    tiempos = []
    for _ in range(2):
        for t in todos:
            a = time.perf_counter()
            r.logits(*r.ids([t]))
            tiempos.append((time.perf_counter() - a) * 1000)
    t = np.array(tiempos)

    # Throughput con micro-lotes de 8 (lo que haría un servidor que agrupa solicitudes simultáneas)
    a = time.perf_counter()
    for i in range(0, len(todos), 8):
        r.logits(*r.ids(todos[i:i + 8]))
    seg_lote8 = time.perf_counter() - a

    res = {
        "variante": VARIANTE, "hilos": HILOS,
        "segundos_carga": round(seg_carga, 2),
        "rss_mb_tras_importar": round(rss_importar, 0),
        "rss_mb_tras_cargar": round(rss_carga, 0),
        "rss_mb_pico": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6, 0),
        "rss_mb_modelo": round(rss_carga - rss_importar, 0),
        "latencia_ms": {"media": round(float(t.mean()), 1), "p50": round(float(np.percentile(t, 50)), 1),
                        "p95": round(float(np.percentile(t, 95)), 1), "p99": round(float(np.percentile(t, 99)), 1),
                        "mediciones": int(len(t))},
        "throughput_msg_s_lote1": round(1000 / float(t.mean()), 1),
        "throughput_msg_s_lote8": round(len(todos) / seg_lote8, 1),
    }
    CARPETA_RES.mkdir(parents=True, exist_ok=True)
    (CARPETA_RES / f"crudo_{VARIANTE}.json").write_text(json.dumps({"medicion": res, "salidas": salida}),
                                                        encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
