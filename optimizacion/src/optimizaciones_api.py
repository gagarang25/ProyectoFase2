"""Componentes optimizados que se conectan a la API de la Fase 2 SIN modificar backend/app.

RouterIntencionOnnx  Misma interfaz que app.router_intencion.RouterIntencion (cargado, info, logits, predecir),
                     pero con ONNX Runtime y la tabla de embeddings en INT8 (variante elegida: onnx_emb8).
                     No importa PyTorch, así que la imagen de Docker puede prescindir de él.
EtaPrecalculada      Inferencia por lotes (batch): calcula de una vez ETA, rango y riesgo de TODAS las órdenes
                     del SAP cuando se refrescan los datos, y en cada solicitud solo consulta el resultado.
                     La post-lógica (fechas, rango que no empieza en el pasado, motivos) es la misma de la Fase 2.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from comun import CARPETA_MODELO, CARPETA_ONNX, HILOS, MAX_LEN


class RouterIntencionOnnx:
    cargado = True

    def __init__(self, ruta_onnx: Path = CARPETA_ONNX / "router_emb8.onnx", carpeta: Path = CARPETA_MODELO,
                 hilos: int = HILOS):
        import json

        import onnxruntime as ort
        from tokenizers import Tokenizer

        from app.politica import decidir, softmax

        inicio = time.perf_counter()
        self._decidir, self._softmax = decidir, softmax
        self.cfg = json.loads((carpeta / "config_cab.json").read_text(encoding="utf-8"))
        op = ort.SessionOptions()
        op.intra_op_num_threads, op.inter_op_num_threads = hilos, 1
        op.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesion = ort.InferenceSession(str(ruta_onnx), op, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(carpeta / "tokenizer.json"))
        self.tok.enable_truncation(max_length=int(self.cfg.get("max_len", MAX_LEN)))
        self.temperatura = float(self.cfg["temperatura"])
        self.umbral = float(self.cfg["umbral_confianza"])
        self.ruta = ruta_onnx
        self.segundos_carga = round(time.perf_counter() - inicio, 2)

    def info(self) -> dict:
        return {"cargado": True, "modelo_base": self.cfg.get("modelo_base"),
                "version": "model-v1.0.0 (ONNX, embeddings INT8)", "runtime": "onnxruntime",
                "archivo": self.ruta.name, "umbral_confianza": self.umbral, "temperatura": self.temperatura,
                "segundos_carga": self.segundos_carga}

    def logits(self, texto: str) -> np.ndarray:
        e = self.tok.encode(texto)
        return self.sesion.run(["logits"], {"input_ids": np.array([e.ids], dtype=np.int64),
                                            "attention_mask": np.array([e.attention_mask], dtype=np.int64)})[0][0]

    def predecir(self, texto: str):
        return self._decidir(self._softmax(self.logits(texto), self.temperatura), self.umbral)


def clave_fila(valores) -> tuple:
    """Clave hashable de una fila (NaN → None, porque NaN != NaN rompería la búsqueda)."""
    return tuple(None if isinstance(v, float) and np.isnan(v) else v for v in valores)


class _Consulta:
    """Sustituye a un modelo de scikit-learn: devuelve el valor precalculado de la fila pedida."""

    def __init__(self, tabla: dict, original, columna: int | None = None):
        self.tabla, self.original, self.columna = tabla, original, columna
        self.aciertos = self.fallos = 0

    def _clave(self, X: pd.DataFrame):
        return clave_fila(X.iloc[0].tolist())

    def predict(self, X):
        k = self._clave(X)
        if k in self.tabla:
            self.aciertos += 1
            return np.array([self.tabla[k]])
        self.fallos += 1                       # orden nueva aún no precalculada: se calcula en línea
        return self.original.predict(X)

    def predict_proba(self, X):
        k = self._clave(X)
        if k in self.tabla:
            self.aciertos += 1
            p = self.tabla[k]
            return np.array([[1 - p, p]])
        self.fallos += 1
        return self.original.predict_proba(X)


def activar_eta_precalculada(entrega, sap) -> dict:
    """Precalcula en lote las 4 salidas para todas las órdenes del SAP y conecta las consultas."""
    ordenes = [sap._a_orden(f) for _, f in sap._df.iterrows()]
    X = pd.DataFrame([{c: o.datos_modelo.get(c, np.nan) for c in entrega.entradas} for o in ordenes])
    X = X.astype({c: "float64" for c in entrega.rangos})
    inicio = time.perf_counter()
    med, inf, sup = entrega.eta.predict(X), entrega.eta_inf.predict(X), entrega.eta_sup.predict(X)
    p = entrega.riesgo.predict_proba(X)[:, 1]
    seg = time.perf_counter() - inicio
    claves = [clave_fila(list(f)) for f in X.itertuples(index=False, name=None)]
    entrega.eta = _Consulta(dict(zip(claves, med)), entrega.eta)
    entrega.eta_inf = _Consulta(dict(zip(claves, inf)), entrega.eta_inf)
    entrega.eta_sup = _Consulta(dict(zip(claves, sup)), entrega.eta_sup)
    entrega.riesgo = _Consulta(dict(zip(claves, p)), entrega.riesgo)
    return {"ordenes_precalculadas": len(ordenes), "segundos_lote": round(seg, 3),
            "ms_por_orden_en_lote": round(seg * 1000 / max(1, len(ordenes)), 3)}
