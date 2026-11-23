"""Utilidades comunes de la Actividad 9 (optimización del router de intención).

Todas las variantes se evalúan con la MISMA política de decisión que usa la API
(backend/app/politica.py), con la misma temperatura y el mismo umbral de model-v1.0.0.
Así, cualquier diferencia de calidad se debe solo a la optimización.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[2]          # ProyectoFase2/
sys.path.insert(0, str(RAIZ / "backend"))

from app.politica import CLASES, decidir, softmax  # noqa: E402

CARPETA_MODELO = RAIZ / "salida_router" / "modelo"
CARPETA_ONNX = RAIZ / "optimizacion" / "modelos_onnx"      # no se sube a Git (pesado); se regenera
CARPETA_RES = RAIZ / "optimizacion" / "resultados"
CFG = json.loads((CARPETA_MODELO / "config_cab.json").read_text(encoding="utf-8"))
TEMPERATURA = float(CFG["temperatura"])
UMBRAL = float(CFG["umbral_confianza"])
MAX_LEN = int(CFG["max_len"])
HILOS = 2   # igual que la API (ROUTER_HILOS=2) y que una instancia de 2 vCPU


def cargar_conjuntos() -> dict[str, pd.DataFrame]:
    d = RAIZ / "datos" / "mensajes_cab"
    prueba = pd.read_csv(d / "prueba.csv")
    prueba["aceptables"] = prueba["intencion"]
    prueba["tipo"] = "plantilla"
    externa = pd.read_csv(d / "prueba_externa_v3.csv")
    return {"prueba": prueba, "externa": externa}


def decisiones(logits: np.ndarray) -> pd.DataFrame:
    filas = []
    for lg in logits:
        dec = decidir(softmax(lg, TEMPERATURA), UMBRAL)
        filas.append({"prediccion": dec.intencion, "confianza": dec.confianza, "accion": dec.accion})
    return pd.DataFrame(filas)


def tabla_decisiones(df: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    """Une mensajes y decisiones con las MISMAS definiciones de la Fase 2 (entrenar_router.evaluar)."""
    x = pd.concat([df.reset_index(drop=True), dec], axis=1)
    acept = [set(str(a).split("|")) for a in x["aceptables"]]
    x["acierto"] = x["prediccion"] == x["intencion"]
    auto = x["accion"] == "automatica"
    x["seguro"] = [(not au) or (p in a) for au, p, a in zip(auto, x["prediccion"], acept)]
    x["auto_erronea"] = auto & ~x["seguro"]
    x["claro"] = x["intencion"].isin(CLASES) & x["tipo"].isin(["claro", "spanglish", "plantilla"])
    # "Automatizable": el cliente pidió algo que el chatbot SÍ puede responder solo (estatus o fecha)
    x["automatizable"] = x["intencion"].isin(["estatus_orden", "fecha_entrega"]) & x["claro"]
    return x


def metricas(df: pd.DataFrame, dec: pd.DataFrame) -> dict:
    x = tabla_decisiones(df, dec)
    auto = x["accion"] == "automatica"
    claros = x[x["claro"]]
    return {
        "mensajes": int(len(x)),
        "accuracy": round(float(claros["acierto"].mean()), 4),
        "f1_macro": round(float(f1_score(claros["intencion"], claros["prediccion"], labels=CLASES,
                                         average="macro", zero_division=0)), 4),
        "cobertura_automatica": round(float(auto.mean()), 4),
        "precision_automatica": round(float(x.loc[auto, "seguro"].mean()), 4) if auto.any() else None,
        "resultados_seguros": round(float(x["seguro"].mean()), 4),
        "respuestas_automaticas_erroneas": int(x["auto_erronea"].sum()),
        "baja_confianza": round(float((x["confianza"] < UMBRAL).mean()), 4),
    }
