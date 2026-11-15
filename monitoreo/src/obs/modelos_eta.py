"""Entrenamiento y predicción del modelo de ETA reutilizando el código de la Fase 2.

Se importan el preprocesador y los hiperparámetros de entrenamiento/entrenar_modelos_entrega.py
para que un reentrenamiento use EXACTAMENTE la misma receta que model-v1.0.0.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .config import RAIZ_BACKEND, RAIZ_REPO

DIR_MOD = RAIZ_BACKEND / "modelos"
SEMILLA = 42


@lru_cache(maxsize=1)
def metadatos() -> dict:
    return json.loads((DIR_MOD / "metadatos_modelos.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def receta():
    spec = importlib.util.spec_from_file_location("entrenar", RAIZ_REPO / "entrenamiento" / "entrenar_modelos_entrega.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@lru_cache(maxsize=1)
def generador():
    spec = importlib.util.spec_from_file_location("generador", RAIZ_REPO / "entrenamiento" / "generar_datos_entregas.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def entradas() -> list[str]:
    return metadatos()["entradas"]


def cargar_v1() -> dict:
    """model-v1.0.0 tal como lo sirve la API de la Fase 2."""
    return {"mediana": joblib.load(DIR_MOD / "eta_mediana.joblib"),
            "inferior": joblib.load(DIR_MOD / "eta_rango_inferior.joblib"),
            "superior": joblib.load(DIR_MOD / "eta_rango_superior.joblib"), "desfase": 0}


def _pipeline(loss: str, cuantil: float | None = None) -> Pipeline:
    p = metadatos()["eta"]["parametros"]
    kw = dict(loss=loss, max_iter=300, early_stopping=False, random_state=SEMILLA, **p)
    if cuantil is not None:
        kw["quantile"] = cuantil
    return Pipeline([("prep", receta().preprocesador()), ("modelo", HistGradientBoostingRegressor(**kw))])


def entrenar(df: pd.DataFrame, pesos: np.ndarray | None = None) -> dict:
    q_lo, q_hi = metadatos()["eta"]["cuantiles_rango"]
    X, y = df[entradas()], df["dias_reales"].to_numpy()
    modelos = {"mediana": _pipeline("absolute_error"), "inferior": _pipeline("quantile", q_lo),
               "superior": _pipeline("quantile", q_hi), "desfase": 0}
    for n in ("mediana", "inferior", "superior"):
        modelos[n].fit(X, y, modelo__sample_weight=pesos)
    return modelos


def predecir(modelos: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ETA en días enteros y rango, con la misma lógica de redondeo que la API."""
    med = modelos["mediana"].predict(X[entradas()])
    eta = np.maximum(1, np.rint(med))
    k = modelos.get("desfase", 0)
    lo = np.maximum(1, np.rint(np.minimum(modelos["inferior"].predict(X[entradas()]), med)) - k)  # nunca < 1 día
    hi = np.rint(np.maximum(modelos["superior"].predict(X[entradas()]), med)) + k
    return eta, lo, hi


def evaluar(modelos: dict, df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"ordenes": 0}
    eta, lo, hi = predecir(modelos, df)
    y = df["dias_reales"].to_numpy()
    return {"ordenes": int(len(df)), "mae": float(np.mean(np.abs(eta - y))),
            "cobertura": float(np.mean((y >= lo) & (y <= hi))), "ancho_rango": float(np.mean(hi - lo)),
            "dentro_1_dia": float(np.mean(np.abs(eta - y) <= 1)), "sesgo": float(np.mean(eta - y))}
