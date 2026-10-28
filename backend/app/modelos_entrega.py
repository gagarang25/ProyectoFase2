"""Modelos tabulares de la API: ETA (regresión con rango) y riesgo de retraso (A6).

Además de la predicción, el servicio explica el "motivo" del ETA con reglas sobre las
mismas variables que ve el modelo (riesgo R08 de la Fase 1: ETA sin rango ni motivo) y
avisa cuando una entrada está fuera de lo que el modelo vio al entrenarse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class EstimacionEntrega:
    eta_dias: int
    eta_min_dias: int
    eta_max_dias: int
    prob_retraso: float
    riesgo_alto: bool
    motivos: list[str]
    advertencias: list[str] = field(default_factory=list)
    fecha_estimada: date | None = None
    fecha_min: date | None = None
    fecha_max: date | None = None
    vencida: bool = False  # la fecha estimada ya pasó y la orden no ha llegado

    def a_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("fecha_estimada", "fecha_min", "fecha_max"):
            d[k] = d[k].isoformat() if d[k] else None
        return d


class ModelosEntrega:
    def __init__(self, carpeta: Path):
        self.meta = json.loads((carpeta / "metadatos_modelos.json").read_text(encoding="utf-8"))
        self.eta = joblib.load(carpeta / "eta_mediana.joblib")
        self.eta_inf = joblib.load(carpeta / "eta_rango_inferior.joblib")
        self.eta_sup = joblib.load(carpeta / "eta_rango_superior.joblib")
        self.riesgo = joblib.load(carpeta / "riesgo_retraso.joblib")
        self.umbral_riesgo = float(self.meta["riesgo_retraso"]["umbral"])
        self.entradas: list[str] = self.meta["entradas"]
        self.categorias: dict[str, list] = self.meta["categoricas"]
        self.rangos: dict[str, list[float]] = self.meta["rangos_numericos_entrenamiento"]

    def info(self) -> dict:
        return {"version": self.meta["version_modelos"], "datos": self.meta["version_datos"],
                "eta_mae_prueba": round(self.meta["eta"]["metricas_prueba"]["mae_dias"], 3),
                "riesgo_auc_prueba": round(self.meta["riesgo_retraso"]["metricas_prueba"]["auc"], 3),
                "umbral_riesgo": self.umbral_riesgo}

    # ------------------------------------------------------------------
    def advertencias(self, datos: dict) -> list[str]:
        """Valores que el modelo nunca vio: la predicción es una extrapolación."""
        avisos = []
        for col, valores in self.categorias.items():
            if datos.get(col) is not None and datos[col] not in valores:
                avisos.append(f"{col}='{datos[col]}' no aparece en los datos de entrenamiento")
        for col, (minimo, maximo) in self.rangos.items():
            v = datos.get(col)
            if v is not None and not (minimo <= float(v) <= maximo):
                avisos.append(f"{col}={v} está fuera del rango de entrenamiento [{minimo:g}, {maximo:g}]")
        return avisos

    @staticmethod
    def motivos(d: dict, idioma: str = "es") -> list[str]:
        es = idioma == "es"
        m = []
        if not d.get("inventario_disponible"):
            m.append(f"las piezas requieren producción ({d.get('produccion_std_dias')} días estándar)" if es
                     else f"parts must be produced ({d.get('produccion_std_dias')} standard days)")
        u = d.get("utilizacion_planta") or 0
        if u > 0.88:
            m.append(f"la planta {d.get('planta')} está al {u:.0%} de su capacidad" if es
                     else f"the {d.get('planta')} plant is at {u:.0%} of capacity")
        if d.get("cruce_frontera"):
            m.append("el envío cruza la frontera (trámite aduanal)" if es else "the shipment crosses the border (customs)")
            if d.get("mes") == 12:
                m.append("en diciembre la aduana es más lenta" if es else "customs is slower in December")
        t = d.get("tasa_retraso_transportista_90d")
        if t is not None and not pd.isna(t) and t >= 0.25:
            m.append(f"el transportista {d.get('transportista')} tuvo {t:.0%} de entregas tardías en 90 días" if es
                     else f"carrier {d.get('transportista')} had {t:.0%} late deliveries in the last 90 days")
        if (d.get("cambios_pedido") or 0) > 0:
            m.append(f"la orden tuvo {d['cambios_pedido']} cambio(s)" if es else f"the order had {d['cambios_pedido']} change(s)")
        if d.get("region_destino") in ("Norte", "Noreste") and d.get("mes") in (6, 7, 8, 9):
            m.append("temporada de lluvias en la ruta" if es else "rainy season on the route")
        if (d.get("holgura_dias") if d.get("holgura_dias") is not None else 1) <= 0:
            m.append("la fecha comprometida no tenía holgura" if es else "the committed date had no buffer")
        return m or (["condiciones normales de producción y transporte"] if es else ["normal production and transport conditions"])

    def estimar(self, datos: dict, idioma: str = "es", fecha_pedido: date | None = None,
                hoy: date | None = None) -> EstimacionEntrega:
        X = pd.DataFrame([{c: datos.get(c, np.nan) for c in self.entradas}])
        X = X.astype({c: "float64" for c in self.rangos})  # None → NaN (lo imputa el modelo)
        mediana = float(self.eta.predict(X)[0])
        eta_dias = max(1, int(round(mediana)))
        eta_min = max(1, int(round(min(float(self.eta_inf.predict(X)[0]), mediana))))
        eta_max = max(eta_dias, int(round(max(float(self.eta_sup.predict(X)[0]), mediana))))
        p = float(self.riesgo.predict_proba(X)[0, 1])
        est = EstimacionEntrega(eta_dias=eta_dias, eta_min_dias=eta_min, eta_max_dias=eta_max,
                                prob_retraso=round(p, 4), riesgo_alto=p >= self.umbral_riesgo,
                                motivos=self.motivos(datos, idioma), advertencias=self.advertencias(datos))
        if fecha_pedido:
            est.fecha_estimada = fecha_pedido + timedelta(days=eta_dias)
            est.fecha_min = fecha_pedido + timedelta(days=eta_min)
            est.fecha_max = fecha_pedido + timedelta(days=eta_max)
            if hoy and est.fecha_min <= hoy:
                # Una orden que no ha llegado no puede llegar antes de mañana: el rango no empieza en el pasado.
                est.fecha_min = hoy + timedelta(days=1)
                est.fecha_max = max(est.fecha_max, est.fecha_min)
            if hoy and est.fecha_estimada <= hoy:
                # El modelo esperaba que ya hubiera llegado: no se informa una fecha pasada.
                est.vencida = True
                est.fecha_estimada = hoy + timedelta(days=1)
                est.fecha_min = hoy + timedelta(days=1)
                est.fecha_max = max(est.fecha_max, hoy + timedelta(days=2))
                est.riesgo_alto = True
        return est
