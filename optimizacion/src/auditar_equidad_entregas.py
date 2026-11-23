"""Paso 6 · Auditoría de equidad (fairness) de los modelos de ETA y de riesgo de retraso por segmento de cliente.

En un chatbot B2B no hay atributos protegidos de personas (género, edad, etnia). El riesgo de trato desigual está
entre CLIENTES: si el modelo avisa menos retrasos a un tipo de cliente o le da fechas menos confiables, ese cliente
recibe peor servicio. Por eso los "grupos" de la auditoría son tipo de cliente (armadora vs tier 1), región de
destino, cruce de frontera y transportista. Datos: prueba temporal de la Fase 2 (5,792 órdenes simuladas).

Métricas por grupo (T22, tabla 4):
  tasa_alerta            paridad demográfica (¿a quién se le avisa más?)
  recall (TPR)           igualdad de oportunidades (de los retrasos reales, ¿cuántos se avisaron?) ← métrica principal
  fpr                    junto con el recall, equalized odds (falsas alarmas)
  precision              paridad predictiva (de las alertas, ¿cuántas fueron retraso real?)
  mae_dias / cobertura   calidad de la fecha informada (KPI R08: cobertura ≥ 85 %)
Regla de alerta de la auditoría (definida antes de calcular): brecha de recall > 0.10 frente al mejor grupo o
razón de recall < 0.80 (regla de las cuatro quintas partes aplicada a la tasa de detección), o cobertura < 0.85.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from comun import CARPETA_RES, RAIZ

GRUPOS = ["tipo_cliente", "region_destino", "cruce_frontera", "transportista", "modalidad"]


def metricas(g: pd.DataFrame) -> dict:
    y, a = g["retraso"].astype(bool), g["alerta"].astype(bool)
    return {
        "ordenes": len(g), "tasa_retraso_real": y.mean(), "tasa_alerta": a.mean(),
        "recall": (a & y).sum() / max(1, y.sum()), "fpr": (a & ~y).sum() / max(1, (~y).sum()),
        "precision": (a & y).sum() / max(1, a.sum()),
        "mae_dias": (g["eta_dias"] - g["dias_reales"]).abs().mean(),
        "cobertura_rango": ((g["dias_reales"] >= g["eta_min"]) & (g["dias_reales"] <= g["eta_max"])).mean(),
    }


def main() -> None:
    d = pd.read_csv(RAIZ / "resultados" / "entrega" / "predicciones_prueba.csv")
    total = {"variable": "TOTAL", "grupo": "todas", **metricas(d)}
    filas = [total]
    hallazgos = []
    for var in GRUPOS:
        sub = pd.DataFrame([{"variable": var, "grupo": str(k), **metricas(g)} for k, g in d.groupby(var)])
        mejor = sub["recall"].max()
        sub["brecha_recall"] = mejor - sub["recall"]
        sub["razon_recall"] = sub["recall"] / mejor
        sub["razon_tasa_alerta"] = sub["tasa_alerta"] / sub["tasa_alerta"].max()
        sub["alerta_auditoria"] = (sub["brecha_recall"] > 0.10) | (sub["razon_recall"] < 0.80) | \
                                  (sub["cobertura_rango"] < 0.85)
        for _, f in sub[sub["alerta_auditoria"]].iterrows():
            hallazgos.append({"variable": var, "grupo": f["grupo"], "recall": round(f["recall"], 3),
                              "brecha_recall": round(f["brecha_recall"], 3), "cobertura": round(f["cobertura_rango"], 3),
                              "tasa_retraso_real": round(f["tasa_retraso_real"], 3)})
        filas.extend(sub.to_dict("records"))
    tabla = pd.DataFrame(filas).round(4)
    tabla.to_csv(CARPETA_RES / "equidad_entregas.csv", index=False)
    (CARPETA_RES / "equidad_entregas_hallazgos.json").write_text(json.dumps(hallazgos, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(tabla.drop(columns=["razon_tasa_alerta"]))
    print(json.dumps(hallazgos, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
