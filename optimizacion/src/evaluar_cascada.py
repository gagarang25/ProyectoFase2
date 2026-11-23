"""Paso 3b · Palanca evaluada: router en cascada (TF-IDF + regresión logística → mDeBERTa solo si hay duda).

Idea: un modelo muy ligero (≈1 ms) responde los mensajes fáciles y el mDeBERTa solo los dudosos. Se entrena con el
conjunto de entrenamiento de data-v3.0.0 (3,200 mensajes, el mismo del router) y se evalúa con las mismas compuertas.
Resultado esperado para el reporte: cuánta carga se quita al modelo pesado y si cambia la calidad o la equidad.

Salidas: resultados/cascada_router.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from app.politica import decidir
from comun import CARPETA_RES, CLASES, RAIZ, UMBRAL, cargar_conjuntos, decisiones, metricas, tabla_decisiones


def main() -> None:
    tr = pd.read_csv(RAIZ / "datos" / "mensajes_cab" / "entrenamiento.csv")
    ligero = make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        LogisticRegression(C=20, max_iter=3000, random_state=42))
    ligero.fit(tr["mensaje"].str.lower(), tr["intencion"])
    idx = [list(ligero.classes_).index(c) for c in CLASES]
    datos = cargar_conjuntos()
    filas = []
    for c, df in datos.items():
        P = ligero.predict_proba(df["mensaje"].str.lower())[:, idx]
        dec_lig = pd.DataFrame([{"prediccion": (d := decidir(p, UMBRAL)).intencion, "confianza": d.confianza,
                                 "accion": d.accion} for p in P])
        base = decisiones(np.array(json.loads((CARPETA_RES / "crudo_onnx_emb8.json").read_text())
                                   ["salidas"][c]["logits"]))
        filas.append({"conjunto": c, "variante": "solo_mdeberta_emb8", "pct_a_mdeberta": 1.0, **metricas(df, base)})
        filas.append({"conjunto": c, "variante": "solo_ligero", "pct_a_mdeberta": 0.0, **metricas(df, dec_lig)})
        for tau in (0.90, 0.95, 0.98):
            usa = P.max(axis=1) >= tau
            cas = base.copy()
            cas.loc[usa] = dec_lig.loc[usa].values
            x = tabla_decisiones(df, cas)
            fila = {"conjunto": c, "variante": f"cascada_tau_{tau}", "pct_a_mdeberta": round(1 - usa.mean(), 4),
                    **metricas(df, cas)}
            for i in ("es", "en"):
                g = x[x["idioma"] == i]
                fila[f"seguros_{i}"] = round(float(g["seguro"].mean()), 4)
                fila[f"pct_a_mdeberta_{i}"] = round(float(1 - usa[(df["idioma"] == i).to_numpy()].mean()), 4)
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_RES / "cascada_router.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tabla)


if __name__ == "__main__":
    main()
