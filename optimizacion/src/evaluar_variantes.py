"""Paso 3 · Compara las variantes medidas contra la línea base (PyTorch fp32, la que sirve hoy la API).

Compuerta de calidad definida ANTES de medir (reporte A9, sección 3):
  Q1  resultados seguros en la prueba externa  ≥ línea base − 1 punto
  Q2  respuestas automáticas erróneas (externa) ≤ línea base
  Q3  accuracy por idioma (es, en)              ≥ línea base − 2 puntos  (equidad entre idiomas, riesgo R02)
  Q4  acciones iguales a la línea base          ≥ 97 % en las 560 frases
Compuerta de desempeño: p95 de latencia del router ≤ línea base (no puede empeorar el SLO de 400 ms).

Salidas: resultados/comparacion_variantes.csv, resultados/equidad_idioma.csv, resultados/resumen_optimizacion.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from comun import CARPETA_RES, RAIZ, cargar_conjuntos, decisiones, metricas, tabla_decisiones

VARIANTES = ["pytorch_fp32", "onnx_fp32", "onnx_emb8", "onnx_int8_matmul", "onnx_int8_pc", "onnx_int8_total"]
ELEGIDA = "onnx_emb8"


def equidad_por_idioma(x: pd.DataFrame) -> pd.DataFrame:
    """Métricas de fairness por grupo de idioma (el atributo sensible disponible en el chatbot)."""
    filas = []
    for idioma, g in x.groupby("idioma"):
        claros = g[g["claro"]]
        aut = g[g["automatizable"]]
        auto = g["accion"] == "automatica"
        filas.append({
            "idioma": idioma, "mensajes": len(g),
            "accuracy_claros": claros["acierto"].mean(),
            # Paridad demográfica: % de mensajes que recibe respuesta inmediata
            "tasa_respuesta_automatica": auto.mean(),
            # Igualdad de oportunidades: de lo que SÍ se podía responder solo, cuánto se respondió solo
            "tpr_automatizable": (aut["accion"] == "automatica").mean() if len(aut) else np.nan,
            # Paridad predictiva: de lo que se respondió solo, cuánto fue correcto
            "precision_automatica": g.loc[auto, "seguro"].mean() if auto.any() else np.nan,
            "resultados_seguros": g["seguro"].mean(),
            # KPI R02 de la Fase 1: mensajes no entendidos (baja confianza) ≤ 10 % por idioma
            "baja_confianza": (g["confianza"] < 0.90).mean(),
        })
    return pd.DataFrame(filas)


def main() -> None:
    datos = cargar_conjuntos()
    crudos = {v: json.loads((CARPETA_RES / f"crudo_{v}.json").read_text(encoding="utf-8")) for v in VARIANTES}

    # Verificación 1: la línea base reproduce las decisiones registradas en la Fase 2 (model-v1.0.0)
    reg = pd.read_csv(RAIZ / "resultados" / "router" / "predicciones_externa.csv")
    base_ext = decisiones(np.array(crudos["pytorch_fp32"]["salidas"]["externa"]["logits"]))
    coincidencia_fase2 = float((base_ext["accion"].to_numpy() == reg["accion"].to_numpy()).mean())
    # Verificación 2: el tokenizador sin PyTorch produce exactamente los mismos tokens
    tokens_iguales = all(crudos[v]["salidas"][c]["ids"] == crudos["pytorch_fp32"]["salidas"][c]["ids"]
                         for v in VARIANTES[1:] for c in ("prueba", "externa"))

    filas, equidad, tablas = [], [], {}
    for v in VARIANTES:
        med = crudos[v]["medicion"]
        fila = {"variante": v, **{k: med[k] for k in ("segundos_carga", "rss_mb_tras_cargar", "rss_mb_pico",
                                                       "throughput_msg_s_lote1", "throughput_msg_s_lote8")},
                **{f"latencia_{k}_ms": med["latencia_ms"][k] for k in ("p50", "p95", "p99")}}
        for c in ("prueba", "externa"):
            lg = np.array(crudos[v]["salidas"][c]["logits"])
            dec = decisiones(lg)
            tablas[(v, c)] = tabla_decisiones(datos[c], dec)
            for k, val in metricas(datos[c], dec).items():
                if k != "mensajes":
                    fila[f"{c}_{k}"] = val
            base = np.array(crudos["pytorch_fp32"]["salidas"][c]["logits"])
            fila[f"{c}_max_dif_logit"] = round(float(np.abs(lg - base).max()), 4)
        acc_v = pd.concat([tablas[(v, c)]["accion"] for c in ("prueba", "externa")]).to_numpy()
        acc_b = pd.concat([tablas[("pytorch_fp32", c)]["accion"] for c in ("prueba", "externa")]).to_numpy()
        fila["acciones_iguales_a_base"] = round(float((acc_v == acc_b).mean()), 4)
        filas.append(fila)
        eq = equidad_por_idioma(pd.concat([tablas[(v, "prueba")], tablas[(v, "externa")]], ignore_index=True))
        eq.insert(0, "conjunto", "prueba+externa")
        eq2 = equidad_por_idioma(tablas[(v, "externa")])
        eq2.insert(0, "conjunto", "externa")
        for e in (eq, eq2):
            e.insert(0, "variante", v)
            equidad.append(e)

    comp = pd.DataFrame(filas)
    eqd = pd.concat(equidad, ignore_index=True).round(4)

    # Compuertas
    b = comp.set_index("variante").loc["pytorch_fp32"]
    eqx = eqd[eqd["conjunto"] == "externa"].set_index(["variante", "idioma"])
    compuertas = {}
    for v in VARIANTES[1:]:
        f = comp.set_index("variante").loc[v]
        q3 = all(eqx.loc[(v, i), "accuracy_claros"] >= eqx.loc[("pytorch_fp32", i), "accuracy_claros"] - 0.02
                 for i in ("es", "en"))
        compuertas[v] = {
            "Q1_seguros": bool(f["externa_resultados_seguros"] >= b["externa_resultados_seguros"] - 0.01),
            "Q2_auto_erroneas": bool(f["externa_respuestas_automaticas_erroneas"]
                                     <= b["externa_respuestas_automaticas_erroneas"]),
            "Q3_equidad_idioma": bool(q3),
            "Q4_acciones_iguales": bool(f["acciones_iguales_a_base"] >= 0.97),
            "D1_latencia_p95": bool(f["latencia_p95_ms"] <= b["latencia_p95_ms"]),
        }
        compuertas[v]["aprobada"] = all(compuertas[v].values())

    # Mensajes donde una variante cambia la acción respecto a la línea base (para el reporte)
    cambios = []
    for c, v in (("prueba", ELEGIDA), ("externa", ELEGIDA), ("prueba", "onnx_int8_pc"), ("externa", "onnx_int8_pc")):
        a, z = tablas[("pytorch_fp32", c)], tablas[(v, c)]
        dif = a["accion"] != z["accion"]
        for i in np.where(dif)[0]:
            cambios.append({"variante": v, "conjunto": c, "mensaje": a.loc[i, "mensaje"], "idioma": a.loc[i, "idioma"],
                            "intencion": a.loc[i, "intencion"], "accion_base": a.loc[i, "accion"],
                            "accion_variante": z.loc[i, "accion"], "conf_base": a.loc[i, "confianza"],
                            "conf_variante": z.loc[i, "confianza"], "seguro_variante": bool(z.loc[i, "seguro"])})

    comp.to_csv(CARPETA_RES / "comparacion_variantes.csv", index=False)
    eqd.to_csv(CARPETA_RES / "equidad_idioma.csv", index=False)
    pd.DataFrame(cambios).to_csv(CARPETA_RES / "cambios_de_accion.csv", index=False)
    resumen = {"coincidencia_base_vs_fase2_externa": coincidencia_fase2, "tokens_iguales_sin_pytorch": tokens_iguales,
               "compuertas": compuertas, "variante_elegida": ELEGIDA,
               "cambios_de_accion": {v: sum(1 for x in cambios if x["variante"] == v) for v in (ELEGIDA, "onnx_int8_pc")}}
    (CARPETA_RES / "resumen_optimizacion.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(comp.T)
    print(eqd[eqd.conjunto == "externa"])
    print(json.dumps(resumen, indent=2))
    print(pd.DataFrame(cambios).groupby("variante").size())


if __name__ == "__main__":
    main()
