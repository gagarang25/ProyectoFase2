"""
estres_drift.py
---------------
Pruebas de estrés y de deriva (drift) de los modelos de ETA y riesgo de retraso, y
análisis de los retrasos que el modelo no anticipa (falsos negativos). Atiende las dos
oportunidades de mejora de la Actividad 6:
  1. Profundizar en los retrasos no anticipados (los 272 falsos negativos de prueba).
  2. Evaluar la estabilidad del modelo ante cambios en las condiciones logísticas.

Cómo se simula el drift
  El generador de la A6 (entrenamiento/generar_datos_entregas.py) se parametrizó con
  un "escenario". Con el escenario base reproduce data-v1.0.0 (mismo SHA-256). Cada
  escenario cambia una condición del mundo (un transportista empeora, la planta se
  satura, la aduana se vuelve lenta, hay más eventos imprevisibles) y se genera un
  trimestre nuevo (oct-dic 2026) con otra semilla. Los modelos NO se reentrenan: se
  mide cuánto empeoran con datos que ya no se parecen a los de entrenamiento.

Uso:
    python pruebas/estres_drift.py
Salida:
    resultados/estres_drift/ (CSV, JSON y figuras)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, roc_auc_score  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DIR_MOD = RAIZ / "backend" / "modelos"
SALIDA = RAIZ / "resultados" / "estres_drift"
COLOR, COLOR_2, GRIS = "#0e5c4a", "#e07b39", "#9aa5a0"
SEMILLA_ESCENARIOS = 7
VENTANA = ("2026-10-01", "2026-12-31")

# Alarmas de monitoreo propuestas (riesgo R10). Se fijan antes de ver los resultados.
ALARMAS = {"mae_dias": 1.30, "cobertura_rango": 0.85, "recall": 0.75, "psi": 0.20}

ESCENARIOS = {
    "Base (otra semilla)": {},
    "Transportista TR-B se deteriora": {"retraso_transportista": {"TR-B": 1.30}},
    "Planta saturada (+10 puntos)": {"utilizacion_extra": 0.10},
    "Aduana lenta todo el año": {"aduana_media": 1.8, "aduana_media_diciembre": 2.5},
    "Más eventos imprevisibles (10 %)": {"prob_imprevisto": 0.10},
    "Combinado moderado": {"retraso_transportista": {"TR-B": 0.8}, "utilizacion_extra": 0.05,
                           "aduana_media": 1.0, "prob_imprevisto": 0.05},
}


def cargar_generador():
    spec = importlib.util.spec_from_file_location("gen", RAIZ / "entrenamiento" / "generar_datos_entregas.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def psi(referencia: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: cuánto cambió la distribución de una variable.
    < 0.1 sin cambio, 0.1-0.2 cambio moderado, > 0.2 cambio importante."""
    referencia, actual = referencia[~np.isnan(referencia)], actual[~np.isnan(actual)]
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    r = np.histogram(referencia, cortes)[0] / len(referencia) + 1e-4
    a = np.histogram(actual, cortes)[0] / len(actual) + 1e-4
    return float(np.sum((a - r) * np.log(a / r)))


class Modelos:
    def __init__(self):
        self.meta = json.loads((DIR_MOD / "metadatos_modelos.json").read_text(encoding="utf-8"))
        self.entradas = self.meta["entradas"]
        self.eta = joblib.load(DIR_MOD / "eta_mediana.joblib")
        self.inf = joblib.load(DIR_MOD / "eta_rango_inferior.joblib")
        self.sup = joblib.load(DIR_MOD / "eta_rango_superior.joblib")
        self.riesgo = joblib.load(DIR_MOD / "riesgo_retraso.joblib")
        self.umbral = self.meta["riesgo_retraso"]["umbral"]

    def evaluar(self, df: pd.DataFrame) -> dict:
        X = df[self.entradas]
        med = self.eta.predict(X)
        eta = np.maximum(1, np.rint(med))
        lo = np.maximum(1, np.rint(np.minimum(self.inf.predict(X), med)))
        hi = np.rint(np.maximum(self.sup.predict(X), med))
        y = df["dias_reales"].to_numpy()
        p = self.riesgo.predict_proba(X)[:, 1]
        alerta = (p >= self.umbral).astype(int)
        r = df["retraso"].to_numpy()
        return {
            "ordenes": len(df), "tasa_retraso_real": float(r.mean()),
            "mae_dias": float(mean_absolute_error(y, eta)), "sesgo_dias": float(np.mean(eta - y)),
            "dentro_1_dia": float(np.mean(np.abs(eta - y) <= 1)),
            "cobertura_rango": float(np.mean((y >= lo) & (y <= hi))),
            "mae_sap": float(mean_absolute_error(y, df["dias_comprometidos"])),
            "auc": float(roc_auc_score(r, p)), "recall": float(recall_score(r, alerta)),
            "precision": float(precision_score(r, alerta, zero_division=0)), "tasa_alertas": float(alerta.mean()),
            "prob_media": float(p.mean()),
        }


def main() -> None:
    (SALIDA / "figuras").mkdir(parents=True, exist_ok=True)
    gen = cargar_generador()
    modelos = Modelos()
    prueba_pred = pd.read_csv(RAIZ / "resultados" / "entrega" / "predicciones_prueba.csv", parse_dates=["fecha_pedido"])

    # ------------------------------------------------------------ 1. Escenarios de drift
    # El PSI se calcula contra el mismo trimestre del escenario base: así mide el efecto del
    # cambio y no la variación natural entre trimestres (la carga de planta es semanal y en un
    # trimestre solo hay 39 valores distintos, por lo que contra todo el histórico siempre "cambia").
    filas, referencia = [], None
    for nombre, esc in ESCENARIOS.items():
        df = gen.simular(SEMILLA_ESCENARIOS, esc)
        ventana = df[(df["fecha_pedido"] >= VENTANA[0]) & (df["fecha_pedido"] <= VENTANA[1])]
        if referencia is None:
            referencia = ventana
        m = modelos.evaluar(ventana)
        for var in ("utilizacion_planta", "tasa_retraso_transportista_90d"):
            m[f"psi_{var}"] = psi(referencia[var].to_numpy(dtype=float), ventana[var].to_numpy(dtype=float))
        alarmas = []
        if m["mae_dias"] > ALARMAS["mae_dias"]:
            alarmas.append("MAE")
        if m["cobertura_rango"] < ALARMAS["cobertura_rango"]:
            alarmas.append("cobertura")
        if m["recall"] < ALARMAS["recall"]:
            alarmas.append("recall")
        if max(m["psi_utilizacion_planta"], m["psi_tasa_retraso_transportista_90d"]) > ALARMAS["psi"]:
            alarmas.append("PSI")
        m["alarmas"] = ", ".join(alarmas) or "ninguna"
        filas.append({"escenario": nombre, **m})
        print(f"{nombre:36s} MAE {m['mae_dias']:.2f} | cobertura {m['cobertura_rango']:.3f} | "
              f"recall {m['recall']:.3f} | AUC {m['auc']:.3f} | retraso real {m['tasa_retraso_real']:.3f} | "
              f"alarmas: {m['alarmas']}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / "escenarios_drift.csv", index=False)

    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.2))
    nombres = [n.replace(" (", "\n(").replace(" se ", "\nse ").replace(" todo", "\ntodo") for n in tabla["escenario"]]
    for ax, col, titulo, limite in ((ejes[0], "mae_dias", "MAE del ETA (días)", ALARMAS["mae_dias"]),
                                    (ejes[1], "cobertura_rango", "Cobertura del rango", ALARMAS["cobertura_rango"]),
                                    (ejes[2], "recall", "Recall del riesgo de retraso", ALARMAS["recall"])):
        colores = [COLOR_2 if ("MAE" in a and col == "mae_dias") or ("cobertura" in a and col == "cobertura_rango")
                   or ("recall" in a and col == "recall") else COLOR for a in tabla["alarmas"]]
        ax.barh(nombres[::-1], tabla[col][::-1], color=colores[::-1])
        ax.axvline(limite, ls="--", color="#555")
        ax.set_title(titulo)
        ax.tick_params(axis="y", labelsize=8)
    for ax in ejes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Escenarios de drift (oct-dic 2026, sin reentrenar). Naranja = dispara la alarma", fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / "escenarios_drift.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------- 2. Estabilidad por segmento y en el tiempo
    p = prueba_pred.copy()
    p["error_abs"] = (p["eta_dias"] - p["dias_reales"]).abs()
    p["mes_pedido"] = p["fecha_pedido"].dt.strftime("%Y-%m")
    segmentos = []
    for col in ("transportista", "region_destino", "cruce_frontera", "modalidad", "mes_pedido"):
        for valor, g in p.groupby(col):
            pos = g["retraso"] == 1
            segmentos.append({"variable": col, "valor": valor, "ordenes": len(g), "tasa_retraso": g["retraso"].mean(),
                              "mae_dias": g["error_abs"].mean(),
                              "cobertura_rango": ((g["dias_reales"] >= g["eta_min"]) & (g["dias_reales"] <= g["eta_max"])).mean(),
                              "recall": (g.loc[pos, "alerta"] == 1).mean() if pos.any() else np.nan})
    seg = pd.DataFrame(segmentos)
    seg.to_csv(SALIDA / "metricas_por_segmento.csv", index=False)

    # ---------------------------------------------- 3. Falsos negativos (retrasos no anticipados)
    base = gen.simular(42, ocultas=True)
    base["id_orden"] = base["id_orden"].astype(str)
    ocultas = base[["id_orden"] + [c for c in base.columns if c.startswith("oculto_")]]
    p = p.merge(ocultas, on="id_orden", how="left")
    p["dias_retraso"] = (p["dias_reales"] - p["dias_comprometidos"]).clip(lower=0)
    p["grupo"] = np.select([(p["retraso"] == 1) & (p["alerta"] == 0), (p["retraso"] == 1) & (p["alerta"] == 1),
                            (p["retraso"] == 0) & (p["alerta"] == 1)], ["FN", "VP", "FP"], "VN")
    retrasadas = p[p["retraso"] == 1]
    comparacion = retrasadas.groupby("grupo").agg(
        ordenes=("id_orden", "size"), prob_media=("prob_retraso", "mean"), holgura_media=("holgura_dias", "mean"),
        utilizacion_media=("utilizacion_planta", "mean"), urgentes=("pedido_urgente", "mean"),
        cruce_frontera=("cruce_frontera", "mean"), con_cambios=("cambios_pedido", lambda s: (s > 0).mean()),
        inventario=("inventario_disponible", "mean"), dias_retraso_medio=("dias_retraso", "mean"),
        retraso_de_1_dia=("dias_retraso", lambda s: (s == 1).mean()),
        con_evento_imprevisto=("oculto_dias_imprevisto", lambda s: (s > 0).mean()),
        aduana_lenta=("oculto_dias_aduana", lambda s: (s > 2).mean()),
    ).T
    comparacion.to_csv(SALIDA / "falsos_negativos_vs_verdaderos_positivos.csv")
    fn = p[p["grupo"] == "FN"]
    explicacion = {
        "falsos_negativos": int(len(fn)),
        "retraso_de_1_dia": float((fn["dias_retraso"] == 1).mean()),
        "con_evento_imprevisto": float((fn["oculto_dias_imprevisto"] > 0).mean()),
        "con_holgura_de_3_o_mas": float((fn["holgura_dias"] >= 3).mean()),
        "inventario_disponible": float(fn["inventario_disponible"].mean()),
        "prob_media": float(fn["prob_retraso"].mean()),
        "costo_fn_mxn": 621.76,
    }
    # Clasificación simple de la causa más probable de cada FN (con las variables ocultas)
    causa = np.select(
        [fn["oculto_dias_imprevisto"] > 0, fn["oculto_dias_aduana"] > 2, fn["dias_retraso"] == 1],
        ["evento imprevisible (bloqueo, clima)", "aduana mucho más lenta de lo normal", "variación de 1 día"],
        "otra combinación de factores")
    explicacion["causas"] = pd.Series(causa).value_counts(normalize=True).round(4).to_dict()
    (SALIDA / "falsos_negativos.json").write_text(json.dumps(explicacion, ensure_ascii=False, indent=2), encoding="utf-8")
    fn[["id_orden", "fecha_pedido", "cliente", "transportista", "region_destino", "holgura_dias", "utilizacion_planta",
        "pedido_urgente", "prob_retraso", "dias_retraso", "oculto_dias_imprevisto", "oculto_dias_aduana"]].to_csv(
        SALIDA / "falsos_negativos_detalle.csv", index=False)
    print("\nFalsos negativos:", json.dumps(explicacion, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    causas = pd.Series(explicacion["causas"]).sort_values()
    ax.barh(causas.index, causas.values, color=COLOR)
    for i, v in enumerate(causas.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center")
    ax.set_xlim(0, max(causas.values) + 0.15)
    ax.set_title(f"Causa más probable de los {len(fn)} retrasos no anticipados")
    ax.set_xlabel("Proporción de falsos negativos")
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / "causas_falsos_negativos.png", dpi=160)
    plt.close(fig)

    resumen = {"alarmas": ALARMAS, "escenarios": tabla.round(4).to_dict(orient="records"),
               "falsos_negativos": explicacion,
               "segmentos_peor_mae": seg.sort_values("mae_dias", ascending=False).head(5).round(4).to_dict(orient="records")}
    (SALIDA / "resumen_estres_drift.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")
    print(comparacion.round(3).to_string())
    print("\nSegmentos con mayor MAE:\n", seg.sort_values("mae_dias", ascending=False).head(6).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
