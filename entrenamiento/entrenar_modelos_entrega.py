"""
entrenar_modelos_entrega.py
---------------------------
Entrena los dos modelos tabulares que usa la API con los datos simulados de
entregas de CAB (data-v1.0.0 de la Actividad 6):

1. Modelo de ETA (nuevo en la Fase 2). Regresión que estima cuántos días tarda
   realmente una orden (dias_reales) desde que se confirma. Se entrena la mediana
   y dos cuantiles (5 % y 95 %) para informar un rango, como pide la Fase 1
   (riesgo R08: "ETA sin rango ni motivo"). Se compara contra la fecha que hoy
   promete SAP (dias_comprometidos), que es la línea base.
2. Clasificador de riesgo de retraso (Actividad 6). Gradient boosting con los
   mejores hiperparámetros de la A6 y el umbral de operación 0.17. Se reentrena con
   la misma división temporal y se comprueba que reproduce las métricas de la A6.

División temporal (igual que la A6): entrenamiento con pedidos anteriores al
01/05/2026 y prueba con los posteriores, para medir con meses que el modelo no vio.

Uso:
    python entrenamiento/entrenar_modelos_entrega.py
Salida:
    backend/modelos/*.joblib y backend/modelos/metadatos_modelos.json
    resultados/entrega/ (métricas, predicciones de prueba y figuras)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (mean_absolute_error, precision_score, recall_score, roc_auc_score,  # noqa: E402
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

SEMILLA = 42
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos" / "entregas" / "entregas_cab.csv"
DIR_MODELOS = RAIZ / "backend" / "modelos"
DIR_RES = RAIZ / "resultados" / "entrega"
CORTE = pd.Timestamp("2026-05-01")
UMBRAL_RIESGO = 0.17  # Actividad 6: recall >= 0.80 en validación cruzada con el menor costo
COLOR, COLOR_2 = "#0e5c4a", "#e07b39"

CATEGORICAS = ["tipo_cliente", "planta", "region_destino", "transportista", "modalidad", "familia_producto"]
NUMERICAS = ["cruce_frontera", "distancia_km", "cantidad_piezas", "peso_kg", "inventario_disponible",
             "utilizacion_planta", "pedido_urgente", "cambios_pedido", "produccion_std_dias", "transito_std_dias",
             "dias_comprometidos", "holgura_dias", "mes", "dia_semana", "envios_previos_ruta",
             "tasa_retraso_transportista_90d"]
ENTRADAS = CATEGORICAS + NUMERICAS


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERICAS),
    ])


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip() \
            or "sin_git"
    except FileNotFoundError:
        return "sin_git"


def main() -> None:
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    (DIR_RES / "figuras").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATOS, parse_dates=["fecha_pedido"])
    entrena = df[df["fecha_pedido"] < CORTE].reset_index(drop=True)
    prueba = df[df["fecha_pedido"] >= CORTE].reset_index(drop=True)
    X_tr, X_te = entrena[ENTRADAS], prueba[ENTRADAS]
    print(f"Entrenamiento {len(entrena):,} órdenes | prueba {len(prueba):,} órdenes")

    # ------------------------------------------------------------------ ETA
    y_tr, y_te = entrena["dias_reales"].to_numpy(), prueba["dias_reales"].to_numpy()
    busqueda = GridSearchCV(
        Pipeline([("prep", preprocesador()),
                  ("modelo", HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                                           early_stopping=False, random_state=SEMILLA))]),
        {"modelo__learning_rate": [0.05, 0.1], "modelo__max_leaf_nodes": [15, 31],
         "modelo__min_samples_leaf": [20, 50]},
        scoring="neg_mean_absolute_error", cv=KFold(5, shuffle=True, random_state=SEMILLA), n_jobs=-1,
    )
    busqueda.fit(X_tr, y_tr)
    eta = busqueda.best_estimator_
    mejores = {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()}
    print(f"ETA · mejores parámetros {mejores} | MAE CV {-busqueda.best_score_:.3f} días")
    cv = pd.DataFrame(busqueda.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    cv["mae_cv"] = -cv["mean_test_score"]
    cv.drop(columns="mean_test_score").to_csv(DIR_RES / "grid_eta.csv", index=False)

    def cuantil(alfa):
        return Pipeline([("prep", preprocesador()),
                         ("modelo", HistGradientBoostingRegressor(loss="quantile", quantile=alfa, max_iter=300,
                                                                  early_stopping=False, random_state=SEMILLA,
                                                                  **mejores))])

    def rango(m_lo, m_hi, X, mediana):
        lo_ = np.maximum(1, np.rint(np.minimum(m_lo.predict(X), mediana)))
        hi_ = np.rint(np.maximum(m_hi.predict(X), mediana))
        return lo_, hi_

    # Selección del rango con validación temporal DENTRO del entrenamiento (feb-abr 2026):
    # el par de cuantiles más angosto que cubra al menos 87 % (KPI de R08 = 85 % + 2 puntos de margen).
    corte_val = pd.Timestamp("2026-02-01")
    sub_tr, sub_val = entrena[entrena["fecha_pedido"] < corte_val], entrena[entrena["fecha_pedido"] >= corte_val]
    med_val = eta.fit(sub_tr[ENTRADAS], sub_tr["dias_reales"]).predict(sub_val[ENTRADAS])
    candidatos = []
    for a, b in ((0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.90), (0.25, 0.90)):
        lo_v, hi_v = rango(cuantil(a).fit(sub_tr[ENTRADAS], sub_tr["dias_reales"]),
                           cuantil(b).fit(sub_tr[ENTRADAS], sub_tr["dias_reales"]), sub_val[ENTRADAS], med_val)
        y_v = sub_val["dias_reales"].to_numpy()
        candidatos.append({"cuantil_inferior": a, "cuantil_superior": b,
                           "cobertura_validacion": float(np.mean((y_v >= lo_v) & (y_v <= hi_v))),
                           "ancho_validacion": float(np.mean(hi_v - lo_v))})
    candidatos = pd.DataFrame(candidatos)
    candidatos.to_csv(DIR_RES / "seleccion_rango_eta.csv", index=False)
    validos = candidatos[candidatos["cobertura_validacion"] >= 0.87]
    elegido = (validos if len(validos) else candidatos).sort_values("ancho_validacion").iloc[0]
    q_lo, q_hi = float(elegido["cuantil_inferior"]), float(elegido["cuantil_superior"])
    print("Selección del rango (validación feb-abr 2026):\n", candidatos.round(3).to_string(index=False))
    print(f"Rango elegido: cuantiles {q_lo:.2f} - {q_hi:.2f}")

    eta.fit(X_tr, y_tr)  # se vuelve a entrenar con todo el periodo de entrenamiento
    cuantiles = {"inferior": cuantil(q_lo).fit(X_tr, y_tr), "superior": cuantil(q_hi).fit(X_tr, y_tr)}

    pred = eta.predict(X_te)
    eta_dias = np.maximum(1, np.rint(pred))
    lo, hi = rango(cuantiles["inferior"], cuantiles["superior"], X_te, pred)
    sap = prueba["dias_comprometidos"].to_numpy()
    nominal = (prueba["produccion_std_dias"] + prueba["transito_std_dias"]).to_numpy()
    error = eta_dias - y_te
    metricas_eta = {
        "mae_dias": float(mean_absolute_error(y_te, eta_dias)),
        "mae_sin_redondeo": float(mean_absolute_error(y_te, pred)),
        "rmse_dias": float(root_mean_squared_error(y_te, eta_dias)),
        "dentro_1_dia": float(np.mean(np.abs(error) <= 1)),
        "sesgo_dias": float(np.mean(error)),
        "cobertura_rango": float(np.mean((y_te >= lo) & (y_te <= hi))),
        "cuantiles_rango": [q_lo, q_hi],
        "ancho_promedio_rango": float(np.mean(hi - lo)),
        "mae_linea_base_sap": float(mean_absolute_error(y_te, sap)),
        "dentro_1_dia_sap": float(np.mean(np.abs(sap - y_te) <= 1)),
        "mae_linea_base_nominal": float(mean_absolute_error(y_te, nominal)),
        "mae_cv": float(-busqueda.best_score_),
    }
    metricas_eta["mejora_vs_sap"] = 1 - metricas_eta["mae_dias"] / metricas_eta["mae_linea_base_sap"]
    print("ETA · prueba:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metricas_eta.items()})

    # ------------------------------------------------------- Riesgo de retraso
    riesgo = Pipeline([("prep", preprocesador()),
                       ("modelo", HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                                                random_state=SEMILLA, learning_rate=0.05,
                                                                max_leaf_nodes=4, min_samples_leaf=100))])
    riesgo.fit(X_tr, entrena["retraso"])
    p_riesgo = riesgo.predict_proba(X_te)[:, 1]
    alerta = (p_riesgo >= UMBRAL_RIESGO).astype(int)
    y_r = prueba["retraso"].to_numpy()
    metricas_riesgo = {
        "auc": float(roc_auc_score(y_r, p_riesgo)),
        "recall": float(recall_score(y_r, alerta)),
        "precision": float(precision_score(y_r, alerta)),
        "tasa_alertas": float(alerta.mean()),
        "falsos_negativos": int(((alerta == 0) & (y_r == 1)).sum()),
        "umbral": UMBRAL_RIESGO,
        "referencia_A6": {"auc": 0.818, "recall": 0.798, "precision": 0.421, "falsos_negativos": 272},
    }
    print("Riesgo · prueba:", {k: v for k, v in metricas_riesgo.items() if k != "referencia_A6"})

    # ----------------------------------------------------- Guardar y documentar
    salida_prueba = prueba[["id_orden", "fecha_pedido", "cliente"] + ENTRADAS + ["dias_reales", "retraso"]].copy()
    salida_prueba["eta_dias"] = eta_dias
    salida_prueba["eta_min"], salida_prueba["eta_max"] = lo, hi
    salida_prueba["prob_retraso"] = p_riesgo.round(4)
    salida_prueba["alerta"] = alerta
    salida_prueba.to_csv(DIR_RES / "predicciones_prueba.csv", index=False)

    rangos = {c: [float(entrena[c].min()), float(entrena[c].max())] for c in NUMERICAS}
    categorias = {c: sorted(entrena[c].dropna().unique().tolist()) for c in CATEGORICAS}
    joblib.dump(eta, DIR_MODELOS / "eta_mediana.joblib")
    for viejo in DIR_MODELOS.glob("eta_q*.joblib"):
        viejo.unlink()
    joblib.dump(cuantiles["inferior"], DIR_MODELOS / "eta_rango_inferior.joblib")
    joblib.dump(cuantiles["superior"], DIR_MODELOS / "eta_rango_superior.joblib")
    joblib.dump(riesgo, DIR_MODELOS / "riesgo_retraso.joblib")
    versiones = {"python": platform.python_version()}
    for lib in ("scikit-learn", "pandas", "numpy", "joblib"):
        versiones[lib] = version(lib)
    metadatos = {
        "version_modelos": "model-v1.0.0",
        "version_datos": "data-v1.0.0 (Actividad 6)",
        "sha256_datos": sha256(DATOS),
        "git_commit": commit_git(),
        "corte_temporal": str(CORTE.date()),
        "entradas": ENTRADAS,
        "categoricas": categorias,
        "rangos_numericos_entrenamiento": rangos,
        "eta": {"algoritmo": f"HistGradientBoostingRegressor (pérdida absoluta) + cuantiles {q_lo:.2f} y {q_hi:.2f}",
                "parametros": mejores, "cuantiles_rango": [q_lo, q_hi], "metricas_prueba": metricas_eta},
        "riesgo_retraso": {"algoritmo": "HistGradientBoostingClassifier (campeón de la A6)",
                           "parametros": {"learning_rate": 0.05, "max_leaf_nodes": 4, "min_samples_leaf": 100,
                                          "max_iter": 200},
                           "umbral": UMBRAL_RIESGO, "metricas_prueba": metricas_riesgo},
        "versiones": versiones,
        "archivos": {p.name: sha256(p) for p in sorted(DIR_MODELOS.glob("*.joblib"))},
    }
    (DIR_MODELOS / "metadatos_modelos.json").write_text(json.dumps(metadatos, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    (DIR_RES / "metricas_entrega.json").write_text(
        json.dumps({"eta": metricas_eta, "riesgo": metricas_riesgo}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Figura: error del ETA frente a la línea base de SAP
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.arange(-8.5, 9.5, 1)
    ax.hist(sap - y_te, bins=bins, alpha=0.55, color=COLOR_2, label=f"Fecha de SAP (MAE {metricas_eta['mae_linea_base_sap']:.2f})")
    ax.hist(error, bins=bins, alpha=0.75, color=COLOR, label=f"Modelo de ETA (MAE {metricas_eta['mae_dias']:.2f})")
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Error en días (estimado − real)")
    ax.set_ylabel("Órdenes de prueba")
    ax.set_title("Error de la fecha estimada frente a la fecha de SAP")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DIR_RES / "figuras" / "error_eta_vs_sap.png", dpi=160)
    plt.close(fig)
    print("Modelos guardados en", DIR_MODELOS)


if __name__ == "__main__":
    main()
