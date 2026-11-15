"""
monitoreo_semanal.py · Plano de MODELO (Actividad 8)
=====================================================
Monitoreo semanal del modelo de ETA del 05/10/2026 al 27/12/2026 con las órdenes que
siguen a data-v1.0.0 (mismo generador y semilla de la Fase 2: la historia es idéntica).

  * 09/11/2026  INC-01: arranca la temporada alta y un programa nuevo de ARM-03; las tres
                plantas suben 15 puntos de carga (escenario de drift de la Fase 2, aplicado
                solo desde esa semana).
  * Detección en dos pasos: primero el drift de entradas (A4, P3, sin etiquetas) y después
    la degradación medida con entregas reales (A5, P2).
  * Respuesta semiautomática: reentrenamiento del challenger, calibración del rango,
    compuerta de validación y aprobación humana antes de mover @champion.

Uso:
    python src/monitoreo_semanal.py          (después de simular_game_day.py: usa el mismo registro)
Salida:
    salidas/semanal/ y MLflow (experimento CAB_Actividad8_Observabilidad).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from obs import drift, modelos_eta as me, tablero  # noqa: E402
from obs.alertas import MotorAlertas  # noqa: E402
from obs.config import carpeta_salida, cargar_config, uri_mlflow  # noqa: E402
from obs.registro import RegistroETA  # noqa: E402
from obs.respuestas import desplegar, reentrenar_challenger  # noqa: E402
from obs.runbooks import EjecucionRunbook  # noqa: E402
from obs.telemetria import Bitacora, Trazas, nuevo_id  # noqa: E402

SALIDA = carpeta_salida("semanal")
INICIO_DRIFT = pd.Timestamp("2026-11-09")
CARGA_EXTRA = 0.15
UMBRAL_CARGA = me.metadatos()["rangos_numericos_entrenamiento"]["utilizacion_planta"][1]   # 0.97


def produccion(con_drift: bool) -> pd.DataFrame:
    """Órdenes desde oct-2026. Con drift, la carga semanal sube 15 puntos SOLO desde el 09/11
    (se parcha la función del generador para no alterar la historia ni la secuencia aleatoria)."""
    gen = me.generador()
    original = gen._utilizacion_semanal

    def con_cambio(rng, semanas):
        u = original(rng, semanas)
        return {k: (float(np.clip(v + CARGA_EXTRA, 0.55, 1.20)) if k[1] >= INICIO_DRIFT else v) for k, v in u.items()}

    if con_drift:
        gen._utilizacion_semanal = con_cambio
    try:
        df = gen.simular(42)
    finally:
        gen._utilizacion_semanal = original
    return df[df["fecha_pedido"] >= "2026-10-01"].reset_index(drop=True)


def predecir_todo(df: pd.DataFrame, modelos: dict, riesgo) -> pd.DataFrame:
    eta, lo, hi = me.predecir(modelos, df)
    p = riesgo.predict_proba(df[me.entradas()])[:, 1]
    return pd.DataFrame({"eta": eta, "lo": lo, "hi": hi, "p_retraso": p}, index=df.index)


def calidad(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"entregas": 0}
    y = g["dias_reales"]
    retrasos = g["retraso"] == 1
    return {"entregas": int(len(g)), "mae": float((g["eta"] - y).abs().mean()),
            "cobertura": float(((y >= g["lo"]) & (y <= g["hi"])).mean()),
            "dentro_1_dia": float(((g["eta"] - y).abs() <= 1).mean()), "sesgo": float((g["eta"] - y).mean()),
            "fuera_rango": int((~((y >= g["lo"]) & (y <= g["hi"]))).sum()),
            "recall_riesgo": float((g.loc[retrasos, "p_retraso"] >= 0.17).mean()) if retrasos.any() else float("nan")}


def main() -> None:
    t_ini = time.time()
    cfg = cargar_config()
    mlflow.set_tracking_uri(uri_mlflow())
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    registro = RegistroETA(cfg["mlflow"]["modelo_registrado"])
    if registro.alias("champion") is None:
        raise SystemExit("Primero ejecuta src/simular_game_day.py (registra model-v1.0.0 como @champion).")
    v1_num = registro.alias("champion")
    v1 = registro.cargar("champion")
    riesgo = joblib.load(me.DIR_MOD / "riesgo_retraso.joblib")

    historico = pd.read_csv(me.RAIZ_REPO / "datos" / "entregas" / "entregas_cab.csv",
                            parse_dates=["fecha_pedido", "fecha_compromiso", "fecha_entrega_real"])
    referencia = historico[historico["fecha_pedido"] < "2026-05-01"]                 # datos de entrenamiento
    prod = produccion(con_drift=True)
    sin_drift = produccion(con_drift=False)                                          # contrafactual
    pred_v1 = predecir_todo(prod, v1, riesgo)
    pred_sin = predecir_todo(sin_drift, v1, riesgo)
    servida = pred_v1.copy()
    servida["version"] = pd.Series(v1_num, index=servida.index, dtype=object)

    cfg_m = cfg["modelo"]
    semanas = pd.date_range(cfg_m["semanas"]["inicio"], cfg_m["semanas"]["fin"], freq="W-MON")
    bit = Bitacora(SALIDA / "monitoring_log.jsonl")
    trazas = Trazas(SALIDA / "traces.jsonl", mlflow_activo=True, max_mlflow=len(semanas))
    motor = MotorAlertas(cfg, "modelo", "INC-SEM", registrar_log=bit)
    filas, drift_vars, alertas_tab, eventos_reg = [], [], [], []
    rb = None
    estado = {"challenger": None, "promovido": None, "intentos": 0, "v3": None}
    pct_sin_drift = []
    snapshots = {"2026-11-09": "01_normal", "2026-11-23": "02_drift_P3", "2026-11-30": "03_mae_P2_reentrenamiento",
                 "2026-12-14": "04_validacion_v3", "2026-12-28": "05_cierre"}

    with mlflow.start_run(run_name="monitoreo_semanal_eta_oct-dic_2026") as run:
        mlflow.set_tags({"tipo_run": "monitoreo_semanal", "plano": "modelo", "version_config": cfg["version_config"],
                         "modelo": cfg["mlflow"]["modelo_registrado"], "champion_inicial": v1_num})
        mlflow.log_params({"slo_mae_dias": cfg["slo"]["eta_mae"]["objetivo_dias"],
                           "slo_cobertura": cfg["slo"]["eta_cobertura"]["objetivo"],
                           "umbral_carga_entrenamiento": UMBRAL_CARGA, "inicio_drift_simulado": str(INICIO_DRIFT.date()),
                           "carga_extra_simulada": CARGA_EXTRA, "ordenes_produccion": len(prod),
                           "referencia_drift": "data-v1.0.0 < 2026-05-01 (entrenamiento)"})
        for i, s in enumerate(semanas):
            ts = (s + timedelta(days=7)).to_pydatetime()          # el job corre el lunes siguiente a las 00:00
            spans, reloj = [], time.perf_counter()

            def marca(nombre, attrs=None):
                nonlocal reloj
                ahora = time.perf_counter()
                spans.append({"nombre": nombre, "inicio_ms": sum(x["dur_ms"] for x in spans),
                              "dur_ms": (ahora - reloj) * 1000, "atributos": attrs or {}})
                reloj = ahora

            # ---------------------------------------------------- datos de la semana
            entregas = prod[(prod["fecha_entrega_real"] >= s) & (prod["fecha_entrega_real"] < s + timedelta(days=7))]
            pedidos = prod[(prod["fecha_pedido"] >= s) & (prod["fecha_pedido"] < s + timedelta(days=7))]
            g = pd.concat([entregas, servida.loc[entregas.index]], axis=1)
            g_v1 = pd.concat([entregas, pred_v1.loc[entregas.index]], axis=1)
            ent_sin = sin_drift[(sin_drift["fecha_entrega_real"] >= s) & (sin_drift["fecha_entrega_real"] < s + timedelta(days=7))]
            g_sin = pd.concat([ent_sin, pred_sin.loc[ent_sin.index]], axis=1)
            marca("cargar_datos", {"entregas": len(entregas), "pedidos": len(pedidos)})

            # ---------------------------------------------------- SLI de calidad (con etiquetas)
            q, q_v1, q_sin = calidad(g), calidad(g_v1), calidad(g_sin)
            seg = {f"mae_requiere_produccion": float((g.loc[g["inventario_disponible"] == 0, "eta"]
                                                       - g.loc[g["inventario_disponible"] == 0, "dias_reales"]).abs().mean()),
                   f"mae_con_inventario": float((g.loc[g["inventario_disponible"] == 1, "eta"]
                                                 - g.loc[g["inventario_disponible"] == 1, "dias_reales"]).abs().mean())}
            filas_4s = filas[-3:] + [{"entregas": q["entregas"], "fuera_rango": q["fuera_rango"]}]
            ent_4s, fuera_4s = sum(f["entregas"] for f in filas_4s), sum(f["fuera_rango"] for f in filas_4s)
            presupuesto = 1 - cfg["slo"]["eta_cobertura"]["objetivo"]
            marca("calcular_sli_calidad", {"mae": round(q["mae"], 3)})

            # ---------------------------------------------------- drift de entradas (sin etiquetas)
            m_drift = {"pct_carga_sobre_rango": float((pedidos["utilizacion_planta"] > UMBRAL_CARGA).mean()),
                       "pct_fuera_rango_todas": drift.fuera_de_rango(pedidos, me.metadatos()["rangos_numericos_entrenamiento"],
                                                                      [c for c in me.metadatos()["rangos_numericos_entrenamiento"]
                                                                       if c != "envios_previos_ruta"]),
                       "utilizacion_media": float(pedidos["utilizacion_planta"].mean()),
                       "ks_utilizacion": drift.ks(referencia["utilizacion_planta"], pedidos["utilizacion_planta"])[0],
                       "nulos_entrada": float(pedidos[me.entradas()].isna().mean().max())}
            for v in cfg_m["variables_drift_numericas"]:
                valor = drift.psi(referencia[v], pedidos[v])
                drift_vars.append({"semana": s.date(), "variable": v, "metodo": "PSI", "valor": valor,
                                   "nivel": drift.nivel_psi(valor)})
            for v in cfg_m["variables_drift_categoricas"]:
                j = drift.jsd(referencia[v], pedidos[v])
                _, p = drift.chi2(referencia[v], pedidos[v])
                drift_vars.append({"semana": s.date(), "variable": v, "metodo": "JSD", "valor": j, "p_chi2": p,
                                   "nivel": "importante" if (p < 0.01 and j >= 0.10) else "estable"})
            dv = pd.DataFrame([d for d in drift_vars if d["semana"] == s.date()])
            m_drift["psi_max_numericas"] = float(dv.loc[dv["metodo"] == "PSI", "valor"].max())
            m_drift["jsd_max_categoricas"] = float(dv.loc[dv["metodo"] == "JSD", "valor"].max())
            m_drift["psi_tasa_retraso_90d"] = drift.psi(referencia["tasa_retraso_transportista_90d"],
                                                        pedidos["tasa_retraso_transportista_90d"])
            ped_sin = sin_drift[(sin_drift["fecha_pedido"] >= s) & (sin_drift["fecha_pedido"] < s + timedelta(days=7))]
            pct_sin_drift.append(float((ped_sin["utilizacion_planta"] > UMBRAL_CARGA).mean()))
            marca("calcular_drift", {"pct_carga_sobre_rango": round(m_drift["pct_carga_sobre_rango"], 3)})

            m = {"mae_semanal": q["mae"], "cobertura_semanal": q["cobertura"], "dentro_1_dia": q["dentro_1_dia"],
                 "sesgo_dias": q["sesgo"], "entregas": q["entregas"], "recall_riesgo": q["recall_riesgo"],
                 "mae_v1_contrafactual": q_v1["mae"], "mae_sin_drift": q_sin["mae"],
                 "cobertura_4s": 1 - fuera_4s / ent_4s, "quema_cobertura_4s": (fuera_4s / ent_4s) / presupuesto,
                 "presupuesto_cobertura_restante_4s": 1 - fuera_4s / (presupuesto * ent_4s), **seg, **m_drift}
            ctx = {"champion": f"v{registro.alias('champion')}", "utilizacion_media": round(m["utilizacion_media"], 3),
                   "entregas_semana": q["entregas"], "mae_requiere_produccion": round(seg["mae_requiere_produccion"], 2),
                   "mae_con_inventario": round(seg["mae_con_inventario"], 2)}
            eventos = motor.evaluar(ts, m, ctx)
            marca("evaluar_alertas", {"eventos": len(eventos)})

            # ---------------------------------------------------- runbook RB-02 y respuesta semiautomática
            for e in eventos:
                if e["tipo"] in ("disparada", "escalada", "agrupada", "resuelta"):
                    alertas_tab.append({"ts": ts.strftime("%d/%m"), "severidad": e["severidad"],
                                        "texto": f"{e['alerta']} · {e['nombre']}", "incidente": e["incidente"],
                                        "estado": e["tipo"]})
                if e["alerta"] == "A4" and e["tipo"] == "disparada":
                    rb = EjecucionRunbook("RB-02", "INC-01", "Data drift y degradación del modelo de ETA", "P3")
                    rb.paso(ts, "identificacion", "Alerta A4 (P3): carga de planta por encima del máximo de "
                            "entrenamiento 2 semanas seguidas", f"{m['pct_carga_sobre_rango']:.0%} de las órdenes con carga "
                            f"> {UMBRAL_CARGA}; carga media {m['utilizacion_media']:.2f} (antes ~0.80); KS {m['ks_utilizacion']:.2f}",
                            "motor de alertas", "alerts.json, drift_por_variable.csv")
                    rb.paso(ts, "diagnostico", "¿Es drift real o error de ingestión? Revisar nulos, esquema y "
                            "contrastar la carga con el plan de producción", f"0 nulos en las 22 entradas; esquema sin "
                            "cambios; Planeación confirma arranque del programa de ARM-03 y temporada alta → drift real",
                            "especialista MLOps", "monitoring_log.jsonl (nulos_entrada)")
                    rb.paso(ts, "diagnostico", "¿Cambió algo más? PSI/KS en numéricas y JSD/χ² en categóricas",
                            f"PSI máx. numéricas {m['psi_max_numericas']:.2f}, JSD máx. categóricas "
                            f"{m['jsd_max_categoricas']:.3f}: el cambio está aislado en la carga de planta",
                            "automatización", "drift_por_variable.csv")
                    rb.paso(ts, "diagnostico", "Cuantificar impacto con las etiquetas disponibles",
                            f"MAE semanal {m['mae_semanal']:.2f} días (SLO ≤ 1.30): todavía dentro, pero las órdenes con "
                            "producción tardan más en entregarse → la degradación aparecerá con retraso",
                            "especialista MLOps", "semanas.csv")
                    rb.paso(ts, "accion", "Decisión P3: no reentrenar aún (faltan etiquetas); vigilar A5 y preparar "
                            "datos para el challenger", "monitoreo semanal continúa", "especialista MLOps", "incidentes.json")
                    inc_inicio_det = ts
                if e["alerta"] == "A5" and e["tipo"] in ("escalada", "disparada"):
                    rb.severidad = "P2"
                    rb.paso(ts, "escalamiento", "Alerta A5: MAE semanal > 1.30 dos semanas seguidas; el incidente sube a P2",
                            f"MAE {m['mae_semanal']:.2f} días; requiere producción {seg['mae_requiere_produccion']:.2f} vs "
                            f"con inventario {seg['mae_con_inventario']:.2f} → la degradación viene de la producción "
                            "en planta saturada", "motor de alertas", "notificaciones.jsonl, semanas.csv")
                    inc_p2 = ts
            if any(e["alerta"] in ("A5", "A6") and e["tipo"] in ("escalada", "disparada", "duplicada_suprimida",
                                                               "renotificada", "agrupada") for e in eventos) \
                    and estado["promovido"] is None and estado["intentos"] < 3:
                estado["intentos"] += 1
                corte = pd.Timestamp(ts) - pd.Timedelta(days=1)
                r = reentrenar_challenger(historico, prod, corte, v1)
                marca("reentrenar_challenger", {"pasa": r["pasa_compuerta"]})
                rb.paso(ts, "accion", f"Reentrenamiento semiautomático del challenger (intento {estado['intentos']}): "
                        f"historia + {r['ordenes_produccion_usadas']} órdenes de producción con peso ×10; rango calibrado "
                        f"+{r['desfase_rango_dias']} día(s)",
                        f"validación (última semana, {r['ordenes_validacion']} entregas): MAE {r['challenger']['mae']:.2f} vs "
                        f"campeón {r['champion']['mae']:.2f} ({r['mejora_mae']:.0%} mejor); cobertura "
                        f"{r['challenger']['cobertura']:.0%} → compuerta {'APROBADA' if r['pasa_compuerta'] else 'NO aprobada'}",
                        "automatización (pipeline T18)", "MLflow: run registro_eta_v3")
                eventos_reg.append({"ts": ts.isoformat(), "evento": "reentrenamiento", **{k: v for k, v in r.items()
                                                                                         if k != "modelos"}})
                if r["pasa_compuerta"]:
                    v3 = registro.registrar(r["modelos"], "Reentrenamiento por drift de carga de planta (INC-01).",
                                            {"version_semantica": "model-v1.2.0", "datos": "data-v1.0.0 + producción oct-nov 2026",
                                             "peso_reciente": 10, "desfase_rango_dias": r["desfase_rango_dias"],
                                             "origen": "RB-02 / INC-01"},
                                            {"mae_validacion": r["challenger"]["mae"], "cobertura_validacion": r["challenger"]["cobertura"],
                                             "mae_campeon_validacion": r["champion"]["mae"]},
                                            ejemplo=prod[me.entradas()].head(3), nombre_run="registro_eta_v3_reentrenamiento_drift")
                    registro.asignar("challenger", v3)
                    aprobacion = ts + timedelta(days=1, hours=10)
                    desplegar(registro, v3, aprobacion, "Especialista MLOps (aprobación manual)")
                    registro.quitar("challenger")
                    estado.update(promovido=aprobacion, v3=v3)
                    pred_v3 = predecir_todo(prod, r["modelos"], riesgo)
                    nuevas = prod["fecha_pedido"] >= pd.Timestamp(aprobacion.date()) + pd.Timedelta(days=1)
                    servida.loc[nuevas, ["eta", "lo", "hi", "p_retraso"]] = pred_v3.loc[nuevas]
                    servida.loc[nuevas, "version"] = v3
                    estado["pred_v3"] = pred_v3
                    rb.paso(aprobacion, "accion", "Aprobación humana y promoción: @champion → v" + v3 + " (v1 queda en "
                            "sombra 7 días; rollback inmediato si la sombra muestra desacuerdo)",
                            f"órdenes nuevas desde el {aprobacion.date() + timedelta(days=1):%d/%m} se estiman con v{v3}",
                            "Especialista MLOps", "MLflow Model Registry (alias champion)")
                    bit({"ts": aprobacion.isoformat(), "evento": "despliegue_modelo", "version": v3, "sombra": v1_num})
                else:
                    rb.paso(ts, "escalamiento", "Compuerta no aprobada: se mantiene v1 y se reintenta con una semana más "
                            "de etiquetas", "sin cambio en producción", "automatización", "MLflow")

            # validación posterior (órdenes pedidas DESPUÉS de la promoción y ya entregadas)
            if estado["promovido"] is not None:
                nuevas = (prod["fecha_pedido"] >= pd.Timestamp(estado["promovido"].date()) + pd.Timedelta(days=1)) & \
                         (prod["fecha_entrega_real"] < s + timedelta(days=7))
                if nuevas.sum() >= 100:
                    gv3 = calidad(pd.concat([prod[nuevas], estado["pred_v3"].loc[nuevas]], axis=1))
                    gv1 = calidad(pd.concat([prod[nuevas], pred_v1.loc[nuevas]], axis=1))
                    m.update({"mae_v3_acumulado": gv3["mae"], "mae_v1_mismas_ordenes": gv1["mae"],
                              "cobertura_v3_acumulada": gv3["cobertura"], "cobertura_v1_mismas_ordenes": gv1["cobertura"]})
                    if ts.strftime("%Y-%m-%d") in ("2026-12-14", "2026-12-28"):
                        rb.paso(ts, "validacion", f"Comparar v{estado['v3']} contra v1 (sombra) en las mismas "
                                f"{gv3['entregas']} órdenes nuevas ya entregadas",
                                f"MAE v{estado['v3']} {gv3['mae']:.2f} vs v1 {gv1['mae']:.2f} "
                                f"({abs(1 - gv3['mae'] / gv1['mae']):.0%} {'mejor' if gv3['mae'] < gv1['mae'] else 'peor'}); "
                                f"cobertura {gv3['cobertura']:.0%} vs {gv1['cobertura']:.0%}",
                                "automatización", "semanas.csv (mae_v3_acumulado)")
            champion_ts = v1_num if estado["promovido"] is None or estado["promovido"] > ts else estado["v3"]
            fila = {"semana": s.date(), "evaluado": ts.date(), **m, "champion": champion_ts,
                    "alertas_activas": len(motor.activas())}
            filas.append({**fila, "fuera_rango": q["fuera_rango"]})
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and v == v},
                               step=i, timestamp=int(ts.timestamp() * 1000))
            marca("registrar_mlflow")
            trazas.registrar(nuevo_id(32), ts, spans, {"status": 200, "job": "monitoreo_semanal_eta",
                                                       "semana": str(s.date()), "accion": "monitoreo"})
            bit({"ts": ts.isoformat(), "evento": "job_monitoreo_semanal", "semana": str(s.date()),
                 "entregas": q["entregas"], "pedidos": len(pedidos), "mae_semanal": round(q["mae"], 3),
                 "cobertura": round(q["cobertura"], 3), "pct_carga_sobre_rango": round(m["pct_carga_sobre_rango"], 3),
                 "nulos_entrada": m["nulos_entrada"], "champion": registro.alias("champion"),
                 "duracion_ms": round(sum(x["dur_ms"] for x in spans), 1)})
            clave = ts.strftime("%Y-%m-%d")
            if clave in snapshots:
                generar_tablero(SALIDA / "tableros" / f"{snapshots[clave]}_{ts:%m%d}.html", filas, alertas_tab, motor,
                                champion_ts, ts)

        # ------------------------------------------------------------ cierre
        tabla = pd.DataFrame(filas)
        ult = tabla.iloc[-1]
        sigue_fuera = bool(ult["mae_semanal"] > cfg["slo"]["eta_mae"]["objetivo_dias"])
        rb.paso(datetime(2026, 12, 28, 10), "escalamiento", "El MAE sigue arriba del SLO con la planta saturada: "
                "se escala al dueño del producto (Gerente de servicio al cliente) con opciones",
                f"decisión: mantener v{estado['v3']}, comunicar siempre rango + motivo ('planta al X % de capacidad'), confirmación "
                "humana de pedidos urgentes en planta > 0.97 (R03) y revisar el SLO por nivel de carga en la Fase 3"
                if sigue_fuera else "no fue necesario", "especialista MLOps → gerente de servicio", "postmortem_INC-01.md")
        rb.paso(datetime(2026, 12, 28, 12), "cierre", "Cerrar el incidente técnico y abrir mejora: variable de "
                "carga proyectada y eventos de rastreo del transportista (pendiente de la Fase 2)",
                "incidente cerrado con mitigación parcial documentada", "especialista MLOps", "postmortem_INC-01.md")
        rb.cerrar(estado="mitigado parcialmente (escalado)" if sigue_fuera else "resuelto",
                  inicio_drift=str(INICIO_DRIFT.date()), deteccion_drift_P3=str(inc_inicio_det.date()),
                  deteccion_degradacion_P2=str(inc_p2.date()),
                  mttd_dias=(inc_inicio_det - INICIO_DRIFT.to_pydatetime()).days,
                  dias_hasta_nuevo_modelo=(estado["promovido"] - inc_p2).days if estado["promovido"] else None,
                  kpi_R10_reaccion="cumplido (≤ 10 días hábiles)" if estado["promovido"] and
                  (estado["promovido"] - inc_p2).days <= 14 else "no cumplido")
        rb.guardar(SALIDA)
        tabla.to_csv(SALIDA / "semanas.csv", index=False)
        pd.DataFrame(drift_vars).to_csv(SALIDA / "drift_por_variable.csv", index=False)
        motor.guardar(SALIDA)
        resumen = {
            "plano": "modelo", "periodo": f"{semanas[0].date()} a {(semanas[-1] + timedelta(days=6)).date()}",
            "ordenes_produccion": int(len(prod)), "semanas": len(tabla), "incidente": rb.resultado,
            "reentrenamientos": eventos_reg, "champion_final": registro.alias("champion"),
            "control_falsas_alarmas_A4_sin_drift": {
                "pct_carga_sobre_rango_por_semana": [round(x, 3) for x in pct_sin_drift],
                "semanas_sobre_umbral": sum(x >= 0.20 for x in pct_sin_drift),
                "alertas_con_regla_de_2_semanas": sum(a >= 0.20 and b >= 0.20 for a, b in zip(pct_sin_drift, pct_sin_drift[1:]))},
            "ruido": motor.resumen_ruido(),
            "mae_final": {"semana": str(ult["semana"]), "servido": float(ult["mae_semanal"]),
                          "v1_contrafactual": float(ult["mae_v1_contrafactual"]), "sin_drift": float(ult["mae_sin_drift"]),
                          "v3_acumulado": float(ult.get("mae_v3_acumulado", np.nan)),
                          "v1_mismas_ordenes": float(ult.get("mae_v1_mismas_ordenes", np.nan)),
                          "cobertura_v3_acumulada": float(ult.get("cobertura_v3_acumulada", np.nan))},
        }
        (SALIDA / "monitoring_summary.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                                         encoding="utf-8")
        bit.cerrar()
        trazas.cerrar()
        postmortem(resumen, tabla, rb)
        mlflow.log_artifacts(str(SALIDA / "tableros"), "tableros")
        for n in ("monitoring_summary.json", "alerts.json", "notificaciones.jsonl", "incidentes.json", "semanas.csv",
                  "drift_por_variable.csv", "monitoring_log.jsonl", "traces.jsonl"):
            mlflow.log_artifact(str(SALIDA / n), "monitoreo")
        with mlflow.start_run(run_name="INC-01_drift_carga_de_planta_ETA", nested=True):
            mlflow.set_tags({"tipo_run": "incidente", "incidente": "INC-01", "runbook": "RB-02", "severidad": "P2"})
            mlflow.log_metrics({"mttd_dias": rb.resultado["mttd_dias"],
                                "dias_hasta_nuevo_modelo": rb.resultado["dias_hasta_nuevo_modelo"] or -1})
            for p in SALIDA.glob("*INC-01*"):
                mlflow.log_artifact(str(p), "evidencia")
        print(json.dumps(resumen, ensure_ascii=False, indent=1, default=str)[:3000])
        print(tabla[["semana", "entregas", "mae_semanal", "mae_v1_contrafactual", "mae_sin_drift", "cobertura_semanal",
                     "quema_cobertura_4s", "pct_carga_sobre_rango", "psi_max_numericas", "jsd_max_categoricas",
                     "recall_riesgo", "champion"]].round(3).to_string())
        print(f"Listo en {time.time() - t_ini:.0f} s · run {run.info.run_id}")


def generar_tablero(ruta, filas, alertas, motor, champion, ts) -> None:
    t = pd.DataFrame(filas)
    x = [pd.Timestamp(d) for d in t["semana"]]
    u = t.iloc[-1]
    activas = motor.activas()
    estado = "CRÍTICO" if any(a["severidad"] == "P1" for a in activas) else "ADVERTENCIA" if activas else "OK"
    S = tablero.semaforo
    tiles = [
        {"capa": "Modelo ETA · con etiquetas", "titulo": "MAE semanal del ETA (entregas de la semana)",
         "valor": f"{u['mae_semanal']:.2f} días", "slo": "SLO ≤ 1.30 · línea base 1.05 · gris = mismo periodo sin drift",
         "color": S(u["mae_semanal"], 1.30, 1.15), "x": x, "fechas": True,
         "series": [{"y": list(t["mae_semanal"]), "nombre": "servido", "marker": "o"},
                    {"y": list(t["mae_v1_contrafactual"]), "nombre": "si siguiera v1", "color": "#e07b39", "ls": "--"},
                    {"y": list(t["mae_sin_drift"]), "nombre": "sin drift", "color": "#9aa5a0"}], "umbral": 1.30},
        {"capa": "Modelo ETA · R08", "titulo": "Cobertura del rango y presupuesto de error (4 semanas)",
         "valor": f"{u['cobertura_semanal']:.0%} · quema {u['quema_cobertura_4s']:.2f}×",
         "slo": f"SLO ≥ 85 % · presupuesto restante {max(0, u['presupuesto_cobertura_restante_4s']):.0%}",
         "color": S(u["cobertura_semanal"], 0.85, 0.87, "min"), "x": x,
         "series": [{"y": list(t["cobertura_semanal"] * 100), "nombre": "cobertura %", "marker": "o"},
                    {"y": list(t["cobertura_4s"] * 100), "nombre": "4 semanas %", "color": "#e07b39"}], "umbral": 85},
        {"capa": "Datos · drift de entrada", "titulo": "Carga de planta arriba del rango de entrenamiento",
         "valor": f"{u['pct_carga_sobre_rango']:.0%} de órdenes", "slo": f"alerta A4 (P3) ≥ 20 % dos semanas · carga media "
         f"{u['utilizacion_media']:.2f}", "color": S(u["pct_carga_sobre_rango"], 0.20, 0.05), "x": x,
         "series": [{"y": list(t["pct_carga_sobre_rango"] * 100), "nombre": "% > 0.97", "marker": "o"},
                    {"y": list(t["utilizacion_media"] * 100), "nombre": "carga media %", "color": "#5b6b65", "ls": "--"}],
         "umbral": 20},
        {"capa": "Datos · otras variables", "titulo": "PSI máx. (numéricas) y JSD máx. (categóricas)",
         "valor": f"PSI {u['psi_max_numericas']:.2f} · JSD {u['jsd_max_categoricas']:.3f}",
         "slo": "PSI > 0.25 o JSD > 0.10 con χ² p < 0.01 = cambio importante",
         "color": S(max(u["psi_max_numericas"], u["jsd_max_categoricas"]), 0.25, 0.10), "x": x,
         "series": [{"y": list(t["psi_max_numericas"]), "nombre": "PSI máx."},
                    {"y": list(t["jsd_max_categoricas"]), "nombre": "JSD máx.", "color": "#e07b39"}], "umbral": 0.25,
         "ylim": (0, 0.3)},
        {"capa": "Diagnóstico · segmentos", "titulo": "MAE por segmento",
         "valor": f"producción {u['mae_requiere_produccion']:.2f} · inventario {u['mae_con_inventario']:.2f}",
         "slo": "la degradación concentrada en un segmento apunta a la causa",
         "color": S(u["mae_requiere_produccion"], 1.6, 1.4), "x": x,
         "series": [{"y": list(t["mae_requiere_produccion"]), "nombre": "requiere producción", "marker": "o"},
                    {"y": list(t["mae_con_inventario"]), "nombre": "con inventario", "color": "#e07b39"}]},
        {"capa": "Modelo riesgo de retraso", "titulo": "Recall semanal del riesgo de retraso",
         "valor": f"{u['recall_riesgo']:.0%}", "slo": "referencia ≥ 75 % (alarma de la Fase 2, informativa)",
         "color": S(u["recall_riesgo"], 0.75, 0.78, "min"), "x": x,
         "series": [{"y": list(t["recall_riesgo"] * 100), "nombre": "recall %", "marker": "o"}], "umbral": 75},
    ]
    ctx = [f"ETA @champion v{champion}" + (" (reentrenamiento en curso)" if any(a["estado"] == "escalada" and
           a["ts"] == f"{ts:%d/%m}" for a in alertas) else ""), f"semana evaluada {t.iloc[-1]['semana']:%d/%m/%Y} "
           f"({int(u['entregas'])} entregas)",
           f"Incidentes abiertos: {', '.join(i.id + ' (' + i.severidad + ')' for i in motor.incidentes.values()) or 'ninguno'}"]
    notas = ["La calidad se mide por semana de ENTREGA (la etiqueta llega cuando la orden se entrega); el drift, por "
             "semana de PEDIDO (se ve en cuanto entra la orden). Por eso el drift avisa antes.",
             "Líneas punteadas: contrafactuales (qué habría pasado sin drift o sin cambiar de modelo)."]
    tablero.generar(ruta, "Tablero operativo · Chatbot CAB · plano de modelo (ETA)",
                    f"Job semanal del {ts:%d/%m/%Y}", estado, ctx, tiles, alertas, notas)


def postmortem(resumen: dict, tabla: pd.DataFrame, rb) -> None:
    r = rb.resultado
    re_ = resumen["reentrenamientos"]
    ok = [x for x in re_ if x["pasa_compuerta"]]
    mf = resumen["mae_final"]
    texto = f"""# Postmortem INC-01 · Data drift por carga de planta y degradación del ETA

**Severidad:** P3 → P2 · **Runbook:** RB-02 · **Formato:** sin culpables

## Qué pasó
Desde el 09/11/2026 las tres plantas operan 15 puntos arriba de lo normal (temporada alta + programa nuevo de
ARM-03). El modelo de ETA nunca vio cargas mayores a {UMBRAL_CARGA}: con la planta saturada la producción se alarga y
el modelo, que no puede extrapolar, subestima los días.

## Detección (proactiva, en dos pasos)
- {r['deteccion_drift_P3']}: A4 (P3) por drift de entrada, **sin esperar etiquetas** (MTTD {r['mttd_dias']} días). La
  semana del 02/11 ya tenía 23.5 % de órdenes arriba de 0.97 por el pico normal de noviembre (quedó "pendiente");
  con la primera semana de drift (100 %) se cumplieron las dos ventanas.
- {r['deteccion_degradacion_P2']}: A5 (P2) cuando el MAE semanal pasó 1.30 dos semanas seguidas. A6 no se activó:
  la cobertura del rango se mantuvo arriba de 85 %.

## Respuesta
- Reentrenamiento semiautomático: {len(re_)} intento(s); compuerta aprobada en {len(ok)}.
- {('Mejora en validación: ' + f"MAE {ok[0]['challenger']['mae']:.2f} vs {ok[0]['champion']['mae']:.2f} del campeón ("
   + f"{ok[0]['mejora_mae']:.0%})") if ok else 'Sin promoción.'}
- Nuevo modelo en producción {r['dias_hasta_nuevo_modelo']} días después de la alerta P2 (KPI R10: {r['kpi_R10_reaccion']}).

## Resultado
- En las órdenes nuevas ya entregadas: MAE v3 {mf['v3_acumulado']:.2f} vs v1 {mf['v1_mismas_ordenes']:.2f} en las mismas
  órdenes; cobertura v3 {mf['cobertura_v3_acumulada']:.0%}.
- Última semana: MAE servido {mf['servido']:.2f}; si siguiera v1 {mf['v1_contrafactual']:.2f}; sin drift {mf['sin_drift']:.2f}.
- Estado: **{r['estado']}**. Reentrenar redujo el error, pero con la planta saturada la incertidumbre real es mayor y
  el SLO de 1.30 días no se recupera. Se escaló al dueño del producto con opciones (ver RB-02).

## Mejoras (acciones)
1. Variable de carga proyectada (plan de producción) para que el modelo vea la saturación antes de que ocurra.
2. SLO de MAE estratificado por nivel de carga, revisado con el negocio.
3. Eventos de rastreo del transportista (pendiente desde la Fase 2) para reducir la incertidumbre del tránsito.
4. Reentrenamiento programado mensual con ventana deslizante, además del disparado por alertas.
"""
    (SALIDA / "postmortem_INC-01.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
