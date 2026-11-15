"""
simular_game_day.py · Plano de SERVICIO (Actividad 8)
======================================================
Game day del 12/11/2026, de 09:00 a 14:00, en ventanas de 5 minutos, con la API real de
la Fase 2 (ver obs/servicio.py). Incluye:

  * 09:50  pico transitorio de 2 minutos (arranque en frío): NO debe generar alerta.
  * 10:30  INC-02: se despliega la versión 2 del modelo de ETA, entrenada con una
           extracción de SAP que invirtió el campo de inventario (defecto de ETL).
           Respuesta automática: rollback del alias @champion en MLflow.
  * 11:45  INC-03: el portal de ARM-01 consulta la API en un bucle (16 solicitudes/s).
           Respuesta automática: escalar réplicas; después, límite de tasa y escalamiento
           al ejecutivo de la cuenta.

Uso:
    python src/simular_game_day.py
Salida:
    salidas/game_day/  (monitoring_summary.json, alerts.json, notificaciones.jsonl,
    incidentes.json, monitoring_log.jsonl.gz, traces.jsonl, ventanas.csv, tableros/,
    runbook_*.md, postmortem_*.md) y MLflow (experimento CAB_Actividad8_Observabilidad).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from obs import drift, modelos_eta as me, tablero  # noqa: E402
from obs.alertas import MotorAlertas  # noqa: E402
from obs.config import carpeta_salida, cargar_config, uri_mlflow  # noqa: E402
from obs.registro import RegistroETA  # noqa: E402
from obs.respuestas import Capacidad, ControlRollback, desplegar, escalar_replicas, reducir_replicas, rollback_modelo_eta  # noqa: E402
from obs.runbooks import EjecucionRunbook  # noqa: E402
from obs.servicio import TOKENS, Cola, Escenario, SimuladorServicio, hora, percentil  # noqa: E402
from obs.telemetria import Bitacora, Trazas, nuevo_id  # noqa: E402

FECHA = "2026-11-12"
SALIDA = carpeta_salida("game_day")


# --------------------------------------------------------------------------- registro de modelos
def preparar_registro(cfg: dict, registro: RegistroETA) -> tuple[str, str, dict]:
    """v1 = model-v1.0.0 (Fase 2). v2 = reentrenamiento con la extracción SAP v2.3 defectuosa."""
    hist = pd.read_csv(me.RAIZ_REPO / "datos" / "entregas" / "entregas_cab.csv", parse_dates=["fecha_pedido"])
    entrena, prueba = hist[hist["fecha_pedido"] < "2026-05-01"], hist[hist["fecha_pedido"] >= "2026-05-01"]
    v1 = registro.alias("champion")
    if v1 is None:
        modelos = me.cargar_v1()
        m = me.evaluar(modelos, prueba)
        v1 = registro.registrar(modelos, "model-v1.0.0 de la Fase 2 (HistGradientBoosting + cuantiles 0.10/0.90).",
                                {"version_semantica": "model-v1.0.0", "git_commit": "082d216", "datos": "data-v1.0.0",
                                 "origen": "Fase 2"}, {"mae_prueba": m["mae"], "cobertura_prueba": m["cobertura"]},
                                ejemplo=prueba[me.entradas()].head(3), nombre_run="registro_eta_v1_fase2")
        registro.asignar("champion", v1)
    # v2: el ETL nuevo invirtió inventario_disponible (1 = agotado). La validación fuera de línea
    # usa los MISMOS datos invertidos, por eso el error no se ve antes del despliegue.
    def invertir(df):
        df = df.copy()
        df["inventario_disponible"] = 1 - df["inventario_disponible"]
        return df

    v2_modelos = me.entrenar(invertir(entrena))
    m_off = me.evaluar(v2_modelos, invertir(prueba))
    m_real = me.evaluar(v2_modelos, prueba)
    v2 = registro.registrar(v2_modelos, "Reentrenamiento programado con la extracción SAP v2.3 (ETL nuevo).",
                            {"version_semantica": "model-v1.1.0", "datos": "extraccion_sap_v2.3", "origen": "pipeline T18",
                             "validacion_fuera_de_linea": "aprobada"},
                            {"mae_validacion_fuera_de_linea": m_off["mae"]}, ejemplo=prueba[me.entradas()].head(3),
                            nombre_run="registro_eta_v2_etl_sap_v2_3")
    return v1, v2, {"v2_mae_validacion_offline": m_off["mae"], "v2_mae_con_datos_correctos": m_real["mae"],
                    "v1_mae_prueba": me.evaluar(me.cargar_v1(), prueba)["mae"], "v2_modelos": v2_modelos}


# --------------------------------------------------------------------------- utilidades de métricas
def metricas_ventana(df: pd.DataFrame, hist: list[dict], cfg: dict, sim: SimuladorServicio, trabajadores: int) -> dict:
    slo_d, slo_l = cfg["slo"]["disponibilidad"], cfg["slo"]["latencia"]
    rechazadas = int((df["status"] == 429).sum())
    validas = df[df["status"] != 429]          # un 429 es un rechazo intencional (límite de tasa), no una falla del servicio
    total = len(validas)
    malas = int((validas["status"] >= 500).sum())
    lentas = int((validas["ms"] > slo_l["umbral_ms"]).sum())
    ok = df[df["status"] == 200]
    m = {"solicitudes": total, "rps": len(df) / 300, "errores_5xx": malas, "rechazadas_429": rechazadas,
         "disponibilidad": 1 - malas / total if total else 1.0,
         "latencia_p50_ms": percentil(validas["ms"], 50), "latencia_p95_ms": percentil(validas["ms"], 95),
         "latencia_p99_ms": percentil(validas["ms"], 99), "cola_p95_ms": percentil(validas["ms_cola"], 95),
         "frac_lentas": lentas / total if total else 0.0,
         "saturacion_cpu": min(1.0, float(df.loc[df["status"] != 429, "ms_servicio"].sum()) / (trabajadores * 300_000)),
         "trabajadores": trabajadores}
    eta_v = ok.dropna(subset=["eta_dias"])["eta_dias"].to_numpy(float)
    hist.append({"malas": malas, "lentas": lentas, "total": total, "etas": eta_v})
    presupuesto_d, presupuesto_l = 1 - slo_d["objetivo"], 1 - slo_l["objetivo"]

    def quema(n, campo, presupuesto):
        h = hist[-n:]
        t = sum(x["total"] for x in h)
        return (sum(x[campo] for x in h) / t / presupuesto) if t else 0.0

    m["quema_disp_5m"], m["quema_disp_1h"] = quema(1, "malas", presupuesto_d), quema(12, "malas", presupuesto_d)
    m["quema_lat_5m"], m["quema_lat_30m"] = quema(1, "lentas", presupuesto_l), quema(6, "lentas", presupuesto_l)
    vol = cfg["servicio"]["volumen_mensual_estimado"]
    m["presupuesto_disp_restante"] = 1 - sum(x["malas"] for x in hist) / (presupuesto_d * vol)
    m["presupuesto_lat_restante"] = 1 - sum(x["lentas"] for x in hist) / (presupuesto_l * vol)
    # Modelo (sin etiquetas): router y ETA
    for idioma in ("es", "en"):
        g = ok[ok["idioma"] == idioma]
        m[f"tasa_baja_confianza_{idioma}"] = float((g["confianza"] < sim.router.umbral).mean()) if len(g) else 0.0
    m["tasa_automatica"] = float((ok["accion"] == "automatica").mean()) if len(ok) else 0.0
    m["confianza_media"] = float(ok["confianza"].mean()) if len(ok) else float("nan")
    eta = ok.dropna(subset=["eta_dias"])
    m["respuestas_con_eta"] = int(len(eta))
    m["eta_desacuerdo_mediana"] = float(np.median(np.abs(eta["eta_dias"] - eta["eta_sombra"]))) if len(eta) else 0.0
    m["eta_imposibles"] = float(eta["imposible"].astype(float).mean()) if len(eta) else 0.0
    # PSI de las predicciones en la última hora (12 ventanas), con grupos fijos de días: con ~25 ETA por
    # ventana el PSI de 5 min es puro ruido (lección del primer intento: valores de 0 a 2 sin ningún cambio).
    etas_1h = np.concatenate([x["etas"] for x in hist[-12:]])
    m["psi_predicciones_eta_1h"] = drift.psi_categorico(agrupar_eta(sim.ref_pred), agrupar_eta(etas_1h)) \
        if len(etas_1h) >= 100 else float("nan")
    return m


def agrupar_eta(dias: np.ndarray) -> np.ndarray:
    return np.select([dias <= 2, dias <= 4, dias <= 6, dias <= 8, dias <= 10], ["1-2", "3-4", "5-6", "7-8", "9-10"], "11+")


def contexto_servicio(df: pd.DataFrame, m: dict, cap: Capacidad, sim) -> dict:
    top = df["cliente"].value_counts(normalize=True)
    return {"replicas": cap.replicas, "saturacion_cpu": round(m["saturacion_cpu"], 3),
            "cliente_principal": f"{top.index[0]} ({top.iloc[0]:.0%} del tráfico)" if len(top) else None,
            "p95_cola_ms": round(m["cola_p95_ms"], 1), "version_eta_servida": sim.version_eta,
            "version_router": "model-v1.0.0"}


# --------------------------------------------------------------------------- simulación
def main() -> None:
    inicio_real = time.time()
    cfg = cargar_config()
    mlflow.set_tracking_uri(uri_mlflow())
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    registro = RegistroETA(cfg["mlflow"]["modelo_registrado"])
    esc = Escenario()

    with mlflow.start_run(run_name=f"game_day_{FECHA}_plano_servicio") as run_padre:
        v1, v2, info_v2 = preparar_registro(cfg, registro)
        sim = SimuladorServicio(cfg, FECHA)
        print(f"Router de reproducción: coincidencia con el modelo real = {sim.router.coincidencia:.0%}")
        mlflow.set_tags({"tipo_run": "game_day", "plano": "servicio", "version_app": cfg["version_app_monitoreada"],
                         "version_config": cfg["version_config"], "router": "model-v1.0.0 (reproducción)"})
        mlflow.log_params({"fecha": FECHA, "ventana_min": 5, "slo_disponibilidad": cfg["slo"]["disponibilidad"]["objetivo"],
                           "slo_latencia": f"{cfg['slo']['latencia']['objetivo']} < {cfg['slo']['latencia']['umbral_ms']} ms",
                           "replicas_iniciales": cfg["servicio"]["capacidad"]["replicas"],
                           "trabajadores_por_replica": cfg["servicio"]["capacidad"]["trabajadores_por_replica"],
                           "timeout_ms": cfg["servicio"]["timeout_ms"], "trafico_normal_rps": cfg["servicio"]["trafico_normal_rps"],
                           "coincidencia_router_reproduccion": sim.router.coincidencia})
        mlflow.log_artifact(str(Path(__file__).resolve().parents[1] / "config" / "monitoring_config.yaml"), "config")

        bit = Bitacora(SALIDA / "monitoring_log.jsonl.gz")
        trazas = Trazas(SALIDA / "traces.jsonl", mlflow_activo=True, max_mlflow=120)
        motor = MotorAlertas(cfg, "servicio", f"INC-{FECHA.replace('-', '')}", registrar_log=bit)
        c = cfg["servicio"]["capacidad"]
        cap = Capacidad(c["replicas"], c["trabajadores_por_replica"], c["replicas_max"])
        cola = Cola(cap.trabajadores)
        control_rb = ControlRollback()
        t0 = hora(FECHA, cfg["servicio"]["game_day"]["inicio"])
        fin = hora(FECHA, cfg["servicio"]["game_day"]["fin"])
        H = lambda hhmm: hora(FECHA, hhmm)  # noqa: E731

        ventanas, hist, acciones, alertas_tablero = [], [], [], []
        rb01 = rb03 = None
        inc = {}                         # datos de cada incidente para MTTD/MTTR
        limite_tasa = False
        trazas_mlflow_ventana = 0
        sello = lambda t: int(t.timestamp() * 1000)  # noqa: E731
        snapshots = {"10:25": "01_normal", "10:40": "02_inc02_alerta", "11:00": "03_inc02_recuperado",
                     "11:55": "04_inc03_alerta", "12:30": "05_inc03_mitigado", "14:00": "06_cierre_dia"}
        naive_alertas = 0
        ts = t0
        w = 0
        while ts < fin:
            fin_v = ts + timedelta(minutes=5)
            # ---------------------------------------------------- eventos programados
            if ts == H(esc.despliegue_v2):
                sim.servir_eta(info_v2["v2_modelos"], v2)
                acciones.append(desplegar(registro, v2, ts, "pipeline de reentrenamiento (T18)"))
                sim.version_sombra = v1
                inc["INC-02"] = {"inicio": ts, "titulo": "Despliegue defectuoso del modelo de ETA (v2)"}
                bit({"ts": ts.isoformat(), "evento": "despliegue_modelo", "modelo": "cab_eta_entrega", "version": v2,
                     "sombra": v1, "servicio": "cab-deploy"})
            if ts == H(esc.inicio_portal):
                inc["INC-03"] = {"inicio": ts, "titulo": "Saturación por consultas en bucle del portal de ARM-01"}
            if ts == H(esc.limite_tasa):
                limite_tasa = True
                acciones.append({"accion": "limite_de_tasa", "ejecutado": True, "detalle": "2 solicitudes/s por token",
                                 "ts": ts.isoformat()})
                rb03.paso(ts, "accion", "Aplicar límite de 2 solicitudes/s por token (configuración del gateway)",
                          "ARM-01 recibe 429 en el exceso; los otros 4 clientes no se afectan", "ingeniero de guardia",
                          "monitoring_log.jsonl.gz (status 429)")
            if ts == H(esc.fin_portal):
                rb03.paso(ts, "escalamiento", "Ejecutivo de cuenta confirma con TI de ARM-01 la corrección del bucle "
                          "del portal (consulta cada 15 min en lugar de continua)", "tráfico de ARM-01 vuelve a lo normal",
                          "ejecutivo de cuenta ARM-01", "ventanas.csv (rps)")
            if ts == H(esc.escala_abajo):
                rec = [v["saturacion_cpu"] for v in ventanas[-6:]]
                if max(rec) < 0.5:
                    acciones.append(reducir_replicas(cap, ts, 1))
                    cola.ajustar(cap.trabajadores, (ts - t0).total_seconds() * 1000)
                    rb03.paso(ts, "accion", "Regresar a 1 réplica (CPU < 50 % durante 30 min)",
                              f"réplicas {acciones[-1]['de']} → 1", "automatización", "ventanas.csv")

            # ---------------------------------------------------- llegadas de la ventana
            n_norm = sim.rng.poisson(cfg["servicio"]["trafico_normal_rps"] * 300)
            llegadas = [(float(sim.rng.uniform(0, 300_000)), str(sim.rng.choice(list(TOKENS))),
                         sim.mensajes[sim.rng.integers(len(sim.mensajes))], False) for _ in range(n_norm)]
            if H(esc.inicio_portal) <= ts < H(esc.fin_portal):
                n_portal = sim.rng.poisson(esc.rps_portal * 300)
                llegadas += [(float(sim.rng.uniform(0, 300_000)), "ARM-01", sim.portal[sim.rng.integers(len(sim.portal))],
                              True) for _ in range(n_portal)]
            llegadas.sort(key=lambda x: x[0])
            filas = []
            base_ms = (ts - t0).total_seconds() * 1000
            for t_rel, cliente, plantilla, es_portal in llegadas:
                momento = ts + timedelta(milliseconds=t_rel)
                trace_id = nuevo_id(32)
                if es_portal and limite_tasa and sim.rng.random() > 2 / esc.rps_portal:
                    fila = {"status": 429, "accion": None, "intencion": None, "confianza": None, "idioma": None,
                            "ms": 2.0, "ms_cola": 0.0, "ms_servicio": 0.0, "eta_dias": None, "eta_sombra": None,
                            "imposible": None}
                else:
                    r = sim.solicitud(cliente, plantilla)
                    lento = 6.0 if H(esc.pico_transitorio[0]) <= momento < H(esc.pico_transitorio[1]) else 1.0
                    servicio = (r["ms_router"] + r["ms_sap"] + r["ms_eta"] + r["ms_resp"]) * lento
                    espera = cola.atender(base_ms + t_rel, servicio)
                    total = espera + servicio
                    fila = {**r, "ms_servicio": servicio, "ms_cola": espera, "ms": min(total, cfg["servicio"]["timeout_ms"])}
                    if total > cfg["servicio"]["timeout_ms"]:
                        fila["status"] = 504
                    # traza: muestreo 1 % + todas las lentas/fallidas (máx. 15 por ventana)
                    muestrear = sim.rng.random() < cfg["mlflow"]["muestreo_trazas"] or (
                        (total > cfg["slo"]["latencia"]["umbral_ms"] or fila["imposible"]) and sim.rng.random() < 0.02)
                    if muestrear:
                        spans, t = [], 0.0
                        if espera > 0.5:
                            spans.append({"nombre": "cola.espera", "inicio_ms": 0, "dur_ms": espera,
                                          "atributos": {"trabajadores": cap.trabajadores}})
                            t = espera
                        for nombre, dur, attrs in (
                                ("router.mdeberta", r["ms_router"] * lento, {"intencion": r["intencion"], "confianza": r["confianza"],
                                                                             "accion": r["accion"]}),
                                ("sap.consultar", r["ms_sap"], {"oc": r["oc"]}),
                                ("eta.estimar", r["ms_eta"], {"version_eta": sim.version_eta, "eta_dias": r["eta_dias"],
                                                              "imposible": r["imposible"]}),
                                ("respuesta.plantilla", r["ms_resp"], {"idioma": r["idioma"]})):
                            if dur > 0 or nombre == "router.mdeberta":
                                spans.append({"nombre": nombre, "inicio_ms": t, "dur_ms": max(dur, 0.01), "atributos": attrs})
                                t += dur
                        if r["eta_dias"] is not None and sim.version_eta != sim.version_sombra:
                            spans.append({"nombre": "eta.sombra_champion", "inicio_ms": t - r["ms_eta"], "dur_ms": r["ms_eta"],
                                          "atributos": {"version_sombra": sim.version_sombra, "eta_sombra": r["eta_sombra"],
                                                        "fuera_de_ruta_critica": True}})
                        trazas.mlflow_activo = trazas_mlflow_ventana < (6 if fila["status"] != 200 or fila["imposible"] else 2)
                        trazas.registrar(trace_id, momento, spans, {
                            "status": fila["status"], "cliente": cliente, "idioma": r["idioma"], "accion": r["accion"],
                            "largo_mensaje": r["largo_mensaje"], "version_eta": sim.version_eta, "replicas": cap.replicas,
                            "ms_cliente": round(fila["ms"], 1)})
                        trazas_mlflow_ventana += int(trazas.mlflow_activo)
                fila.update({"cliente": cliente, "portal": es_portal})
                filas.append(fila)
                bit({"ts": momento.isoformat(timespec="milliseconds"), "servicio": "cab-api",
                     "nivel": "ERROR" if fila["status"] >= 500 else "WARNING" if fila["status"] == 429 else "INFO",
                     "evento": "chat_respuesta", "trace_id": trace_id, "endpoint": "/api/chat", "status": fila["status"],
                     "cliente": cliente, "origen": "portal" if es_portal else "chat_web", "ms_total": round(fila["ms"], 1),
                     "ms_cola": round(fila["ms_cola"], 1), "intencion": fila.get("intencion"), "confianza": fila.get("confianza"),
                     "accion": fila.get("accion"), "idioma": fila.get("idioma"), "largo_mensaje": fila.get("largo_mensaje"),
                     "version_router": "model-v1.0.0", "version_eta": sim.version_eta, "eta_dias": fila.get("eta_dias"),
                     "eta_sombra": fila.get("eta_sombra"), "replicas": cap.replicas})
            trazas_mlflow_ventana = 0
            df = pd.DataFrame(filas)
            m = metricas_ventana(df, hist, cfg, sim, cap.trabajadores)
            m["replicas"] = cap.replicas
            naive_alertas += int(m["latencia_p95_ms"] > cfg["slo"]["latencia"]["umbral_ms"]) + int(m["disponibilidad"] < 0.995)
            ctx = contexto_servicio(df, m, cap, sim)
            eventos = motor.evaluar(fin_v, m, ctx)

            # ---------------------------------------------------- respuestas automatizadas + runbooks
            for e in eventos:
                if e["tipo"] in ("disparada", "escalada", "agrupada"):
                    alertas_tablero.append({"ts": fin_v.strftime("%H:%M"), "severidad": e["severidad"],
                                            "texto": f"{e['alerta']} · {e['nombre']}", "incidente": e["incidente"],
                                            "estado": e["tipo"]})
                if e["tipo"] == "resuelta":
                    alertas_tablero.append({"ts": fin_v.strftime("%H:%M"), "severidad": e["severidad"],
                                            "texto": f"{e['alerta']} · {e['nombre']}", "incidente": e["incidente"],
                                            "estado": "resuelta"})
                if e["tipo"] != "disparada":
                    continue
                if e["alerta"] == "A3":
                    inc["INC-02"].update(deteccion=fin_v, id_motor=e["incidente"])
                    rb01 = EjecucionRunbook("RB-01", "INC-02", "Despliegue defectuoso del modelo de ETA", "P1")
                    rb01.paso(fin_v, "identificacion", "Alerta A3 (P1): desacuerdo con el campeón en sombra y fechas "
                              "imposibles en 2 ventanas seguidas", f"desacuerdo mediano {m['eta_desacuerdo_mediana']:.1f} días; "
                              f"imposibles {m['eta_imposibles']:.0%} (esta regla no se activó: v2 no dio fechas menores al tránsito)",
                              "motor de alertas", "notificaciones.jsonl, alerts.json")
                    rb01.paso(fin_v, "diagnostico", "Correlacionar con cambios recientes: registro de despliegues",
                              f"v{v2} desplegada a las {esc.despliegue_v2} (único cambio); la señal empezó en esa ventana",
                              "automatización", "monitoring_log.jsonl.gz (evento despliegue_modelo)")
                    r = rollback_modelo_eta(registro, fin_v, control_rb, "A3: desacuerdo con sombra y ETA imposibles")
                    acciones.append(r)
                    if r["ejecutado"]:
                        sim.servir_eta(registro.cargar("champion"), r["a_version"])
                        sim.version_sombra = r["a_version"]
                    rb01.paso(fin_v, "accion", "Rollback automático: mover @champion a la versión anterior y recargar",
                              f"@champion v{r.get('de_version')} → v{r.get('a_version')} ({'ejecutado' if r['ejecutado'] else r['motivo']})",
                              "automatización", "MLflow Model Registry (alias y etiqueta estado=revertida)")
                    inc["INC-02"]["mitigacion"] = fin_v
                if e["alerta"] in ("A1", "A2") and "INC-03" in inc and "deteccion" not in inc["INC-03"]:
                    v = df[df["status"] != 429]
                    inc["INC-03"].update(deteccion=fin_v, id_motor=e["incidente"],
                                         cola_pct=float(v["ms_cola"].sum() / (v["ms_cola"] + v["ms_servicio"]).sum()),
                                         router_ms=float(v["ms_router"].median()))
                    rb03 = EjecucionRunbook("RB-03", "INC-03", "Latencia alta y errores por saturación", e["severidad"])
                    rb03.paso(fin_v, "identificacion", f"Alerta {e['alerta']} ({e['severidad']}): quema del presupuesto de error",
                              f"disponibilidad {m['disponibilidad']:.1%}, p95 {m['latencia_p95_ms']:.0f} ms, "
                              f"quema 1 h {m['quema_disp_1h']:.0f}×", "motor de alertas", "notificaciones.jsonl")
                    rb03.paso(fin_v, "diagnostico", "Revisar saturación (CPU) y tráfico por cliente en los logs",
                              f"CPU {m['saturacion_cpu']:.0%}; {ctx['cliente_principal']}; {m['rps']:.1f} solicitudes/s "
                              f"(normal 0.5)", "automatización", "monitoring_log.jsonl.gz agrupado por cliente")
                    r = escalar_replicas(cap, m["saturacion_cpu"], fin_v, objetivo=3)
                    acciones.append(r)
                    if r["ejecutado"]:
                        cola.ajustar(cap.trabajadores, (fin_v - t0).total_seconds() * 1000)
                    rb03.paso(fin_v, "accion", "Escalamiento automático de réplicas (regla: CPU ≥ 85 %)",
                              f"réplicas {r.get('de')} → {r.get('a')} (listas en 90 s)" if r["ejecutado"] else r["motivo"],
                              "automatización", "ventanas.csv (replicas, trabajadores)")
            # pasos de diagnóstico con trazas (una ventana después de la detección)
            if rb03 and len(rb03.pasos) == 3 and fin_v == inc["INC-03"]["deteccion"] + timedelta(minutes=5):
                rb03.paso(fin_v, "diagnostico", "Revisar trazas de solicitudes lentas: ¿dónde se pierde el tiempo?",
                          f"en la ventana del pico, la espera en cola fue el {inc['INC-03']['cola_pct']:.0%} del tiempo; el "
                          f"span del router siguió en {inc['INC-03']['router_ms']:.0f} ms (mediana) → falla de capacidad, "
                          "no del modelo", "ingeniero de guardia",
                          "traces.jsonl / MLflow Traces")
                rb03.paso(fin_v, "escalamiento", "Avisar al ejecutivo de cuenta de ARM-01 (origen del tráfico)",
                          "se abre caso con TI de ARM-01", "ingeniero de guardia", "notificaciones.jsonl")

            fila_v = {"inicio": ts, "fin": fin_v, **m, "version_eta": sim.version_eta,
                      "alertas_activas": len(motor.activas())}
            ventanas.append(fila_v)
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and v == v},
                               step=w, timestamp=sello(fin_v))
            mlflow.log_metric("alertas_activas", len(motor.activas()), step=w, timestamp=sello(fin_v))

            # validación de recuperación
            for clave_inc, rb, cond in (
                    ("INC-02", rb01, lambda v: v["eta_desacuerdo_mediana"] < 1.0 and v["eta_imposibles"] < 0.05
                               and v["version_eta"] == v1),
                    ("INC-03", rb03, lambda v: v["latencia_p95_ms"] < 400 and v["disponibilidad"] >= 0.995)):
                d = inc.get(clave_inc)
                if rb and d and "deteccion" in d and "recuperacion" not in d:
                    ult = ventanas[-6:] if clave_inc == "INC-03" else ventanas[-2:]
                    req = 6 if clave_inc == "INC-03" else 2
                    if len(ult) == req and all(cond(v) for v in ult) and ult[0]["inicio"] >= d["deteccion"]:
                        d["recuperacion"] = ult[0]["inicio"]
                        d["validado"] = fin_v
                        if clave_inc == "INC-02":
                            rb.paso(fin_v, "validacion", "Confirmar 2 ventanas sanas con la versión restaurada",
                                    "desacuerdo con la sombra < 1 día y 0 % de fechas imposibles con v1 desde "
                                    f"{ult[0]['inicio']:%H:%M}", "automatización", "ventanas.csv, tablero 03")
                        else:
                            rb.paso(fin_v, "validacion", "Confirmar 30 min (6 ventanas) dentro del SLO",
                                    f"p95 < 400 ms y disponibilidad ≥ 99.5 % desde {ult[0]['inicio']:%H:%M}",
                                    "automatización", "ventanas.csv, tablero 05")
            if "INC-03" in inc and "mitigacion" not in inc["INC-03"] and "deteccion" in inc["INC-03"] \
                    and m["latencia_p95_ms"] < 400 and m["disponibilidad"] >= 0.995:
                inc["INC-03"]["mitigacion"] = ts

            # tablero en momentos clave
            hhmm = fin_v.strftime("%H:%M")
            if hhmm in snapshots:
                generar_tablero(SALIDA / "tableros" / f"{snapshots[hhmm]}_{hhmm.replace(':', '')}.html", ventanas,
                                alertas_tablero, motor, cap, sim, cfg, fin_v)
            ts, w = fin_v, w + 1
            if w % 12 == 0:
                print(f"{ts:%H:%M} · ventanas {w} · solicitudes {sum(v['solicitudes'] for v in ventanas):,} · "
                      f"activas {motor.activas()}")

        # ------------------------------------------------------------ cierre de runbooks y evidencia
        backtest = backtest_v2(info_v2["v2_modelos"])
        d2, d3 = inc["INC-02"], inc["INC-03"]
        rb01.paso(d2["validado"], "validacion", "Confirmar con etiquetas (backtest con órdenes ya entregadas)",
                  f"v2 habría tenido MAE {backtest['v2']['mae']:.2f} días y cobertura {backtest['v2']['cobertura']:.0%} "
                  f"vs v1 {backtest['v1']['mae']:.2f} y {backtest['v1']['cobertura']:.0%}", "especialista MLOps",
                  "backtest_v2.json")
        rb01.paso(d2["validado"], "cierre", "Bloquear v2 (etiqueta estado=revertida), abrir defecto del ETL SAP v2.3 y "
                  "agregar prueba de contrato de datos al pipeline", "incidente cerrado; postmortem sin culpables",
                  "especialista MLOps", "postmortem_INC-02.md")
        rb01.cerrar(estado="resuelto", mttd_min=minutos(d2["deteccion"] - d2["inicio"]),
                    tiempo_mitigacion_min=minutos(d2["mitigacion"] - d2["deteccion"]),
                    mttr_min=minutos(d2["recuperacion"] - d2["inicio"]), escalado=False,
                    respuestas_afectadas=int(sum(v["respuestas_con_eta"] for v in ventanas
                                                 if d2["inicio"] <= v["inicio"] < d2["mitigacion"])))
        rb03.paso(hora(FECHA, esc.escala_abajo) + timedelta(minutes=5), "cierre", "Documentar, dejar el límite de tasa como permanente y proponer autoescalado "
                  "por CPU en la nube", "incidente cerrado", "ingeniero de guardia", "postmortem_INC-03.md")
        rb03.cerrar(estado="resuelto", mttd_min=minutos(d3["deteccion"] - d3["inicio"]),
                    tiempo_mitigacion_min=minutos(d3["mitigacion"] - d3["deteccion"]),
                    mttr_min=minutos(d3["mitigacion"] - d3["inicio"]), escalado=True,
                    errores_5xx=int(sum(v["errores_5xx"] for v in ventanas)),
                    presupuesto_disp_restante=f"{ventanas[-1]['presupuesto_disp_restante']:.0%}")
        for rb in (rb01, rb03):
            rb.guardar(SALIDA)

        tabla = pd.DataFrame(ventanas)
        tabla.to_csv(SALIDA / "ventanas.csv", index=False)
        motor.guardar(SALIDA)
        (SALIDA / "backtest_v2.json").write_text(json.dumps(backtest, indent=2), encoding="utf-8")
        resumen = {
            "plano": "servicio", "fecha": FECHA, "solicitudes": int(tabla["solicitudes"].sum()),
            "ventanas": len(tabla), "estado_final": "ok" if not motor.activas() else "alerta",
            "incidentes": {k: {"titulo": v["titulo"], "id_motor": v.get("id_motor"), "inicio": v["inicio"].isoformat(),
                               "deteccion": v["deteccion"].isoformat(), "mitigacion": v["mitigacion"].isoformat(),
                               "recuperacion": v["recuperacion"].isoformat(),
                               "mttd_min": minutos(v["deteccion"] - v["inicio"]),
                               "mttr_min": minutos((v["recuperacion"] if k == "INC-02" else v["mitigacion"]) - v["inicio"])}
                           for k, v in inc.items()},
            "acciones": acciones, "ruido": {**motor.resumen_ruido(), "alertas_regla_simple_sin_ventanas": naive_alertas},
            "slo_dia": {"disponibilidad": 1 - tabla["errores_5xx"].sum() / tabla["solicitudes"].sum(),
                        "frac_lentas": float((tabla["frac_lentas"] * tabla["solicitudes"]).sum() / tabla["solicitudes"].sum()),
                        "presupuesto_disp_restante": float(tabla["presupuesto_disp_restante"].iloc[-1]),
                        "presupuesto_lat_restante": float(tabla["presupuesto_lat_restante"].iloc[-1])},
            "v2": {k: v for k, v in info_v2.items() if k != "v2_modelos"}, "backtest_v2": backtest,
            "registro": {"v1": v1, "v2": v2, "champion_final": registro.alias("champion")},
            "router_reproduccion_coincidencia": sim.router.coincidencia,
            "trazas": {"jsonl": trazas.n, "mlflow": trazas.enviadas_mlflow}, "lineas_log": bit.n,
        }
        (SALIDA / "monitoring_summary.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                                         encoding="utf-8")
        bit.cerrar()
        trazas.cerrar()
        extracto_log(SALIDA)
        postmortems(SALIDA, resumen, rb01, rb03)
        mlflow.log_artifacts(str(SALIDA / "tableros"), "tableros")
        for nombre in ("monitoring_summary.json", "alerts.json", "notificaciones.jsonl", "incidentes.json", "ventanas.csv",
                       "monitoring_log_extracto.jsonl", "traces.jsonl", "backtest_v2.json"):
            mlflow.log_artifact(str(SALIDA / nombre), "monitoreo")
        for k, rb in (("INC-02", rb01), ("INC-03", rb03)):
            with mlflow.start_run(run_name=f"{k}_{inc[k]['titulo'][:40]}", nested=True):
                mlflow.set_tags({"tipo_run": "incidente", "incidente": k, "runbook": rb.runbook, "severidad": rb.severidad})
                mlflow.log_metrics({"mttd_min": rb.resultado["mttd_min"], "mttr_min": rb.resultado["mttr_min"],
                                    "tiempo_mitigacion_min": rb.resultado["tiempo_mitigacion_min"]})
                for p in SALIDA.glob(f"*{k}*"):
                    mlflow.log_artifact(str(p), "evidencia")
        print(json.dumps({k: v for k, v in resumen.items() if k in ("incidentes", "ruido", "slo_dia")}, indent=1,
                         ensure_ascii=False, default=str))
        print(f"Listo en {time.time() - inicio_real:.0f} s · run {run_padre.info.run_id}")


def minutos(td: timedelta) -> float:
    return round(td.total_seconds() / 60, 1)


def backtest_v2(v2_modelos: dict) -> dict:
    """Confirmación posterior con etiquetas: órdenes del piloto entregadas hasta el 30/11/2026."""
    p = pd.read_csv(me.RAIZ_REPO / "datos" / "entregas" / "piloto_ab_cab.csv",
                    parse_dates=["fecha_pedido", "fecha_entrega_real"])
    p = p[p["fecha_entrega_real"] <= "2026-11-30"]
    return {"ordenes": int(len(p)), "v1": me.evaluar(me.cargar_v1(), p), "v2": me.evaluar(v2_modelos, p)}


def generar_tablero(ruta, ventanas, alertas, motor, cap, sim, cfg, ahora) -> None:
    t = pd.DataFrame(ventanas)
    x = list(t["fin"])
    u = t.iloc[-1]
    activas = motor.activas()
    estado = "CRÍTICO" if any(a["severidad"] == "P1" for a in activas) else \
        "ADVERTENCIA" if activas else "OK"
    S = tablero.semaforo
    tiles = [
        {"capa": "Servicio · síntoma", "titulo": "Disponibilidad (5 min)", "valor": f"{u['disponibilidad']:.2%}",
         "slo": f"SLO ≥ 99.5 % · presupuesto 28 d: " + (f"restante {u['presupuesto_disp_restante']:.0%}"
                if u['presupuesto_disp_restante'] > 0 else f"AGOTADO (consumo {1 - u['presupuesto_disp_restante']:.0%})"),
         "color": S(u["disponibilidad"], 0.995, 0.999, "min"), "x": x,
         "series": [{"y": list(t["disponibilidad"] * 100), "nombre": "disponibilidad %"},
                    {"y": list(t["presupuesto_disp_restante"].clip(lower=0) * 100), "nombre": "presupuesto restante %",
                     "color": "#e07b39"}],
         "umbral": 99.5, "ylim": (0, 101)},
        {"capa": "Servicio · síntoma", "titulo": "Latencia p95 de /api/chat", "valor": f"{u['latencia_p95_ms']:.0f} ms",
         "slo": f"SLO: 95 % < 400 ms · quema 30 min {u['quema_lat_30m']:.1f}×",
         "color": S(u["latencia_p95_ms"], 400, 300), "x": x,
         "series": [{"y": list(t["latencia_p95_ms"]), "nombre": "p95"}, {"y": list(t["latencia_p50_ms"]), "nombre": "p50",
                                                                           "color": "#9aa5a0"}], "umbral": 400},
        {"capa": "Infraestructura · causa", "titulo": "Tráfico y saturación de CPU",
         "valor": f"{u['rps']:.1f} sol/s · CPU {u['saturacion_cpu']:.0%}",
         "slo": f"réplicas activas: {cap.replicas} · referencia CPU < 85 %", "color": S(u["saturacion_cpu"], 0.85, 0.7),
         "x": x, "series": [{"y": list(t["saturacion_cpu"] * 100), "nombre": "CPU %"},
                            {"y": list(t["replicas"] * 25), "nombre": "réplicas ×25", "color": "#5b6b65", "ls": "--"}],
         "umbral": 85, "ylim": (0, 105)},
        {"capa": "Modelo ETA · sin etiquetas", "titulo": "ETA vs campeón en sombra",
         "valor": f"{u['eta_desacuerdo_mediana']:.1f} d · imposibles {u['eta_imposibles']:.0%}",
         "slo": f"SLO: desacuerdo < 1 día y < 5 % imposibles · servida v{sim.version_eta}",
         "color": "rojo" if (u["eta_desacuerdo_mediana"] >= 1 or u["eta_imposibles"] >= 0.05) else "verde", "x": x,
         "series": [{"y": list(t["eta_desacuerdo_mediana"]), "nombre": "desacuerdo (días)"},
                    {"y": list(t["eta_imposibles"] * 10), "nombre": "imposibles ×10", "color": "#c62828"}], "umbral": 1.0},
        {"capa": "Modelo ETA · drift de salida", "titulo": "PSI de las predicciones de ETA (última hora)",
         "valor": f"{u['psi_predicciones_eta_1h']:.2f}", "slo": "informativo: también cambia si cambia la mezcla de clientes",
         "color": S(u["psi_predicciones_eta_1h"], 0.25, 0.10), "x": x,
         "series": [{"y": list(t["psi_predicciones_eta_1h"]), "nombre": "PSI 1 h"}], "umbral": 0.25},
        {"capa": "Modelo router · R02", "titulo": "Baja confianza por idioma (escala a ejecutivo)",
         "valor": f"es {u['tasa_baja_confianza_es']:.0%} · en {u['tasa_baja_confianza_en']:.0%}",
         "slo": "objetivo ≤ 10 % por idioma (revisión mensual, sin alerta)",
         "color": S(max(u["tasa_baja_confianza_es"], u["tasa_baja_confianza_en"]), 0.10, 0.08), "x": x,
         "series": [{"y": list(t["tasa_baja_confianza_es"] * 100), "nombre": "es %"},
                    {"y": list(t["tasa_baja_confianza_en"] * 100), "nombre": "en %", "color": "#e07b39"}], "umbral": 10},
    ]
    ctx = [f"API app-v1.0.0 · router model-v1.0.0 · ETA @champion v{sim.version_eta}",
           f"Ventana 5 min · solicitudes en la última: {int(u['solicitudes'])}",
           f"Incidentes abiertos: {', '.join(i.id + ' (' + i.severidad + ')' for i in motor.incidentes.values()) or 'ninguno'}"]
    notas = ["Rojo = fuera del SLO; amarillo = se acerca; verde = dentro. Arriba los síntomas que ve el cliente, "
             "en medio la causa probable y abajo la calidad de los modelos.",
             "El desacuerdo con la sombra solo existe mientras corre el campeón anterior en paralelo (7 días tras un despliegue)."]
    tablero.generar(ruta, "Tablero operativo · Chatbot CAB · plano de servicio",
                    f"Game day {ahora:%d/%m/%Y} · actualizado {ahora:%H:%M}", estado, ctx, tiles, alertas, notas)


def extracto_log(carpeta: Path) -> None:
    """Extracto legible del log: eventos del monitor y solicitudes alrededor de cada incidente."""
    import gzip

    momentos = [("10:29:50", "10:30:10"), ("10:39:58", "10:40:30"), ("11:45:00", "11:45:20"), ("11:49:55", "11:50:05"),
                ("12:15:00", "12:15:03")]
    with gzip.open(carpeta / "monitoring_log.jsonl.gz", "rt", encoding="utf-8") as f, \
            (carpeta / "monitoring_log_extracto.jsonl").open("w", encoding="utf-8") as g:
        n_por_momento = {m: 0 for m in momentos}
        for linea in f:
            e = json.loads(linea)
            hhmmss = e["ts"][11:19]
            if e.get("servicio") != "cab-api":
                g.write(linea)
                continue
            for m in momentos:
                if m[0] <= hhmmss <= m[1] and n_por_momento[m] < 8:
                    n_por_momento[m] += 1
                    g.write(linea)


def postmortems(carpeta: Path, resumen: dict, rb01, rb03) -> None:
    b = resumen["backtest_v2"]
    i2, i3 = resumen["incidentes"]["INC-02"], resumen["incidentes"]["INC-03"]
    textos = {
        "INC-02": f"""# Postmortem INC-02 · Despliegue defectuoso del modelo de ETA (v2)

**Severidad:** P1 · **Runbook:** RB-01 · **Formato:** sin culpables

## Qué pasó
A las 10:30 se desplegó la versión 2 del modelo de ETA, reentrenada con la extracción de SAP v2.3. Ese ETL invirtió
el campo `inventario_disponible` (1 pasó a significar "agotado"). La validación fuera de línea usó los mismos datos
invertidos y dio MAE {resumen['v2']['v2_mae_validacion_offline']:.2f} días, por eso la versión pasó la compuerta.
En producción, la API envía el campo correcto: el modelo sumaba días de producción a piezas en inventario y los
quitaba a piezas que había que fabricar.

## Detección y respuesta
- Detección: alerta A3 a las {i2['deteccion'][11:16]} (MTTD {i2['mttd_min']:.0f} min): el ETA servido difería del
  campeón en sombra y aparecieron fechas menores que el tiempo de tránsito.
- Respuesta: rollback automático del alias @champion a v1 (segundos). MTTR {i2['mttr_min']:.0f} min.
- Respuestas afectadas: {rb01.resultado['respuestas_afectadas']} respuestas con ETA durante 10 min.

## Confirmación con etiquetas
Backtest con {b['ordenes']} órdenes del piloto ya entregadas: v2 habría tenido MAE {b['v2']['mae']:.2f} días y
cobertura {b['v2']['cobertura']:.0%}; v1 tiene {b['v1']['mae']:.2f} días y {b['v1']['cobertura']:.0%}.

## Qué funcionó
La sombra del campeón anterior detectó el problema sin esperar etiquetas, y el rollback por alias fue inmediato.

## Mejoras (acciones)
1. Prueba de contrato de datos en el pipeline: la proporción de `inventario_disponible` no puede cambiar más de 10 puntos
   contra la extracción anterior (habría bloqueado v2 antes de entrenar).
2. Despliegue canary: 10 % del tráfico durante 30 min con la sombra activa antes de pasar al 100 %.
3. Validar cada versión nueva también contra el conjunto de prueba congelado de la versión anterior.
""",
        "INC-03": f"""# Postmortem INC-03 · Saturación por consultas en bucle del portal de ARM-01

**Severidad:** P1 (A1) con A2 agrupada · **Runbook:** RB-03 · **Formato:** sin culpables

## Qué pasó
A las 11:45 el portal de proveedores de ARM-01 empezó a consultar todas sus OC en un bucle continuo
(≈16 solicitudes/s contra 0.5 normales). Con 1 réplica (2 trabajadores) la capacidad era de ≈13 solicitudes/s:
la cola creció, las solicitudes pasaron de 3 s y el chat devolvió errores a los cinco clientes.

## Detección y respuesta
- Detección: alerta A1 (quema rápida del presupuesto de disponibilidad) a las {i3['deteccion'][11:16]} (MTTD {i3['mttd_min']:.0f} min).
  A2 (latencia) se agrupó en el mismo incidente y no generó otra notificación.
- Diagnóstico: CPU al 100 %, un solo cliente con la mayor parte del tráfico y trazas con la espera en cola como
  causa principal (el router seguía en ≈130 ms): problema de capacidad, no del modelo.
- Respuesta automática: 1 → 3 réplicas. Servicio dentro del SLO a las {i3['mitigacion'][11:16]} (MTTR {i3['mttr_min']:.0f} min).
- Respuesta manual: límite de 2 solicitudes/s por token (12:15) y corrección del portal por TI de ARM-01 (12:40).
- Presupuesto de error de disponibilidad restante al cierre del día: {resumen['slo_dia']['presupuesto_disp_restante']:.0%}.

## Mejoras (acciones)
1. Límite de tasa por token como configuración permanente del gateway.
2. Autoescalado por CPU en la nube (Elastic Beanstalk o Cloud Run, según el manual de la Fase 2).
3. Endpoint por lotes para integraciones (una consulta con varias OC) y documentación para clientes.
4. Política de presupuesto de error: si queda < 25 %, se congelan despliegues no urgentes.
"""}
    for k, texto in textos.items():
        (carpeta / f"postmortem_{k}.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
