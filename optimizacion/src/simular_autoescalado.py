"""Paso 5 · Simulación de capacidad y autoescalado con la carga del game day de la A8 (12/11/2026, 09:00–14:00).

Se reproduce la misma línea de tiempo de la A8: tráfico normal del chat web (0.5 sol/s) y la ráfaga del portal de
ARM-01 (16 sol/s de 11:45 a 12:40, incidente INC-03). Los tiempos de servicio NO son supuestos: se remuestrean de
las 480 latencias medidas de punta a punta en medir_api.py (modo actual y modo propuesta).

Modelo de colas: cada réplica atiende una solicitud a la vez (la inferencia usa los 2 vCPU con 2 hilos); el
balanceador envía cada solicitud a la réplica que se libera primero (least outstanding requests). Si la respuesta
tarda más de 3 s, el chat web abandona (error para el SLO de disponibilidad), aunque el servidor la procese igual.

Configuraciones comparadas:
  A  actual_A8          Software actual, 1× t3.medium, escalado reactivo de la A8 (alerta de 5 min → 3 réplicas).
  B  vertical           Software actual, 1× c7i.xlarge (4 vCPU = 2 inferencias en paralelo), sin réplicas.
  C  horizontal_sin_opt Software actual, t3.medium, autoescalado propuesto (mín 2, máx 4).
  D  propuesta          Software optimizado (ONNX + ETA por lotes), t3.small, autoescalado propuesto (mín 2, máx 4).
  E  propuesta_limite   D + límite de tasa de 2 sol/s por token desde el inicio (lección del postmortem INC-03).
Cada una se corre con la demanda actual (0.5 sol/s) y con crecimiento ×4 (2 sol/s: los 85 clientes de la Fase 1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from comun import CARPETA_RES

DURACION_S = 5 * 3600                      # 09:00–14:00
PORTAL = (2 * 3600 + 45 * 60, 3 * 3600 + 40 * 60)   # 11:45–12:40
RPS_PORTAL = 16.0
TIMEOUT_MS = 3000
SLO_LAT_MS = 400
PRECIO_H = {"t3.small": 0.0208, "t3.medium": 0.0416, "c7i.xlarge": 0.1785}   # USD/h on-demand us-east-1
PRECIO_ALB_H = 0.0225


@dataclass
class Config:
    nombre: str
    software: str          # actual | propuesta
    instancia: str
    servidores_por_replica: int
    politica: str          # a8 | propuesta | fija
    minimo: int
    maximo: int
    limite_tasa: float | None = None
    alb: bool = True


CONFIGS = [
    Config("A_actual_A8", "actual", "t3.medium", 1, "a8", 1, 4, alb=True),
    Config("B_vertical", "actual", "c7i.xlarge", 2, "fija", 1, 1, alb=False),
    Config("C_horizontal_sin_opt", "actual", "t3.medium", 1, "propuesta", 2, 4),
    Config("D_propuesta", "propuesta", "t3.small", 1, "propuesta", 2, 4),
    Config("E_propuesta_limite", "propuesta", "t3.small", 1, "propuesta", 2, 4, limite_tasa=2.0),
]


def llegadas(rng, rps_base: float, limite: float | None, rps_portal: float = RPS_PORTAL):
    """Devuelve (tiempos, es_portal, rechazada_429)."""
    n = rng.poisson(rps_base * DURACION_S)
    t_chat = np.sort(rng.uniform(0, DURACION_S, n))
    dur = PORTAL[1] - PORTAL[0]
    t_portal = np.sort(rng.uniform(PORTAL[0], PORTAL[1], rng.poisson(rps_portal * dur)))
    rechazo = np.zeros(len(t_portal), dtype=bool)
    if limite:  # token bucket de 1 s por token (el portal usa un solo token)
        # El límite es POR TOKEN: con varios portales (escenario de estrés) cada uno tiene su propio límite
        n_portales = max(1, round(rps_portal / RPS_PORTAL))
        segundo = np.floor(t_portal).astype(int) * 10 + rng.integers(0, n_portales, len(t_portal))
        orden_en_seg = np.zeros(len(t_portal), dtype=int)
        vistos: dict[int, int] = {}
        for i, s in enumerate(segundo):
            orden_en_seg[i] = vistos.get(s, 0)
            vistos[s] = orden_en_seg[i] + 1
        rechazo = orden_en_seg >= limite
    t = np.concatenate([t_chat, t_portal])
    portal = np.concatenate([np.zeros(len(t_chat), bool), np.ones(len(t_portal), bool)])
    r429 = np.concatenate([np.zeros(len(t_chat), bool), rechazo])
    o = np.argsort(t, kind="stable")
    return t[o], portal[o], r429[o]


def simular(cfg: Config, rps_base: float, servicio_ms: np.ndarray, rps_portal: float = RPS_PORTAL,
            semilla: int = 42) -> dict:
    rng = np.random.default_rng(semilla)
    t, portal, r429 = llegadas(rng, rps_base, cfg.limite_tasa, rps_portal)
    s = rng.choice(servicio_ms, size=len(t)) / 1000.0
    replicas = cfg.minimo
    libres = [[0.0] * cfg.servidores_por_replica for _ in range(replicas)]   # hora en que se libera cada servidor
    pendientes: list[tuple[float, int]] = []                                   # (hora_lista, +n)
    lat = np.full(len(t), np.nan)
    ocupado_min = np.zeros(DURACION_S // 60 + 1)
    replicas_min = np.zeros(DURACION_S // 60 + 1)
    eventos = []
    i = 0
    racha_alta = racha_baja = 0
    ultimo_cambio = -1e9
    for minuto in range(DURACION_S // 60):
        fin = (minuto + 1) * 60.0
        # Réplicas que terminan de arrancar en este minuto
        for p in [p for p in pendientes if p[0] <= minuto * 60]:
            for _ in range(p[1]):
                libres.append([minuto * 60.0] * cfg.servidores_por_replica)
            pendientes.remove(p)
        replicas = len(libres)
        lat_min = []
        while i < len(t) and t[i] < fin:
            if r429[i]:
                i += 1
                continue
            # servidor que se libera primero
            ri, si = min(((r, k) for r in range(len(libres)) for k in range(len(libres[r]))),
                         key=lambda x: libres[x[0]][x[1]])
            inicio = max(t[i], libres[ri][si])
            libres[ri][si] = inicio + s[i]
            lat[i] = (libres[ri][si] - t[i]) * 1000
            lat_min.append(lat[i])
            ocupado_min[minuto] += s[i]
            i += 1
        capacidad = 60.0 * sum(len(x) for x in libres)
        util = min(1.0, ocupado_min[minuto] / capacidad)
        p95 = float(np.percentile(lat_min, 95)) if lat_min else 0.0
        replicas_min[minuto] = len(libres)
        en_camino = sum(p[1] for p in pendientes)

        if cfg.politica == "propuesta":
            # Escalar hacia afuera: 2 minutos seguidos con utilización > 60 % o p95 > 300 ms
            racha_alta = racha_alta + 1 if (util > 0.60 or p95 > 300) else 0
            racha_baja = racha_baja + 1 if (util < 0.25 and p95 < 300) else 0
            total = len(libres) + en_camino
            if racha_alta >= 2 and total < cfg.maximo and fin - ultimo_cambio >= 120:
                deseadas = min(cfg.maximo, max(total + 1, math.ceil(len(libres) * util / 0.5)))
                pendientes.append((fin + 180, deseadas - total))   # 3 min: instancia + imagen + modelo
                eventos.append({"min": minuto, "accion": f"+{deseadas - total}", "util": round(util, 2),
                                "p95": round(p95)})
                ultimo_cambio, racha_alta = fin, 0
            elif racha_baja >= 15 and len(libres) > cfg.minimo and not pendientes and fin - ultimo_cambio >= 900:
                libres.pop()
                eventos.append({"min": minuto, "accion": "-1", "util": round(util, 2), "p95": round(p95)})
                ultimo_cambio, racha_baja = fin, 0
        elif cfg.politica == "a8" and (minuto + 1) % 5 == 0:
            # A8: se evalúa por ventanas de 5 min; la alerta A1 dispara el escalado a 3 réplicas
            ventana = lat[(t >= fin - 300) & (t < fin) & ~r429]
            ventana = ventana[~np.isnan(ventana)]
            err = float((ventana > TIMEOUT_MS).mean()) if len(ventana) else 0.0
            if err > 0.005 and util >= 0.85 and len(libres) + en_camino < 3:
                pendientes.append((fin + 300, 3 - len(libres) - en_camino))
                eventos.append({"min": minuto, "accion": "→3 (A1)", "util": round(util, 2), "p95": round(p95)})
                ultimo_cambio = fin
            racha_baja = racha_baja + 5 if util < 0.20 else 0
            if racha_baja >= 30 and len(libres) > cfg.minimo and fin - ultimo_cambio >= 900:
                while len(libres) > cfg.minimo:
                    libres.pop()
                eventos.append({"min": minuto, "accion": f"→{cfg.minimo}", "util": round(util, 2), "p95": round(p95)})
                ultimo_cambio, racha_baja = fin, 0

    validas = ~r429
    l = lat[validas]
    errores = int((l > TIMEOUT_MS).sum())
    ok = l[l <= TIMEOUT_MS]
    horas_replica = replicas_min.sum() / 60
    costo = horas_replica * PRECIO_H[cfg.instancia] + (DURACION_S / 3600 * PRECIO_ALB_H if cfg.alb else 0)
    ventana_portal = (t >= PORTAL[0]) & (t < PORTAL[1]) & validas
    return {
        "config": cfg.nombre, "rps_base": rps_base, "rps_portal": rps_portal, "solicitudes_validas": int(validas.sum()),
        "rechazadas_429": int(r429.sum()), "errores_timeout": errores,
        "errores_chat_web": int(((lat > TIMEOUT_MS) & ~portal & validas).sum()),
        "solicitudes_chat_web": int((~portal).sum()),
        "disponibilidad": round(1 - errores / validas.sum(), 5),
        "pct_bajo_400ms": round(float((l <= SLO_LAT_MS).mean()), 4),
        # Percentiles acotados al timeout: más de 3 s ya es un error para el cliente
        "p50_ms": round(float(np.percentile(ok, 50)), 1),
        "p95_ms": round(min(float(np.percentile(l, 95)), TIMEOUT_MS), 1),
        "p95_ms_rafaga": round(min(float(np.percentile(lat[ventana_portal], 95)), TIMEOUT_MS), 1),
        "replicas_max": int(replicas_min.max()), "horas_replica_5h": round(horas_replica, 2),
        "costo_usd_5h": round(costo, 3), "eventos": eventos,
    }


def main() -> None:
    serv = {m: np.array(json.loads((CARPETA_RES / f"api_{m}.json").read_text())["latencias_ms"])
            for m in ("actual", "propuesta")}
    filas = []
    escenarios = {"actual (A8)": (0.5, 16.0), "crecimiento x4": (2.0, 16.0), "estres: 3 portales": (2.0, 48.0)}
    for nombre_esc, (rps, rps_portal) in escenarios.items():
        for cfg in CONFIGS:
            r = simular(cfg, rps, serv[cfg.software], rps_portal)
            r = {"escenario": nombre_esc, **r}
            filas.append(r)
            print(f"{nombre_esc[:10]:10s} {cfg.nombre:22s}: disp={r['disponibilidad']:.4f} ≤400ms={r['pct_bajo_400ms']:.3f} "
                  f"p95={r['p95_ms']:.0f} p95_rafaga={r['p95_ms_rafaga']:.0f} réplicas_max={r['replicas_max']} "
                  f"h_réplica={r['horas_replica_5h']} errores={r['errores_timeout']} 429={r['rechazadas_429']} "
                  f"eventos={r['eventos']}")
    df = pd.DataFrame([{k: v for k, v in f.items() if k != "eventos"} for f in filas])
    df.to_csv(CARPETA_RES / "simulacion_autoescalado.csv", index=False)
    (CARPETA_RES / "simulacion_autoescalado_eventos.json").write_text(
        json.dumps([{"escenario": f["escenario"], "config": f["config"], "eventos": f["eventos"]} for f in filas],
                   indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
