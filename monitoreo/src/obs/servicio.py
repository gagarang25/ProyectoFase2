"""Plano de servicio: game day del 12/11/2026 con la API REAL de la Fase 2.

Cómo funciona:
  * Se importa backend/app/main.py sin modificarlo y se inicializa con un router de
    reproducción: devuelve la decisión que el mDeBERTa-v3 real tomó para ese mismo
    mensaje (logits registrados en resultados/router/predicciones_externa.csv) y su
    latencia medida en Docker (pruebas/resultados/.../prueba_externa_api.csv). Así se
    usa la salida real del modelo sin necesitar Hugging Face.
  * El resto de la ruta es real: autenticación por token, SAP simulado, modelos de ETA y
    riesgo, plantillas de respuesta y la traza de pasos de la API.
  * Las llegadas son un proceso de Poisson y la cola se simula con varios trabajadores
    (réplicas × trabajadores); así aparece la saturación cuando el tráfico supera la capacidad.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import RAIZ_REPO

PATRON_OC = re.compile(r"\b45\d{8}\b")
TOKENS = {"ARM-01": "demo-arm01", "ARM-03": "demo-arm03", "ARM-04": "demo-arm04", "ARM-07": "demo-arm07",
          "TIER-02": "demo-tier02"}


def clave(texto: str) -> str:
    return PATRON_OC.sub("<OC>", texto.strip().lower())


class RouterReproduccion:
    """Reproduce las decisiones REALES del mDeBERTa-v3 (model-v1.0.0) con la política de la API."""

    cargado = True

    def __init__(self):
        from app.politica import decidir, softmax

        self._decidir, self._softmax = decidir, softmax
        cfg = json.loads((RAIZ_REPO / "resultados" / "router" / "config_cab.json").read_text(encoding="utf-8"))
        self.temperatura, self.umbral = float(cfg["temperatura"]), float(cfg["umbral_confianza"])
        pred = pd.read_csv(RAIZ_REPO / "resultados" / "router" / "predicciones_externa.csv")
        lat = pd.read_csv(RAIZ_REPO / "pruebas" / "resultados" / "20261028_1551" / "prueba_externa_api.csv")[["id", "ms"]]
        self.tabla = pred.merge(lat, on="id")
        self.logits = {clave(m): np.array(json.loads(lg)) for m, lg in zip(self.tabla["mensaje"], self.tabla["logits"])}
        self.ms = {clave(m): float(ms) for m, ms in zip(self.tabla["mensaje"], self.tabla["ms"])}
        self.ultimo_ms = 0.0
        # Verificación: la reproducción debe dar la misma acción que el modelo real
        acciones = [self.predecir(m).accion for m in self.tabla["mensaje"]]
        self.coincidencia = float(np.mean(np.array(acciones) == self.tabla["accion"].to_numpy()))

    def info(self) -> dict:
        return {"cargado": True, "version": "model-v1.0.0", "modo": "reproducción de salidas registradas",
                "umbral_confianza": self.umbral, "temperatura": self.temperatura}

    def predecir(self, texto: str):
        k = clave(texto)
        self.ultimo_ms = self.ms[k]
        return self._decidir(self._softmax(self.logits[k], self.temperatura), self.umbral)


@dataclass
class Escenario:
    """Línea de tiempo del game day (hora local, ventanas de 5 min)."""
    pico_transitorio: tuple = ("09:50", "09:52")        # arranque en frío de un worker: un solo pico
    despliegue_v2: str = "10:30"                         # INC-02: despliegue de ETA v2 con defecto
    inicio_portal: str = "11:45"                         # INC-03: el portal de ARM-01 consulta en bucle
    limite_tasa: str = "12:15"                           # acción manual aprobada: 2 solicitudes/s por token
    fin_portal: str = "12:40"                            # ARM-01 corrige su integración
    rps_portal: float = 16.0
    escala_abajo: str = "13:20"


class SimuladorServicio:
    def __init__(self, cfg: dict, fecha: str = "2026-11-12", semilla: int = 20261112):
        os.environ.setdefault("DATOS_SAP_RUTA", str(RAIZ_REPO / "datos" / "entregas" / "piloto_ab_cab.csv"))
        os.environ.setdefault("MODELOS_ENTREGA_RUTA", str(RAIZ_REPO / "backend" / "modelos"))
        os.environ["FECHA_REFERENCIA"] = fecha
        os.environ["BITACORA_RUTA"] = ""
        import logging

        from fastapi.testclient import TestClient

        from app import main
        from app.modelos_entrega import ModelosEntrega

        for nombre in ("cab.api", "cab.bitacora", "cab.router", "httpx"):
            logging.getLogger(nombre).setLevel(logging.WARNING)
        self.cfg = cfg
        self.rng = np.random.default_rng(semilla)
        self.router = RouterReproduccion()
        main.inicializar(router=self.router)
        self.main = main
        self.api = TestClient(main.app)
        self.sombra = ModelosEntrega(RAIZ_REPO / "backend" / "modelos")      # campeón validado (v1) en sombra
        self.version_eta = "1"
        self.version_sombra = "1"
        self._cache: dict = {}
        # Mensajes del piloto (los 160 de la prueba externa) y OC por cliente
        self.mensajes = self.router.tabla[["mensaje", "intencion", "idioma", "tipo"]].to_dict("records")
        self.portal = [m for m in self.mensajes if m["intencion"] == "estatus_orden" and PATRON_OC.search(m["mensaje"])]
        self.oc_cliente = {c: [o.oc for o in main.estado.sap.ordenes_de(c, limite=1000)] for c in TOKENS}
        self.oc_abiertas = {c: [o.oc for o in main.estado.sap.ordenes_de(c, solo_abiertas=True, limite=1000)]
                            for c in TOKENS}
        # Distribución de referencia de las predicciones de ETA (v1 sobre las órdenes abiertas)
        self.ref_pred = np.array([self.sombra.estimar(main.estado.sap.buscar(oc, c).datos_modelo).eta_dias
                                  for c in TOKENS for oc in self.oc_abiertas[c]])

    # ------------------------------------------------------------------ modelo de ETA servido
    def servir_eta(self, modelos: dict, version: str) -> None:
        e = self.main.estado.entrega
        e.eta, e.eta_inf, e.eta_sup = modelos["mediana"], modelos["inferior"], modelos["superior"]
        self.version_eta = version
        self._cache = {k: v for k, v in self._cache.items() if k[2] == version}

    # ------------------------------------------------------------------ una solicitud real a la API
    def solicitud(self, cliente: str, plantilla: dict) -> dict:
        texto = plantilla["mensaje"]
        if PATRON_OC.search(texto):
            pool = self.oc_abiertas[cliente] if self.rng.random() < 0.8 else self.oc_cliente[cliente]
            texto = PATRON_OC.sub(str(self.rng.choice(pool)), texto)
        k = (cliente, texto, self.version_eta)
        if k not in self._cache:
            r = self.api.post("/api/chat", json={"mensaje": texto}, headers={"Authorization": f"Bearer {TOKENS[cliente]}"})
            cuerpo = r.json()
            pasos = {p["paso"]: p["ms"] for p in cuerpo.get("traza", [])}
            est = (cuerpo.get("datos") or {}).get("estimacion")
            fila = {"status": r.status_code, "accion": cuerpo.get("accion"), "intencion": cuerpo.get("intencion"),
                    "confianza": cuerpo.get("confianza"), "idioma": cuerpo.get("idioma"), "oc": cuerpo.get("oc"),
                    "largo_mensaje": len(texto), "ms_router": self.router.ultimo_ms,
                    "ms_sap": pasos.get("sap", 0.0), "ms_eta": pasos.get("eta", 0.0), "ms_resp": pasos.get("respuesta", 0.0),
                    "eta_dias": est["eta_dias"] if est else None, "eta_sombra": None, "imposible": None}
            if est:
                orden = self.main.estado.sap.buscar(cuerpo["oc"], cliente)
                fila["imposible"] = bool(est["eta_dias"] < orden.datos_modelo["transito_std_dias"])
                fila["eta_sombra"] = (est["eta_dias"] if self.version_eta == self.version_sombra
                                      else self.sombra.estimar(orden.datos_modelo).eta_dias)
            self._cache[k] = fila
        return dict(self._cache[k])


def hora(fecha: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{fecha}T{hhmm}:00")


def percentil(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


class Cola:
    """Cola FIFO con k trabajadores. Las solicitudes que esperan más que el timeout fallan
    para el cliente, pero el servidor igual las procesa (eso agrava la saturación)."""

    def __init__(self, trabajadores: int):
        self.libres = [0.0] * trabajadores
        self.pendientes_arranque: list[tuple[float, int]] = []

    def ajustar(self, trabajadores: int, t_ms: float, arranque_ms: float = 90_000) -> None:
        actual = len(self.libres) + sum(n for _, n in self.pendientes_arranque)
        if trabajadores > actual:          # réplicas nuevas: disponibles después del arranque
            self.pendientes_arranque.append((t_ms + arranque_ms, trabajadores - actual))
        elif trabajadores < actual:
            self.libres = sorted(self.libres)[:trabajadores]

    def atender(self, llegada_ms: float, servicio_ms: float) -> float:
        for t, n in list(self.pendientes_arranque):
            if llegada_ms >= t:
                self.libres += [t] * n
                self.pendientes_arranque.remove((t, n))
        i = int(np.argmin(self.libres))
        inicio = max(llegada_ms, self.libres[i])
        self.libres[i] = inicio + servicio_ms
        return inicio - llegada_ms           # espera en cola
