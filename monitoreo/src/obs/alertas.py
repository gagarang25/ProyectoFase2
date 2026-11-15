"""Motor de alertas inteligentes basado en SLO (Tema 20).

Técnicas para reducir ruido que implementa:
    1. Ventanas de evaluación: una regla dispara solo si se cumple N ventanas seguidas.
    2. Quema del presupuesto de error en dos ventanas (larga y corta): alerta cuando el
       SLO está en riesgo, no ante cualquier desviación.
    3. Agrupación y deduplicación: alertas del mismo `grupo` forman UN incidente; las
       repeticiones no se vuelven a notificar (se cuentan como suprimidas).
    4. Severidad (P1/P2/P3/INFO) con ruteo distinto y re-notificación espaciada.
    5. Histéresis: una alerta se resuelve tras `resolver_tras` ventanas sanas seguidas.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

OPERADORES = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
ORDEN_SEVERIDAD = {"INFO": 0, "P3": 1, "P2": 2, "P1": 3}


@dataclass
class EstadoRegla:
    seguidas_si: int = 0
    seguidas_no: int = 0
    disparada: bool = False
    desde: datetime | None = None


@dataclass
class Incidente:
    id: str
    grupo: str
    abierto: datetime
    severidad: str
    reglas: set = field(default_factory=set)
    notificaciones: int = 0
    suprimidas: int = 0
    ultima_notificacion: datetime | None = None
    cerrado: datetime | None = None
    acciones: list = field(default_factory=list)

    def a_dict(self) -> dict:
        return {"id": self.id, "grupo": self.grupo, "abierto": self.abierto.isoformat(), "severidad": self.severidad,
                "reglas": sorted(self.reglas), "notificaciones": self.notificaciones,
                "notificaciones_suprimidas": self.suprimidas,
                "cerrado": self.cerrado.isoformat() if self.cerrado else None, "acciones": self.acciones}


class MotorAlertas:
    def __init__(self, config: dict, plano: str, prefijo: str, registrar_log: Callable[[dict], None] | None = None):
        self.cfg = config
        self.reglas = [r for r in config["alertas"] if r["plano"] == plano]
        self.sev = config["severidades"]
        self.resolver_tras = int(config["resolucion"]["resolver_tras"])
        self.renotificar = bool(config["resolucion"].get("renotificar_por_tiempo", {}).get(plano, True))
        self.estado = {r["id"]: EstadoRegla() for r in self.reglas}
        self.incidentes: dict[str, Incidente] = {}      # grupo → incidente activo
        self.historial: list[Incidente] = []
        self.eventos: list[dict] = []                   # alerts.json
        self.notificaciones: list[dict] = []            # notificaciones.jsonl
        self.prefijo = prefijo
        self._n = 0
        self.log = registrar_log or (lambda e: None)

    # ------------------------------------------------------------------
    @staticmethod
    def _cumple(cond: dict, metricas: dict) -> bool:
        valor = metricas.get(cond["metrica"])
        return valor is not None and valor == valor and OPERADORES[cond["op"]](valor, cond["umbral"])

    def _condicion(self, regla: dict, metricas: dict) -> bool:
        if "condiciones" in regla:
            return all(self._cumple(c, metricas) for c in regla["condiciones"])
        return any(self._cumple(c, metricas) for c in regla["condiciones_alguna"])

    def _valores(self, regla: dict, metricas: dict) -> dict:
        conds = regla.get("condiciones") or regla.get("condiciones_alguna")
        return {c["metrica"]: (round(metricas.get(c["metrica"]), 4) if metricas.get(c["metrica"]) is not None else None)
                for c in conds}

    def _umbrales(self, regla: dict) -> dict:
        conds = regla.get("condiciones") or regla.get("condiciones_alguna")
        return {c["metrica"]: f"{c['op']} {c['umbral']}" for c in conds}

    # ------------------------------------------------------------------
    def evaluar(self, ts: datetime, metricas: dict, contexto: dict | None = None) -> list[dict]:
        """Evalúa todas las reglas en una ventana. Devuelve los eventos nuevos."""
        contexto = contexto or {}
        nuevos = []
        for regla in self.reglas:
            st = self.estado[regla["id"]]
            if self._condicion(regla, metricas):
                st.seguidas_si += 1
                st.seguidas_no = 0
                if not st.disparada and st.seguidas_si >= int(regla.get("ventanas", 1)):
                    st.disparada, st.desde = True, ts
                    nuevos.append(self._disparar(ts, regla, metricas, contexto))
                elif st.disparada:
                    nuevos.append(self._repetida(ts, regla, metricas, contexto))
                elif st.seguidas_si >= 1:
                    nuevos.append(self._evento(ts, regla, "pendiente", metricas,
                                               f"condición cumplida {st.seguidas_si} de {regla.get('ventanas', 1)} "
                                               "ventanas: todavía no se notifica"))
            else:
                st.seguidas_si = 0
                if st.disparada:
                    st.seguidas_no += 1
                    if st.seguidas_no >= self.resolver_tras:
                        st.disparada = False
                        nuevos.append(self._resolver(ts, regla, metricas))
        return [e for e in nuevos if e]

    def _evento(self, ts, regla, tipo, metricas, detalle, incidente=None) -> dict:
        e = {"ts": ts.isoformat(), "tipo": tipo, "alerta": regla["id"], "nombre": regla["nombre"],
             "severidad": regla["severidad"], "slo": regla["slo"], "valores": self._valores(regla, metricas),
             "umbrales": self._umbrales(regla), "ventanas_requeridas": regla.get("ventanas", 1),
             "runbook": regla["runbook"], "accion_automatica": regla.get("accion_automatica"),
             "incidente": incidente, "detalle": detalle}
        self.eventos.append(e)
        self.log({"evento": f"alerta_{tipo}", **{k: e[k] for k in ("alerta", "severidad", "incidente", "valores")},
                  "ts": e["ts"]})
        return e

    def _disparar(self, ts, regla, metricas, contexto) -> dict:
        grupo = regla["grupo"]
        inc = self.incidentes.get(grupo)
        if inc is None:
            self._n += 1
            inc = Incidente(id=f"{self.prefijo}-{self._n:02d}", grupo=grupo, abierto=ts, severidad=regla["severidad"])
            self.incidentes[grupo] = inc
            self.historial.append(inc)
            inc.reglas.add(regla["id"])
            e = self._evento(ts, regla, "disparada", metricas, "nuevo incidente", inc.id)
            self._notificar(ts, inc, regla, metricas, contexto, motivo="incidente abierto")
            return e
        inc.reglas.add(regla["id"])
        if ORDEN_SEVERIDAD[regla["severidad"]] > ORDEN_SEVERIDAD[inc.severidad]:
            inc.severidad = regla["severidad"]
            e = self._evento(ts, regla, "escalada", metricas, f"el incidente sube a {regla['severidad']}", inc.id)
            self._notificar(ts, inc, regla, metricas, contexto, motivo=f"severidad escalada a {regla['severidad']}")
            return e
        inc.suprimidas += 1
        return self._evento(ts, regla, "agrupada", metricas,
                            f"se agrupa en {inc.id} (misma causa probable): no se notifica de nuevo", inc.id)

    def _repetida(self, ts, regla, metricas, contexto) -> dict | None:
        inc = self.incidentes.get(regla["grupo"])
        if inc is None:
            return None
        minutos = self.sev[inc.severidad]["renotificar_min"] if self.renotificar else 0
        if minutos and inc.ultima_notificacion and (ts - inc.ultima_notificacion).total_seconds() / 60 >= minutos:
            self._notificar(ts, inc, regla, metricas, contexto, motivo=f"sigue activo después de {minutos} min")
            return self._evento(ts, regla, "renotificada", metricas, "recordatorio", inc.id)
        inc.suprimidas += 1
        return self._evento(ts, regla, "duplicada_suprimida", metricas, "alerta repetida: no se notifica", inc.id)

    def _resolver(self, ts, regla, metricas) -> dict:
        inc = self.incidentes.get(regla["grupo"])
        e = self._evento(ts, regla, "resuelta", metricas,
                         f"{self.resolver_tras} ventanas seguidas dentro del umbral", inc.id if inc else None)
        if inc and not any(self.estado[r].disparada for r in inc.reglas):
            inc.cerrado = ts
            del self.incidentes[regla["grupo"]]
            self._notificar(ts, inc, regla, metricas, {}, motivo="incidente resuelto (validar con el runbook)")
        return e

    def _notificar(self, ts, inc: Incidente, regla, metricas, contexto, motivo: str) -> None:
        canales = self.sev[inc.severidad]["canales"]
        msg = {
            "ts": ts.isoformat(), "incidente": inc.id, "motivo": motivo, "severidad": inc.severidad,
            "titulo": f"[{inc.severidad}] {regla['nombre']}", "slo": regla["slo"],
            "metricas": self._valores(regla, metricas), "umbrales": self._umbrales(regla),
            "ventanas_evaluadas": regla.get("ventanas", 1), "contexto": contexto,
            "runbook": self._ruta_runbook(regla["runbook"]), "accion_automatica": regla.get("accion_automatica"),
            "canales": canales, "respuesta_esperada": self.sev[inc.severidad]["respuesta"],
        }
        inc.notificaciones += 1
        inc.ultima_notificacion = ts
        self.notificaciones.append(msg)
        self.log({"evento": "notificacion_enviada", "incidente": inc.id, "severidad": inc.severidad,
                  "canales": canales, "ts": ts.isoformat()})

    @staticmethod
    def _ruta_runbook(clave: str) -> str:
        from .config import DIR_RUNBOOKS

        encontrados = sorted(DIR_RUNBOOKS.glob(f"{clave}_*.md"))
        return f"monitoreo/runbooks/{encontrados[0].name}" if encontrados else f"monitoreo/runbooks/{clave}.md"

    # ------------------------------------------------------------------
    def activas(self) -> list[dict]:
        return [{"id": r["id"], "nombre": r["nombre"], "severidad": r["severidad"]} for r in self.reglas
                if self.estado[r["id"]].disparada]

    def guardar(self, carpeta) -> None:
        (carpeta / "alerts.json").write_text(json.dumps(self.eventos, ensure_ascii=False, indent=2), encoding="utf-8")
        with (carpeta / "notificaciones.jsonl").open("w", encoding="utf-8") as f:
            for n in self.notificaciones:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        (carpeta / "incidentes.json").write_text(
            json.dumps([i.a_dict() for i in self.historial], ensure_ascii=False, indent=2), encoding="utf-8")

    def resumen_ruido(self) -> dict:
        tipos = {}
        for e in self.eventos:
            tipos[e["tipo"]] = tipos.get(e["tipo"], 0) + 1
        return {"evaluaciones_con_condicion": sum(v for k, v in tipos.items() if k != "resuelta"),
                "notificaciones_enviadas": len(self.notificaciones), "por_tipo": tipos}
