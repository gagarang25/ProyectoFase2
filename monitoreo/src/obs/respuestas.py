"""Respuestas automatizadas (Tema 20) con sus límites de seguridad.

    rollback_modelo_eta   automático: mueve @champion a la versión anterior.
    escalar_replicas      automático: agrega réplicas de la API si la CPU está saturada.
    reentrenar_challenger semiautomático: entrena, calibra y valida un challenger; la
                          promoción a @champion la aprueba una persona.

Cada función devuelve un diccionario con lo que hizo (o por qué NO lo hizo), para que
el runbook y la bitácora lo registren.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import modelos_eta as me
from .registro import RegistroETA


# ----------------------------------------------------------------------------- despliegue y rollback
def desplegar(registro: RegistroETA, version: str, ts: datetime, quien: str) -> dict:
    anterior = registro.alias("champion")
    if anterior:
        registro.etiquetar(version, "champion_anterior", anterior)
    registro.asignar("champion", version)
    registro.etiquetar(version, "desplegado", ts.isoformat())
    registro.etiquetar(version, "desplegado_por", quien)
    return {"accion": "despliegue", "version": version, "anterior": anterior, "ts": ts.isoformat()}


@dataclass
class ControlRollback:
    ultimos: list = field(default_factory=list)

    def permitido(self, ts: datetime) -> bool:
        return not any(ts - t < timedelta(hours=1) for t in self.ultimos)


def rollback_modelo_eta(registro: RegistroETA, ts: datetime, control: ControlRollback, motivo: str) -> dict:
    actual = registro.alias("champion")
    previo = registro.cliente.get_model_version(registro.nombre, actual).tags.get("champion_anterior") if actual else None
    if not previo:
        return {"accion": "rollback", "ejecutado": False, "motivo": "no hay versión anterior: escalar a humano (P1)"}
    if not control.permitido(ts):
        return {"accion": "rollback", "ejecutado": False,
                "motivo": "ya hubo un rollback en la última hora (evita ciclos): escalar a humano"}
    registro.asignar("champion", previo)
    registro.etiquetar(actual, "estado", "revertida")
    registro.etiquetar(actual, "motivo_rollback", motivo)
    control.ultimos.append(ts)
    return {"accion": "rollback", "ejecutado": True, "de_version": actual, "a_version": previo, "ts": ts.isoformat(),
            "motivo": motivo}


# ----------------------------------------------------------------------------- escalamiento
@dataclass
class Capacidad:
    replicas: int
    trabajadores_por_replica: int
    replicas_max: int
    ultimo_cambio: datetime | None = None

    @property
    def trabajadores(self) -> int:
        return self.replicas * self.trabajadores_por_replica


def escalar_replicas(cap: Capacidad, saturacion: float, ts: datetime, objetivo: int | None = None) -> dict:
    if saturacion < 0.85:
        return {"accion": "escalar_replicas", "ejecutado": False,
                "motivo": f"CPU al {saturacion:.0%}: la causa no es falta de capacidad, escalar no ayuda → humano"}
    if cap.replicas >= cap.replicas_max:
        return {"accion": "escalar_replicas", "ejecutado": False, "motivo": "ya está en el máximo de réplicas → humano"}
    if cap.ultimo_cambio and ts - cap.ultimo_cambio < timedelta(minutes=15):
        return {"accion": "escalar_replicas", "ejecutado": False, "motivo": "enfriamiento de 15 min"}
    antes = cap.replicas
    cap.replicas = min(cap.replicas_max, objetivo or cap.replicas + 2)
    cap.ultimo_cambio = ts
    return {"accion": "escalar_replicas", "ejecutado": True, "de": antes, "a": cap.replicas, "ts": ts.isoformat()}


def reducir_replicas(cap: Capacidad, ts: datetime, a: int) -> dict:
    antes, cap.replicas, cap.ultimo_cambio = cap.replicas, a, ts
    return {"accion": "reducir_replicas", "ejecutado": True, "de": antes, "a": a, "ts": ts.isoformat()}


# ----------------------------------------------------------------------------- reentrenamiento
def reentrenar_challenger(historico: pd.DataFrame, produccion: pd.DataFrame, corte: pd.Timestamp,
                          campeon: dict, peso_reciente: float = 10.0, cobertura_meta: float = 0.87) -> dict:
    """Entrena con lo entregado hasta `corte - 7 días`, calibra el rango y valida con la última semana.

    - Pesos: las órdenes de producción (oct 2026 en adelante) pesan `peso_reciente` veces más,
      para que el modelo aprenda la condición nueva sin olvidar la historia.
    - Calibración del rango (conformal simple): se ensancha k días hasta cubrir ≥ 87 %
      (85 % del KPI R08 + 2 puntos de margen, mismo criterio que en la Fase 2).
    - Compuerta: MAE ≤ 95 % del campeón Y cobertura ≥ 85 % en la semana de validación.
    """
    limite = corte - pd.Timedelta(days=7)
    entregadas = produccion[produccion["fecha_entrega_real"] <= corte]
    entrena = pd.concat([historico, entregadas[entregadas["fecha_entrega_real"] <= limite]], ignore_index=True)
    valida = entregadas[entregadas["fecha_entrega_real"] > limite]
    pesos = np.where(entrena["fecha_pedido"] >= pd.Timestamp("2026-10-01"), peso_reciente, 1.0)
    reto = me.entrenar(entrena, pesos)
    for k in range(0, 4):
        reto["desfase"] = k
        if me.evaluar(reto, valida)["cobertura"] >= cobertura_meta:
            break
    m_reto, m_camp = me.evaluar(reto, valida), me.evaluar(campeon, valida)
    pasa = m_reto["mae"] <= 0.95 * m_camp["mae"] and m_reto["cobertura"] >= 0.85
    return {"modelos": reto, "ordenes_entrenamiento": int(len(entrena)), "ordenes_validacion": int(len(valida)),
            "ordenes_produccion_usadas": int((entrena["fecha_pedido"] >= pd.Timestamp("2026-10-01")).sum()),
            "desfase_rango_dias": reto["desfase"], "challenger": m_reto, "champion": m_camp, "pasa_compuerta": bool(pasa),
            "mejora_mae": 1 - m_reto["mae"] / m_camp["mae"]}
