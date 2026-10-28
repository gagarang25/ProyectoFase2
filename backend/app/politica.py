"""
Política de decisión del router de intención.

Este módulo es la única fuente de la regla que convierte las probabilidades del
modelo en una acción del chatbot. Lo usan tanto el cuaderno de entrenamiento
(para elegir el umbral y medir la cobertura) como la API (para decidir en
producción), de modo que la regla evaluada y la regla desplegada son la misma.

Acciones posibles
  automatica  El chatbot responde solo, con datos de SAP o del modelo de ETA.
              Solo aplica a estatus_orden y fecha_entrega.
  aclarar     El modelo duda entre estatus y fecha: se pregunta al cliente cuál
              de las dos quiere (no hay riesgo de inventar nada).
  escalar     Queja, petición de hablar con una persona o baja confianza: la
              conversación pasa a un ejecutivo con el contexto ya reunido.
  canalizar   Tema fuera del alcance del chatbot (facturas, pagos, etc.): se le
              indica al cliente a qué área dirigirse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

CLASES: list[str] = ["estatus_orden", "fecha_entrega", "queja", "contactar_ejecutivo", "otro"]
AUTOMATICAS = {"estatus_orden", "fecha_entrega"}
ESCALAR = {"queja", "contactar_ejecutivo"}
UMBRAL_POR_DEFECTO = 0.90  # definido en la Fase 1 (escalar en caso de duda)


@dataclass
class Decision:
    intencion: str
    confianza: float
    accion: str
    probabilidades: dict[str, float] = field(default_factory=dict)
    segunda_intencion: str | None = None


def softmax(logits: np.ndarray, temperatura: float = 1.0) -> np.ndarray:
    """Convierte las salidas del modelo (logits) en probabilidades que suman 1."""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperatura), 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decidir(probabilidades: Sequence[float], umbral: float = UMBRAL_POR_DEFECTO,
            clases: Sequence[str] = CLASES) -> Decision:
    """Aplica la regla de decisión a un vector de probabilidades."""
    p = np.asarray(probabilidades, dtype=np.float64)
    orden = np.argsort(p)[::-1]
    i1, i2 = int(orden[0]), int(orden[1])
    top1, top2 = clases[i1], clases[i2]
    p1, p2 = float(p[i1]), float(p[i2])

    if p1 >= umbral:
        if top1 in AUTOMATICAS:
            accion = "automatica"
        elif top1 in ESCALAR:
            accion = "escalar"
        else:
            accion = "canalizar"
    elif {top1, top2} <= AUTOMATICAS and p1 + p2 >= umbral:
        accion = "aclarar"  # duda entre estatus y fecha: se pregunta, no se adivina
    else:
        accion = "escalar"  # en caso de duda, una persona

    return Decision(
        intencion=top1,
        confianza=round(p1, 4),
        accion=accion,
        probabilidades={c: round(float(v), 4) for c, v in zip(clases, p)},
        segunda_intencion=top2,
    )


def es_resultado_seguro(decision: Decision, aceptables: Iterable[str]) -> bool:
    """Un resultado es seguro si el chatbot no responde en automático algo equivocado.

    Se usa en la evaluación: responder en automático con una intención fuera de las
    aceptables es el error grave (riesgo R05); aclarar, escalar o canalizar son
    resultados seguros aunque no sean la intención exacta.
    """
    aceptables = {a for a in aceptables if a}
    if decision.accion == "automatica":
        return decision.intencion in aceptables
    return True
