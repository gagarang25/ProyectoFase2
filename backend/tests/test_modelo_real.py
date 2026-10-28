"""Pruebas del router mDeBERTa-v3 real. Se ejecutan donde el modelo está disponible
(dentro del contenedor: docker compose exec api pytest -q). Fuera de él se omiten."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

RUTA = Path(os.getenv("MODELO_ROUTER_RUTA", Path(__file__).resolve().parents[1] / "modelo_router"))
pytestmark = pytest.mark.skipif(not (RUTA / "config.json").exists(), reason=f"modelo no disponible en {RUTA}")


@pytest.fixture(scope="module")
def router():
    from app.router_intencion import RouterIntencion

    return RouterIntencion(RUTA, hilos=int(os.getenv("HILOS_CPU", "2")))


def test_modelo_coincide_con_la_politica(router):
    from app.politica import CLASES

    assert [router.modelo.config.id2label[i] for i in range(len(CLASES))] == CLASES
    assert 0.5 <= router.umbral < 1 and router.temperatura > 0


@pytest.mark.parametrize("mensaje,intencion", [
    ("¿cuál es el estatus de la OC 4500127273?", "estatus_orden"),
    ("¿cuándo llega la OC 4500127273?", "fecha_entrega"),
    ("where is my order 4500127273?", "estatus_orden"),
    ("what is the delivery date for PO 4500127273?", "fecha_entrega"),
    ("quiero poner una queja, llegaron piezas dañadas", "queja"),
    ("quiero hablar con mi ejecutivo de cuenta", "contactar_ejecutivo"),
    ("necesito la factura de la OC 4500127273", "otro"),
])
def test_mensajes_claros(router, mensaje, intencion):
    assert router.predecir(mensaje).intencion == intencion


def test_probabilidades_validas(router):
    d = router.predecir("hola")
    assert abs(sum(d.probabilidades.values()) - 1) < 1e-3


def test_determinista(router):
    a = router.logits("¿cuándo llega mi pedido?")
    b = router.logits("¿cuándo llega mi pedido?")
    assert np.allclose(a, b)


def test_latencia_cpu(router):
    textos = ["¿cuándo llega la OC 4500127273?", "quiero hablar con un humano", "where is my order?"] * 10
    router.predecir(textos[0])
    tiempos = []
    for t in textos:
        inicio = time.perf_counter()
        router.predecir(t)
        tiempos.append((time.perf_counter() - inicio) * 1000)
    assert np.median(tiempos) < 500, f"mediana {np.median(tiempos):.0f} ms"
