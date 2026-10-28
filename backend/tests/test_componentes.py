"""Pruebas unitarias de la política de decisión y de la extracción de datos."""

from __future__ import annotations

import pytest

from app.entidades import detectar_idioma, extraer_oc, normalizar_oc
from app.politica import CLASES, decidir, es_resultado_seguro, softmax


@pytest.mark.parametrize("texto,esperado", [
    ("¿cómo va la OC 4500127273?", "4500127273"),
    ("OC-4500127273", "4500127273"),
    ("PO#4500127273 please", "4500127273"),
    ("#4500127273", "4500127273"),
    ("pedido 45001272731", None),       # 11 dígitos: no es una OC
    ("4600127273", None),               # no empieza con 45
    ("NP 4410-C22", None),              # número de parte
    ("sin número", None),
])
def test_extraer_oc(texto, esperado):
    assert extraer_oc(texto) == esperado


def test_normalizar_oc():
    assert normalizar_oc("OC-4500127273") == normalizar_oc("4500127273") == "4500127273"


@pytest.mark.parametrize("texto,idioma", [
    ("¿cuándo llega mi pedido?", "es"), ("cuando llega mi pedido", "es"), ("when will my order arrive?", "en"),
    ("4500127273", "es"), ("what's the ETA de la orden", "es"),
])
def test_detectar_idioma(texto, idioma):
    assert detectar_idioma(texto) == idioma


def test_softmax_y_temperatura():
    p = softmax([2.0, 1.0, 0.0, 0.0, 0.0])
    assert abs(p.sum() - 1) < 1e-9
    assert softmax([2.0, 1.0, 0, 0, 0], temperatura=3).max() < p.max()  # T > 1 reduce la confianza


@pytest.mark.parametrize("p,accion", [
    ([0.95, 0.02, 0.01, 0.01, 0.01], "automatica"),
    ([0.02, 0.95, 0.01, 0.01, 0.01], "automatica"),
    ([0.01, 0.01, 0.95, 0.02, 0.01], "escalar"),
    ([0.01, 0.01, 0.02, 0.95, 0.01], "escalar"),
    ([0.01, 0.01, 0.01, 0.02, 0.95], "canalizar"),
    ([0.55, 0.40, 0.02, 0.02, 0.01], "aclarar"),     # estatus vs fecha
    ([0.55, 0.05, 0.38, 0.01, 0.01], "escalar"),     # estatus vs queja: no se pregunta, se escala
    ([0.30, 0.20, 0.20, 0.20, 0.10], "escalar"),
])
def test_politica(p, accion):
    assert decidir(p, 0.90).accion == accion


def test_umbral_exacto_se_acepta():
    assert decidir([0.90, 0.04, 0.03, 0.02, 0.01], 0.90).accion == "automatica"


def test_resultado_seguro():
    auto_bien = decidir([0.95, 0.02, 0.01, 0.01, 0.01], 0.9)
    assert es_resultado_seguro(auto_bien, {"estatus_orden"})
    assert not es_resultado_seguro(auto_bien, {"queja"})
    assert es_resultado_seguro(decidir([0.3, 0.2, 0.2, 0.2, 0.1], 0.9), {"queja"})
    assert len(decidir([0.2] * 5, 0.9).probabilidades) == len(CLASES)
