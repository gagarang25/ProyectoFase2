"""Configuración de pytest.

Las pruebas de la API usan un router simulado con reglas fijas para probar la lógica de
integración (SAP, ETA, plantillas, seguridad) de forma determinista y sin cargar el
modelo de lenguaje. Las pruebas del modelo real están en test_modelo_real.py y se
ejecutan solo donde el modelo está disponible (dentro del contenedor).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

RAIZ_BACKEND = Path(__file__).resolve().parents[1]
RAIZ_REPO = RAIZ_BACKEND.parent
sys.path.insert(0, str(RAIZ_BACKEND))

# Fuera del contenedor, las rutas apuntan a los archivos del repositorio.
if not (RAIZ_BACKEND / "datos" / "piloto_ab_cab.csv").exists():
    os.environ.setdefault("DATOS_SAP_RUTA", str(RAIZ_REPO / "datos" / "entregas" / "piloto_ab_cab.csv"))
os.environ.setdefault("FECHA_REFERENCIA", "2026-10-28")
os.environ.setdefault("BITACORA_RUTA", "")

from app.politica import decidir  # noqa: E402


class RouterPrueba:
    """Router determinista: decide por palabras clave. Solo para pruebas de integración."""

    cargado = True

    def info(self) -> dict:
        return {"cargado": True, "version": "router-de-prueba"}

    def predecir(self, texto: str):
        t = texto.lower()
        if "duda" in t:  # estatus y fecha casi empatados → debe preguntar
            p = [0.50, 0.44, 0.02, 0.02, 0.02]
        elif "galimatías" in t:  # nada claro → debe escalar
            p = [0.30, 0.10, 0.20, 0.20, 0.20]
        elif "queja" in t or "complain" in t:
            p = [0.01, 0.01, 0.96, 0.01, 0.01]
        elif "ejecutivo" in t or "human" in t:
            p = [0.01, 0.01, 0.01, 0.96, 0.01]
        elif "factura" in t or "invoice" in t:
            p = [0.01, 0.01, 0.01, 0.01, 0.96]
        elif "cuándo" in t or "when" in t or "fecha" in t:
            p = [0.02, 0.95, 0.01, 0.01, 0.01]
        else:
            p = [0.95, 0.02, 0.01, 0.01, 0.01]
        return decidir(p, 0.90)


@pytest.fixture(scope="session")
def api():
    from fastapi.testclient import TestClient

    from app import main

    main.inicializar(router=RouterPrueba())
    with TestClient(main.app) as cliente:
        yield cliente


@pytest.fixture(scope="session")
def sap():
    from app import main

    return main.estado.sap


AUTH = {"Authorization": "Bearer demo-arm01"}  # cliente ARM-01
AUTH_OTRO = {"Authorization": "Bearer demo-arm03"}  # cliente ARM-03
