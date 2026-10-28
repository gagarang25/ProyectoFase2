"""Configuración de la API. Todo se lee de variables de entorno para que la misma imagen
de Docker funcione en local y en la nube sin cambiar código (12-factor app)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

RAIZ_APP = Path(__file__).resolve().parents[1]

# Clientes del piloto de la Fase 1 (5 clientes) con un token de demostración.
# En producción esto lo resolvería un proveedor de identidad (OAuth2 / API gateway).
CLIENTES_DEMO = {
    "demo-arm01": "ARM-01",
    "demo-arm03": "ARM-03",
    "demo-arm04": "ARM-04",
    "demo-arm07": "ARM-07",
    "demo-tier02": "TIER-02",
}


def _ruta(variable: str, por_defecto: Path) -> Path:
    return Path(os.getenv(variable, str(por_defecto)))


@dataclass(frozen=True)
class Configuracion:
    modelo_router: Path = field(default_factory=lambda: _ruta("MODELO_ROUTER_RUTA", RAIZ_APP / "modelo_router"))
    modelos_entrega: Path = field(default_factory=lambda: _ruta("MODELOS_ENTREGA_RUTA", RAIZ_APP / "modelos"))
    datos_sap: Path = field(default_factory=lambda: _ruta("DATOS_SAP_RUTA", RAIZ_APP / "datos" / "piloto_ab_cab.csv"))
    frontend: Path = field(default_factory=lambda: _ruta("FRONTEND_RUTA", RAIZ_APP / "frontend"))
    bitacora: str = field(default_factory=lambda: os.getenv("BITACORA_RUTA", ""))
    # Fecha "de hoy" del SAP simulado. Fija por defecto para que las pruebas sean reproducibles.
    fecha_referencia: date = field(default_factory=lambda: date.fromisoformat(os.getenv("FECHA_REFERENCIA", "2026-10-28")))
    umbral_confianza: float | None = field(
        default_factory=lambda: float(os.environ["UMBRAL_CONFIANZA"]) if os.getenv("UMBRAL_CONFIANZA") else None)
    hilos_cpu: int = field(default_factory=lambda: int(os.getenv("HILOS_CPU", "2")))
    cors_origenes: list[str] = field(default_factory=lambda: [o.strip() for o in os.getenv(
        "CORS_ORIGENES", "http://localhost:8080,http://127.0.0.1:8080").split(",") if o.strip()])
    clientes: dict[str, str] = field(default_factory=lambda: json.loads(os.getenv("CLIENTES_JSON", "null")) or CLIENTES_DEMO)
    git_commit: str = field(default_factory=lambda: os.getenv("GIT_COMMIT", "desconocido"))
    version_app: str = field(default_factory=lambda: os.getenv("VERSION_APP", "app-v1.0.0"))


def cargar() -> Configuracion:
    return Configuracion()
