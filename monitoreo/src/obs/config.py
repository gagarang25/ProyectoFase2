"""Rutas y configuración compartidas."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

RAIZ_MON = Path(__file__).resolve().parents[2]          # monitoreo/
RAIZ_REPO = RAIZ_MON.parent                             # ProyectoFase2/
RAIZ_BACKEND = RAIZ_REPO / "backend"
DIR_CONFIG = RAIZ_MON / "config"
DIR_SALIDAS = RAIZ_MON / "salidas"
DIR_RUNBOOKS = RAIZ_MON / "runbooks"
DIR_MLFLOW = RAIZ_MON / "mlflow"                        # mlflow.db + mlartifacts/ (no se versiona; se respalda en zip)

# La API de la Fase 2 se importa tal cual (sin modificarla): se agrega backend/ al path.
if str(RAIZ_BACKEND) not in sys.path:
    sys.path.insert(0, str(RAIZ_BACKEND))


def cargar_config(ruta: Path | None = None) -> dict:
    ruta = ruta or DIR_CONFIG / "monitoring_config.yaml"
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def uri_mlflow() -> str:
    DIR_MLFLOW.mkdir(parents=True, exist_ok=True)
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(DIR_MLFLOW / 'mlflow.db').as_posix()}")


def carpeta_salida(nombre: str) -> Path:
    ruta = DIR_SALIDAS / nombre
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta
