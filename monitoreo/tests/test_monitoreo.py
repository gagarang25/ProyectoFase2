"""Pruebas unitarias del paquete de monitoreo (rápidas, sin MLflow ni la API).

    python -m pytest monitoreo/tests -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obs import drift  # noqa: E402
from obs.alertas import MotorAlertas  # noqa: E402
from obs.config import cargar_config  # noqa: E402
from obs.respuestas import Capacidad, escalar_replicas  # noqa: E402

RNG = np.random.default_rng(0)
T0 = datetime(2026, 11, 12, 9, 0)


def test_psi_estable_y_con_cambio():
    ref = RNG.normal(0, 1, 5000)
    assert drift.psi(ref, RNG.normal(0, 1, 2000)) < 0.10
    assert drift.psi(ref, RNG.normal(1.0, 1, 2000)) > 0.25


def test_jsd_y_chi2_categoricas():
    ref = pd.Series(RNG.choice(list("ABC"), 3000, p=[0.5, 0.3, 0.2]))
    igual = pd.Series(RNG.choice(list("ABC"), 1000, p=[0.5, 0.3, 0.2]))
    distinta = pd.Series(RNG.choice(list("ABC"), 1000, p=[0.1, 0.3, 0.6]))
    assert drift.jsd(ref, igual) < 0.01 and drift.jsd(ref, distinta) > 0.10
    assert drift.chi2(ref, distinta)[1] < 0.01


def test_ks_detecta_desplazamiento():
    estadistico, p = drift.ks(RNG.normal(0, 1, 2000), RNG.normal(0.5, 1, 2000))
    assert estadistico > 0.1 and p < 0.01


def test_fuera_de_rango():
    df = pd.DataFrame({"x": [0.5, 0.9, 1.1, 0.95]})
    assert drift.fuera_de_rango(df, {"x": [0.7, 0.97]}) == 0.5


def _motor():
    return MotorAlertas(cargar_config(), "servicio", "PRUEBA")


def _sano():
    return {"quema_disp_1h": 0, "quema_disp_5m": 0, "quema_lat_30m": 0, "quema_lat_5m": 0,
            "eta_desacuerdo_mediana": 0, "eta_imposibles": 0}


def test_pico_aislado_no_notifica():
    """A2 pide 2 ventanas seguidas: un solo pico queda como 'pendiente' y no se notifica."""
    motor = _motor()
    motor.evaluar(T0, {**_sano(), "quema_lat_30m": 7, "quema_lat_5m": 9})
    motor.evaluar(T0 + timedelta(minutes=5), _sano())
    assert motor.notificaciones == []
    assert any(e["tipo"] == "pendiente" for e in motor.eventos)


def test_agrupacion_y_deduplicacion():
    """A1 y A2 del mismo grupo forman UN incidente; las repeticiones no se vuelven a notificar."""
    motor = _motor()
    malo = {**_sano(), "quema_disp_1h": 50, "quema_disp_5m": 50, "quema_lat_30m": 10, "quema_lat_5m": 10}
    for i in range(3):
        motor.evaluar(T0 + timedelta(minutes=5 * i), malo)
    assert len(motor.historial) == 1
    assert len(motor.notificaciones) == 1                       # solo "incidente abierto"
    tipos = [e["tipo"] for e in motor.eventos]
    assert "agrupada" in tipos and "duplicada_suprimida" in tipos


def test_resolucion_con_histeresis():
    motor = _motor()
    malo = {**_sano(), "eta_desacuerdo_mediana": 4}
    motor.evaluar(T0, malo)
    motor.evaluar(T0 + timedelta(minutes=5), malo)              # dispara A3
    motor.evaluar(T0 + timedelta(minutes=10), _sano())          # 1 ventana sana: sigue activa
    assert motor.activas()
    motor.evaluar(T0 + timedelta(minutes=15), _sano())          # 2 ventanas sanas: se resuelve
    assert not motor.activas() and motor.historial[0].cerrado is not None


def test_escalar_solo_si_cpu_saturada():
    cap = Capacidad(1, 2, 4)
    assert not escalar_replicas(cap, 0.40, T0)["ejecutado"]    # CPU normal: escalar no ayuda
    assert escalar_replicas(cap, 0.95, T0, objetivo=3)["ejecutado"] and cap.replicas == 3
    assert not escalar_replicas(cap, 0.95, T0 + timedelta(minutes=5))["ejecutado"]   # enfriamiento


def test_plano_semanal_sin_recordatorios_repetidos():
    """En el plano semanal solo se notifica al abrir, al subir de severidad y al cerrar."""
    motor = MotorAlertas(cargar_config(), "modelo", "PRUEBA")
    for i in range(6):
        motor.evaluar(T0 + timedelta(days=7 * i), {"pct_carga_sobre_rango": 1.0, "mae_semanal": 1.0,
                                                    "quema_cobertura_4s": 0.5})
    assert len(motor.notificaciones) == 1
