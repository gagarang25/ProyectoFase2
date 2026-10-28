"""Pruebas de integración de la API (router simulado, SAP simulado y modelos de ETA reales)."""

from __future__ import annotations

import json
import logging
import re
from datetime import date

import pytest

from .conftest import AUTH, AUTH_OTRO

FECHA = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")


def orden_de(sap, cliente: str, abierta: bool):
    ordenes = sap.ordenes_de(cliente, limite=500)
    return next(o for o in ordenes if (o.estatus != "entregada") == abierta)


# ----------------------------------------------------------------- salud y validación
def test_salud_reporta_versiones(api):
    r = api.get("/api/salud")
    assert r.status_code == 200
    s = r.json()
    assert s["estado"] == "ok"
    assert s["ordenes_sap_simulado"] > 0
    assert s["modelos_entrega"]["version"] == "model-v1.0.0"
    assert "X-Tiempo-ms" in r.headers


@pytest.mark.parametrize("mensaje", ["", "   ", "x" * 1001])
def test_mensajes_invalidos_dan_422(api, mensaje):
    assert api.post("/api/intencion", json={"mensaje": mensaje}).status_code == 422
    assert api.post("/api/chat", json={"mensaje": mensaje}, headers=AUTH).status_code == 422


def test_cuerpo_mal_formado_da_422(api):
    r = api.post("/api/chat", content="esto no es json", headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 422


def test_intencion_detecta_oc_e_idioma(api):
    r = api.post("/api/intencion", json={"mensaje": "When will PO#4500127273 arrive?"}).json()
    assert r["oc_detectada"] == "4500127273"
    assert r["idioma"] == "en"
    assert r["intencion"] == "fecha_entrega"
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-3


# ----------------------------------------------------------------- seguridad (R04)
@pytest.mark.parametrize("encabezados", [{}, {"Authorization": "Bearer token-falso"}, {"Authorization": "demo-arm01"}])
def test_chat_sin_autenticacion_da_401(api, encabezados):
    assert api.post("/api/chat", json={"mensaje": "hola"}, headers=encabezados).status_code == 401


def test_no_se_ven_ordenes_de_otro_cliente(api, sap):
    ajena = orden_de(sap, "ARM-03", abierta=True)
    r = api.get(f"/api/ordenes/{ajena.oc}", headers=AUTH)
    assert r.status_code == 404
    inexistente = api.get("/api/ordenes/4599999999", headers=AUTH)
    assert inexistente.json() == r.json()  # misma respuesta: no se revela que la OC existe
    assert api.get(f"/api/ordenes/{ajena.oc}", headers=AUTH_OTRO).status_code == 200


def test_chat_con_oc_ajena_responde_igual_que_inexistente(api, sap):
    ajena = orden_de(sap, "ARM-03", abierta=True)
    r1 = api.post("/api/chat", json={"mensaje": f"¿cómo va la OC {ajena.oc}?"}, headers=AUTH).json()
    r2 = api.post("/api/chat", json={"mensaje": "¿cómo va la OC 4599999999?"}, headers=AUTH).json()
    assert r1["accion"] == r2["accion"] == "oc_no_encontrada"
    assert r1["respuesta"].replace(ajena.oc, "X") == r2["respuesta"].replace("4599999999", "X")
    assert r1["datos"] is None


def test_lista_de_ordenes_solo_del_cliente(api):
    r = api.get("/api/ordenes?solo_abiertas=false", headers=AUTH).json()
    assert r["cliente"] == "ARM-01"
    assert r["ordenes"] and all(o["cliente"] == "ARM-01" for o in r["ordenes"])


# ----------------------------------------------------------------- flujo del chat
def test_estatus_de_orden_abierta(api, sap):
    o = orden_de(sap, "ARM-01", abierta=True)
    r = api.post("/api/chat", json={"mensaje": f"¿cómo va la OC-{o.oc}?"}, headers=AUTH).json()
    assert r["accion"] == "automatica" and r["intencion"] == "estatus_orden"
    assert o.oc in r["respuesta"]
    assert r["datos"]["orden"]["estatus"] == o.estatus
    assert [p["paso"] for p in r["traza"]][:3] == ["router", "sap", "eta"]


def test_fecha_no_inventa_fechas(api, sap):
    """R05: toda fecha del texto debe venir de SAP o del modelo de ETA."""
    o = orden_de(sap, "ARM-01", abierta=True)
    r = api.post("/api/chat", json={"mensaje": f"¿cuándo llega la {o.oc}?"}, headers=AUTH).json()
    assert r["intencion"] == "fecha_entrega"
    est, orden = r["datos"]["estimacion"], r["datos"]["orden"]
    permitidas = {est["fecha_estimada"], est["fecha_min"], est["fecha_max"], orden["fecha_compromiso"]}
    en_texto = {date(int(a), int(m), int(d)).isoformat() for d, m, a in FECHA.findall(r["respuesta"])}
    assert en_texto and en_texto <= permitidas
    assert est["eta_min_dias"] <= est["eta_dias"] <= est["eta_max_dias"]
    assert est["fecha_estimada"] > "2026-10-28"  # nunca una fecha estimada en el pasado
    assert est["fecha_min"] > "2026-10-28"  # ni un rango que empiece en el pasado (defecto visto en el chat)
    assert "estimación" in r["respuesta"]


def test_rango_nunca_empieza_en_el_pasado(api, sap):
    """Para todas las órdenes abiertas de los 5 clientes, el rango informado empieza después de hoy."""
    for token in ("demo-arm01", "demo-arm03", "demo-arm04", "demo-arm07", "demo-tier02"):
        for o in api.get("/api/ordenes?solo_abiertas=true", headers={"Authorization": f"Bearer {token}"}).json()["ordenes"]:
            r = api.get(f"/api/ordenes/{o['oc']}/eta", headers={"Authorization": f"Bearer {token}"}).json()
            assert "2026-10-28" < r["fecha_min"] <= r["fecha_estimada"] <= r["fecha_max"], (o["oc"], r)


def test_orden_entregada_da_fecha_real(api, sap):
    o = orden_de(sap, "ARM-01", abierta=False)
    r = api.post("/api/chat", json={"mensaje": f"¿cuándo llega la {o.oc}?"}, headers=AUTH).json()
    assert "se entregó" in r["respuesta"]
    assert o.fecha_entrega.strftime("%d/%m/%Y") in r["respuesta"]
    assert r["datos"]["estimacion"] is None


def test_sin_oc_pide_el_dato_y_luego_usa_el_contexto(api, sap):
    r = api.post("/api/chat", json={"mensaje": "¿cómo va mi pedido?"}, headers=AUTH).json()
    assert r["accion"] == "pedir_dato"
    o = orden_de(sap, "ARM-01", abierta=True)
    r2 = api.post("/api/chat", json={"mensaje": "¿y cuándo llega?", "oc_contexto": o.oc}, headers=AUTH).json()
    assert r2["accion"] == "automatica" and r2["oc"] == o.oc


def test_duda_entre_estatus_y_fecha_pregunta(api, sap):
    r = api.post("/api/chat", json={"mensaje": "duda con la 4500127273"}, headers=AUTH).json()
    assert r["accion"] == "aclarar"
    assert [op["intencion"] for op in r["opciones"]] == ["estatus_orden", "fecha_entrega", "contactar_ejecutivo"]
    o = orden_de(sap, "ARM-01", abierta=True)
    r2 = api.post("/api/chat", json={"mensaje": "duda", "oc_contexto": o.oc, "intencion_elegida": "fecha_entrega"},
                  headers=AUTH).json()
    assert r2["accion"] == "automatica" and r2["intencion"] == "fecha_entrega"


def test_baja_confianza_escala(api):
    r = api.post("/api/chat", json={"mensaje": "galimatías"}, headers=AUTH).json()
    assert r["accion"] == "escalar" and r["requiere_ejecutivo"] and r["folio"].startswith("CAB-")


@pytest.mark.parametrize("mensaje,accion,ejecutivo", [
    ("quiero poner una queja, llegó dañado", "escalar", True),
    ("quiero hablar con mi ejecutivo", "escalar", True),
    ("necesito la factura de octubre", "canalizar", False),
])
def test_acciones_que_no_responden_en_automatico(api, mensaje, accion, ejecutivo):
    r = api.post("/api/chat", json={"mensaje": mensaje}, headers=AUTH).json()
    assert r["accion"] == accion and r["requiere_ejecutivo"] is ejecutivo
    assert r["datos"] is None


def test_respuesta_en_ingles(api, sap):
    o = orden_de(sap, "ARM-01", abierta=True)
    r = api.post("/api/chat", json={"mensaje": f"When will order {o.oc} arrive?"}, headers=AUTH).json()
    assert r["idioma"] == "en" and r["respuesta"].startswith("I estimate")


def test_pedido_urgente_con_riesgo_requiere_confirmacion_humana(api, sap):
    """R03: un pedido urgente con riesgo alto se confirma con un ejecutivo."""
    for cliente, token in (("ARM-01", "demo-arm01"), ("ARM-03", "demo-arm03"), ("ARM-04", "demo-arm04"),
                           ("ARM-07", "demo-arm07"), ("TIER-02", "demo-tier02")):
        for o in sap.ordenes_de(cliente, solo_abiertas=True, limite=500):
            if o.pedido_urgente:
                r = api.post("/api/chat", json={"mensaje": f"¿cuándo llega la {o.oc}?"},
                             headers={"Authorization": f"Bearer {token}"}).json()
                if r["datos"]["estimacion"]["riesgo_alto"]:
                    assert r["requiere_ejecutivo"] and r["datos"]["critico"]
                    assert "ejecutivo confirmará" in r["respuesta"]
                    return
    pytest.skip("No hay pedidos urgentes con riesgo alto a la fecha de referencia")


# ----------------------------------------------------------------- ETA directo
def test_eta_estimar_valores_normales(api):
    r = api.post("/api/eta/estimar", json={}).json()
    assert r["eta_dias"] >= 1 and r["advertencias"] == []
    assert 0 <= r["prob_retraso"] <= 1


def test_eta_valores_raros_dan_advertencia(api):
    r = api.post("/api/eta/estimar", json={"utilizacion_planta": 1.4, "transportista": "TR-Z",
                                           "distancia_km": 4500}).json()
    assert len(r["advertencias"]) == 3
    assert r["eta_dias"] >= 1


@pytest.mark.parametrize("campo,valor", [("utilizacion_planta", 2.5), ("cruce_frontera", 3), ("mes", 13),
                                         ("cantidad_piezas", 0), ("tasa_retraso_transportista_90d", 1.5)])
def test_eta_valores_imposibles_dan_422(api, campo, valor):
    assert api.post("/api/eta/estimar", json={campo: valor}).status_code == 422


def test_eta_aumenta_si_hay_que_producir(api):
    """Prueba metamórfica: si las piezas deben producirse, la entrega no puede ser más rápida."""
    inventario = api.post("/api/eta/estimar", json={"inventario_disponible": 1, "produccion_std_dias": 0}).json()
    producir = api.post("/api/eta/estimar", json={"inventario_disponible": 0, "produccion_std_dias": 8,
                                                  "dias_comprometidos": 12}).json()
    assert producir["eta_dias"] > inventario["eta_dias"]


def test_eta_valores_faltantes_se_imputan(api):
    r = api.post("/api/eta/estimar", json={"tasa_retraso_transportista_90d": None})
    assert r.status_code == 200


# ----------------------------------------------------------------- bitácora (R09)
def test_bitacora_no_guarda_el_texto(api, sap, caplog):
    secreto = "mi contraseña es 12345"
    with caplog.at_level(logging.INFO, logger="cab.bitacora"):
        api.post("/api/chat", json={"mensaje": secreto}, headers=AUTH)
    registros = [json.loads(r.message) for r in caplog.records if r.name == "cab.bitacora"]
    assert registros and registros[-1]["largo_mensaje"] == len(secreto)
    assert all(secreto not in r.message for r in caplog.records)


# ----------------------------------------------------------------- falla segura
def test_sin_router_escala_todo(sap):
    from fastapi.testclient import TestClient

    from app import main
    from app.router_intencion import RouterNoDisponible

    anterior = main.estado.router
    main.estado.router = RouterNoDisponible("prueba")
    try:
        with TestClient(main.app) as c:
            assert c.get("/api/salud").json()["estado"] == "degradado"
            r = c.post("/api/chat", json={"mensaje": "¿cómo va la 4500127273?"}, headers=AUTH).json()
            assert r["accion"] == "escalar" and r["requiere_ejecutivo"]
    finally:
        main.estado.router = anterior
