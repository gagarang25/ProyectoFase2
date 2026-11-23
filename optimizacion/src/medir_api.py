"""Paso 4 · Medición de punta a punta de /api/chat: arquitectura actual vs propuesta.

Se usa la API REAL de la Fase 2 (backend/app sin cambios) con FastAPI TestClient, en un proceso limpio
por modo, con 2 hilos (como una instancia de 2 vCPU):
  actual     Router PyTorch fp32 (app-v1.0.0) + ETA calculada en línea en cada solicitud.
  onnx       Router ONNX con embeddings INT8 + ETA en línea.
  propuesta  Router ONNX con embeddings INT8 + ETA precalculada por lotes (consulta en memoria).

Carga: los 160 mensajes de la prueba externa, con números de OC reales del SAP simulado de cada cliente,
3 pasadas (480 solicitudes). Se verifica que las respuestas sean IDÉNTICAS entre modos.

Uso:  python optimizacion/src/medir_api.py <modo>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import sys
import time

import numpy as np
import psutil

MODO = sys.argv[1]
PROC = psutil.Process()
PATRON_OC = re.compile(r"\b45\d{8}\b")
TOKENS = {"ARM-01": "demo-arm01", "ARM-03": "demo-arm03", "ARM-04": "demo-arm04", "ARM-07": "demo-arm07",
          "TIER-02": "demo-tier02"}


def main() -> None:
    from comun import CARPETA_MODELO, CARPETA_RES, RAIZ, cargar_conjuntos

    os.environ.update({"MODELO_ROUTER_RUTA": str(CARPETA_MODELO), "MODELOS_ENTREGA_RUTA": str(RAIZ / "backend" / "modelos"),
                       "DATOS_SAP_RUTA": str(RAIZ / "datos" / "entregas" / "piloto_ab_cab.csv"),
                       "FECHA_REFERENCIA": "2026-10-28", "BITACORA_RUTA": "", "HILOS_CPU": "2"})
    import logging

    from fastapi.testclient import TestClient

    from app import main as api

    for n in ("cab.api", "cab.bitacora", "cab.router", "httpx"):
        logging.getLogger(n).setLevel(logging.WARNING)
    rss_base = PROC.memory_info().rss / 1e6
    inicio = time.perf_counter()
    extra = {}
    if MODO == "actual":
        api.inicializar()
    else:
        from optimizaciones_api import RouterIntencionOnnx, activar_eta_precalculada

        api.inicializar(router=RouterIntencionOnnx())
        if MODO == "propuesta":
            extra = activar_eta_precalculada(api.estado.entrega, api.estado.sap)
    seg_arranque = time.perf_counter() - inicio
    rss_listo = PROC.memory_info().rss / 1e6
    cliente_http = TestClient(api.app)

    externa = cargar_conjuntos()["externa"]
    rng = np.random.default_rng(20261120)
    oc_abiertas = {c: [o.oc for o in api.estado.sap.ordenes_de(c, solo_abiertas=True, limite=1000)] for c in TOKENS}
    solicitudes = []
    for pasada in range(3):
        for i, texto in enumerate(externa["mensaje"]):
            c = list(TOKENS)[(i + pasada) % 5]
            if PATRON_OC.search(texto):
                texto = PATRON_OC.sub(str(rng.choice(oc_abiertas[c])), texto)
            solicitudes.append((c, texto))
    for c, t in solicitudes[:20]:  # calentamiento
        cliente_http.post("/api/chat", json={"mensaje": t}, headers={"Authorization": f"Bearer {TOKENS[c]}"})

    tiempos, pasos, firmas, confianzas = [], {"router": [], "eta": []}, [], []
    for c, t in solicitudes:
        a = time.perf_counter()
        r = cliente_http.post("/api/chat", json={"mensaje": t}, headers={"Authorization": f"Bearer {TOKENS[c]}"})
        tiempos.append((time.perf_counter() - a) * 1000)
        cuerpo = r.json()
        for p in cuerpo.get("traza", []):
            if p["paso"] in pasos:
                pasos[p["paso"]].append(p["ms"])
        sin_tiempos = {k: v for k, v in cuerpo.items() if k not in ("traza", "folio")}
        sin_tiempos["respuesta"] = re.sub(r"CAB-[0-9A-F]{8}", "CAB-folio", sin_tiempos.get("respuesta", ""))
        # La confianza se compara aparte (puede variar en el 4.º decimal); todo lo demás debe ser idéntico
        confianzas.append(sin_tiempos.pop("confianza", None))
        firmas.append(hashlib.sha256(json.dumps(sin_tiempos, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    t = np.array(tiempos)
    res = {
        "modo": MODO, "solicitudes": len(t), "segundos_arranque": round(seg_arranque, 2),
        "rss_mb_antes_de_cargar": round(rss_base), "rss_mb_listo": round(rss_listo),
        "rss_mb_pico": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6),
        "latencia_ms": {k: round(float(np.percentile(t, q)), 1) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "latencia_media_ms": round(float(t.mean()), 1),
        "capacidad_sol_s_por_trabajador": round(1000 / float(t.mean()), 1),
        "ms_router_media": round(float(np.mean(pasos["router"])), 1) if pasos["router"] else None,
        "ms_eta_media": round(float(np.mean(pasos["eta"])), 2) if pasos["eta"] else None,
        "solicitudes_con_eta": len(pasos["eta"]), **extra,
    }
    (CARPETA_RES / f"api_{MODO}.json").write_text(json.dumps({"medicion": res, "latencias_ms": [round(x, 2) for x in tiempos], "firmas": firmas,
                    "confianzas": confianzas}, indent=1),
                                                 encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
