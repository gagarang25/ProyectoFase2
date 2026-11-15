"""Logs estructurados y trazas.

Logs: una línea JSON por evento (monitoring_log.jsonl). Mismos campos que la bitácora
de la Fase 2 (riesgo R09) más `trace_id`, `status`, `ms_cola` y la versión de cada
modelo. Igual que en la Fase 2, NUNCA se guarda el texto del mensaje (LFPDPPP).

Trazas: cada solicitud muestreada genera una traza con un span por etapa
(cola → router → SAP → ETA → plantilla). Se escriben en traces.jsonl con un formato
parecido al de OpenTelemetry y, además, en MLflow Tracing para verlas en la interfaz.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path


def nuevo_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


class Bitacora:
    def __init__(self, ruta: Path):
        import gzip

        self.ruta = ruta
        self.f = gzip.open(ruta, "wt", encoding="utf-8") if ruta.suffix == ".gz" else ruta.open("w", encoding="utf-8")
        self.n = 0

    def __call__(self, evento: dict) -> None:
        evento.setdefault("nivel", "INFO")
        evento.setdefault("servicio", "cab-monitor")
        self.f.write(json.dumps(evento, ensure_ascii=False, default=str) + "\n")
        self.n += 1

    def cerrar(self) -> None:
        self.f.close()


class Trazas:
    """Escribe spans en JSONL y, si se indica, también en MLflow Tracing."""

    def __init__(self, ruta: Path, mlflow_activo: bool = False, max_mlflow: int = 120):
        self.f = ruta.open("w", encoding="utf-8")
        self.mlflow_activo = mlflow_activo
        self.max_mlflow = max_mlflow
        self.enviadas_mlflow = 0
        self.n = 0

    def registrar(self, trace_id: str, inicio: datetime, spans: list[dict], atributos: dict) -> None:
        """spans: [{nombre, inicio_ms, dur_ms, atributos}] relativos al inicio de la solicitud."""
        raiz = nuevo_id(8)
        total = max(s["inicio_ms"] + s["dur_ms"] for s in spans)
        filas = [{"trace_id": trace_id, "span_id": raiz, "parent_span_id": None, "nombre": "POST /api/chat",
                  "inicio": inicio.isoformat(timespec="milliseconds"), "duracion_ms": round(total, 2),
                  "estado": atributos.get("status"), "atributos": atributos}]
        for s in spans:
            filas.append({"trace_id": trace_id, "span_id": nuevo_id(8), "parent_span_id": raiz, "nombre": s["nombre"],
                          "inicio": (inicio + timedelta(milliseconds=s["inicio_ms"])).isoformat(timespec="milliseconds"),
                          "duracion_ms": round(s["dur_ms"], 2), "estado": "ok", "atributos": s.get("atributos", {})})
        for fila in filas:
            self.f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        self.n += 1
        if self.mlflow_activo and self.enviadas_mlflow < self.max_mlflow:
            self._mlflow(inicio, spans, atributos, total)

    def _mlflow(self, inicio: datetime, spans: list[dict], atributos: dict, total: float) -> None:
        import mlflow

        t0 = int(inicio.timestamp() * 1e9)
        raiz = mlflow.start_span_no_context(
            name="POST /api/chat", span_type="CHAIN", start_time_ns=t0,
            inputs={"cliente": atributos.get("cliente"), "idioma": atributos.get("idioma"),
                    "largo_mensaje": atributos.get("largo_mensaje")},
            attributes={k: str(v) for k, v in atributos.items()})
        tipos = {"cola.espera": "UNKNOWN", "router.mdeberta": "LLM", "sap.consultar": "RETRIEVER",
                 "eta.estimar": "TOOL", "eta.sombra_champion": "TOOL", "respuesta.plantilla": "PARSER"}
        for s in spans:
            hijo = mlflow.start_span_no_context(
                name=s["nombre"], span_type=tipos.get(s["nombre"], "UNKNOWN"), parent_span=raiz,
                start_time_ns=t0 + int(s["inicio_ms"] * 1e6), attributes={k: str(v) for k, v in s.get("atributos", {}).items()})
            hijo.end(end_time_ns=t0 + int((s["inicio_ms"] + s["dur_ms"]) * 1e6))
        estado = "OK" if atributos.get("status") == 200 else "ERROR"
        raiz.end(outputs={"accion": atributos.get("accion"), "status": atributos.get("status"),
                          "ms_servidor": round(total, 1), "ms_cliente": atributos.get("ms_cliente")}, status=estado, end_time_ns=t0 + int(total * 1e6))
        self.enviadas_mlflow += 1

    def cerrar(self) -> None:
        self.f.close()
