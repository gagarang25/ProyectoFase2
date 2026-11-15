"""Ejecución de un runbook con evidencia (Tema 20: identificar → diagnosticar → actuar →
validar → cerrar). Cada paso queda con hora, quién lo hizo (persona o automatización),
qué se hizo, el resultado y la evidencia. Se guarda en Markdown y JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ETAPAS = ("identificacion", "diagnostico", "accion", "escalamiento", "validacion", "cierre")


class EjecucionRunbook:
    def __init__(self, runbook: str, incidente: str, titulo: str, severidad: str):
        self.runbook, self.incidente, self.titulo, self.severidad = runbook, incidente, titulo, severidad
        self.pasos: list[dict] = []
        self.resultado: dict = {}

    def paso(self, ts: datetime, etapa: str, descripcion: str, resultado: str, ejecuta: str = "automatización",
             evidencia: str | None = None) -> None:
        assert etapa in ETAPAS, etapa
        self.pasos.append({"n": len(self.pasos) + 1, "ts": ts.isoformat(timespec="minutes"), "etapa": etapa,
                           "ejecuta": ejecuta, "descripcion": descripcion, "resultado": resultado,
                           "evidencia": evidencia})

    def cerrar(self, **resultado) -> None:
        self.resultado = resultado

    def guardar(self, carpeta: Path) -> Path:
        base = carpeta / f"runbook_{self.runbook}_{self.incidente}"
        base.with_suffix(".json").write_text(json.dumps(
            {"runbook": self.runbook, "incidente": self.incidente, "titulo": self.titulo, "severidad": self.severidad,
             "pasos": self.pasos, "resultado": self.resultado}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        def md(texto) -> str:   # "<" se interpreta como etiqueta HTML en algunos visores de Markdown
            return str(texto or "").replace("<", "&lt;").replace("|", "/")

        filas = ["| # | Hora | Etapa | Ejecuta | Qué se hizo | Resultado | Evidencia |", "|---|---|---|---|---|---|---|"]
        for p in self.pasos:
            filas.append(f"| {p['n']} | {p['ts'].replace('T', ' ')} | {p['etapa']} | {md(p['ejecuta'])} | "
                         f"{md(p['descripcion'])} | {md(p['resultado'])} | {md(p['evidencia'])} |")
        res = "\n".join(f"- **{k.replace('_', ' ')}:** {v}" for k, v in self.resultado.items())
        texto = (f"# Ejecución del runbook {self.runbook} · {self.incidente}\n\n**{self.titulo}** · severidad "
                 f"{self.severidad}\n\n" + "\n".join(filas) + f"\n\n## Resultado\n\n{res}\n")
        base.with_suffix(".md").write_text(texto, encoding="utf-8")
        return base.with_suffix(".md")
