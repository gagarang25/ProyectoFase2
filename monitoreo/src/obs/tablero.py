"""Tablero operativo (Tema 19): pocas métricas accionables, lo crítico arriba, colores de
semáforo, agrupado por capa y con la lista de alertas recientes. Se genera como HTML
autocontenido (abre sin internet y MLflow lo muestra como artefacto) y como PNG."""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COLORES = {"verde": "#2e7d32", "amarillo": "#e0a100", "rojo": "#c62828", "gris": "#7a8580"}
FONDO = {"verde": "#e8f5e9", "amarillo": "#fff8e1", "rojo": "#ffebee", "gris": "#f1f3f2"}
SEV_COLOR = {"P1": "rojo", "P2": "rojo", "P3": "amarillo", "INFO": "gris"}


def semaforo(valor, slo, advertencia, direccion: str = "max") -> str:
    """direccion='max': el valor debe quedarse por debajo del SLO; 'min': por encima."""
    if valor is None or valor != valor:
        return "gris"
    if direccion == "max":
        return "rojo" if valor > slo else "amarillo" if valor > advertencia else "verde"
    return "rojo" if valor < slo else "amarillo" if valor < advertencia else "verde"


def _grafica(x, series: list[dict], umbral=None, alto=1.25, ancho=3.6, fechas=True, ylim=None) -> str:
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=110)
    for s in series:
        ax.plot(x, s["y"], color=s.get("color", "#0e5c4a"), lw=1.6, label=s.get("nombre"),
                ls=s.get("ls", "-"), marker=s.get("marker"))
    if umbral is not None:
        ax.axhline(umbral, color="#c62828", ls="--", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if fechas:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if (x[-1] - x[0]).days < 1 else "%d/%m"))
    if len(series) > 1:
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generar(ruta: Path, titulo: str, momento: str, estado: str, contexto: list[str], tiles: list[dict],
            alertas: list[dict], notas: list[str] | None = None) -> Path:
    """tiles: [{titulo, valor, slo, color, x, series, umbral, capa, ylim}]"""
    color_estado = {"OK": "verde", "ADVERTENCIA": "amarillo", "CRÍTICO": "rojo"}[estado]
    bloques = []
    for t in tiles:
        img = _grafica(t["x"], t["series"], t.get("umbral"), ylim=t.get("ylim"), fechas=t.get("fechas", True))
        bloques.append(f"""
        <div class="tile" style="border-top:6px solid {COLORES[t['color']]};background:{FONDO[t['color']]}">
          <div class="capa">{escape(t.get('capa', ''))}</div>
          <div class="ttl">{escape(t['titulo'])}</div>
          <div class="val" style="color:{COLORES[t['color']]}">{escape(t['valor'])}</div>
          <div class="slo">{escape(t['slo'])}</div>
          <img src="{img}"/>
        </div>""")
    filas = "".join(
        f"<tr><td>{escape(a['ts'])}</td><td><span class='sev' style='background:{COLORES[SEV_COLOR.get(a['severidad'], 'gris')]}'>"
        f"{escape(a['severidad'])}</span></td><td>{escape(a['texto'])}</td><td>{escape(a.get('incidente') or '')}</td>"
        f"<td>{escape(a['estado'])}</td></tr>" for a in alertas[-7:][::-1]) or \
        "<tr><td colspan='5'>Sin alertas en el periodo</td></tr>"
    notas_html = "".join(f"<li>{escape(n)}</li>" for n in (notas or []))
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>{escape(titulo)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#fafbfa;color:#1d2a26}}
.cab{{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:#0e3b30;color:#fff}}
.cab h1{{font-size:18px;margin:0}} .cab .m{{font-size:12px;opacity:.85}}
.estado{{font-size:15px;font-weight:bold;padding:6px 14px;border-radius:6px;background:{COLORES[color_estado]};color:#fff}}
.ctx{{padding:6px 18px;font-size:12px;background:#e6ece9}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:12px 18px}}
.tile{{border-radius:6px;padding:8px 10px;box-shadow:0 1px 2px rgba(0,0,0,.08)}}
.capa{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#5b6b65}}
.ttl{{font-size:13px;font-weight:bold}} .val{{font-size:24px;font-weight:bold;margin:2px 0}}
.slo{{font-size:11px;color:#44524d;margin-bottom:4px}} .tile img{{width:100%}}
table{{border-collapse:collapse;width:calc(100% - 36px);margin:0 18px 8px;font-size:12px;background:#fff}}
th,td{{border-bottom:1px solid #dde3e0;padding:5px 6px;text-align:left}} th{{background:#e6ece9}}
.sev{{color:#fff;padding:1px 6px;border-radius:3px;font-weight:bold}}
h2{{font-size:14px;margin:6px 18px}} ul{{font-size:11.5px;margin:4px 18px 14px;color:#44524d}}
</style></head><body>
<div class="cab"><div><h1>{escape(titulo)}</h1><div class="m">{escape(momento)}</div></div>
<div class="estado">Estado general: {estado}</div></div>
<div class="ctx">{' · '.join(escape(c) for c in contexto)}</div>
<div class="grid">{''.join(bloques)}</div>
<h2>Alertas recientes</h2>
<table><tr><th>Hora</th><th>Sev.</th><th>Alerta</th><th>Incidente</th><th>Estado</th></tr>{filas}</table>
<ul>{notas_html}</ul>
</body></html>"""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(html, encoding="utf-8")
    return ruta
