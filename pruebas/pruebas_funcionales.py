"""
pruebas_funcionales.py
----------------------
Pruebas funcionales de la solución YA DESPLEGADA (contenedores en local o en la nube).
Consume los endpoints igual que el navegador, a través del frontend (nginx), así que
también comprueba la integración frontend → backend.

Qué se prueba
  1. Casos funcionales y extremos del chat y de la API (casos_funcionales.json).
  2. El router dentro del contenedor con los 160 mensajes de la prueba externa, y la
     coincidencia con las predicciones de Colab (mismo modelo → mismas respuestas).
  3. Latencia de /api/chat con 100 solicitudes seguidas.

Solo usa la librería estándar de Python (no hay que instalar nada).

Uso:
    python pruebas/pruebas_funcionales.py                          # http://localhost:8080
    python pruebas/pruebas_funcionales.py --url https://usuario-cab-chatbot-fase2.hf.space
Salida:
    pruebas/resultados/<fecha>/ con CSV de cada prueba y resumen.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CASOS = Path(__file__).with_name("casos_funcionales.json")
EXTERNA = RAIZ / "datos" / "mensajes_cab" / "prueba_externa_v3.csv"
PRED_COLAB = RAIZ / "resultados" / "router" / "predicciones_externa.csv"


def llamar(url: str, metodo: str = "GET", cuerpo=None, token: str | None = None, crudo: str | None = None):
    datos = crudo.encode() if crudo is not None else (json.dumps(cuerpo).encode() if cuerpo is not None else None)
    req = urllib.request.Request(url, data=datos, method=metodo)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            estado, texto = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        estado, texto = e.code, e.read().decode()
    ms = (time.perf_counter() - inicio) * 1000
    try:
        return estado, json.loads(texto), ms
    except json.JSONDecodeError:
        return estado, texto, ms


def obtener(d, ruta: str):
    """Lee un campo con notación de puntos: 'datos.orden.estatus'."""
    for parte in ruta.split("."):
        if d is None:
            return None
        d = d[int(parte)] if isinstance(d, list) else d.get(parte) if isinstance(d, dict) else None
    return d


def evaluar_caso(caso: dict, respuesta, estado: int) -> tuple[bool, str]:
    if estado != caso["esperado"]["status"]:
        return False, f"status {estado} ≠ {caso['esperado']['status']}"
    for campo, esperado in caso["esperado"].get("campos", {}).items():
        valor = obtener(respuesta, campo)
        if isinstance(esperado, list):
            if valor not in esperado:
                return False, f"{campo}={valor!r} no está en {esperado}"
        elif valor != esperado:
            return False, f"{campo}={valor!r} ≠ {esperado!r}"
    for campo, texto in caso["esperado"].get("contiene", {}).items():
        valor = str(obtener(respuesta, campo))
        if texto not in valor:
            return False, f"{campo} no contiene {texto!r}"
    for campo in caso["esperado"].get("no_contiene_datos_de", []):
        if campo in json.dumps(respuesta, ensure_ascii=False):
            return False, f"la respuesta contiene {campo!r}"
    return True, "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080", help="URL del frontend o de la API")
    p.add_argument("--latencia", type=int, default=100, help="solicitudes para medir latencia")
    args = p.parse_args()
    base = args.url.rstrip("/")
    salida = RAIZ / "pruebas" / "resultados" / datetime.now().strftime("%Y%m%d_%H%M")
    salida.mkdir(parents=True, exist_ok=True)

    # 0. Salud
    estado, salud, _ = llamar(f"{base}/api/salud")
    print(f"Salud: {estado} · {salud.get('estado')} · router {salud.get('router', {}).get('version')} "
          f"· commit {salud.get('git_commit')}")
    (salida / "salud.json").write_text(json.dumps(salud, ensure_ascii=False, indent=2), encoding="utf-8")

    # Órdenes reales del SAP simulado para armar los casos
    _, abiertas, _ = llamar(f"{base}/api/ordenes?solo_abiertas=true", token="demo-arm01")
    _, todas, _ = llamar(f"{base}/api/ordenes?solo_abiertas=false", token="demo-arm01")
    _, ajenas, _ = llamar(f"{base}/api/ordenes?solo_abiertas=true", token="demo-arm03")
    variables = {
        "OC_ABIERTA": abiertas["ordenes"][0]["oc"],
        "OC_ENTREGADA": next(o["oc"] for o in todas["ordenes"] if o["estatus"] == "entregada"),
        "OC_AJENA": ajenas["ordenes"][0]["oc"],
    }

    # 1. Casos funcionales y extremos
    plantilla = CASOS.read_text(encoding="utf-8")
    for k, v in variables.items():
        plantilla = plantilla.replace("{" + k + "}", v)
    casos = json.loads(plantilla)
    filas = []
    for caso in casos:
        estado, resp, ms = llamar(f"{base}{caso['ruta']}", caso.get("metodo", "GET"), caso.get("cuerpo"),
                                  caso.get("token"), caso.get("crudo"))
        ok, motivo = evaluar_caso(caso, resp, estado)
        filas.append({"id": caso["id"], "categoria": caso["categoria"], "descripcion": caso["descripcion"],
                      "status": estado, "accion": obtener(resp, "accion") if isinstance(resp, dict) else "",
                      "intencion": obtener(resp, "intencion") if isinstance(resp, dict) else "",
                      "confianza": obtener(resp, "confianza") if isinstance(resp, dict) else "",
                      "ms": round(ms, 1), "resultado": "PASA" if ok else "FALLA", "detalle": motivo,
                      "respuesta": (obtener(resp, "respuesta") or "")[:300] if isinstance(resp, dict) else str(resp)[:300]})
        print(f"[{'PASA' if ok else 'FALLA'}] {caso['id']:5s} {caso['descripcion']} ({ms:.0f} ms) {'' if ok else motivo}")
    with open(salida / "casos_funcionales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # 2. Prueba externa a través de la API + coincidencia con Colab
    with open(EXTERNA, encoding="utf-8") as f:
        externa = list(csv.DictReader(f))
    colab = {}
    if PRED_COLAB.exists():
        with open(PRED_COLAB, encoding="utf-8") as f:
            colab = {r["id"]: r for r in csv.DictReader(f)}
    filas_ext = []
    for fila in externa:
        estado, r, ms = llamar(f"{base}/api/intencion", "POST", {"mensaje": fila["mensaje"]})
        c = colab.get(fila["id"], {})
        filas_ext.append({**fila, "prediccion_api": r["intencion"], "confianza_api": r["confianza"],
                          "accion_api": r["accion"], "prediccion_colab": c.get("prediccion", ""),
                          "accion_colab": c.get("accion", ""),
                          "coincide_colab": (c.get("prediccion") == r["intencion"] and c.get("accion") == r["accion"])
                          if c else "", "ms": round(ms, 1)})
    with open(salida / "prueba_externa_api.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_ext[0]))
        w.writeheader()
        w.writerows(filas_ext)
    claros = [r for r in filas_ext if r["tipo"] in ("claro", "spanglish")]
    acc = sum(r["prediccion_api"] == r["intencion"] for r in claros) / len(claros)
    auto = [r for r in filas_ext if r["accion_api"] == "automatica"]
    erroneas = [r for r in auto if r["prediccion_api"] not in set((r["aceptables"] or "").split("|"))]
    coincidencias = [r["coincide_colab"] for r in filas_ext if r["coincide_colab"] != ""]

    # 3. Latencia del flujo completo
    tiempos = []
    for i in range(args.latencia):
        _, _, ms = llamar(f"{base}/api/chat", "POST", {"mensaje": f"¿cuándo llega la OC {variables['OC_ABIERTA']}?"},
                          token="demo-arm01")
        tiempos.append(ms)
    tiempos.sort()
    lat = {"mediana_ms": statistics.median(tiempos), "p95_ms": tiempos[int(0.95 * len(tiempos)) - 1],
           "max_ms": tiempos[-1], "solicitudes": len(tiempos)}

    pasan = sum(f["resultado"] == "PASA" for f in filas)
    resumen = [
        f"# Resultados de pruebas funcionales · {datetime.now():%d/%m/%Y %H:%M}",
        "", f"- URL probada: {base}", f"- Commit desplegado: {salud.get('git_commit')}",
        f"- Router: {salud.get('router', {}).get('version')} (umbral {salud.get('router', {}).get('umbral_confianza')})",
        f"- Casos funcionales y extremos: **{pasan}/{len(filas)} pasan**",
        f"- Prueba externa ({len(filas_ext)} mensajes) vía API: accuracy en mensajes claros {acc:.3f}; "
        f"respondidos en automático {len(auto) / len(filas_ext):.3f}; automáticas erróneas {len(erroneas)}",
        f"- Coincidencia con Colab: {sum(coincidencias)}/{len(coincidencias)}" if coincidencias
        else "- Coincidencia con Colab: no se encontró resultados/router/predicciones_externa.csv",
        f"- Latencia /api/chat ({lat['solicitudes']} solicitudes): mediana {lat['mediana_ms']:.0f} ms, "
        f"p95 {lat['p95_ms']:.0f} ms, máximo {lat['max_ms']:.0f} ms",
    ]
    (salida / "resumen.md").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    (salida / "resumen.json").write_text(json.dumps({
        "url": base, "casos_pasan": pasan, "casos_total": len(filas), "accuracy_externa_api": acc,
        "cobertura_automatica": len(auto) / len(filas_ext), "automaticas_erroneas": len(erroneas),
        "coincidencia_colab": [sum(coincidencias), len(coincidencias)], "latencia": lat, "variables": variables,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(resumen))
    print(f"\nArchivos en {salida}")


if __name__ == "__main__":
    main()
