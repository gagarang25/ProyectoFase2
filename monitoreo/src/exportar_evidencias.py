"""
exportar_evidencias.py
======================
1. Convierte los tableros HTML a PNG con Chrome o Edge en modo headless (si están instalados).
2. Crea un respaldo portable de MLflow (mlflow.db + artefactos) en salidas/mlflow_respaldo_A8.zip,
   para abrir el panel sin volver a ejecutar la simulación (mejora pedida en la retroalimentación de la A5).

Uso:
    python src/exportar_evidencias.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from obs.config import DIR_MLFLOW, DIR_SALIDAS  # noqa: E402

CANDIDATOS = ["google-chrome", "chromium", "chromium-browser", "msedge", "chrome",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def navegador() -> str | None:
    for c in CANDIDATOS:
        ruta = shutil.which(c) or (c if Path(c).exists() else None)
        if ruta:
            return ruta
    return None


def main() -> None:
    nav = navegador()
    htmls = sorted(DIR_SALIDAS.glob("*/tableros/*.html"))
    if nav:
        for h in htmls:
            png = h.with_suffix(".png")
            subprocess.run([nav, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                            "--window-size=1280,790", f"--screenshot={png}", h.resolve().as_uri()],
                           capture_output=True, timeout=60)
            print("PNG:", png.relative_to(DIR_SALIDAS))
    else:
        print("No se encontró Chrome/Edge: abre los HTML de salidas/*/tableros/ en el navegador.")
    destino = DIR_SALIDAS / "mlflow_respaldo_A8.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for p in DIR_MLFLOW.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(DIR_MLFLOW.parent))
    print(f"Respaldo de MLflow: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
