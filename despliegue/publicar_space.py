"""
publicar_space.py
-----------------
Publica la aplicación en Hugging Face Spaces (PaaS de contenedores). El Space construye
la imagen con el MISMO Dockerfile del repositorio y la expone con una URL pública.

Qué sube: Dockerfile, backend/ (código, modelos tabulares y pruebas), frontend/public,
el CSV del SAP simulado y un README con la configuración del Space (sdk: docker,
app_port: 8000). El router mDeBERTa-v3 NO se sube: el Dockerfile lo descarga de su
repositorio de modelo con la etiqueta model-v1.0.0.

Requisitos: pip install huggingface_hub y haber iniciado sesión con un token de
escritura (hf auth login, o la variable de entorno HF_TOKEN).

Uso (desde la raíz del repositorio):
    python despliegue/publicar_space.py --space usuario/cab-chatbot-fase2 --modelo usuario/router-intencion-cab-mdeberta
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parents[1]

README_SPACE = """---
title: Chatbot CAB Fase 2
emoji: 🚚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Estatus de órdenes y ETA (prototipo académico, datos simulados)
---

# Chatbot de estatus de órdenes y ETA · CAB (prototipo Fase 2)

Prototipo académico de la Maestría en Inteligencia Artificial (Tecmilenio). Componentes
Automotrices del Bajío es una empresa ficticia y todos los datos son simulados.

- Código fuente y manual de despliegue: {repo_github}
- Router de intención: [{modelo}](https://huggingface.co/{modelo}) (etiqueta model-v1.0.0)
- Commit desplegado: `{commit}`

Para probarlo, elige un cliente del piloto en la parte superior y pregunta, por ejemplo,
"¿cuándo llega la OC …?" usando una de las órdenes abiertas del panel derecho. La API
está documentada en `/docs`.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="usuario/nombre-del-space")
    p.add_argument("--modelo", required=True, help="usuario/router-intencion-cab-mdeberta")
    p.add_argument("--revision", default="model-v1.0.0")
    p.add_argument("--repo-github", default="https://github.com/gagarang25/ProyectoFase2")
    args = p.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip() \
        or "desconocido"
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    # Variables de construcción: el Space pasa sus "variables" como build args al Dockerfile.
    api.add_space_variable(args.space, "MODELO_ROUTER", args.modelo)
    api.add_space_variable(args.space, "MODELO_REVISION", args.revision)
    api.add_space_variable(args.space, "GIT_COMMIT", commit)
    api.add_space_variable(args.space, "CORS_ORIGENES", f"https://{args.space.replace('/', '-').lower()}.hf.space")

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(README_SPACE.format(repo_github=args.repo_github, modelo=args.modelo, commit=commit),
                          encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.space, repo_type="space")
    api.upload_folder(
        folder_path=str(RAIZ), repo_id=args.space, repo_type="space",
        allow_patterns=["Dockerfile", "backend/app/**", "backend/modelos/**", "backend/tests/**",
                        "backend/requirements.txt", "frontend/public/**", "datos/entregas/piloto_ab_cab.csv"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message=f"Despliegue de app-v1.0.0 (git {commit[:7]})",
    )
    print(f"Space publicado: https://huggingface.co/spaces/{args.space}")
    print(f"URL de la aplicación (cuando termine de construir): https://{args.space.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
