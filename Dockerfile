# syntax=docker/dockerfile:1
# =============================================================================
# Imagen de la API del chatbot de CAB (backend FastAPI + modelos + frontend estático)
#
# - En local, docker compose usa esta imagen para el servicio "api" y sirve el
#   frontend aparte con nginx (frontend/Dockerfile).
# - En la nube (Hugging Face Spaces, Cloud Run, Azure Container Apps) esta imagen
#   sola es la aplicación completa: la API también entrega el chat en "/".
#
# Construir:  docker build -t cab-chatbot-api:1.0.0 .
# Ejecutar:   docker run --rm -p 8000:8000 cab-chatbot-api:1.0.0
# =============================================================================

# 1. Imagen base ligera y compatible: Python 3.12 sobre Debian slim.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# 2. Directorio de trabajo dentro del contenedor.
WORKDIR /app

# 3. Dependencias. PyTorch para CPU desde su índice oficial (sin librerías de GPU, ~5 veces más ligero).
#    Se instalan antes de copiar el código para aprovechar la caché de capas de Docker.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# 4. Router de intención: se descarga de Hugging Face Hub la versión etiquetada.
#    Cambiar el modelo o la versión no requiere tocar código, solo estos argumentos.
ARG MODELO_ROUTER=gagarang/router-intencion-cab-mdeberta
ARG MODELO_REVISION=model-v1.0.0
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODELO_ROUTER}', revision='${MODELO_REVISION}', local_dir='/app/modelo_router', \
allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])" \
 && ls -lh /app/modelo_router

# 5. Código, modelos tabulares, datos del SAP simulado, pruebas y frontend.
COPY backend/app ./app
COPY backend/modelos ./modelos
COPY backend/tests ./tests
COPY datos/entregas/piloto_ab_cab.csv ./datos/piloto_ab_cab.csv
COPY frontend/public ./frontend

# Trazabilidad: el commit de Git y la versión quedan dentro de la imagen y se ven en /api/salud.
ARG GIT_COMMIT=desconocido
ENV GIT_COMMIT=${GIT_COMMIT} \
    VERSION_APP=app-v1.0.0 \
    MODELO_ROUTER_RUTA=/app/modelo_router \
    MODELOS_ENTREGA_RUTA=/app/modelos \
    DATOS_SAP_RUTA=/app/datos/piloto_ab_cab.csv \
    FRONTEND_RUTA=/app/frontend \
    PORT=8000

# 6. Usuario sin privilegios (uid 1000, el que exige Hugging Face Spaces).
RUN useradd --create-home --uid 1000 cab && chown -R cab:cab /app
USER cab

# 7. Puerto del servicio y verificación de salud (Docker marca el contenedor como "healthy").
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/api/salud', timeout=4)"

# 8. Comando de ejecución. PORT permite que la plataforma en la nube elija el puerto.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
