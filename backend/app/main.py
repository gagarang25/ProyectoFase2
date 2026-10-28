"""API del chatbot de estatus de órdenes y ETA de CAB (prototipo, Fase 2).

Arquitectura (Fase 1):
  mensaje del cliente → router de intención (mDeBERTa-v3) → decisión → consulta segura
  a SAP (simulado) y modelo de ETA, o escalamiento → respuesta con plantilla.

Endpoints principales (documentación interactiva en /docs):
  GET  /api/salud                  Estado de la API y versiones de modelos, datos y código.
  POST /api/intencion              Clasifica un mensaje (router + política de decisión).
  GET  /api/ordenes                Órdenes del cliente autenticado.
  GET  /api/ordenes/{oc}           Estatus de una OC del cliente autenticado.
  GET  /api/ordenes/{oc}/eta       Fecha estimada, rango, motivos y riesgo de retraso.
  POST /api/eta/estimar            ETA y riesgo para datos de una orden (simulaciones y pruebas).
  POST /api/chat                   Flujo completo de una conversación.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as cfg_mod
from .entidades import detectar_idioma, extraer_oc, normalizar_oc
from .esquemas import ChatEntrada, DatosOrden, MensajeEntrada, Paso, RespuestaChat, RespuestaIntencion
from .modelos_entrega import ModelosEntrega
from .politica import Decision
from .respuestas import respuesta_estatus, respuesta_fecha, texto
from .router_intencion import crear_router
from .sap_simulado import SapSimulado

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cab.api")
log_bitacora = logging.getLogger("cab.bitacora")


class Estado:
    """Componentes cargados una sola vez al arrancar (los modelos tardan segundos en cargar)."""
    cfg = None
    router = None
    entrega = None
    sap = None
    inicio = None


estado = Estado()


def inicializar(router=None) -> None:
    estado.cfg = cfg_mod.cargar()
    estado.entrega = ModelosEntrega(estado.cfg.modelos_entrega)
    estado.sap = SapSimulado(estado.cfg.datos_sap, estado.cfg.fecha_referencia, estado.entrega.entradas)
    estado.router = router or crear_router(estado.cfg.modelo_router, estado.cfg.umbral_confianza, estado.cfg.hilos_cpu)
    estado.inicio = datetime.now(timezone.utc)
    if estado.cfg.bitacora and not log_bitacora.handlers:
        try:
            manejador = logging.FileHandler(estado.cfg.bitacora, encoding="utf-8")
            manejador.setFormatter(logging.Formatter("%(message)s"))
            log_bitacora.addHandler(manejador)
        except OSError as error:  # sin permiso de escritura: la bitácora sigue en la salida estándar
            log.warning("No se pudo abrir la bitácora %s (%s); se registra solo en consola", estado.cfg.bitacora, error)
    log.info("API lista: %d órdenes en SAP simulado, router cargado=%s", len(estado.sap), estado.router.cargado)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    if estado.sap is None:  # las pruebas pueden inicializar con un router simulado
        inicializar()
    yield


app = FastAPI(
    title="API del chatbot de CAB (prototipo Fase 2)",
    description="Estatus de órdenes y fecha estimada de entrega para clientes B2B de Componentes Automotrices "
                "del Bajío (empresa ficticia). Datos simulados.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)
_cors = cfg_mod.cargar().cors_origenes
app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    respuesta.headers["X-Tiempo-ms"] = f"{(time.perf_counter() - inicio) * 1000:.1f}"
    return respuesta


# ---------------------------------------------------------------------------
# Autenticación simulada: un token por cliente del piloto (riesgo R04)
# ---------------------------------------------------------------------------
def cliente_autenticado(authorization: str | None = Header(None, description="Bearer <token del cliente>")) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Falta el encabezado Authorization: Bearer <token>")
    cliente = estado.cfg.clientes.get(authorization.split(" ", 1)[1].strip())
    if not cliente:
        raise HTTPException(401, "Token no válido")
    return cliente


def registrar(evento: dict) -> None:
    """Bitácora de cada conversación (riesgo R09). No guarda el texto del mensaje, solo su largo,
    para no almacenar datos personales (LFPDPPP)."""
    evento["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_bitacora.info(json.dumps(evento, ensure_ascii=False))


# ---------------------------------------------------------------------------
@app.get("/api/salud", tags=["operación"])
def salud():
    return {
        "estado": "ok" if estado.router.cargado else "degradado",
        "version_app": estado.cfg.version_app,
        "git_commit": estado.cfg.git_commit,
        "inicio_utc": estado.inicio.isoformat(timespec="seconds"),
        "fecha_referencia_sap": estado.cfg.fecha_referencia.isoformat(),
        "ordenes_sap_simulado": len(estado.sap),
        "router": estado.router.info(),
        "modelos_entrega": estado.entrega.info(),
    }


@app.post("/api/intencion", response_model=RespuestaIntencion, tags=["modelos"])
def clasificar(entrada: MensajeEntrada):
    d = estado.router.predecir(entrada.mensaje)
    return RespuestaIntencion(intencion=d.intencion, confianza=d.confianza, accion=d.accion,
                              segunda_intencion=d.segunda_intencion, probabilidades=d.probabilidades,
                              idioma=detectar_idioma(entrada.mensaje), oc_detectada=extraer_oc(entrada.mensaje))


@app.get("/api/ordenes", tags=["SAP simulado"])
def listar_ordenes(solo_abiertas: bool = True, cliente: str = Depends(cliente_autenticado)):
    ordenes = estado.sap.ordenes_de(cliente, solo_abiertas=solo_abiertas)
    return {"cliente": cliente, "ordenes": [o.publica() for o in ordenes]}


@app.get("/api/ordenes/{oc}", tags=["SAP simulado"])
def consultar_orden(oc: str, cliente: str = Depends(cliente_autenticado)):
    orden = estado.sap.buscar(oc, cliente)
    if not orden:
        raise HTTPException(404, "Orden no encontrada en la cuenta del cliente")
    return orden.publica()


@app.get("/api/ordenes/{oc}/eta", tags=["modelos"])
def eta_de_orden(oc: str, idioma: str = "es", cliente: str = Depends(cliente_autenticado)):
    orden = estado.sap.buscar(oc, cliente)
    if not orden:
        raise HTTPException(404, "Orden no encontrada en la cuenta del cliente")
    if orden.estatus == "entregada":
        return {"oc": orden.oc, "estatus": "entregada", "fecha_entrega": orden.fecha_entrega.isoformat()}
    est = estado.entrega.estimar(orden.datos_modelo, idioma, orden.fecha_pedido, estado.cfg.fecha_referencia)
    return {"oc": orden.oc, "estatus": orden.estatus, "fecha_compromiso": orden.fecha_compromiso.isoformat(),
            **est.a_dict()}


@app.post("/api/eta/estimar", tags=["modelos"])
def estimar_eta(datos: DatosOrden, idioma: str = "es"):
    return estado.entrega.estimar(datos.a_dict(), idioma).a_dict()


# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=RespuestaChat, tags=["chat"])
def chat(entrada: ChatEntrada, cliente: str = Depends(cliente_autenticado)):
    traza: list[Paso] = []
    reloj = time.perf_counter()

    def marcar(paso: str, detalle: str):
        nonlocal reloj
        ahora = time.perf_counter()
        traza.append(Paso(paso=paso, detalle=detalle, ms=round((ahora - reloj) * 1000, 2)))
        reloj = ahora

    idioma = detectar_idioma(entrada.mensaje)
    oc = extraer_oc(entrada.mensaje) or normalizar_oc(entrada.oc_contexto or "")
    folio = f"CAB-{uuid.uuid4().hex[:8].upper()}"

    # 1. Intención: elegida por el cliente (botón de aclaración) o del router
    if entrada.intencion_elegida:
        d = Decision(intencion=entrada.intencion_elegida, confianza=1.0,
                     accion="automatica" if entrada.intencion_elegida in ("estatus_orden", "fecha_entrega")
                     else "escalar" if entrada.intencion_elegida in ("queja", "contactar_ejecutivo") else "canalizar")
        marcar("intencion", f"elegida por el cliente: {d.intencion}")
    else:
        d = estado.router.predecir(entrada.mensaje)
        marcar("router", f"{d.intencion} ({d.confianza:.2f}) → {d.accion}")

    r = RespuestaChat(respuesta="", accion=d.accion, intencion=d.intencion, confianza=d.confianza, idioma=idioma,
                      oc=oc)

    # 2. Acción
    if not estado.router.cargado and not entrada.intencion_elegida:
        r.respuesta, r.requiere_ejecutivo, r.folio = texto("router_no_disponible", idioma, folio=folio), True, folio
    elif d.accion == "aclarar":
        r.respuesta = texto("aclarar", idioma)
        r.opciones = [{"intencion": "estatus_orden", "texto": "Estatus de la orden" if idioma == "es" else "Order status"},
                      {"intencion": "fecha_entrega", "texto": "Fecha de entrega" if idioma == "es" else "Delivery date"},
                      {"intencion": "contactar_ejecutivo",
                       "texto": "Hablar con un ejecutivo" if idioma == "es" else "Talk to an executive"}]
    elif d.accion == "escalar":
        clave = "escalar_queja" if d.intencion == "queja" else "escalar_contacto" \
            if d.intencion == "contactar_ejecutivo" else "escalar_duda"
        r.respuesta, r.requiere_ejecutivo, r.folio = texto(clave, idioma, folio=folio), True, folio
    elif d.accion == "canalizar":
        r.respuesta = texto("canalizar", idioma)
        r.opciones = [{"intencion": "contactar_ejecutivo",
                       "texto": "Hablar con un ejecutivo" if idioma == "es" else "Talk to an executive"}]
    elif not oc:
        r.accion, r.respuesta = "pedir_dato", texto("pedir_oc", idioma)
    else:
        orden = estado.sap.buscar(oc, cliente)
        marcar("sap", f"OC {oc}: {'encontrada, ' + orden.estatus if orden else 'no encontrada en la cuenta'}")
        if not orden:
            r.accion, r.respuesta = "oc_no_encontrada", texto("oc_no_encontrada", idioma, oc=oc)
            r.opciones = [{"intencion": "contactar_ejecutivo",
                           "texto": "Hablar con un ejecutivo" if idioma == "es" else "Talk to an executive"}]
        else:
            est = None
            if orden.estatus != "entregada":
                est = estado.entrega.estimar(orden.datos_modelo, idioma, orden.fecha_pedido, estado.cfg.fecha_referencia)
                marcar("eta", f"{est.eta_dias} días [{est.eta_min_dias}-{est.eta_max_dias}], "
                              f"p(retraso)={est.prob_retraso:.2f}")
            critico = bool(orden.pedido_urgente and est and est.riesgo_alto)  # R03: confirmación humana
            if d.intencion == "fecha_entrega":
                r.respuesta = respuesta_fecha(orden, idioma, est, critico)
            else:
                r.respuesta = respuesta_estatus(orden, idioma, est)
            r.requiere_ejecutivo = critico
            r.folio = folio if critico else None
            r.datos = {"orden": orden.publica(), "estimacion": est.a_dict() if est else None, "critico": critico}

    marcar("respuesta", r.accion)
    r.traza = traza
    registrar({"cliente": cliente, "intencion": r.intencion, "confianza": r.confianza, "accion": r.accion,
               "oc": r.oc, "idioma": idioma, "largo_mensaje": len(entrada.mensaje), "folio": r.folio,
               "ms_total": round(sum(p.ms for p in traza), 2),
               "router": estado.router.info().get("version") if estado.router.cargado else "no_disponible"})
    return r


@app.exception_handler(Exception)
async def error_no_controlado(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno. Se registró en la bitácora."})


# ---------------------------------------------------------------------------
# Frontend estático (solo en la imagen "todo en uno" que se despliega en la nube).
# En local, docker compose sirve el frontend con nginx en su propio contenedor.
# ---------------------------------------------------------------------------
_frontend = cfg_mod.cargar().frontend
if (_frontend / "index.html").exists():
    app.mount("/static", StaticFiles(directory=_frontend / "static"), name="frontend")

    @app.get("/", include_in_schema=False)
    def inicio():
        return FileResponse(_frontend / "index.html")
