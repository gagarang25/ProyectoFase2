"""Esquemas de entrada y salida de la API (validación automática con Pydantic).

Si una solicitud no cumple el esquema (mensaje vacío, texto demasiado largo, valores
fuera de rango), FastAPI responde 422 con el detalle del error antes de llegar al modelo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intencion = Literal["estatus_orden", "fecha_entrega", "queja", "contactar_ejecutivo", "otro"]


class MensajeEntrada(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=1000, examples=["¿Cuándo llega la OC 4500127273?"])

    @field_validator("mensaje")
    @classmethod
    def no_vacio(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("el mensaje no puede estar vacío")
        return v.strip()


class ChatEntrada(MensajeEntrada):
    oc_contexto: str | None = Field(None, description="Última OC mencionada en la conversación (la envía el frontend).")
    intencion_elegida: Intencion | None = Field(
        None, description="Respuesta del cliente a una pregunta de aclaración (botones del chat).")


class Paso(BaseModel):
    paso: str
    detalle: str
    ms: float


class RespuestaIntencion(BaseModel):
    intencion: str
    confianza: float
    accion: Literal["automatica", "aclarar", "escalar", "canalizar"]
    segunda_intencion: str | None = None
    probabilidades: dict[str, float] = {}
    idioma: str
    oc_detectada: str | None = None


class RespuestaChat(BaseModel):
    respuesta: str
    accion: str
    intencion: str
    confianza: float
    idioma: str
    oc: str | None = None
    requiere_ejecutivo: bool = False
    opciones: list[dict[str, str]] = []
    datos: dict | None = None
    folio: str | None = None
    traza: list[Paso] = []


class DatosOrden(BaseModel):
    """Variables de una orden para estimar ETA y riesgo sin consultar SAP (pruebas y simulaciones).
    Los límites son de validez física; los valores raros pero posibles se aceptan y se
    reportan como advertencia de extrapolación."""

    tipo_cliente: str = "armadora"
    planta: str = "Queretaro"
    region_destino: str = "Bajio"
    transportista: str = "TR-A"
    modalidad: str = "Dedicado"
    familia_producto: str = "Estampados"
    cruce_frontera: int = Field(0, ge=0, le=1)
    distancia_km: float = Field(200, ge=0, le=5000)
    cantidad_piezas: int = Field(500, ge=1, le=1_000_000)
    peso_kg: float = Field(800, ge=0, le=100_000)
    inventario_disponible: int = Field(1, ge=0, le=1)
    utilizacion_planta: float = Field(0.8, ge=0, le=1.5)
    pedido_urgente: int = Field(0, ge=0, le=1)
    cambios_pedido: int = Field(0, ge=0, le=20)
    produccion_std_dias: int = Field(0, ge=0, le=60)
    transito_std_dias: int = Field(1, ge=0, le=30)
    dias_comprometidos: int = Field(4, ge=0, le=120)
    holgura_dias: int | None = Field(None, ge=-30, le=120,
                                     description="Si se omite, se calcula como comprometidos − producción − tránsito.")
    mes: int = Field(10, ge=1, le=12)
    dia_semana: int = Field(2, ge=0, le=6)
    envios_previos_ruta: int = Field(50, ge=0, le=100_000)
    tasa_retraso_transportista_90d: float | None = Field(None, ge=0, le=1)

    def a_dict(self) -> dict:
        d = self.model_dump()
        if d["holgura_dias"] is None:
            d["holgura_dias"] = d["dias_comprometidos"] - d["produccion_std_dias"] - d["transito_std_dias"]
        return d
