"""SAP simulado de solo lectura.

Sustituye la conexión con SAP S/4HANA del caso de negocio (CAB es ficticia). Usa las
órdenes del piloto A/B de la Actividad 6 (5 clientes, oct-dic 2026) y calcula su
estatus respecto a una fecha de referencia ("hoy").

Reglas de seguridad del diseño (Fase 1):
- R04 (fuga entre clientes): toda consulta se filtra por el cliente autenticado. Si la
  OC existe pero es de otro cliente, la respuesta es idéntica a una OC inexistente, para
  no revelar que existe.
- Solo lectura: no hay ninguna operación que modifique datos.
- El resultado real de las órdenes abiertas (fecha_entrega_real) nunca sale de este
  módulo: en la simulación existe, pero en la vida real todavía no ha ocurrido.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .entidades import normalizar_oc

ETAPAS = {
    "en_produccion": {"es": "en producción", "en": "in production"},
    "lista_para_embarque": {"es": "lista para embarque", "en": "ready to ship"},
    "en_transito": {"es": "en tránsito", "en": "in transit"},
    "entregada": {"es": "entregada", "en": "delivered"},
}


@dataclass
class Orden:
    oc: str
    cliente: str
    estatus: str
    fecha_pedido: date
    fecha_compromiso: date
    fecha_entrega: date | None
    planta: str
    transportista: str
    familia_producto: str
    cantidad_piezas: int
    pedido_urgente: bool
    datos_modelo: dict  # variables de entrada de los modelos de ETA y riesgo

    def publica(self) -> dict:
        """Lo que se puede mostrar al cliente (sin variables internas)."""
        return {
            "oc": self.oc, "cliente": self.cliente, "estatus": self.estatus,
            "fecha_pedido": self.fecha_pedido.isoformat(), "fecha_compromiso": self.fecha_compromiso.isoformat(),
            "fecha_entrega": self.fecha_entrega.isoformat() if self.fecha_entrega else None,
            "planta": self.planta, "transportista": self.transportista, "familia_producto": self.familia_producto,
            "cantidad_piezas": self.cantidad_piezas, "pedido_urgente": self.pedido_urgente,
        }


class SapSimulado:
    def __init__(self, ruta: Path, hoy: date, entradas: list[str]):
        df = pd.read_csv(ruta, parse_dates=["fecha_pedido", "fecha_compromiso", "fecha_entrega_real"])
        self.hoy = hoy
        self.entradas = entradas
        ref = pd.Timestamp(hoy)
        df = df[df["fecha_pedido"] <= ref].copy()  # las órdenes futuras todavía no existen
        df["oc"] = df["id_orden"].str.replace("OC-", "", regex=False)
        self._df = df.set_index("oc", drop=False)

    def __len__(self) -> int:
        return len(self._df)

    def _estatus(self, fila: pd.Series) -> str:
        ref = pd.Timestamp(self.hoy)
        if fila["fecha_entrega_real"] <= ref:
            return "entregada"
        fin_produccion = fila["fecha_pedido"] + pd.Timedelta(days=int(fila["produccion_std_dias"]))
        if ref < fin_produccion:
            return "en_produccion"
        if ref < fin_produccion + pd.Timedelta(days=1):
            return "lista_para_embarque"
        return "en_transito"

    def _a_orden(self, fila: pd.Series) -> Orden:
        estatus = self._estatus(fila)
        return Orden(
            oc=fila["oc"], cliente=fila["cliente"], estatus=estatus,
            fecha_pedido=fila["fecha_pedido"].date(), fecha_compromiso=fila["fecha_compromiso"].date(),
            fecha_entrega=fila["fecha_entrega_real"].date() if estatus == "entregada" else None,
            planta=fila["planta"], transportista=fila["transportista"], familia_producto=fila["familia_producto"],
            cantidad_piezas=int(fila["cantidad_piezas"]), pedido_urgente=bool(fila["pedido_urgente"]),
            datos_modelo={c: (None if pd.isna(fila[c]) else fila[c].item() if hasattr(fila[c], "item") else fila[c])
                          for c in self.entradas},
        )

    def buscar(self, oc: str, cliente: str) -> Orden | None:
        """Busca una OC del cliente autenticado. Otra cuenta o inexistente → None (misma respuesta)."""
        clave = normalizar_oc(oc)
        if not clave or clave not in self._df.index:
            return None
        fila = self._df.loc[clave]
        if fila["cliente"] != cliente:
            return None
        return self._a_orden(fila)

    def ordenes_de(self, cliente: str, solo_abiertas: bool = False, limite: int = 20) -> list[Orden]:
        sub = self._df[self._df["cliente"] == cliente].sort_values("fecha_pedido", ascending=False)
        ordenes = [self._a_orden(f) for _, f in sub.iterrows()]
        if solo_abiertas:
            ordenes = [o for o in ordenes if o.estatus != "entregada"]
        return ordenes[:limite]

    def dias_desde_pedido(self, orden: Orden) -> int:
        return (self.hoy - orden.fecha_pedido).days

    @staticmethod
    def sumar_dias(fecha: date, dias: int) -> date:
        return fecha + timedelta(days=int(dias))
