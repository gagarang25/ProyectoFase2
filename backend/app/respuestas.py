"""Plantillas de respuesta en español e inglés.

Decisión de diseño (riesgo R05, alucinación): el chatbot no redacta fechas ni estatus
por su cuenta. Cada dato que aparece en una respuesta sale de SAP (simulado) o del
modelo de ETA, y se inserta en una plantilla fija. La Fase 1 dejó el modelo de lenguaje
generativo como opcional; en el prototipo se usan plantillas, que son la "respuesta de
respaldo" del diseño y no dependen de un proveedor externo (riesgo R12).
"""

from __future__ import annotations

from datetime import date

from .modelos_entrega import EstimacionEntrega
from .sap_simulado import ETAPAS, Orden


def f(fecha: date | None) -> str:
    return fecha.strftime("%d/%m/%Y") if fecha else "—"


TEXTOS = {
    "pedir_oc": {
        "es": "Con gusto lo reviso. ¿Me compartes el número de orden de compra? Son 10 dígitos y empiezan con 45, por ejemplo 4500127273.",
        "en": "Happy to check. Could you share the purchase order number? It has 10 digits and starts with 45, e.g. 4500127273.",
    },
    "oc_no_encontrada": {
        "es": "No encontré la orden {oc} en tu cuenta. Revisa el número o, si prefieres, te comunico con tu ejecutivo de servicio.",
        "en": "I couldn't find order {oc} in your account. Please check the number or I can connect you with your service executive.",
    },
    "aclarar": {
        "es": "¿Quieres saber en qué etapa va la orden o la fecha estimada de entrega?",
        "en": "Would you like the order status or the estimated delivery date?",
    },
    "escalar_queja": {
        "es": "Lamento lo ocurrido. Ya pasé tu caso a un ejecutivo de servicio con el detalle de la conversación; te contactará en menos de 4 horas hábiles. Folio: {folio}.",
        "en": "I'm sorry about this. Your case is now with a service executive, who will contact you within 4 business hours. Ticket: {folio}.",
    },
    "escalar_contacto": {
        "es": "Claro. Te comunico con tu ejecutivo de servicio; ya tiene el contexto de la conversación. Folio: {folio}.",
        "en": "Sure. I'm connecting you with your service executive, who already has the context. Ticket: {folio}.",
    },
    "escalar_duda": {
        "es": "No estoy seguro de haber entendido tu mensaje. Para no darte información equivocada, lo paso a un ejecutivo de servicio. Folio: {folio}.",
        "en": "I'm not sure I understood your message. To avoid giving you wrong information, I'm passing it to a service executive. Ticket: {folio}.",
    },
    "canalizar": {
        "es": "Ese tema está fuera de lo que puedo resolver (solo consulto estatus y fechas de entrega). Te recomiendo escribir a cuentasporcobrar@cab.example para facturación y pagos, o puedo comunicarte con tu ejecutivo.",
        "en": "That's outside what I can handle (I only check order status and delivery dates). For invoices and payments please write to cuentasporcobrar@cab.example, or I can connect you with your executive.",
    },
    "router_no_disponible": {
        "es": "En este momento no puedo procesar mensajes automáticamente. Te comunico con un ejecutivo de servicio. Folio: {folio}.",
        "en": "I can't process messages automatically right now. I'm connecting you with a service executive. Ticket: {folio}.",
    },
}


def texto(clave: str, idioma: str, **valores) -> str:
    return TEXTOS[clave][idioma].format(**valores)


def respuesta_estatus(o: Orden, idioma: str, est: EstimacionEntrega | None) -> str:
    etapa = ETAPAS[o.estatus][idioma]
    if idioma == "es":
        if o.estatus == "entregada":
            return (f"La orden {o.oc} ({o.cantidad_piezas:,} piezas de {o.familia_producto.lower()}) se entregó el "
                    f"{f(o.fecha_entrega)}. La fecha comprometida era el {f(o.fecha_compromiso)}.")
        base = (f"La orden {o.oc} ({o.cantidad_piezas:,} piezas de {o.familia_producto.lower()}) está {etapa} "
                f"desde la planta {o.planta}, con el transportista {o.transportista}.")
        if est:
            base += f" La fecha estimada de entrega es el {f(est.fecha_estimada)}."
        return base
    if o.estatus == "entregada":
        return (f"Order {o.oc} ({o.cantidad_piezas:,} {o.familia_producto.lower()} parts) was delivered on "
                f"{f(o.fecha_entrega)}. The committed date was {f(o.fecha_compromiso)}.")
    base = (f"Order {o.oc} ({o.cantidad_piezas:,} {o.familia_producto.lower()} parts) is {etapa} from the "
            f"{o.planta} plant, carrier {o.transportista}.")
    if est:
        base += f" The estimated delivery date is {f(est.fecha_estimada)}."
    return base


def respuesta_fecha(o: Orden, idioma: str, est: EstimacionEntrega | None, critico: bool) -> str:
    if o.estatus == "entregada" or est is None:
        return respuesta_estatus(o, idioma, None)
    es = idioma == "es"
    motivos = "; ".join(est.motivos[:3])
    if es:
        partes = [f"Estimo que la orden {o.oc} llegue el {f(est.fecha_estimada)}, "
                  f"probablemente entre el {f(est.fecha_min)} y el {f(est.fecha_max)}.",
                  f"La fecha comprometida en SAP es el {f(o.fecha_compromiso)}.",
                  f"Lo que más influye: {motivos}."]
        if est.vencida:
            partes.insert(1, "La orden ya superó la fecha que el modelo esperaba, por eso viene con retraso.")
        if est.riesgo_alto:
            partes.append(f"El modelo de riesgo marca probabilidad de retraso de {est.prob_retraso:.0%}.")
        if critico:
            partes.append("Por ser un pedido urgente con riesgo de retraso, un ejecutivo confirmará la fecha en menos de 4 horas.")
        partes.append("Es una estimación, no un compromiso de entrega.")
        return " ".join(partes)
    partes = [f"I estimate order {o.oc} will arrive on {f(est.fecha_estimada)}, "
              f"most likely between {f(est.fecha_min)} and {f(est.fecha_max)}.",
              f"The committed date in SAP is {f(o.fecha_compromiso)}.",
              f"Main factors: {motivos}."]
    if est.vencida:
        partes.insert(1, "The order is already past the date the model expected, so it is running late.")
    if est.riesgo_alto:
        partes.append(f"The delay-risk model shows a {est.prob_retraso:.0%} probability of delay.")
    if critico:
        partes.append("Because this is an urgent order at risk, an executive will confirm the date within 4 hours.")
    partes.append("This is an estimate, not a delivery commitment.")
    return " ".join(partes)
