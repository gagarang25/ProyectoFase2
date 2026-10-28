"""Extracción de datos del mensaje: número de orden de compra (OC) e idioma.

Se hace con reglas y no con el modelo, porque un número de OC tiene un formato fijo
(45 + 8 dígitos, como en SAP) y una regla es exacta y fácil de auditar.
"""

from __future__ import annotations

import re
import unicodedata

# 10 dígitos que empiezan con 45, con o sin prefijo (OC, PO, #, OC-...), sin estar pegados a otros dígitos.
_PATRON_OC = re.compile(r"(?<!\d)(45\d{8})(?!\d)")

_PALABRAS_ES = {"el", "la", "los", "las", "de", "que", "mi", "mis", "cuándo", "cuando", "dónde", "donde", "cómo",
                "como", "por", "favor", "pedido", "orden", "llega", "quiero", "hablar", "con", "una", "ya", "está",
                "esta", "para", "qué", "hola", "gracias", "necesito", "cuál", "cual", "va", "piezas", "entrega"}
_PALABRAS_EN = {"the", "my", "is", "when", "where", "what", "order", "will", "arrive", "please", "want", "talk",
                "to", "with", "can", "you", "i", "status", "delivery", "hi", "thanks", "need", "shipment", "how"}


def extraer_oc(texto: str) -> str | None:
    """Regresa el primer número de OC del mensaje (solo dígitos) o None."""
    encontrados = _PATRON_OC.findall(texto or "")
    return encontrados[0] if encontrados else None


def normalizar_oc(oc: str) -> str | None:
    """Acepta 'OC-4500123456', 'PO 4500123456', '#4500123456'… y regresa '4500123456'."""
    return extraer_oc(re.sub(r"[\s\-#]", " ", oc or ""))


def detectar_idioma(texto: str) -> str:
    """Español o inglés con un conteo simple de palabras frecuentes. Ante empate: español."""
    t = unicodedata.normalize("NFC", (texto or "").lower())
    if any(c in t for c in "¿¡ñáéíóú"):
        return "es"
    palabras = re.findall(r"[a-záéíóúñ]+", t)
    es = sum(p in _PALABRAS_ES for p in palabras)
    en = sum(p in _PALABRAS_EN for p in palabras)
    return "en" if en > es else "es"
