"""Router de intención: mDeBERTa-v3 ajustado (Actividad 4 + data-v3.0.0 de la Fase 2).

El modelo se descarga de Hugging Face Hub al construir la imagen de Docker (etiqueta
model-v1.0.0) y se carga una sola vez al arrancar la API. Si el modelo no está
disponible, la API arranca igual y el router escala todo a un ejecutivo: es preferible
un chatbot que no responde solo a uno que adivina (falla segura).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .politica import CLASES, UMBRAL_POR_DEFECTO, Decision, decidir, softmax

log = logging.getLogger("cab.router")


class RouterNoDisponible:
    """Sustituto cuando el modelo no cargó: nunca responde en automático."""

    cargado = False

    def __init__(self, motivo: str):
        self.motivo = motivo

    def info(self) -> dict:
        return {"cargado": False, "motivo": self.motivo}

    def predecir(self, texto: str) -> Decision:
        return Decision(intencion="desconocida", confianza=0.0, accion="escalar")


class RouterIntencion:
    cargado = True

    def __init__(self, carpeta: Path, umbral: float | None = None, hilos: int = 2):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        inicio = time.perf_counter()
        torch.set_num_threads(max(1, hilos))
        self._torch = torch
        ruta_cfg = carpeta / "config_cab.json"
        self.cfg = json.loads(ruta_cfg.read_text(encoding="utf-8")) if ruta_cfg.exists() else {}
        self.tokenizer = AutoTokenizer.from_pretrained(carpeta)
        # Los pesos se publican en 16 bits; en CPU se usan en 32 bits (igual que en la evaluación de Colab).
        self.modelo = AutoModelForSequenceClassification.from_pretrained(carpeta, dtype=torch.float32).eval()
        etiquetas = [self.modelo.config.id2label[i] for i in range(self.modelo.config.num_labels)]
        if etiquetas != CLASES:
            raise ValueError(f"Las etiquetas del modelo {etiquetas} no coinciden con las de la política {CLASES}")
        self.temperatura = float(self.cfg.get("temperatura", 1.0))
        self.umbral = float(umbral if umbral is not None else self.cfg.get("umbral_confianza", UMBRAL_POR_DEFECTO))
        self.max_len = int(self.cfg.get("max_len", 64))
        self.segundos_carga = round(time.perf_counter() - inicio, 2)
        log.info("Router cargado en %.1f s (umbral %.2f, temperatura %.3f)", self.segundos_carga, self.umbral,
                 self.temperatura)

    def info(self) -> dict:
        return {
            "cargado": True,
            "modelo_base": self.cfg.get("modelo_base"),
            "version": self.cfg.get("version_modelo"),
            "version_datos": self.cfg.get("version_datos"),
            "git_commit_entrenamiento": self.cfg.get("git_commit"),
            "umbral_confianza": self.umbral,
            "temperatura": self.temperatura,
            "segundos_carga": self.segundos_carga,
        }

    def logits(self, texto: str):
        with self._torch.no_grad():
            entrada = self.tokenizer(texto, truncation=True, max_length=self.max_len, return_tensors="pt")
            return self.modelo(**entrada).logits[0].float().numpy()

    def predecir(self, texto: str) -> Decision:
        return decidir(softmax(self.logits(texto), self.temperatura), self.umbral)


def crear_router(carpeta: Path, umbral: float | None, hilos: int):
    if not (carpeta / "config.json").exists():
        log.warning("No se encontró el modelo del router en %s: se escalará todo a un ejecutivo", carpeta)
        return RouterNoDisponible(f"modelo no encontrado en {carpeta}")
    try:
        return RouterIntencion(carpeta, umbral, hilos)
    except Exception as error:  # noqa: BLE001 - cualquier falla de carga → falla segura
        log.exception("No se pudo cargar el router")
        return RouterNoDisponible(f"{type(error).__name__}: {error}")
