"""
publicar_modelo_hf.py
---------------------
Publica el router ajustado en Hugging Face Hub con su ficha (model card) y una
etiqueta de versión, para que el Dockerfile descargue siempre la misma versión.

En la A4 el modelo ajustado no se publicó y había que reentrenarlo cada vez; aquí
queda versionado: la etiqueta model-v1.0.0 del Hub corresponde al commit de Git
con el que se entrenó (guardado en config_cab.json).

Uso (requiere haber iniciado sesión con un token de escritura):
    python entrenamiento/publicar_modelo_hf.py --carpeta salida_router --repo usuario/router-intencion-cab-mdeberta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ETIQUETA = "model-v1.0.0"


def ficha(config: dict, metricas: dict, repo: str) -> str:
    ext = metricas["resultados"]["externa"]
    pru = metricas["resultados"]["prueba"]
    lat = metricas["latencia"]
    por_idioma = ", ".join(f"{k}: {v:.3f}" for k, v in ext.get("accuracy_por_idioma", {}).items())
    return f"""---
language: [es, en]
license: mit
base_model: {config['modelo_base']}
pipeline_tag: text-classification
tags: [intent-classification, customer-service, chatbot, cab, tecmilenio]
---

# Router de intención del chatbot de CAB (mDeBERTa-v3 ajustado)

Prototipo académico (Maestría en Inteligencia Artificial, Tecmilenio). Clasifica el mensaje de un
cliente B2B de **Componentes Automotrices del Bajío** (empresa ficticia) en cinco intenciones para
decidir si el chatbot responde solo o pasa la conversación a un ejecutivo.

| Intención | Acción del chatbot |
|---|---|
| estatus_orden | Respuesta automática con el estatus de SAP (simulado) |
| fecha_entrega | Respuesta automática con el modelo de ETA |
| queja | Escalar a un ejecutivo |
| contactar_ejecutivo | Escalar a un ejecutivo |
| otro | Canalizar al área correspondiente |

## Datos y entrenamiento

- Datos: `{config['version_datos']}`: muestra de Bitext (CDLA-Sharing-1.0) + mensajes simulados de CAB en español e inglés.
- Commit de entrenamiento: `{config['git_commit']}` (árbol limpio: {config['git_limpio']}).
- Hiperparámetros: {json.dumps(config['hiperparametros'], ensure_ascii=False)}.
- Calibración: temperatura {config['temperatura']}; umbral de confianza {config['umbral_confianza']} (si la confianza es menor, se pregunta o se escala).

## Resultados

| Conjunto | Accuracy | F1 macro | Respondidas en automático | Automáticas erróneas |
|---|---|---|---|---|
| Prueba (data-v3.0.0, {pru['mensajes']} mensajes) | {pru['accuracy']:.3f} | {pru['f1_macro']:.3f} | {pru['cobertura_automatica']:.3f} | {pru['respuestas_automaticas_erroneas']} |
| Prueba externa escrita a mano ({ext['mensajes']} mensajes) | {ext['accuracy']:.3f} | {ext['f1_macro']:.3f} | {ext['cobertura_automatica']:.3f} | {ext['respuestas_automaticas_erroneas']} |

Accuracy de la prueba externa por idioma: {por_idioma}. Latencia en CPU: mediana {lat['latencia_cpu_mediana_ms']:.1f} ms.

## Limitaciones

Los mensajes de CAB son simulados y la prueba externa la escribió el autor; los resultados deben
confirmarse con conversaciones reales en el piloto. Los pesos se guardan en 16 bits; para usarlos en
CPU se cargan en 32 bits (`dtype=torch.float32`). El archivo `config_cab.json` contiene la temperatura,
el umbral y las versiones de datos y código.

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("{repo}", revision="{ETIQUETA}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}", revision="{ETIQUETA}", dtype=torch.float32)
```
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", default="salida_router")
    p.add_argument("--repo", required=True, help="usuario/nombre-del-modelo")
    p.add_argument("--privado", action="store_true")
    args = p.parse_args()

    carpeta = Path(args.carpeta)
    config = json.loads((carpeta / "modelo" / "config_cab.json").read_text(encoding="utf-8"))
    metricas = json.loads((carpeta / "resultados" / "metricas_router.json").read_text(encoding="utf-8"))
    (carpeta / "modelo" / "README.md").write_text(ficha(config, metricas, args.repo), encoding="utf-8")
    (carpeta / "modelo" / "metricas_router.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.privado, exist_ok=True)
    info = api.upload_folder(folder_path=str(carpeta / "modelo"), repo_id=args.repo, repo_type="model",
                             commit_message=f"{ETIQUETA}: router mDeBERTa-v3 ajustado con {config['version_datos']} "
                                            f"(git {config['git_commit'][:7]})")
    api.create_tag(args.repo, tag=ETIQUETA, revision=info.oid, exist_ok=True,
                   tag_message=f"Router de intención CAB, datos {config['version_datos']}")
    print(f"Publicado: https://huggingface.co/{args.repo}  (etiqueta {ETIQUETA}, commit del Hub {info.oid[:7]})")


if __name__ == "__main__":
    main()
