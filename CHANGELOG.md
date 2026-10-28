# Registro de cambios

## app-v1.0.0 · Proyecto Fase 2
- API REST (FastAPI) con router de intención, SAP simulado de solo lectura, modelo de ETA y riesgo de retraso.
- Chat web (nginx) que consume la API; `docker compose` con dos contenedores.
- `Dockerfile` único para la nube (API + chat estático); demo en Hugging Face Spaces.
- Pruebas: pytest (unitarias e integración), 35 casos funcionales y extremos contra el contenedor, estrés y drift.

## model-v1.0.0
- Router mDeBERTa-v3 ajustado con data-v3.0.0, calibrado con temperatura y publicado en Hugging Face Hub.
- Modelo de ETA (HistGradientBoostingRegressor, pérdida absoluta) con rango por cuantiles 10 % y 90 %.
- Clasificador de riesgo de retraso de la A6 reentrenado con los mismos hiperparámetros (umbral 0.17).

## data-v3.0.0
- Mensajes de CAB en español e inglés + Bitext (A4) con marcadores reemplazados y reetiquetado del "ETA".
- Prueba externa de 160 mensajes escritos a mano (incluye los 50 de la A5).

## code-v1.0.0
- Código de la API, del frontend, de entrenamiento y de pruebas.
