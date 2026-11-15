"""Versiones del modelo de ETA en el MLflow Model Registry.

El alias @champion indica qué versión sirve la API. Un rollback NO copia archivos:
solo mueve el alias a la versión anterior, por eso tarda segundos (Tema 20).
Cada versión guarda la mediana (modelo registrado) y los dos cuantiles del rango
como artefactos de la misma ejecución, para restaurarlos juntos.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

NOMBRES = ("mediana", "inferior", "superior")


class RegistroETA:
    def __init__(self, nombre: str):
        self.nombre = nombre
        self.cliente = MlflowClient()

    def registrar(self, modelos: dict, descripcion: str, etiquetas: dict, metricas: dict | None = None,
                  ejemplo=None, nombre_run: str | None = None) -> str:
        """Registra una versión nueva y devuelve su número."""
        with mlflow.start_run(run_name=nombre_run or f"registro_{etiquetas.get('version_semantica', 'eta')}",
                              nested=mlflow.active_run() is not None) as run:
            mlflow.set_tags({"tipo_run": "registro_modelo", **etiquetas})
            if metricas:
                mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
            info = mlflow.sklearn.log_model(modelos["mediana"], name="modelo_eta", input_example=ejemplo,
                                            serialization_format="cloudpickle", registered_model_name=self.nombre)
            with tempfile.TemporaryDirectory() as tmp:
                for n in ("inferior", "superior"):
                    joblib.dump(modelos[n], Path(tmp) / f"eta_rango_{n}.joblib")
                (Path(tmp) / "calibracion.json").write_text(json.dumps({"desfase_dias": modelos.get("desfase", 0)}))
                mlflow.log_artifacts(tmp, artifact_path="rango")
        version = str(info.registered_model_version)
        self.cliente.update_model_version(self.nombre, version, description=descripcion)
        for k, v in etiquetas.items():
            self.cliente.set_model_version_tag(self.nombre, version, k, str(v))
        self.cliente.set_model_version_tag(self.nombre, version, "run_id", run.info.run_id)
        return version

    def alias(self, alias: str) -> str | None:
        try:
            return self.cliente.get_model_version_by_alias(self.nombre, alias).version
        except Exception:  # noqa: BLE001 - el alias todavía no existe
            return None

    def asignar(self, alias: str, version: str) -> None:
        self.cliente.set_registered_model_alias(self.nombre, alias, version)

    def quitar(self, alias: str) -> None:
        try:
            self.cliente.delete_registered_model_alias(self.nombre, alias)
        except Exception:  # noqa: BLE001
            pass

    def etiquetar(self, version: str, clave: str, valor: str) -> None:
        self.cliente.set_model_version_tag(self.nombre, version, clave, valor)

    def cargar(self, ref: str) -> dict:
        """ref = alias ('champion') o número de versión. Devuelve las tres partes del modelo."""
        mv = (self.cliente.get_model_version_by_alias(self.nombre, ref) if not ref.isdigit()
              else self.cliente.get_model_version(self.nombre, ref))
        modelos = {"mediana": mlflow.sklearn.load_model(f"models:/{self.nombre}/{mv.version}"), "version": mv.version}
        run_id = mv.tags.get("run_id") or mv.run_id
        carpeta = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="rango"))
        modelos["inferior"] = joblib.load(carpeta / "eta_rango_inferior.joblib")
        modelos["superior"] = joblib.load(carpeta / "eta_rango_superior.joblib")
        modelos["desfase"] = json.loads((carpeta / "calibracion.json").read_text())["desfase_dias"]
        return modelos
