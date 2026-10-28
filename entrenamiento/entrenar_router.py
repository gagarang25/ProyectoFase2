"""
entrenar_router.py
------------------
Ajusta (fine-tuning) mDeBERTa-v3 como router de intención del chatbot de CAB con
data-v3.0.0, calibra su confianza, elige el umbral de decisión y lo evalúa con la
prueba externa escrita a mano. Lo ejecuta el cuaderno router_mdeberta_colab.ipynb
(GPU de Colab), pero también corre en CPU con un modelo pequeño para pruebas.

Configuración base: la misma de la Actividad 4 (tasa de aprendizaje 2e-5, lotes de
8, 3 épocas, weight decay 0.01, conversión a 32 bits antes de entrenar). Cambio
justificado: label smoothing de 0.1, porque en la A5 el router se equivocaba con
confianza de 0.97; suavizar las etiquetas evita que el modelo aprenda a dar
probabilidades de 1.0 a todo.

Uso:
    python entrenamiento/entrenar_router.py --salida salida_router
Salida (carpeta --salida):
    modelo/               modelo ajustado + tokenizador + config_cab.json
    resultados/           métricas, predicciones, historial y figuras
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, ConfusionMatrixDisplay)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "backend"))
from app.politica import CLASES, UMBRAL_POR_DEFECTO, decidir, es_resultado_seguro, softmax  # noqa: E402

DIR_DATOS = RAIZ / "datos" / "mensajes_cab"
MODELO_BASE = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
PRECISION_OBJETIVO = 0.98  # KPI del riesgo R05: exactitud del estatus >= 98 %
label2id = {c: i for i, c in enumerate(CLASES)}
id2label = {i: c for c, i in label2id.items()}
COLOR = "#0e5c4a"  # verde de los documentos de la Fase 1


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo-base", default=MODELO_BASE)
    p.add_argument("--salida", default="salida_router")
    p.add_argument("--epocas", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--guardar-fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="Guarda los pesos en 16 bits (la mitad de tamaño); la API los vuelve a 32 bits al cargar.")
    return p.parse_args()


def commit_git() -> dict:
    """Commit del repositorio con el que se entrena (retroalimentación de la A5)."""
    def correr(*cmd):
        return subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    try:
        commit = correr("git", "rev-parse", "HEAD")
        if not commit:
            return {"commit": "sin_git", "limpio": False}
        return {"commit": commit, "limpio": correr("git", "status", "--porcelain") == ""}
    except FileNotFoundError:
        return {"commit": "sin_git", "limpio": False}


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(args, tokenizer, entrena: pd.DataFrame, valida: pd.DataFrame, dir_trabajo: Path):
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.semilla)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.modelo_base, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,  # la capa final pasa de 3 salidas (NLI) a 5 intenciones
    )
    model = model.float()  # mDeBERTa se publica en 16 bits: entrenar en 16 bits da pérdida NaN (lección de la A4)

    def a_dataset(df):
        ds = Dataset.from_pandas(pd.DataFrame({"mensaje": df["mensaje"], "labels": df["intencion"].map(label2id)}),
                                 preserve_index=False)
        return ds.map(lambda lote: tokenizer(lote["mensaje"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["mensaje"])

    def metricas(eval_pred):
        logits, etiquetas = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(etiquetas, pred),
                "f1_macro": f1_score(etiquetas, pred, average="macro")}

    config = TrainingArguments(
        output_dir=str(dir_trabajo / "puntos_control"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.lote,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epocas,
        weight_decay=0.01,
        label_smoothing_factor=args.label_smoothing,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        logging_steps=50,
        seed=args.semilla,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=config, train_dataset=a_dataset(entrena), eval_dataset=a_dataset(valida),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metricas,
    )
    inicio = time.time()
    trainer.train()
    segundos = time.time() - inicio
    historial = pd.DataFrame([r for r in trainer.state.log_history if "eval_f1_macro" in r])
    return trainer.model, segundos, historial


# ---------------------------------------------------------------------------
# Inferencia, calibración y umbral
# ---------------------------------------------------------------------------
@torch.no_grad()
def logits_de(model, tokenizer, textos, max_len, lote=32, dispositivo="cpu"):
    model.eval().to(dispositivo)
    salida = []
    for i in range(0, len(textos), lote):
        entrada = tokenizer(list(textos[i:i + lote]), truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt").to(dispositivo)
        salida.append(model(**entrada).logits.float().cpu().numpy())
    return np.concatenate(salida)


def ajustar_temperatura(logits: np.ndarray, etiquetas: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): un solo número T que divide los logits
    para que la confianza se parezca a la proporción real de aciertos."""
    def nll(log_t):
        p = softmax(logits, np.exp(log_t))
        return -np.mean(np.log(p[np.arange(len(etiquetas)), etiquetas] + 1e-12))
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(r.x))


def ece(probas: np.ndarray, etiquetas: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado: diferencia promedio entre confianza y aciertos."""
    conf, pred = probas.max(1), probas.argmax(1)
    bordes = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs((pred[m] == etiquetas[m]).mean() - conf[m].mean())
    return float(total)


def curva_umbral(probas: np.ndarray, aceptables: list[set], umbrales=np.round(np.arange(0.50, 0.995, 0.01), 2)):
    filas = []
    for u in umbrales:
        dec = [decidir(p, u) for p in probas]
        auto = [d for d in dec]
        n_auto = sum(d.accion == "automatica" for d in auto)
        correctas = sum(d.accion == "automatica" and d.intencion in a for d, a in zip(dec, aceptables))
        filas.append({"umbral": float(u), "cobertura_automatica": n_auto / len(dec),
                      "precision_automatica": correctas / n_auto if n_auto else np.nan,
                      "seguros": np.mean([es_resultado_seguro(d, a) for d, a in zip(dec, aceptables)])})
    return pd.DataFrame(filas)


def elegir_umbral(curva_val: pd.DataFrame) -> tuple[float, float]:
    ok = curva_val[curva_val["precision_automatica"] >= PRECISION_OBJETIVO]
    minimo_val = float(ok["umbral"].min()) if len(ok) else 0.99
    return max(UMBRAL_POR_DEFECTO, minimo_val), minimo_val


def medir_latencia(model, tokenizer, textos, max_len, repeticiones=3):
    model.eval().to("cpu")
    tiempos = []
    with torch.no_grad():
        model(**tokenizer(textos[0], return_tensors="pt"))  # calentamiento
        for _ in range(repeticiones):
            for t in textos:
                inicio = time.perf_counter()
                model(**tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt"))
                tiempos.append((time.perf_counter() - inicio) * 1000)
    t = np.array(tiempos)
    return {"latencia_cpu_mediana_ms": float(np.median(t)), "latencia_cpu_p95_ms": float(np.percentile(t, 95)),
            "mediciones": int(len(t)), "hilos_cpu": torch.get_num_threads()}


def evaluar(probas: np.ndarray, df: pd.DataFrame, umbral: float) -> tuple[pd.DataFrame, dict]:
    decisiones = [decidir(p, umbral) for p in probas]
    salida = df.copy()
    salida["prediccion"] = [d.intencion for d in decisiones]
    salida["confianza"] = [d.confianza for d in decisiones]
    salida["accion"] = [d.accion for d in decisiones]
    salida["segunda"] = [d.segunda_intencion for d in decisiones]
    aceptables = [set(str(a).split("|")) if isinstance(a, str) else {i} for a, i in
                  zip(salida.get("aceptables", salida["intencion"]), salida["intencion"])]
    salida["acierto"] = [p == i for p, i in zip(salida["prediccion"], salida["intencion"])]
    salida["seguro"] = [es_resultado_seguro(d, a) for d, a in zip(decisiones, aceptables)]
    auto = salida["accion"] == "automatica"
    # La exactitud se mide solo con mensajes que tienen una intención clara (claro o spanglish);
    # los ambiguos y fuera de dominio se evalúan con "resultado seguro".
    tipos = salida["tipo"] if "tipo" in salida else pd.Series("claro", index=salida.index)
    claros = salida[salida["intencion"].isin(CLASES) & tipos.isin(["claro", "spanglish"])]
    res = {
        "mensajes": int(len(salida)),
        "accuracy": float(claros["acierto"].mean()),
        "f1_macro": float(f1_score(claros["intencion"], claros["prediccion"], labels=CLASES, average="macro",
                                   zero_division=0)),
        "cobertura_automatica": float(auto.mean()),
        "precision_automatica": float(salida.loc[auto, "seguro"].mean()) if auto.any() else None,
        "resultados_seguros": float(salida["seguro"].mean()),
        "respuestas_automaticas_erroneas": int((auto & ~salida["seguro"]).sum()),
        "acciones": salida["accion"].value_counts().to_dict(),
    }
    if "idioma" in salida:
        res["accuracy_por_idioma"] = claros.groupby("idioma")["acierto"].mean().round(4).to_dict()
    if "tipo" in salida:
        res["seguros_por_tipo"] = salida.groupby("tipo")["seguro"].mean().round(4).to_dict()
    return salida, res


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_matriz(df, titulo, ruta):
    claros = df[df["intencion"].isin(CLASES)]
    m = confusion_matrix(claros["intencion"], claros["prediccion"], labels=CLASES)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ConfusionMatrixDisplay(m, display_labels=["estatus", "fecha", "queja", "ejecutivo", "otro"]).plot(
        ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title(titulo)
    ax.set_xlabel("Intención predicha")
    ax.set_ylabel("Intención real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def figura_calibracion(conf_antes, conf_despues, aciertos, ruta, bins=10):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    bordes = np.linspace(0, 1, bins + 1)
    for conf, etiqueta, color in ((conf_antes, "Sin calibrar", "#e07b39"), (conf_despues, "Con temperatura", COLOR)):
        xs, ys = [], []
        for a, b in zip(bordes[:-1], bordes[1:]):
            m = (conf > a) & (conf <= b)
            if m.sum() >= 3:
                xs.append(conf[m].mean())
                ys.append(aciertos[m].mean())
        ax.plot(xs, ys, marker="o", label=etiqueta, color=color)
    ax.plot([0, 1], [0, 1], ls="--", color="#888", label="Calibración perfecta")
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Proporción de aciertos")
    ax.set_title("Calibración en la prueba externa")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def figura_umbral(curva_ext, umbral, ruta):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(curva_ext["umbral"], curva_ext["cobertura_automatica"], label="Respondidas en automático", color=COLOR)
    ax.plot(curva_ext["umbral"], curva_ext["precision_automatica"], label="Precisión de las automáticas",
            color="#e07b39")
    ax.axvline(umbral, ls="--", color="#555")
    ax.text(umbral + 0.005, 0.05, f"umbral {umbral:.2f}", color="#555")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cobertura frente a precisión (prueba externa)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = argumentos()
    salida = Path(args.salida)
    dir_modelo, dir_res = salida / "modelo", salida / "resultados"
    dir_modelo.mkdir(parents=True, exist_ok=True)
    (dir_res / "figuras").mkdir(parents=True, exist_ok=True)

    entrena = pd.read_csv(DIR_DATOS / "entrenamiento.csv")
    valida = pd.read_csv(DIR_DATOS / "validacion.csv")
    prueba = pd.read_csv(DIR_DATOS / "prueba.csv")
    externa = pd.read_csv(DIR_DATOS / "prueba_externa_v3.csv")
    manifiesto = json.loads((DIR_DATOS / "versiones.json").read_text(encoding="utf-8"))
    git = commit_git()
    print(f"Datos {manifiesto['version']} | entrenamiento {len(entrena)}, validación {len(valida)}, "
          f"prueba {len(prueba)}, externa {len(externa)} | commit {git['commit'][:7]} (limpio={git['limpio']})")

    tokenizer = AutoTokenizer.from_pretrained(args.modelo_base)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Fine-tuning
    model, segundos, historial = entrenar(args, tokenizer, entrena, valida, salida)
    historial.to_csv(dir_res / "historial_entrenamiento.csv", index=False)
    print(f"Entrenamiento terminado en {segundos:.0f} s")

    # 2. Calibración en validación (nunca con la prueba)
    y_val = valida["intencion"].map(label2id).to_numpy()
    logits_val = logits_de(model, tokenizer, valida["mensaje"].tolist(), args.max_len, dispositivo=gpu)
    temperatura = ajustar_temperatura(logits_val, y_val)
    probas_val = softmax(logits_val, temperatura)
    curva_val = curva_umbral(probas_val, [{c} for c in valida["intencion"]])
    umbral, umbral_min_val = elegir_umbral(curva_val)
    print(f"Temperatura {temperatura:.3f} | umbral mínimo en validación {umbral_min_val:.2f} | umbral final {umbral:.2f}")

    # 3. Guardar el modelo con su configuración de decisión
    if args.guardar_fp16:
        model = model.half()
    model.save_pretrained(dir_modelo)
    tokenizer.save_pretrained(dir_modelo)
    versiones = {"python": platform.python_version()}
    for lib in ("torch", "transformers", "tokenizers", "sentencepiece", "datasets", "scikit-learn"):
        try:
            versiones[lib] = version(lib)
        except Exception:  # noqa: BLE001
            pass

    # 4. Evaluación final con el modelo YA GUARDADO, recargado en CPU y 32 bits (igual que en la API)
    modelo_cpu = AutoModelForSequenceClassification.from_pretrained(dir_modelo, dtype=torch.float32)
    tok_cpu = AutoTokenizer.from_pretrained(dir_modelo)
    resultados = {}
    for nombre, df in (("prueba", prueba), ("externa", externa)):
        logits = logits_de(modelo_cpu, tok_cpu, df["mensaje"].tolist(), args.max_len)
        probas = softmax(logits, temperatura)
        pred_df, res = evaluar(probas, df, umbral)
        pred_df["logits"] = [json.dumps(np.round(f, 5).tolist()) for f in logits]
        pred_df.to_csv(dir_res / f"predicciones_{nombre}.csv", index=False)
        figura_matriz(pred_df, f"Router mDeBERTa-v3 · {nombre}", dir_res / "figuras" / f"matriz_{nombre}.png")
        resultados[nombre] = res
        if nombre == "externa":
            claros = pred_df[pred_df["intencion"].isin(CLASES) & pred_df["tipo"].isin(["claro", "spanglish"])]
            y = claros["intencion"].map(label2id).to_numpy()
            lg = logits[claros.index]
            p_antes, p_despues = softmax(lg, 1.0), softmax(lg, temperatura)
            aciertos = (p_despues.argmax(1) == y).astype(float)
            res["ece_sin_calibrar"] = ece(p_antes, y)
            res["ece_calibrado"] = ece(p_despues, y)
            figura_calibracion(p_antes.max(1), p_despues.max(1), aciertos, dir_res / "figuras" / "calibracion_externa.png")
            aceptables = [set(str(a).split("|")) for a in pred_df["aceptables"].fillna("")]
            curva_ext = curva_umbral(probas, aceptables)
            curva_ext.to_csv(dir_res / "curva_umbral_externa.csv", index=False)
            figura_umbral(curva_ext, umbral, dir_res / "figuras" / "umbral_externa.png")
            reporte = classification_report(claros["intencion"], claros["prediccion"], labels=CLASES,
                                            output_dict=True, zero_division=0)
            pd.DataFrame(reporte).T.to_csv(dir_res / "reporte_por_clase_externa.csv")
        print(f"{nombre:8s} accuracy={res['accuracy']:.3f}  f1_macro={res['f1_macro']:.3f}  "
              f"cobertura={res['cobertura_automatica']:.3f}  automáticas_erróneas={res['respuestas_automaticas_erroneas']}")

    latencia = medir_latencia(modelo_cpu, tok_cpu, externa["mensaje"].tolist()[:50], args.max_len)
    print(f"Latencia en CPU: mediana {latencia['latencia_cpu_mediana_ms']:.1f} ms, "
          f"p95 {latencia['latencia_cpu_p95_ms']:.1f} ms")

    config_cab = {
        "nombre": "router_intencion_cab",
        "version_modelo": "model-v1.0.0",
        "modelo_base": args.modelo_base,
        "clases": CLASES,
        "temperatura": round(temperatura, 4),
        "umbral_confianza": umbral,
        "umbral_minimo_validacion": umbral_min_val,
        "precision_objetivo": PRECISION_OBJETIVO,
        "max_len": args.max_len,
        "version_datos": manifiesto["version"],
        "sha256_datos": {k: v["sha256"] for k, v in manifiesto["archivos"].items()},
        "git_commit": git["commit"],
        "git_limpio": git["limpio"],
        "hiperparametros": {"lr": args.lr, "lote": args.lote, "epocas": args.epocas, "weight_decay": 0.01,
                            "label_smoothing": args.label_smoothing, "semilla": args.semilla},
        "pesos": "float16" if args.guardar_fp16 else "float32",
        "versiones": versiones,
        "entrenado_en": torch.cuda.get_device_name(0) if gpu == "cuda" else "CPU",
        "segundos_entrenamiento": round(segundos, 1),
    }
    (dir_modelo / "config_cab.json").write_text(json.dumps(config_cab, ensure_ascii=False, indent=2), encoding="utf-8")
    metricas = {"config": config_cab, "resultados": resultados, "latencia": latencia,
                "historial": historial.round(4).to_dict(orient="records")}
    (dir_res / "metricas_router.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2, default=float),
                                                  encoding="utf-8")
    curva_val.to_csv(dir_res / "curva_umbral_validacion.csv", index=False)
    print(f"Listo: modelo en {dir_modelo} y resultados en {dir_res}")


if __name__ == "__main__":
    main()
