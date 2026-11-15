"""Métodos para comparar la distribución de referencia (baseline) con la de producción.

Cada método sirve para un tipo de variable (Tema 19):
    PSI   numéricas o categóricas; umbrales interpretables (< 0.10 estable,
          0.10–0.25 cambio moderado, > 0.25 cambio importante).
    KS    numéricas continuas: máxima distancia entre las distribuciones acumuladas.
    JSD   categóricas o discretas: medida simétrica y acotada entre 0 y 1.
    χ²    categóricas: ¿la diferencia de frecuencias es estadísticamente significativa?

Nota de diseño (lección de la Fase 2): una variable que solo cambia una vez por semana
y por planta (utilizacion_planta) tiene muy pocos valores distintos en una ventana, y
el PSI con cuantiles "detecta" cambios aunque no los haya. Para esa variable se usa la
proporción de órdenes fuera del rango visto en el entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4


def psi(referencia, actual, bins: int = 10) -> float:
    """Population Stability Index con cortes en los cuantiles de la referencia."""
    r = pd.to_numeric(pd.Series(referencia), errors="coerce").dropna().to_numpy(float)
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy(float)
    if len(r) == 0 or len(a) == 0:
        return float("nan")
    cortes = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf
    pr = np.histogram(r, cortes)[0] / len(r) + EPS
    pa = np.histogram(a, cortes)[0] / len(a) + EPS
    return float(np.sum((pa - pr) * np.log(pa / pr)))


def psi_categorico(referencia, actual) -> float:
    categorias = sorted(set(pd.Series(referencia).dropna()) | set(pd.Series(actual).dropna()))
    pr = pd.Series(referencia).value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy() + EPS
    pa = pd.Series(actual).value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy() + EPS
    return float(np.sum((pa - pr) * np.log(pa / pr)))


def ks(referencia, actual) -> tuple[float, float]:
    """Estadístico KS (0 = iguales, 1 = totalmente distintas) y su valor p."""
    r = pd.to_numeric(pd.Series(referencia), errors="coerce").dropna()
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    if len(r) == 0 or len(a) == 0:
        return float("nan"), float("nan")
    resultado = stats.ks_2samp(r, a)
    return float(resultado.statistic), float(resultado.pvalue)


def jsd(referencia, actual) -> float:
    """Divergencia de Jensen-Shannon en base 2 (0 = iguales, 1 = sin traslape)."""
    categorias = sorted(set(pd.Series(referencia).dropna()) | set(pd.Series(actual).dropna()))
    p = pd.Series(referencia).value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    q = pd.Series(actual).value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    m = (p + q) / 2

    def kl(x, y):
        mascara = x > 0
        return float(np.sum(x[mascara] * np.log2(x[mascara] / y[mascara])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def chi2(referencia, actual) -> tuple[float, float]:
    """Prueba χ² de bondad de ajuste: frecuencias actuales vs proporciones de la referencia."""
    categorias = sorted(set(pd.Series(referencia).dropna()) | set(pd.Series(actual).dropna()))
    esperadas_p = pd.Series(referencia).value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy() + EPS
    esperadas_p /= esperadas_p.sum()
    observadas = pd.Series(actual).value_counts().reindex(categorias, fill_value=0).to_numpy()
    estadistico, p = stats.chisquare(observadas, esperadas_p * observadas.sum())
    return float(estadistico), float(p)


def fuera_de_rango(df: pd.DataFrame, rangos: dict[str, list[float]], columnas: list[str] | None = None) -> float:
    """Proporción de filas con al menos una variable fuera del rango [mín, máx] del entrenamiento."""
    columnas = columnas or list(rangos)
    mascara = np.zeros(len(df), dtype=bool)
    for c in columnas:
        lo, hi = rangos[c]
        v = pd.to_numeric(df[c], errors="coerce")
        mascara |= ((v < lo) | (v > hi)).fillna(False).to_numpy()
    return float(mascara.mean()) if len(df) else 0.0


def nivel_psi(valor: float) -> str:
    if np.isnan(valor):
        return "sin datos"
    return "estable" if valor < 0.10 else "moderado" if valor < 0.25 else "importante"
