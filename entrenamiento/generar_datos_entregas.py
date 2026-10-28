"""
Genera el conjunto simulado de entregas de Componentes Automotrices del Bajio (CAB).

CAB es una empresa ficticia y no hay acceso a un SAP real, por eso los datos se
simulan con un proceso fisico sencillo: cada orden tiene un tiempo de produccion
y de transito "reales" que dependen de la planta, el transportista, la aduana,
la carga de la planta, etc. Si la suma supera los dias comprometidos con el
cliente, la orden llega tarde (retraso = 1).

El modelo de aprendizaje automatico NO ve los tiempos reales: solo ve lo que se
conoce al confirmar la orden (columnas de entrada del diccionario de datos).

Salidas (data-v1.0.0):
  datos/entregas_cab.csv     historico oct-2024 a sep-2026 (entrenamiento y prueba)
  datos/piloto_ab_cab.csv    oct-dic 2026, 5 clientes del piloto (prueba A/B simulada)
  datos/versiones.json       huella SHA-256 de cada archivo

Uso:  python entrenamiento/generar_datos_entregas.py

Copia del generador de la Actividad 6. En la Fase 2 se le agrego el parametro
"escenario" para simular cambios en las condiciones logisticas (pruebas de drift).
Con el escenario por defecto produce exactamente data-v1.0.0 (mismo SHA-256).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
VERSION_DATOS = "data-v1.0.0"
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "datos" / "entregas"

INICIO = pd.Timestamp("2024-10-01")
FIN_HISTORICO = pd.Timestamp("2026-09-30")
FIN_PILOTO = pd.Timestamp("2026-12-31")
ORDENES_POR_DIA = 41  # ~1,250 ordenes al mes

# ---------------------------------------------------------------------------
# Catalogos (todos ficticios)
# ---------------------------------------------------------------------------
PLANTAS = {"Queretaro": 0, "San Juan del Rio": 50, "Celaya": 60}  # km extra al destino

REGIONES = {  # distancia base desde Queretaro (km)
    "Bajio": (90, 220),
    "Centro": (220, 380),
    "Noreste": (650, 820),
    "Norte": (850, 1100),
    "EE.UU.": (1150, 2300),
}

# cliente: (tipo, region, peso en el volumen, transportistas que usa)
CLIENTES = {
    "ARM-01": ("armadora", "Bajio", 9, ["TR-A", "TR-B"]),
    "ARM-02": ("armadora", "Bajio", 8, ["TR-A", "TR-C"]),
    "ARM-03": ("armadora", "Centro", 7, ["TR-B", "TR-C"]),
    "ARM-04": ("armadora", "Noreste", 7, ["TR-B", "TR-D"]),
    "ARM-05": ("armadora", "Norte", 6, ["TR-C", "TR-D"]),
    "ARM-06": ("armadora", "Bajio", 6, ["TR-A", "TR-D"]),
    "ARM-07": ("armadora", "EE.UU.", 5, ["TR-E", "TR-B"]),
    "ARM-08": ("armadora", "EE.UU.", 4, ["TR-E"]),
    "TIER-01": ("tier1", "Bajio", 5, ["TR-A", "TR-C"]),
    "TIER-02": ("tier1", "Centro", 5, ["TR-B", "TR-D"]),
    "TIER-03": ("tier1", "Noreste", 4, ["TR-C", "TR-D"]),
    "TIER-04": ("tier1", "Bajio", 4, ["TR-A", "TR-B", "TR-D"]),
    "TIER-05": ("tier1", "Norte", 4, ["TR-D", "TR-C"]),
    "TIER-06": ("tier1", "Centro", 3, ["TR-A", "TR-C"]),
    "TIER-07": ("tier1", "EE.UU.", 3, ["TR-E"]),
    "TIER-08": ("tier1", "Bajio", 3, ["TR-B", "TR-D"]),
    "TIER-09": ("tier1", "Noreste", 2, ["TR-B", "TR-C"]),
    "TIER-10": ("tier1", "Norte", 2, ["TR-C", "TR-D"]),
    "TIER-11": ("tier1", "Centro", 2, ["TR-D"]),
    "TIER-12": ("tier1", "Bajio", 1, ["TR-A", "TR-D"]),
}
CLIENTES_PILOTO = ["ARM-01", "ARM-03", "ARM-04", "ARM-07", "TIER-02"]

# Retraso promedio extra de cada transportista (dias) -> no visible para el modelo
TRANSPORTISTAS = {"TR-A": 0.15, "TR-B": 0.35, "TR-C": 0.55, "TR-D": 0.95, "TR-E": 0.45}

# familia: (dias de produccion estandar, kg por pieza, planta preferida)
FAMILIAS = {
    "Estampados": (3, 1.8, "Queretaro"),
    "Arneses": (5, 0.6, "Celaya"),
    "Plasticos inyectados": (4, 0.4, "San Juan del Rio"),
    "Maquinados": (6, 2.5, "Queretaro"),
    "Ensambles": (8, 3.2, "Celaya"),
}


def _utilizacion_semanal(rng: np.random.Generator, semanas: pd.DatetimeIndex) -> dict:
    """Carga (utilizacion de capacidad) de cada planta por semana."""
    util = {}
    for planta in PLANTAS:
        nivel = 0.80
        for s in semanas:
            # caminata aleatoria con regreso a la media + picos en nov-dic
            pico = 0.07 if s.month in (11, 12) else (0.03 if s.month in (3, 9) else 0.0)
            nivel = 0.75 * nivel + 0.25 * 0.80 + rng.normal(0, 0.035)
            util[(planta, s)] = float(np.clip(nivel + pico, 0.55, 1.08))
    return util


# Condiciones logisticas de data-v1.0.0. Las pruebas de drift cambian estos valores.
ESCENARIO_BASE = {
    "retraso_transportista": {},   # media extra de transito por transportista (reemplaza TRANSPORTISTAS)
    "utilizacion_extra": 0.0,      # puntos de carga de planta que se suman cada semana
    "aduana_media": 0.6,           # dias promedio extra en aduana (ene-nov)
    "aduana_media_diciembre": 1.4,
    "prob_imprevisto": 0.03,       # bloqueos, clima, robos
}


def simular(semilla: int = SEMILLA, escenario: dict | None = None, ocultas: bool = False) -> pd.DataFrame:
    """ocultas=True agrega columnas con los tiempos que el modelo NO ve (solo para analizar errores)."""
    esc = {**ESCENARIO_BASE, **(escenario or {})}
    retraso_tr = {**TRANSPORTISTAS, **esc["retraso_transportista"]}
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(INICIO, FIN_PILOTO, freq="D")
    semanas = pd.date_range(INICIO - pd.Timedelta(days=7), FIN_PILOTO, freq="W-MON")
    util = _utilizacion_semanal(rng, semanas)
    if esc["utilizacion_extra"]:
        util = {k: float(np.clip(v + esc["utilizacion_extra"], 0.55, 1.20)) for k, v in util.items()}

    clientes = list(CLIENTES)
    pesos = np.array([CLIENTES[c][2] for c in clientes], dtype=float)
    pesos /= pesos.sum()
    familias = list(FAMILIAS)

    filas = []
    for dia in dias:
        # menos pedidos en fin de semana y en la ultima quincena de diciembre
        factor = 0.35 if dia.dayofweek >= 5 else 1.15
        if dia.month == 12 and dia.day > 18:
            factor *= 0.6
        n = rng.poisson(ORDENES_POR_DIA * factor)
        for _ in range(n):
            cliente = rng.choice(clientes, p=pesos)
            tipo, region, _, trans_cliente = CLIENTES[cliente]
            familia = familias[rng.integers(len(familias))]
            prod_std, kg_pieza, planta_pref = FAMILIAS[familia]
            planta = planta_pref if rng.random() < 0.75 else rng.choice(list(PLANTAS))
            transportista = rng.choice(trans_cliente)
            lo, hi = REGIONES[region]
            distancia = float(rng.uniform(lo, hi) + PLANTAS[planta] + rng.normal(0, 15))
            cruce = region == "EE.UU."
            piezas = int(np.clip(rng.lognormal(6.2, 0.9), 20, 20000))
            peso = round(piezas * kg_pieza * rng.uniform(0.9, 1.1), 1)
            if peso < 250:
                modalidad = rng.choice(["Paqueteria", "Consolidado"], p=[0.7, 0.3])
            else:
                modalidad = rng.choice(["Dedicado", "Consolidado"], p=[0.5, 0.5])
            inventario = rng.random() < 0.55
            urgente = rng.random() < 0.12
            cambios = int(min(rng.poisson(0.35), 4))
            semana = dia - pd.Timedelta(days=dia.dayofweek)
            u = util.get((planta, semana), 0.8)

            # plan nominal (lo que SAP usa para comprometer la fecha)
            transito_std = int(np.ceil(distancia / 650)) + (1 if cruce else 0)
            produccion_std = 0 if inventario else prod_std
            if urgente:
                colchon = int(rng.choice([-1, 0, 0, 1]))
            else:
                colchon = int(rng.choice([1, 1, 2, 2, 2, 3, 3, 3, 4]))
            dias_comprometidos = max(1, produccion_std + transito_std + colchon)

            # tiempos reales (ocultos para el modelo)
            prod_real = 0.0
            if not inventario:
                saturacion = 1 + 5 * max(0.0, u - 0.88)
                prod_real = prod_std * rng.lognormal(0, 0.22) * saturacion
            prod_real += cambios * rng.uniform(0.3, 1.2)
            if inventario and rng.random() < 0.05:  # inventario en sistema que no existia
                prod_real += prod_std * rng.uniform(0.5, 1.0)
            trans_real = transito_std * rng.lognormal(-0.15, 0.18)
            trans_real += rng.exponential(retraso_tr[transportista])
            if modalidad == "Consolidado":
                trans_real += rng.exponential(0.45)
            elif modalidad == "Paqueteria":
                trans_real += rng.exponential(0.25)
            dias_aduana = 0.0
            if cruce:
                dias_aduana = rng.exponential(esc["aduana_media_diciembre"] if dia.month == 12 else esc["aduana_media"])
                trans_real += dias_aduana
            if region in ("Norte", "Noreste") and dia.month in (6, 7, 8, 9):
                trans_real += rng.exponential(0.35)
            imprevisto = 0.0
            if rng.random() < esc["prob_imprevisto"]:  # bloqueos, clima, robos: no se pueden prever
                imprevisto = rng.uniform(1.5, 5.0)
                trans_real += imprevisto
            total_real = prod_real + trans_real + rng.normal(0, 0.35)
            dias_reales = max(1, int(round(total_real)))

            filas.append(
                {
                    "fecha_pedido": dia,
                    "cliente": cliente,
                    "tipo_cliente": tipo,
                    "planta": planta,
                    "region_destino": region,
                    "cruce_frontera": int(cruce),
                    "distancia_km": round(distancia, 0),
                    "transportista": transportista,
                    "modalidad": modalidad,
                    "familia_producto": familia,
                    "cantidad_piezas": piezas,
                    "peso_kg": peso,
                    "inventario_disponible": int(inventario),
                    "utilizacion_planta": round(u, 3),
                    "pedido_urgente": int(urgente),
                    "cambios_pedido": cambios,
                    "produccion_std_dias": produccion_std,
                    "transito_std_dias": transito_std,
                    "dias_comprometidos": dias_comprometidos,
                    "holgura_dias": dias_comprometidos - produccion_std - transito_std,
                    "dias_reales": dias_reales,
                    **({"oculto_dias_imprevisto": round(imprevisto, 2), "oculto_dias_aduana": round(dias_aduana, 2),
                        "oculto_produccion_real": round(prod_real, 2), "oculto_transito_real": round(trans_real, 2)}
                       if ocultas else {}),
                }
            )

    df = pd.DataFrame(filas).sort_values("fecha_pedido", kind="stable").reset_index(drop=True)
    df["fecha_compromiso"] = df["fecha_pedido"] + pd.to_timedelta(df["dias_comprometidos"], "D")
    df["fecha_entrega_real"] = df["fecha_pedido"] + pd.to_timedelta(df["dias_reales"], "D")
    df["dias_retraso"] = (df["dias_reales"] - df["dias_comprometidos"]).clip(lower=0)
    df["retraso"] = (df["dias_reales"] > df["dias_comprometidos"]).astype(int)
    df["mes"] = df["fecha_pedido"].dt.month
    df["dia_semana"] = df["fecha_pedido"].dt.dayofweek
    df = _agregar_historial(df)
    df.insert(0, "id_orden", [f"OC-{4500100000 + i}" for i in range(len(df))])
    return df


def _agregar_historial(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de historial calculadas solo con informacion anterior al pedido."""
    fechas = df["fecha_pedido"].values
    # 1) envios previos del mismo cliente con el mismo transportista (ultimos 365 dias)
    previos = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby(["cliente", "transportista"]).groups.items():
        idx = np.array(sorted(idx))
        f = fechas[idx]
        izquierda = np.searchsorted(f, f - np.timedelta64(365, "D"), side="left")
        actual = np.searchsorted(f, f, side="left")
        previos[idx] = actual - izquierda
    df["envios_previos_ruta"] = previos

    # 2) tasa de retraso del transportista en entregas YA REALIZADAS en los ultimos 90 dias
    tasa = np.full(len(df), np.nan)
    for _, sub in df.groupby("transportista"):
        entregas = sub.sort_values("fecha_entrega_real")
        f_ent = entregas["fecha_entrega_real"].values
        acum = np.concatenate([[0], np.cumsum(entregas["retraso"].values)])
        for i, f_ped in zip(sub.index, sub["fecha_pedido"].values):
            fin = np.searchsorted(f_ent, f_ped, side="left")  # entregadas antes del pedido
            ini = np.searchsorted(f_ent, f_ped - np.timedelta64(90, "D"), side="left")
            n = fin - ini
            tasa[i] = (acum[fin] - acum[ini]) / n if n >= 30 else np.nan
    df["tasa_retraso_transportista_90d"] = np.round(tasa, 4)
    return df


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    DIR_DATOS.mkdir(exist_ok=True)
    df = simular()
    fecha_txt = lambda s: s.dt.strftime("%Y-%m-%d")  # noqa: E731
    for c in ("fecha_pedido", "fecha_compromiso", "fecha_entrega_real"):
        df[c] = fecha_txt(df[c])

    historico = df[df["fecha_pedido"] <= FIN_HISTORICO.strftime("%Y-%m-%d")]
    piloto = df[(df["fecha_pedido"] > FIN_HISTORICO.strftime("%Y-%m-%d"))
                & (df["cliente"].isin(CLIENTES_PILOTO))]

    rutas = {"entregas_cab.csv": historico, "piloto_ab_cab.csv": piloto}
    manifiesto = {"version": VERSION_DATOS, "semilla": SEMILLA,
                  "generador": "entrenamiento/generar_datos_entregas.py", "archivos": {}}
    for nombre, tabla in rutas.items():
        ruta = DIR_DATOS / nombre
        tabla.to_csv(ruta, index=False, lineterminator="\n")
        manifiesto["archivos"][nombre] = {
            "filas": int(len(tabla)),
            "columnas": int(tabla.shape[1]),
            "tasa_retraso": round(float(tabla["retraso"].mean()), 4),
            "periodo": [tabla["fecha_pedido"].min(), tabla["fecha_pedido"].max()],
            "sha256": sha256(ruta),
        }
        print(f"{nombre}: {len(tabla):,} filas, retraso {tabla['retraso'].mean():.3f}")
    (DIR_DATOS / "versiones.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
