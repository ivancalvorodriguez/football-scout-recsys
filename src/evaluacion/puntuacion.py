"""Score compuesto de un barrido: agrega las metricas de una combinacion en un numero.

Las tablas de `resumen_barrido.md` dan una metrica por tabla, asi que la mejor
combinacion segun el top-1 puede no serlo segun el RBO ni segun la cobertura. Este
modulo agrega todas las metricas en un unico score por (modelo, combinacion) para
poder ver de un vistazo QUE modelo y con QUE hiperparametros va mejor.

**El score es una heuristica de lectura, no un criterio validado.** El protocolo de
`docs/Como_evaluar.pdf` juzga por CONVERGENCIA de señales, no por un numero: aqui
se colapsan a la fuerza metricas que miden cosas distintas (validez, estabilidad,
utilidad downstream, diversidad) con unos pesos elegidos a mano. Sirve para ordenar
y para ver la forma del optimo; no sustituye a las tablas por metrica, y ninguna
conclusion de la memoria deberia apoyarse solo en el.

Como se calcula:

1. Cada metrica se orienta (`MEJOR`): la asimetria de la S es "menor es mejor", asi
   que entra con signo negativo. El resto entra tal cual.
2. Cada metrica se lleva a **z-score dentro de su entidad**, sobre todos los pares
   (modelo, combinacion) de esa entidad. Es lo que hace comparables cosas en
   unidades distintas (un top-1 de 0.25 y una diversidad de 2.6) y lo que permite
   comparar F2 con F5 y `por_liga` con `global` en el mismo grafico. **Nunca se
   mezclan jugador y equipo**: distintas features, distinto pool y distinto metodo,
   como declara `docs/evaluacion.md`.
3. El score es la media ponderada (`PESOS`) de esas z, renormalizada sobre las
   metricas realmente disponibles: una metrica que no existe para ese modelo (la
   pureza posicional del equipo) no cuenta ni penaliza.

Una metrica constante en toda la entidad (varianza 0) aporta z=0 a todos: no
discrimina, y meterla como NaN dejaria sin score a modelos perfectamente evaluados.

**Cada metrica cuenta DOS veces: dentro y fuera de muestra.** La Fase 7
(`generalizacion`) vuelve a medir lo mismo sobre una liga excluida del ajuste, y
esa gemela entra en el score con su valor absoluto y el MISMO peso que su original
(`FACTOR_GENERALIZACION` = 1.0): con las 8 metricas de siempre y sus 7 gemelas, el
score agrega 15. Lo que se gana midiendo fuera de muestra es una senal que ninguna
metrica en muestra detecta —todas se miden sobre las entidades del entrenamiento—,
y con el mismo peso el score dice exactamente eso: cuenta la senal, no la pondera.

Tres consecuencias que hay que tener presentes:

- **No es una brecha, es el nivel.** Un modelo malo dentro y fuera tendria brecha
  cero sin generalizar nada; la caida se sigue viendo en el informe de la fase,
  que publica los dos ambitos uno al lado del otro.
- **Solo las tienen los modelos que la fase puede medir.** El equipo no tiene
  etiqueta de rol (ni pureza ni k-NN) y la F5 de equipo (Sinkhorn) no se sabe
  proyectar, asi que ahi el score se renormaliza sobre las demas metricas, como
  ya hacia con la pureza. Un barrido sin `--holdout` no tiene ninguna gemela y el
  score es exactamente el de antes.
- **Heredan la circularidad de su original**: `gen_pureza_top1` y
  `gen_knn_accuracy` usan la posicion como etiqueta y la posicion es una feature
  del jugador. Las que no son circulares son `gen_top1` y `gen_mrr` (nadie le ha
  dicho al modelo que dos mitades de partidos son la misma persona). Es el motivo
  por el que `FACTOR_GENERALIZACION` vale 1 y no 1.5: ponderar al alza la mitad
  fuera de muestra pondera al alza tambien su parte circular.
"""

from __future__ import annotations

import pandas as pd

from . import generalizacion

# --------------------------------------------------------------------------- #
# Orientacion y pesos                                                           #
# --------------------------------------------------------------------------- #
# Orientacion: mismo criterio que las tablas de `barrido.escribir_resumen_barrido`.

MEJOR: dict[str, str] = {"asimetria": "min"}
MEJOR_POR_DEFECTO = "max"


def orientacion(metrica: str) -> str:
    return MEJOR.get(metrica, MEJOR_POR_DEFECTO)


# Pesos: reflejan el orden de importancia del protocolo, no una calibracion.
# - Validez de la similitud (que el vecino recuperado sea el correcto): lo que de
#   verdad mide si el modelo sirve para recomendar.
# - Estabilidad y utilidad downstream: condiciones necesarias, algo por debajo.
# - Diversidad, cobertura y asimetria: higiene. Pesan poco a proposito, porque
#   son facilmente "ganables" por un modelo malo (una S casi aleatoria da
#   cobertura y diversidad altisimas).
PESOS_EN_MUESTRA: dict[str, float] = {
    "top1": 1.0,
    "mrr": 1.0,
    "pureza_top1": 1.0,
    "rbo_medio": 0.75,
    "knn_accuracy": 0.75,
    "coverage": 0.25,
    "diversity": 0.25,
    "asimetria": 0.25,
}

# Cuanto pesa una metrica medida FUERA de muestra respecto de la misma metrica
# medida sobre las entidades del ajuste. Vale 1: las dos mitades de cada metrica
# pesan igual. Llego a valer 1.5 —aguantar el nivel en una liga que el modelo no
# vio exige mas que aguantarlo en la que se ajusto—, pero ponderar al alza la
# mitad fuera de muestra amplifica tambien las metricas CIRCULARES de esa mitad
# (`gen_pureza_top1` y `gen_knn_accuracy` usan la posicion como etiqueta y la
# posicion es una feature), y eso es justo lo que el score no debe premiar. Con
# 1.0 la exigencia sigue contando —la gemela entra en el score— pero no se
# multiplica. Un factor y no una tabla aparte, para que no puedan desincronizarse:
# cambiar el peso de una metrica cambia el de su gemela.
FACTOR_GENERALIZACION = 1.0

# El diccionario final: cada metrica y, si la Fase 7 sabe medirla fuera de
# muestra, su gemela `gen_<metrica>` con el peso multiplicado. `asimetria` no
# tiene gemela (no hay S fuera de muestra: ver `generalizacion.SIN_GEMELA`).
PESOS: dict[str, float] = {
    **PESOS_EN_MUESTRA,
    **{generalizacion.nombre_metrica(m): p * FACTOR_GENERALIZACION
       for m, p in PESOS_EN_MUESTRA.items() if m in generalizacion.METRICAS},
}

NOMBRE = "score_compuesto"


def _signo(metrica: str) -> float:
    return -1.0 if orientacion(metrica) == "min" else 1.0


def _z(valores: list[float]) -> list[float]:
    """z-score de una lista; ceros si no hay dispersion (la metrica no discrimina)."""
    n = len(valores)
    media = sum(valores) / n
    var = sum((v - media) ** 2 for v in valores) / n
    if var <= 0:
        return [0.0] * n
    sd = var ** 0.5
    return [(v - media) / sd for v in valores]


def puntuar(df: pd.DataFrame, pesos: dict[str, float] | None = None) -> pd.DataFrame:
    """Score por (entidad, modelo, combinacion) a partir de la tabla larga del barrido.

    `df` es `barrido_metricas.csv` (columnas: entidad, modelo, combinacion,
    metrica, valor). Devuelve una tabla con el score y las z que lo componen (una
    columna `z_<metrica>` por metrica), para que el numero sea auditable y no un
    veredicto opaco.
    """
    pesos = pesos or PESOS
    usables = df[df["metrica"].isin(pesos)].copy()
    usables = usables[pd.notna(usables["valor"])]
    if usables.empty:
        return pd.DataFrame(columns=["entidad", "modelo", "combinacion", NOMBRE])

    # z por (entidad, metrica) sobre todos los pares modelo-combinacion.
    zetas: dict[tuple[str, str, str, str], float] = {}
    for (entidad, metrica), bloque in usables.groupby(["entidad", "metrica"]):
        claves = list(zip(bloque["modelo"], bloque["combinacion"]))
        for (modelo, combi), z in zip(claves, _z([float(v) for v in bloque["valor"]])):
            zetas[(entidad, modelo, combi, metrica)] = z * _signo(str(metrica))

    filas: list[dict] = []
    llaves = usables[["entidad", "modelo", "combinacion"]].drop_duplicates()
    for entidad, modelo, combi in llaves.itertuples(index=False):
        fila = {"entidad": entidad, "modelo": modelo, "combinacion": combi}
        num = den = 0.0
        n_metricas = 0
        for metrica, peso in pesos.items():
            z = zetas.get((entidad, modelo, combi, metrica))
            fila[f"z_{metrica}"] = z
            if z is None:
                continue
            num += peso * z
            den += peso
            n_metricas += 1
        fila[NOMBRE] = num / den if den else float("nan")
        fila["n_metricas"] = n_metricas
        filas.append(fila)

    columnas = ["entidad", "modelo", "combinacion", NOMBRE, "n_metricas"]
    columnas += [f"z_{m}" for m in pesos]
    return pd.DataFrame(filas)[columnas].sort_values(
        ["entidad", NOMBRE], ascending=[True, False]
    ).reset_index(drop=True)


def como_metrica(scores: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los scores en filas con el formato de `barrido_metricas.csv`.

    Asi el score viaja por el mismo camino que cualquier otra metrica (rejilla,
    figuras, CSV) sin duplicar codigo: es una metrica mas, llamada `score_compuesto`.
    """
    columnas = ["modelo", "formulacion", "entidad", "normalizacion", "distancia"]
    ejes = df[[c for c in columnas if c in df.columns]] \
        .drop_duplicates().set_index("modelo")
    filas = []
    for r in scores.itertuples(index=False):
        if r.modelo not in ejes.index:
            continue
        e = ejes.loc[r.modelo]
        filas.append({
            "combinacion": r.combinacion, "modelo": r.modelo,
            "formulacion": e["formulacion"], "entidad": e["entidad"],
            "normalizacion": e["normalizacion"],
            "distancia": e.get("distancia", "euclidea"),
            "fase": "score", "metrica": NOMBRE,
            "valor": getattr(r, NOMBRE),
        })
    return pd.DataFrame(filas)


def ranking(scores: pd.DataFrame, n: int = 3) -> list[str]:
    """Lineas de texto con las `n` mejores (modelo, combinacion) por entidad."""
    lineas: list[str] = []
    for entidad, bloque in scores.groupby("entidad"):
        mejores = bloque.sort_values(NOMBRE, ascending=False).head(n)
        lineas.append(f"  {entidad}:")
        for r in mejores.itertuples(index=False):
            lineas.append(f"    {getattr(r, NOMBRE):+.3f}  {r.modelo}  ({r.combinacion})")
    return lineas
