"""Capa de features por-partido: derivadas + estandarizacion (por liga o global).

Convierte los CONTEOS CRUDOS de la BD en el vector de features de cada
observacion (no de cada entidad):

1. per-90 de los conteos (volumen) — solo jugador; el equipo usa el conteo
   por-partido tal cual (juega el partido completo).
2. ratios (eficiencia) = exito / intentos, NaN si el denominador es 0.
3. diferencias tipo rendimiento - modelo (p. ej. np_goals - npxg) sobre per-90.
4. z-score POR LIGA (competition-season) **o GLOBAL** (mezclando todas las ligas),
   con nanmean/nanstd; los NaN de ratios sin definir quedan en 0 (la media
   estandarizada), separando volumen de eficiencia sin romper el ajuste.

Al jugador se le añaden ademas 25 columnas `pos_*` con la fraccion de minutos
por posicion (one-hot ponderado), salvo que se apague con
``config.USE_POSITION_FEATURES``. Como entran depende de
``config.POSITION_SCALING``:

- ``"zscore_sqrt"`` (por defecto): pasan por el z-score con el resto y despues el
  bloque se divide por ``sqrt(nº de posiciones)``, de forma que las 25 columnas
  juntas pesan como UNA feature en vez de como ~25.
- ``"zscore"``: solo el z-score, sin dividir -> el bloque pesa como ~25 features.
- ``"cruda"``: la fraccion se queda en [0,1] fuera del z-score y del winsorizado
  (el bloque tambien pesa como una feature, pero sin estandarizar).

Modos de normalizacion:
- ``por_liga`` (por defecto): estandariza dentro de cada (competition_id,
  season_id). Evita que las ligas dominen la comparacion, pero pierde toda
  comparacion INTER-liga (Depay en La Liga queda en un espacio distinto de
  Shaqiri en Bundesliga).
- ``global``: estandariza con la media/std de TODO el dataset, mezclando ligas.
  Habilita la comparacion inter-liga a costa de juntar ligas de niveles muy
  distintos (efecto mitigado por la ponderacion por minutos).

Devuelve una `MatrizFeatures`: X (M x d, sin NaN), metadatos por fila
(entity_id, entity_name, peso por minutos, liga), los nombres de columna y las
`EstadisticasNorm` (mu/sd) con las que se estandarizo. NUNCA agrega filas por
entidad: M filas entran y M filas salen.

Esas estadisticas se pueden CONGELAR y volver a pasar a `construir` al anadir
datos nuevos (flujo incremental, `src.incremental`): sin congelarlas, meter una
liga mas mueve la media/desviacion y, con ello, el vector de TODAS las
observaciones antiguas, aunque sus conteos crudos no hayan cambiado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import LEAGUE_KEY

# Clave unica de `EstadisticasNorm` cuando la normalizacion es global (no hay
# una clave por liga que usar).
CLAVE_GLOBAL = "__global__"


@dataclass
class EstadisticasNorm:
    """mu/sd por grupo con las que se hizo el z-score (una por liga, o una sola).

    `mu`/`sd` van indexadas por clave de grupo (la `league_key` en `por_liga`,
    `CLAVE_GLOBAL` en `global`) y cada valor es un vector alineado con
    `columnas`. Es serializable a JSON tal cual, para viajar dentro del artefacto
    del modelo y poder reestandarizar datos nuevos EN EL MISMO espacio.
    """

    normalizacion: str
    columnas: list[str] = field(default_factory=list)
    mu: dict[str, list[float]] = field(default_factory=dict)
    sd: dict[str, list[float]] = field(default_factory=dict)

    def como_dict(self) -> dict:
        return {
            "normalizacion": self.normalizacion,
            "columnas": list(self.columnas),
            "mu": {k: [float(x) for x in v] for k, v in self.mu.items()},
            "sd": {k: [float(x) for x in v] for k, v in self.sd.items()},
        }

    @classmethod
    def desde_dict(cls, datos: dict | None) -> "EstadisticasNorm | None":
        if not datos:
            return None
        return cls(
            normalizacion=str(datos["normalizacion"]),
            columnas=list(datos.get("columnas", [])),
            mu={str(k): list(v) for k, v in (datos.get("mu") or {}).items()},
            sd={str(k): list(v) for k, v in (datos.get("sd") or {}).items()},
        )

    def _serie(self, tabla: dict[str, list[float]], clave: str,
               columnas: pd.Index) -> pd.Series:
        """Vector guardado para `clave`, realineado a `columnas` (NaN si falta)."""
        valores = tabla.get(clave)
        if valores is None or not self.columnas:
            return pd.Series(np.nan, index=columnas, dtype=float)
        serie = pd.Series(list(valores), index=list(self.columnas), dtype=float)
        return serie.reindex(columnas)

    def serie_mu(self, clave: str, columnas: pd.Index) -> pd.Series:
        return self._serie(self.mu, clave, columnas)

    def serie_sd(self, clave: str, columnas: pd.Index) -> pd.Series:
        return self._serie(self.sd, clave, columnas)


@dataclass
class MatrizFeatures:
    """Salida de la capa de features (una fila por observacion)."""

    X: np.ndarray            # (M, d) features z-scored por liga, sin NaN
    feat_names: list[str]    # d nombres de columna
    entity_id: np.ndarray    # (M,) id de la entidad de cada fila
    entity_name: np.ndarray  # (M,) nombre de la entidad de cada fila
    weight: np.ndarray       # (M,) masa de la observacion (minutos/90 o 1.0)
    league: np.ndarray       # (M,) clave de liga de cada fila
    match_id: np.ndarray | None = None        # (M,) partido de cada fila
    estadisticas: EstadisticasNorm | None = None  # mu/sd usadas en el z-score


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den con NaN cuando den == 0 (sin divisiones por cero silenciosas)."""
    den = den.astype(float)
    out = num.astype(float) / den.where(den != 0.0, np.nan)
    return out


def _per90(df: pd.DataFrame, cols: list[str], minutes: pd.Series) -> pd.DataFrame:
    """valor * 90 / minutos; minutos<=0 -> NaN (no deberia ocurrir en la BD)."""
    m = minutes.astype(float).where(minutes > 0.0, np.nan)
    return df[cols].astype(float).mul(90.0 / m, axis=0)


def _derivar_jugador(df: pd.DataFrame) -> pd.DataFrame:
    minutes = df["minutes_played"]
    feats = _per90(df, config.PLAYER_COUNT_FEATURES, minutes)

    # Denominadores compuestos para algunos ratios.
    aux = pd.DataFrame(index=df.index)
    aux["__aerials_total__"] = df["aerial_won"] + df["aerial_lost"]
    aux["__tackle_duels__"] = df["tackles_won"] + df["dribbled_past"]
    src = pd.concat([df, aux], axis=1)

    for name, num, den in config.PLAYER_RATIO_FEATURES:
        feats[name] = _safe_ratio(src[num], src[den])

    # Diferencias sobre valores per-90 (rendimiento - modelo).
    for name, a, b in config.PLAYER_DIFF_FEATURES:
        feats[name] = feats[a] - feats[b]

    # Posicion (ver config.USE_POSITION_FEATURES). En los modos con z-score se
    # concatena aqui, para pasar por la estandarizacion con el resto; en modo
    # "cruda" se añade despues, ya en `construir`.
    if config.USE_POSITION_FEATURES and config.POSITION_SCALING in _CON_ZSCORE:
        feats = pd.concat([feats, _posiciones(df)], axis=1)
    return feats


def _posiciones(df: pd.DataFrame) -> pd.DataFrame:
    """Las 25 columnas `pos_*` (fraccion de minutos por posicion) del df.

    Las aporta `data.cargar_jugadores`; si faltasen (un df construido a mano sin
    esa capa) se rellenan a 0 = sin senal de posicion.
    """
    return df.reindex(columns=config.POSITION_FEATURES).astype(float).fillna(0.0)


def _derivar_equipo(df: pd.DataFrame) -> pd.DataFrame:
    # El equipo juega el partido completo: se usa el conteo por-partido directo.
    feats = df[config.TEAM_COUNT_FEATURES].astype(float).copy()
    for col in config.TEAM_RATE_FEATURES:
        feats[col] = df[col].astype(float)
    for name, num, den in config.TEAM_RATIO_FEATURES:
        feats[name] = _safe_ratio(df[num], df[den])
    for name, a, b in config.TEAM_DIFF_FEATURES:
        feats[name] = df[a].astype(float) - df[b].astype(float)
    return feats


def _mu_sd(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """mu/sd de un bloque de filas (nanmean/nanstd, ddof=0).

    Std 0 (feature constante en el bloque) -> 1 para no dividir por cero. Una
    columna entera a NaN da mu NaN -> 0: sus filas son NaN igualmente y acaban en
    0 tras el `nan_to_num`, asi que el resultado no cambia y las estadisticas
    quedan serializables (sin NaN).
    """
    mu = g.mean(skipna=True).fillna(0.0)
    sd = g.std(skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mu, sd


def _mu_sd_con_previas(
    g: pd.DataFrame, previas: EstadisticasNorm | None, clave: str
) -> tuple[pd.Series, pd.Series]:
    """mu/sd del bloque, dando prioridad a las CONGELADAS de `previas`.

    Lo que `previas` no cubra (una liga que no existia, una feature nueva) se
    calcula del propio bloque: es lo unico que se puede hacer, y coincide con lo
    que haria un ajuste en frio para esos casos.
    """
    mu, sd = _mu_sd(g)
    if previas is None:
        return mu, sd
    return (
        previas.serie_mu(clave, g.columns).combine_first(mu),
        previas.serie_sd(clave, g.columns).combine_first(sd),
    )


def _zscore_por_liga(
    feats: pd.DataFrame, league: pd.Series, previas: EstadisticasNorm | None = None
) -> tuple[pd.DataFrame, EstadisticasNorm]:
    """Estandariza cada feature dentro de su liga (nanmean/nanstd, ddof=0).

    Los NaN (ratios sin definir) se mantienen aqui y se rellenan a 0 despues.
    Devuelve tambien las mu/sd por liga efectivamente aplicadas.
    """
    stats = EstadisticasNorm("por_liga", columnas=list(feats.columns))
    salida = pd.DataFrame(np.nan, index=feats.index, columns=feats.columns, dtype=float)
    for clave, filas in feats.groupby(league, sort=False).groups.items():
        bloque = feats.loc[filas]
        mu, sd = _mu_sd_con_previas(bloque, previas, str(clave))
        salida.loc[filas] = ((bloque - mu) / sd).to_numpy()
        stats.mu[str(clave)] = mu.tolist()
        stats.sd[str(clave)] = sd.tolist()
    return salida, stats


def _zscore_global(
    feats: pd.DataFrame, previas: EstadisticasNorm | None = None
) -> tuple[pd.DataFrame, EstadisticasNorm]:
    """Estandariza cada feature con la media/std de TODO el dataset (sin agrupar).

    Misma convencion que ``_zscore_por_liga``: std 0 -> 1, NaN se conservan.
    Habilita comparacion inter-liga a costa de mezclar ligas heterogeneas.
    """
    mu, sd = _mu_sd_con_previas(feats, previas, CLAVE_GLOBAL)
    stats = EstadisticasNorm(
        "global", columnas=list(feats.columns),
        mu={CLAVE_GLOBAL: mu.tolist()}, sd={CLAVE_GLOBAL: sd.tolist()},
    )
    return (feats - mu) / sd, stats


# Modos validos de normalizacion (los que `construir` acepta).
NORMALIZACIONES_VALIDAS = ("por_liga", "global")

# Modos validos de escalado del bloque de posicion (config.POSITION_SCALING).
POSITION_SCALINGS_VALIDAS = ("cruda", "zscore", "zscore_sqrt")

# Los que estandarizan el bloque con el resto de features (se concatena antes del
# z-score); "cruda" es el unico que se añade despues.
_CON_ZSCORE = ("zscore", "zscore_sqrt")


def _bloque_posicion_crudo(df: pd.DataFrame) -> np.ndarray:
    """Las 25 columnas `pos_*` sin tocar: fraccion de minutos en [0,1].

    A proposito NO pasan por el z-score ni por el winsorizado. Cada fila suma 1
    (one-hot "blando"), asi que dos observaciones en posiciones distintas distan
    sqrt(2) en este bloque y aportan 2.0 a la distancia^2 — la misma aportacion
    esperada que una unica feature z-scoreada de varianza 1. El bloque entero
    influye por tanto como UNA feature, que es el criterio elegido; no se divide
    por sqrt(n_posiciones) porque eso lo dejaria n_posiciones veces por debajo.
    """
    pos = _posiciones(df).to_numpy(dtype=float)
    return np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)


def _reescalar_bloque_zscoreado(X: np.ndarray, feat_names: list[str]) -> None:
    """Divide IN PLACE el bloque `pos_*` ya z-scoreado por sqrt(nº de posiciones).

    Cada columna estandarizada tiene varianza 1 y aporta E[(xi-xj)^2] = 2 a la
    distancia^2, asi que las P columnas juntas aportan 2P: la posicion pasaria a
    pesar como ~P features y a gobernar el ranking. Dividir el bloque por sqrt(P)
    divide su distancia^2 por P y lo deja en 2.0 — exactamente lo que aporta UNA
    feature — sin renunciar al z-score (a diferencia del modo "cruda", que pesa
    igual pero deja la fraccion sin estandarizar ni winsorizar).

    El divisor es el tamaño del CATALOGO (`config.POSITION_FEATURES`, 25), no el
    nº de posiciones observadas: asi el peso del bloque no depende de que
    posiciones aparezcan en el subconjunto de datos que se cargue.
    """
    cols = [i for i, c in enumerate(feat_names) if c in set(config.POSITION_FEATURES)]
    if cols:
        X[:, cols] /= np.sqrt(len(config.POSITION_FEATURES))


def derivar(df: pd.DataFrame, entidad: str) -> pd.DataFrame:
    """Features derivadas de cada observacion SIN estandarizar.

    Son los pasos 1-3 de `construir` (per-90 o conteo por partido, ratios y
    diferencias) parando justo antes del z-score: valores en sus unidades reales
    (pases por 90', % de acierto, xG...). El modelo NO usa esto — se ajusta con
    la salida de `construir`; existe para poder MOSTRAR una metrica en una
    interfaz, donde un z-score no dice nada a un ojeador.

    Devuelve las columnas en el mismo orden que `construir().feat_names`, sea
    cual sea `config.POSITION_SCALING`; hay un test que lo fija, porque de esa
    correspondencia depende que un valor crudo se pueda emparejar con la columna
    del artefacto que le toca.
    """
    if entidad == "jugador":
        feats = _derivar_jugador(df)
        # En los modos con z-score, `_derivar_jugador` ya concatena el bloque de
        # posicion; en "cruda" lo hace `construir` despues de estandarizar. Aqui
        # no hay z-score que respetar, asi que se añade en el mismo sitio: al
        # final, que es donde acaba en los dos casos.
        if config.USE_POSITION_FEATURES and config.POSITION_SCALING == "cruda":
            feats = pd.concat([feats, _posiciones(df)], axis=1)
        return feats
    if entidad == "equipo":
        return _derivar_equipo(df)
    raise ValueError(f"entidad desconocida: {entidad!r}")


def masa(df: pd.DataFrame, entidad: str) -> np.ndarray:
    """Peso de cada observacion al agregar por entidad.

    Minutos/90 en el jugador (un cameo de 5' no puede pesar como un partido
    entero) y masa uniforme en el equipo, que juega el partido completo. Es el
    mismo criterio que usa `construir`, expuesto aparte para que quien agregue
    los valores de `derivar` no tenga que reimplementarlo.
    """
    if entidad == "jugador":
        return (df["minutes_played"].astype(float) / 90.0).to_numpy()
    if entidad == "equipo":
        return np.ones(len(df), dtype=float)
    raise ValueError(f"entidad desconocida: {entidad!r}")


def construir(
    df: pd.DataFrame,
    entidad: str,
    normalizacion: str = "por_liga",
    estadisticas: EstadisticasNorm | None = None,
) -> MatrizFeatures:
    """Construye la matriz de features por-partido para una entidad.

    ``normalizacion``:
    - ``"por_liga"`` (default): z-score por (competition_id, season_id).
    - ``"global"``: z-score sobre todo el dataset (mezclando ligas).

    ``estadisticas``: mu/sd CONGELADAS de un ajuste anterior (las que devuelve
    esta misma funcion en ``mf.estadisticas``). Con ellas, anadir datos nuevos no
    mueve el vector de las observaciones que ya estaban — condicion para que el
    reentrenamiento incremental pueda reaprovechar el ajuste previo y para que
    las recomendaciones antiguas no cambien por debajo sin motivo. Lo que no
    cubran (liga nueva, feature nueva) se calcula del dato. None = recalcular
    todo, el comportamiento de un ajuste en frio.
    """
    if normalizacion not in NORMALIZACIONES_VALIDAS:
        raise ValueError(
            f"normalizacion desconocida: {normalizacion!r} "
            f"(usa {NORMALIZACIONES_VALIDAS})"
        )
    if config.POSITION_SCALING not in POSITION_SCALINGS_VALIDAS:
        raise ValueError(
            f"config.POSITION_SCALING desconocido: {config.POSITION_SCALING!r} "
            f"(usa {POSITION_SCALINGS_VALIDAS})"
        )

    if entidad == "jugador":
        feats = _derivar_jugador(df)
    elif entidad == "equipo":
        feats = _derivar_equipo(df)
    else:
        raise ValueError(f"entidad desconocida: {entidad!r}")
    weight = masa(df, entidad)

    feat_names = list(feats.columns)
    league = df[LEAGUE_KEY]
    if estadisticas is not None and estadisticas.normalizacion != normalizacion:
        raise ValueError(
            f"las estadisticas congeladas son de normalizacion "
            f"{estadisticas.normalizacion!r} y se pidio {normalizacion!r}"
        )
    if normalizacion == "por_liga":
        feats_z, stats = _zscore_por_liga(feats, league, estadisticas)
    else:  # "global"
        feats_z, stats = _zscore_global(feats, estadisticas)
    # NaN restantes (ratios sin denominador) -> 0 = media estandarizada.
    X = feats_z[feat_names].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Winsorizado: tapa z-scores extremos de per-90 en cameos de pocos minutos.
    X = np.clip(X, -config.F_CLIP_Z, config.F_CLIP_Z)

    # Posicion en modo "zscore_sqrt" (el default): el bloque ya paso por el
    # z-score y el clip con el resto; ahora se reescala para que las 25 columnas
    # pesen como una sola feature (ver _reescalar_bloque_zscoreado).
    if (
        entidad == "jugador"
        and config.USE_POSITION_FEATURES
        and config.POSITION_SCALING == "zscore_sqrt"
    ):
        _reescalar_bloque_zscoreado(X, feat_names)

    # Posicion en modo "cruda": se concatena DESPUES, para que no la toquen ni el
    # z-score ni el clip y conserve su escala [0,1] (ver _bloque_posicion_crudo).
    if (
        entidad == "jugador"
        and config.USE_POSITION_FEATURES
        and config.POSITION_SCALING == "cruda"
    ):
        X = np.hstack([X, _bloque_posicion_crudo(df)])
        feat_names = feat_names + list(config.POSITION_FEATURES)

    return MatrizFeatures(
        X=X,
        feat_names=feat_names,
        entity_id=df["entity_id"].to_numpy(),
        entity_name=df["entity_name"].to_numpy().astype(str),
        weight=weight,
        league=league.to_numpy().astype(str),
        # El partido identifica la observacion dentro de la entidad: es la clave
        # con la que el reentrenamiento incremental empareja las filas de esta X
        # con las del ajuste anterior. Puede faltar en un df construido a mano.
        match_id=(df["match_id"].to_numpy() if "match_id" in df.columns else None),
        estadisticas=stats,
    )
