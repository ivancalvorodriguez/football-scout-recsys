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

Modos de normalizacion:
- ``por_liga`` (por defecto): estandariza dentro de cada (competition_id,
  season_id). Evita que las ligas dominen la comparacion, pero pierde toda
  comparacion INTER-liga (Depay en La Liga queda en un espacio distinto de
  Shaqiri en Bundesliga).
- ``global``: estandariza con la media/std de TODO el dataset, mezclando ligas.
  Habilita la comparacion inter-liga a costa de juntar ligas de niveles muy
  distintos (efecto mitigado por la ponderacion por minutos).

Devuelve una `MatrizFeatures`: X (M x d, sin NaN), metadatos por fila
(entity_id, entity_name, peso por minutos, liga) y los nombres de columna. NUNCA
agrega filas por entidad: M filas entran y M filas salen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .data import LEAGUE_KEY


@dataclass
class MatrizFeatures:
    """Salida de la capa de features (una fila por observacion)."""

    X: np.ndarray            # (M, d) features z-scored por liga, sin NaN
    feat_names: list[str]    # d nombres de columna
    entity_id: np.ndarray    # (M,) id de la entidad de cada fila
    entity_name: np.ndarray  # (M,) nombre de la entidad de cada fila
    weight: np.ndarray       # (M,) masa de la observacion (minutos/90 o 1.0)
    league: np.ndarray       # (M,) clave de liga de cada fila


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
    return feats


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


def _zscore_por_liga(feats: pd.DataFrame, league: pd.Series) -> pd.DataFrame:
    """Estandariza cada feature dentro de su liga (nanmean/nanstd, ddof=0).

    Std 0 (feature constante en la liga) -> 1 para no dividir por cero. Los NaN
    (ratios sin definir) se mantienen aqui y se rellenan a 0 despues.
    """
    def z(g: pd.DataFrame) -> pd.DataFrame:
        mu = g.mean(skipna=True)
        sd = g.std(skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return (g - mu) / sd

    return feats.groupby(league, group_keys=False).apply(z)


def _zscore_global(feats: pd.DataFrame) -> pd.DataFrame:
    """Estandariza cada feature con la media/std de TODO el dataset (sin agrupar).

    Misma convencion que ``_zscore_por_liga``: std 0 -> 1, NaN se conservan.
    Habilita comparacion inter-liga a costa de mezclar ligas heterogeneas.
    """
    mu = feats.mean(skipna=True)
    sd = feats.std(skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return (feats - mu) / sd


# Modos validos de normalizacion (los que `construir` acepta).
NORMALIZACIONES_VALIDAS = ("por_liga", "global")


def construir(
    df: pd.DataFrame, entidad: str, normalizacion: str = "por_liga"
) -> MatrizFeatures:
    """Construye la matriz de features por-partido para una entidad.

    ``normalizacion``:
    - ``"por_liga"`` (default): z-score por (competition_id, season_id).
    - ``"global"``: z-score sobre todo el dataset (mezclando ligas).
    """
    if normalizacion not in NORMALIZACIONES_VALIDAS:
        raise ValueError(
            f"normalizacion desconocida: {normalizacion!r} "
            f"(usa {NORMALIZACIONES_VALIDAS})"
        )

    if entidad == "jugador":
        feats = _derivar_jugador(df)
        weight = (df["minutes_played"].astype(float) / 90.0).to_numpy()
    elif entidad == "equipo":
        feats = _derivar_equipo(df)
        weight = np.ones(len(df), dtype=float)  # masa uniforme (partido completo)
    else:
        raise ValueError(f"entidad desconocida: {entidad!r}")

    feat_names = list(feats.columns)
    league = df[LEAGUE_KEY]
    if normalizacion == "por_liga":
        feats_z = _zscore_por_liga(feats, league)
    else:  # "global"
        feats_z = _zscore_global(feats)
    # NaN restantes (ratios sin denominador) -> 0 = media estandarizada.
    X = feats_z[feat_names].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Winsorizado: tapa z-scores extremos de per-90 en cameos de pocos minutos.
    X = np.clip(X, -config.F_CLIP_Z, config.F_CLIP_Z)

    return MatrizFeatures(
        X=X,
        feat_names=feat_names,
        entity_id=df["entity_id"].to_numpy(),
        entity_name=df["entity_name"].to_numpy().astype(str),
        weight=weight,
        league=league.to_numpy().astype(str),
    )
