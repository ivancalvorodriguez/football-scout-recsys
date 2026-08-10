"""Estadisticas de normalizacion: las mu/sd del z-score, y congelarlas.

Son la condicion para que el flujo incremental tenga sentido: si al anadir una
liga se recalculan, el vector de un jugador que no ha jugado cambia igualmente y
sus recomendaciones se mueven sin motivo. Congeladas, las filas antiguas salen
identicas y el ajuste anterior sigue siendo valido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similitud import data, features
from src.similitud.features import CLAVE_GLOBAL, EstadisticasNorm


@pytest.fixture
def df_dos_ligas(bd_sintetica) -> pd.DataFrame:
    return data.cargar(bd_sintetica, "equipo")


def _liga_a(df: pd.DataFrame) -> pd.DataFrame:
    """Solo la primera liga: simula "la BD antes de que llegara la segunda"."""
    primera = df[data.LEAGUE_KEY].iloc[0]
    return df[df[data.LEAGUE_KEY] == primera].reset_index(drop=True)


# --- Lo que se devuelve -------------------------------------------------------
def test_construir_devuelve_las_estadisticas_que_uso(df_dos_ligas):
    mf = features.construir(df_dos_ligas, "equipo", normalizacion="global")
    stats = mf.estadisticas
    assert stats is not None
    assert stats.normalizacion == "global"
    assert stats.columnas == mf.feat_names
    assert list(stats.mu) == [CLAVE_GLOBAL]
    assert len(stats.mu[CLAVE_GLOBAL]) == len(mf.feat_names)


def test_por_liga_guarda_una_entrada_por_liga(df_dos_ligas):
    mf = features.construir(df_dos_ligas, "equipo", normalizacion="por_liga")
    assert set(mf.estadisticas.mu) == set(np.unique(mf.league).tolist())


def test_las_estadisticas_viajan_por_json(df_dos_ligas):
    mf = features.construir(df_dos_ligas, "equipo", normalizacion="por_liga")
    copia = EstadisticasNorm.desde_dict(mf.estadisticas.como_dict())
    assert copia.normalizacion == mf.estadisticas.normalizacion
    assert copia.columnas == mf.estadisticas.columnas
    for clave, valores in mf.estadisticas.mu.items():
        assert copia.mu[clave] == pytest.approx(valores)


def test_desde_dict_de_nada_es_nada():
    assert EstadisticasNorm.desde_dict(None) is None
    assert EstadisticasNorm.desde_dict({}) is None


def test_la_matriz_lleva_el_partido_de_cada_fila(df_dos_ligas):
    mf = features.construir(df_dos_ligas, "equipo")
    assert mf.match_id is not None
    assert np.array_equal(mf.match_id, df_dos_ligas["match_id"].to_numpy())


# --- Congelar: el efecto que motiva todo esto --------------------------------
def test_sin_congelar_anadir_una_liga_mueve_las_filas_antiguas(df_dos_ligas):
    """Con z-score global, la media se recalcula con las dos ligas y todo se mueve."""
    solo_a = _liga_a(df_dos_ligas)
    antes = features.construir(solo_a, "equipo", normalizacion="global")
    despues = features.construir(df_dos_ligas, "equipo", normalizacion="global")
    n = antes.X.shape[0]
    assert not np.allclose(antes.X, despues.X[:n])


def test_congelando_las_filas_antiguas_no_se_mueven(df_dos_ligas):
    solo_a = _liga_a(df_dos_ligas)
    antes = features.construir(solo_a, "equipo", normalizacion="global")
    despues = features.construir(
        df_dos_ligas, "equipo", normalizacion="global",
        estadisticas=antes.estadisticas)
    n = antes.X.shape[0]
    assert np.array_equal(antes.X, despues.X[:n])


def test_por_liga_ya_aisla_a_las_ligas_que_no_se_tocan(df_dos_ligas):
    """`por_liga` estandariza dentro de cada liga: anadir OTRA no la afecta."""
    solo_a = _liga_a(df_dos_ligas)
    antes = features.construir(solo_a, "equipo", normalizacion="por_liga")
    despues = features.construir(df_dos_ligas, "equipo", normalizacion="por_liga")
    n = antes.X.shape[0]
    assert np.array_equal(antes.X, despues.X[:n])


def test_una_liga_nueva_se_estandariza_con_sus_propios_datos(df_dos_ligas):
    """Lo que las mu/sd congeladas no cubren se calcula del dato, sin fallar."""
    solo_a = _liga_a(df_dos_ligas)
    antes = features.construir(solo_a, "equipo", normalizacion="por_liga")
    despues = features.construir(
        df_dos_ligas, "equipo", normalizacion="por_liga",
        estadisticas=antes.estadisticas)
    nuevas = set(despues.estadisticas.mu) - set(antes.estadisticas.mu)
    assert nuevas, "la segunda liga deberia aparecer como grupo nuevo"
    assert np.isfinite(despues.X).all()


def test_congelar_conserva_las_estadisticas_en_la_salida(df_dos_ligas):
    antes = features.construir(df_dos_ligas, "equipo", normalizacion="global")
    despues = features.construir(
        df_dos_ligas, "equipo", normalizacion="global",
        estadisticas=antes.estadisticas)
    assert despues.estadisticas.mu[CLAVE_GLOBAL] == pytest.approx(
        antes.estadisticas.mu[CLAVE_GLOBAL])


def test_estadisticas_de_otra_normalizacion_se_rechazan(df_dos_ligas):
    stats = features.construir(df_dos_ligas, "equipo", normalizacion="global").estadisticas
    with pytest.raises(ValueError, match="normalizacion"):
        features.construir(df_dos_ligas, "equipo", normalizacion="por_liga",
                           estadisticas=stats)


def test_columnas_desconocidas_en_las_congeladas_se_ignoran(df_dos_ligas):
    """Un vector de features que ha cambiado no debe romper la reestandarizacion."""
    mf = features.construir(df_dos_ligas, "equipo", normalizacion="global")
    stats = mf.estadisticas
    stats.columnas = stats.columnas + ["feature_que_ya_no_existe"]
    stats.mu[CLAVE_GLOBAL] = list(stats.mu[CLAVE_GLOBAL]) + [0.0]
    stats.sd[CLAVE_GLOBAL] = list(stats.sd[CLAVE_GLOBAL]) + [1.0]
    otra = features.construir(df_dos_ligas, "equipo", normalizacion="global",
                              estadisticas=stats)
    assert np.array_equal(mf.X, otra.X)


def test_el_zscore_congelado_es_el_de_siempre(df_dos_ligas):
    """Congelar con las estadisticas del propio dataset no cambia nada."""
    mf = features.construir(df_dos_ligas, "equipo", normalizacion="por_liga")
    otra = features.construir(df_dos_ligas, "equipo", normalizacion="por_liga",
                              estadisticas=mf.estadisticas)
    assert np.array_equal(mf.X, otra.X)
