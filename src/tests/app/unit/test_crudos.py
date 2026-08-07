"""Valor real de cada métrica por entidad, reconstruido desde la BD."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.app.crudos import CatalogoCrudos, TablaCrudos


@pytest.fixture
def crudos(bd: Path) -> CatalogoCrudos:
    return CatalogoCrudos(bd)


def _valor(catalogo: CatalogoCrudos, entidad: str, entity_id: int, feature: str) -> float:
    tabla = catalogo.tabla(entidad)
    fila = int(np.flatnonzero(tabla.entity_ids == entity_id)[0])
    return float(tabla.valores[fila, tabla.feat_names.index(feature)])


class TestTabla:
    def test_las_columnas_son_las_del_pipeline(self, crudos: CatalogoCrudos) -> None:
        """Se derivan con `features.derivar`, así que casan con el artefacto."""
        nombres = crudos.tabla("jugador").feat_names
        assert "passes" in nombres and "pass_completion_pct" in nombres
        assert any(n.startswith("pos_") for n in nombres)

    def test_promedia_el_per90_ponderando_por_minutos(
        self, crudos: CatalogoCrudos
    ) -> None:
        """El 10 hace 50, 60 y 70 pases en tres partidos completos."""
        assert _valor(crudos, "jugador", 10, "passes") == pytest.approx(60.0)

    def test_un_partido_a_medias_no_pesa_como_uno_entero(
        self, crudos: CatalogoCrudos
    ) -> None:
        """El 20 juega 45' con la mitad de pases: su per-90 no se mueve."""
        assert _valor(crudos, "jugador", 20, "passes") == pytest.approx(30.0)

    def test_el_equipo_va_por_partido(self, crudos: CatalogoCrudos) -> None:
        assert _valor(crudos, "equipo", 1, "passes") == pytest.approx(500.0)

    def test_una_metrica_ausente_en_la_bd_queda_en_nan(
        self, crudos: CatalogoCrudos
    ) -> None:
        """NULL no es 0: «sin dato» y «cero intercepciones» no son lo mismo."""
        assert np.isnan(_valor(crudos, "jugador", 10, "interceptions"))

    def test_un_jugador_sin_partidos_no_esta_en_la_tabla(
        self, crudos: CatalogoCrudos
    ) -> None:
        assert 40 not in crudos.tabla("jugador").entity_ids


class TestAlinear:
    def test_reordena_a_las_filas_del_modelo(self, crudos: CatalogoCrudos) -> None:
        tabla = crudos.tabla("jugador")
        matriz = tabla.alinear(np.array([20, 10]), ["passes"])
        assert list(matriz[:, 0]) == pytest.approx([30.0, 60.0])

    def test_una_entidad_que_no_esta_en_la_bd_queda_en_nan(
        self, crudos: CatalogoCrudos
    ) -> None:
        """El artefacto puede ser de un universo distinto al de la BD actual."""
        matriz = crudos.tabla("jugador").alinear(np.array([10, 999]), ["passes"])
        assert matriz[0, 0] == pytest.approx(60.0)
        assert np.isnan(matriz[1, 0])

    def test_una_columna_que_no_existe_queda_en_nan(
        self, crudos: CatalogoCrudos
    ) -> None:
        matriz = crudos.tabla("jugador").alinear(np.array([10]), ["passes", "inventada"])
        assert not np.isnan(matriz[0, 0]) and np.isnan(matriz[0, 1])

    def test_sin_ninguna_coincidencia_devuelve_none(self) -> None:
        """Si no casa nada, la tabla no es de este modelo: mejor no enseñar nada."""
        tabla = TablaCrudos(
            feat_names=["passes"],
            entity_ids=np.array([1]),
            valores=np.array([[10.0]]),
        )
        assert tabla.alinear(np.array([999]), ["passes"]) is None
        assert tabla.alinear(np.array([1]), ["inventada"]) is None

    def test_la_matriz_del_catalogo_encaja_con_el_modelo(
        self, crudos: CatalogoCrudos
    ) -> None:
        matriz = crudos.matriz("jugador", np.array([10, 20, 30]), ["passes", "npxg"])
        assert matriz.shape == (3, 2)


class TestDegradacion:
    def test_sin_bd_no_hay_valores_reales(self, tmp_path: Path) -> None:
        """La interfaz vuelve al z-score, que siempre está en el artefacto."""
        catalogo = CatalogoCrudos(tmp_path / "no_existe.db")
        assert catalogo.tabla("jugador") is None
        assert catalogo.matriz("jugador", np.array([10]), ["passes"]) is None

    def test_una_bd_sin_las_tablas_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "vacia.db"
        with sqlite3.connect(ruta) as conn:
            conn.execute("CREATE TABLE otra (x INTEGER)")
        assert CatalogoCrudos(ruta).tabla("jugador") is None

    def test_un_fichero_que_no_es_sqlite_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "basura.db"
        ruta.write_text("esto no es una base de datos")
        assert CatalogoCrudos(ruta).tabla("equipo") is None

    def test_una_entidad_desconocida_no_rompe(self, crudos: CatalogoCrudos) -> None:
        assert crudos.tabla("arbitro") is None


class TestCache:
    def test_no_vuelve_a_derivar_en_cada_consulta(
        self, bd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Derivar las observaciones de jugador tarda; no se hace por petición."""
        catalogo = CatalogoCrudos(bd)
        catalogo.tabla("jugador")
        monkeypatch.setattr(
            catalogo, "_cargar",
            lambda entidad: (_ for _ in ()).throw(AssertionError("recargo")),
        )
        assert catalogo.tabla("jugador") is not None

    def test_cachea_cada_entidad_por_separado(self, bd: Path) -> None:
        catalogo = CatalogoCrudos(bd)
        assert catalogo.tabla("jugador") is not None
        assert catalogo.tabla("equipo") is not None

    def test_recarga_si_la_bd_cambia(self, bd: Path, tmp_path: Path) -> None:
        ruta = tmp_path / "copia.db"
        ruta.write_bytes(bd.read_bytes())
        catalogo = CatalogoCrudos(ruta)
        assert _valor(catalogo, "jugador", 30, "passes") == pytest.approx(20.0)
        with sqlite3.connect(ruta) as conn:
            conn.execute("UPDATE player_match_stats SET passes = 40 WHERE player_id = 30")
        # Escritura y relectura caen en la misma marca de tiempo del sistema de
        # ficheros: se envejece a mano para probar la invalidación en sí.
        os.utime(ruta, (0, 0))
        assert _valor(catalogo, "jugador", 30, "passes") == pytest.approx(40.0)


class TestMediaPonderada:
    def test_los_nan_no_arrastran_la_media(self, tmp_path: Path) -> None:
        """Cada columna promedia sobre las filas en las que SÍ está definida."""
        from src.app.crudos import _media_ponderada

        X = np.array([[1.0, np.nan], [3.0, 8.0]])
        salida = _media_ponderada(X, np.array([1.0, 1.0]), np.array([0, 0]), 1)
        assert list(salida[0]) == pytest.approx([2.0, 8.0])

    def test_una_columna_sin_ningun_valor_queda_en_nan(self) -> None:
        from src.app.crudos import _media_ponderada

        X = np.array([[np.nan], [np.nan]])
        salida = _media_ponderada(X, np.array([1.0, 1.0]), np.array([0, 0]), 1)
        assert np.isnan(salida[0, 0])

    def test_pondera_por_la_masa(self) -> None:
        from src.app.crudos import _media_ponderada

        X = np.array([[10.0], [20.0]])
        salida = _media_ponderada(X, np.array([3.0, 1.0]), np.array([0, 0]), 1)
        assert salida[0, 0] == pytest.approx(12.5)
