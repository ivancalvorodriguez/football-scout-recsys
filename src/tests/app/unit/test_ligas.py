"""Traduccion de la clave de liga (`11-27`) a nombre legible."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.app.ligas import CatalogoLigas


class TestNombre:
    def test_traduce_competicion_y_temporada(self, bd_ligas: Path) -> None:
        assert CatalogoLigas(bd_ligas).nombre("11-27") == "La Liga 2015/2016"

    def test_una_clave_desconocida_se_devuelve_tal_cual(self, bd_ligas: Path) -> None:
        assert CatalogoLigas(bd_ligas).nombre("99-99") == "99-99"

    def test_sin_bd_cae_a_la_clave(self, tmp_path: Path) -> None:
        """La BD es opcional: la app solo necesita `outputs/modelo/`."""
        assert CatalogoLigas(tmp_path / "no_existe.db").nombre("11-27") == "11-27"

    def test_una_bd_sin_las_tablas_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "vacia.db"
        with sqlite3.connect(ruta) as conn:
            conn.execute("CREATE TABLE otra (x INTEGER)")
        assert CatalogoLigas(ruta).nombre("11-27") == "11-27"

    def test_un_fichero_que_no_es_sqlite_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "basura.db"
        ruta.write_text("esto no es una base de datos")
        assert CatalogoLigas(ruta).nombre("11-27") == "11-27"


class TestTexto:
    def test_une_las_ligas_de_la_entidad(self, bd_ligas: Path) -> None:
        texto = CatalogoLigas(bd_ligas).texto(("11-27", "7-27"))
        assert texto == "La Liga 2015/2016, Ligue 1 2015/2016"

    def test_sin_ligas_lo_dice(self, bd_ligas: Path) -> None:
        assert CatalogoLigas(bd_ligas).texto(()) == "sin liga registrada"


class TestCache:
    def test_no_reabre_la_bd_en_cada_consulta(self, bd_ligas: Path, monkeypatch) -> None:
        catalogo = CatalogoLigas(bd_ligas)
        catalogo.nombre("11-27")
        monkeypatch.setattr(
            catalogo, "_cargar", lambda: (_ for _ in ()).throw(AssertionError("recargo"))
        )
        assert catalogo.nombre("7-27") == "Ligue 1 2015/2016"

    def test_recarga_si_la_bd_cambia(self, tmp_path: Path) -> None:
        ruta = tmp_path / "scouting.db"
        with sqlite3.connect(ruta) as conn:
            conn.execute(
                "CREATE TABLE competitions (competition_id INTEGER PRIMARY KEY, "
                "competition_name TEXT, country_name TEXT, competition_gender TEXT)"
            )
            conn.execute(
                "CREATE TABLE seasons (season_id INTEGER PRIMARY KEY, season_name TEXT)"
            )
            conn.execute("INSERT INTO seasons VALUES (27, '2015/2016')")
        catalogo = CatalogoLigas(ruta)
        assert catalogo.nombre("11-27") == "11-27"
        with sqlite3.connect(ruta) as conn:
            conn.execute("INSERT INTO competitions VALUES (11, 'La Liga', 'Spain', 'male')")
        # Escritura y relectura caen en la misma marca de tiempo del sistema de
        # ficheros: se envejece a mano para probar la invalidacion en si.
        os.utime(ruta, (0, 0))
        assert catalogo.nombre("11-27") == "La Liga 2015/2016"
