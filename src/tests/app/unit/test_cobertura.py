"""Coherencia entre lo que cubre un modelo y lo que tiene su base de datos.

El bug que motiva el modulo es un SILENCIO: un modelo ajustado sobre una BD mas
pequena se sirve sin quejarse y las entidades que le faltan sencillamente no
aparecen. Asi que aqui se comprueba sobre todo que la incoherencia se DETECTA y
se pone en palabras, y que la ausencia de datos no se confunde con coherencia.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from src.app.cobertura import CatalogoUniverso, Cobertura, ids_de_artefacto, medir


# --- medir --------------------------------------------------------------------
def test_modelo_que_cubre_la_bd_entera_es_coherente():
    c = medir("jugador", [10, 20, 30], frozenset({10, 20, 30}))
    assert (c.en_modelo, c.en_datos, c.cubiertas) == (3, 3, 3)
    assert c.faltan == 0 and c.ajenas == 0
    assert c.coherente and not c.incompleta
    assert c.texto == ""


def test_modelo_al_que_le_faltan_entidades():
    """El caso de produccion: la BD crecio y el modelo se quedo atras."""
    c = medir("jugador", [10, 20], frozenset({10, 20, 30, 40}))
    assert (c.cubiertas, c.faltan) == (2, 4 - 2)
    assert c.incompleta and not c.coherente
    assert c.texto == "el modelo cubre 2 de 4 jugadores de estos datos"


def test_modelo_con_entidades_que_la_bd_no_tiene():
    """Sobrar no impide buscar, asi que NO es «incompleta»; pero si se cuenta."""
    c = medir("equipo", [1, 2, 3], frozenset({1, 2}))
    assert (c.ajenas, c.faltan) == (1, 0)
    assert not c.incompleta
    assert not c.coherente
    assert c.texto == "1 equipos del modelo no están en estos datos"


def test_las_dos_direcciones_a_la_vez_se_dicen_las_dos():
    c = medir("jugador", [10, 99], frozenset({10, 20, 30}))
    assert "cubre 1 de 3 jugadores" in c.texto
    assert "1 jugadores del modelo no están" in c.texto


@pytest.mark.parametrize("ids_datos, ids_modelo", [(None, [1, 2]), (frozenset({1}), None)])
def test_sin_uno_de_los_dos_lados_no_se_afirma_nada(ids_datos, ids_modelo):
    """Sin BD (o sin artefacto legible) callar es lo correcto: no hay comparacion.

    Es distinto de «no falta nadie»: `coherente` sale True para que la interfaz
    no pinte avisos, pero `medible` deja claro que no se ha medido.
    """
    c = medir("jugador", ids_modelo, ids_datos)
    assert not c.medible
    assert c.coherente and not c.incompleta
    assert c.texto == ""


def test_una_bd_vacia_si_es_un_dato():
    """Un conjunto SIN estadisticas no es lo mismo que un conjunto ilegible."""
    c = medir("equipo", [1, 2], frozenset())
    assert c.medible
    assert c.ajenas == 2 and c.faltan == 0


def test_el_plural_sale_del_catalogo_de_etiquetas():
    assert medir("jugador", [], frozenset()).plural == "jugadores"
    assert medir("equipo", [], frozenset()).plural == "equipos"


# --- ids_de_artefacto ---------------------------------------------------------
def test_ids_de_artefacto_no_carga_la_matriz(tmp_path: Path):
    """Se leen los ids sin tocar la S: es lo que permite avisar de cada modelo.

    La S del artefacto real de jugador ocupa 51 MB; la portada y la pagina de
    datos preguntan por VARIOS modelos en cada carga.
    """
    ruta = tmp_path / "m.npz"
    np.savez_compressed(ruta, S=np.zeros((3, 3)), entity_ids=np.array([7, 8, 9]))
    assert ids_de_artefacto(ruta) == [7, 8, 9]


def test_ids_de_artefacto_de_un_modelo_de_verdad(dir_modelos: Path):
    from src.app.catalogo import ClaveModelo

    ruta = dir_modelos / f"{ClaveModelo('equipo').stem}.npz"
    assert ids_de_artefacto(ruta) == [1, 2, 3]


@pytest.mark.parametrize("contenido", [None, b"esto no es un npz"])
def test_ids_de_artefacto_ilegible_es_none(tmp_path: Path, contenido):
    ruta = tmp_path / "roto.npz"
    if contenido is not None:
        ruta.write_bytes(contenido)
    assert ids_de_artefacto(ruta) is None


def test_ids_de_artefacto_sin_entity_ids_es_none(tmp_path: Path):
    """Un `.npz` de otra cosa no debe reventar la pagina."""
    ruta = tmp_path / "otro.npz"
    np.savez_compressed(ruta, algo=np.zeros(3))
    assert ids_de_artefacto(ruta) is None


# --- CatalogoUniverso ---------------------------------------------------------
def test_universo_cuenta_como_lo_haria_el_pipeline(bd: Path):
    """La tabla que manda es la de estadisticas, no `players`.

    En la BD sintetica el jugador 40 esta dado de alta pero no jugo ningun
    partido: `similitud.data` no lo veria, asi que contarlo daria un aviso falso
    permanente («falta el 40») que nadie podria arreglar reentrenando.
    """
    universo = CatalogoUniverso(bd)
    assert 40 not in universo.ids("jugador")
    assert universo.ids("jugador") == frozenset({10, 20, 30, 50, 60, 70})
    assert universo.ids("equipo") == frozenset({1, 2, 3})


def test_universo_sin_bd_es_none(tmp_path: Path):
    assert CatalogoUniverso(tmp_path / "no_existe.db").ids("jugador") is None


def test_universo_de_un_fichero_que_no_es_sqlite(tmp_path: Path):
    ruta = tmp_path / "falsa.db"
    ruta.write_text("no soy una base de datos", encoding="utf-8")
    assert CatalogoUniverso(ruta).ids("jugador") is None


def test_universo_de_una_entidad_desconocida_es_none(bd: Path):
    assert CatalogoUniverso(bd).ids("arbitro") is None


def test_universo_se_relee_si_la_bd_cambia(bd: Path, tmp_path: Path):
    """Ampliar la BD con el servidor levantado tiene que notarse sin reiniciar."""
    copia = tmp_path / "scouting.db"
    copia.write_bytes(bd.read_bytes())
    universo = CatalogoUniverso(copia)
    assert 999 not in universo.ids("equipo")

    with closing(sqlite3.connect(copia)) as conn:
        conn.execute("INSERT INTO teams (team_id, team_name) VALUES (999, 'Nuevo')")
        conn.execute(
            "INSERT INTO team_match_stats (team_id, match_id, passes) VALUES (999, 1, 1)"
        )
        conn.commit()
    # `st_mtime` tiene resolucion de segundos en algunos sistemas de ficheros: se
    # fuerza la huella en vez de dormir un segundo dentro de la prueba.
    import os
    estado = copia.stat()
    os.utime(copia, ns=(estado.st_atime_ns, estado.st_mtime_ns + 2_000_000_000))

    assert 999 in universo.ids("equipo")


def test_el_universo_se_cachea_entre_llamadas(bd: Path, monkeypatch: pytest.MonkeyPatch):
    """La comparacion es por peticion; la consulta a la BD, no."""
    universo = CatalogoUniverso(bd)
    universo.ids("jugador")
    monkeypatch.setattr(
        universo, "_consultar",
        lambda entidad: pytest.fail("no deberia volver a consultar la BD"),
    )
    assert universo.ids("jugador") == frozenset({10, 20, 30, 50, 60, 70})


# --- Composicion --------------------------------------------------------------
def test_cobertura_de_un_modelo_real_contra_su_bd(dir_modelos: Path, bd: Path):
    """El modelo sintetico base incluye al jugador 40, que la BD no respalda."""
    from src.app.catalogo import ClaveModelo

    c = medir(
        "jugador",
        ids_de_artefacto(dir_modelos / f"{ClaveModelo('jugador').stem}.npz"),
        CatalogoUniverso(bd).ids("jugador"),
    )
    assert isinstance(c, Cobertura)
    assert c.en_datos == 6 and c.cubiertas == 6 and c.ajenas == 1
    assert not c.incompleta
