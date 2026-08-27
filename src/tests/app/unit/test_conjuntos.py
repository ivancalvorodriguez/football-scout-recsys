"""Conjuntos de datos: descubrimiento, resolucion y alta de uno nuevo.

Lo que de verdad importa aqui es que crear un conjunto NO toque al de origen:
todo el planteamiento (modelos que siguen casando con sus datos, poder comparar
antes y despues) se apoya en eso.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.conjuntos import CatalogoDatos, ConjuntoNoDisponible
from src.app.nombres import NombreInvalido


def _partidos(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


# Cuenta con la que se consultan los catalogos: lo que se crea en un `tmp_path`
# no tiene dueno apuntado y es COMPARTIDO, asi que se ve desde cualquiera. Lo
# que cambia es que enumerar exige decir quien pregunta.
USUARIO = "ana"

@pytest.fixture
def catalogo(bd: Path, tmp_path: Path) -> CatalogoDatos:
    """Catalogo sobre una COPIA de la BD sintetica: aqui se escribe."""
    copia = tmp_path / "db" / "scouting.db"
    copia.parent.mkdir(parents=True)
    copia.write_bytes(bd.read_bytes())
    return CatalogoDatos(copia)


class TestDescubrimiento:
    def test_solo_el_base_al_principio(self, catalogo: CatalogoDatos) -> None:
        conjuntos = catalogo.conjuntos(USUARIO)
        assert [c.slug for c in conjuntos] == [config_app.VARIANTE_BASE]
        assert conjuntos[0].es_base and conjuntos[0].existe

    def test_el_base_se_lista_aunque_no_haya_bd(self, tmp_path: Path) -> None:
        """Es el destino de la extraccion: tiene que salir para poder explicarlo."""
        conjuntos = CatalogoDatos(tmp_path / "no_existe.db").conjuntos(USUARIO)
        assert [c.slug for c in conjuntos] == [config_app.VARIANTE_BASE]
        assert not conjuntos[0].existe
        assert CatalogoDatos(tmp_path / "no_existe.db").disponibles(USUARIO) == []

    def test_el_base_va_primero(self, catalogo: CatalogoDatos) -> None:
        catalogo.crear("Ampliado", USUARIO)
        assert catalogo.conjuntos(USUARIO)[0].es_base

    def test_sin_metadatos_se_cae_al_nombre_de_la_carpeta(
        self, catalogo: CatalogoDatos
    ) -> None:
        carpeta = catalogo.dir_conjuntos / "a-mano"
        carpeta.mkdir(parents=True)
        (carpeta / "scouting.db").write_bytes(catalogo.db_path.read_bytes())
        (carpeta / config_app.FICHERO_CONJUNTO).write_text("{ roto", encoding="utf-8")
        assert catalogo.conjunto("a-mano", USUARIO).nombre == "a-mano"

    def test_una_carpeta_sin_bd_se_lista_pero_no_esta_disponible(
        self, catalogo: CatalogoDatos
    ) -> None:
        """Es la incorporacion en curso (o fallida)."""
        (catalogo.dir_conjuntos / "a-medias").mkdir(parents=True)
        assert not catalogo.conjunto("a-medias", USUARIO).existe
        assert [c.slug for c in catalogo.disponibles(USUARIO)] == [config_app.VARIANTE_BASE]

    def test_un_slug_inventado(self, catalogo: CatalogoDatos) -> None:
        with pytest.raises(ConjuntoNoDisponible):
            catalogo.conjunto("fantasma", USUARIO)


class TestResolver:
    def test_sin_nada_pedido_es_el_base(self, catalogo: CatalogoDatos) -> None:
        assert catalogo.resolver(None).es_base

    def test_lo_pedido_manda(self, catalogo: CatalogoDatos) -> None:
        catalogo.crear("Ampliado", USUARIO)
        assert catalogo.resolver("ampliado").nombre == "Ampliado"

    def test_un_conjunto_desaparecido_cae_al_base(self, catalogo: CatalogoDatos) -> None:
        """Lo pide un MODELO: mejor servirlo sin algun adorno que no servirlo."""
        assert catalogo.resolver("borrado-a-mano").es_base


class TestCrear:
    def test_copia_la_bd_del_origen(self, catalogo: CatalogoDatos) -> None:
        nuevo = catalogo.crear("Ampliado", USUARIO)
        assert nuevo.existe
        assert _partidos(nuevo.ruta) == _partidos(catalogo.db_path)

    def test_el_origen_no_se_toca(self, catalogo: CatalogoDatos) -> None:
        """Toda la idea: los modelos ya entrenados siguen casando con sus datos."""
        antes = catalogo.db_path.stat().st_mtime_ns
        nuevo = catalogo.crear("Ampliado", USUARIO)
        # Escribir en la copia no puede alcanzar al original.
        with closing(sqlite3.connect(nuevo.ruta)) as conn:
            conn.execute("DELETE FROM matches")
            conn.commit()
        assert _partidos(nuevo.ruta) == 0
        assert _partidos(catalogo.db_path) == 4
        assert catalogo.db_path.stat().st_mtime_ns == antes

    def test_deja_sus_metadatos(self, catalogo: CatalogoDatos) -> None:
        nuevo = catalogo.crear("Con la Bundesliga", USUARIO, paquete="open-data/data")
        meta = json.loads(
            (nuevo.ruta.parent / config_app.FICHERO_CONJUNTO).read_text("utf-8"))
        assert meta["nombre"] == "Con la Bundesliga"
        assert meta["slug"] == "con-la-bundesliga"
        assert meta["origen"] == config_app.VARIANTE_BASE
        assert meta["paquete"] == "open-data/data"

    def test_se_puede_encadenar_desde_otro_conjunto(self, catalogo: CatalogoDatos) -> None:
        catalogo.crear("Primero", USUARIO)
        segundo = catalogo.crear("Segundo", USUARIO, origen="primero")
        assert segundo.origen == "primero"

    def test_un_nombre_repetido_se_rechaza(self, catalogo: CatalogoDatos) -> None:
        catalogo.crear("Ampliado", USUARIO)
        with pytest.raises(NombreInvalido, match="Ya hay"):
            catalogo.crear("ampliado", USUARIO)

    def test_no_se_puede_llamar_como_el_de_fabrica(self, catalogo: CatalogoDatos) -> None:
        with pytest.raises(NombreInvalido, match="fábrica"):
            catalogo.crear("Base", USUARIO)

    @pytest.mark.parametrize("nombre", ["", "   ", "¿?"])
    def test_un_nombre_vacio_se_rechaza(self, catalogo: CatalogoDatos, nombre: str) -> None:
        with pytest.raises(NombreInvalido):
            catalogo.crear(nombre, USUARIO)

    def test_no_se_parte_de_un_conjunto_sin_bd(self, tmp_path: Path) -> None:
        catalogo = CatalogoDatos(tmp_path / "no_existe.db")
        with pytest.raises(NombreInvalido, match="no tiene base de datos"):
            catalogo.crear("Ampliado", USUARIO)

    def test_descartar_libera_el_nombre(self, catalogo: CatalogoDatos) -> None:
        catalogo.crear("Ampliado", USUARIO)
        catalogo.descartar("ampliado")
        assert [c.slug for c in catalogo.conjuntos(USUARIO)] == [config_app.VARIANTE_BASE]
        catalogo.crear("Ampliado", USUARIO)      # el nombre vuelve a estar libre

    def test_el_base_no_se_descarta(self, catalogo: CatalogoDatos) -> None:
        with pytest.raises(ValueError):
            catalogo.descartar(config_app.VARIANTE_BASE)
