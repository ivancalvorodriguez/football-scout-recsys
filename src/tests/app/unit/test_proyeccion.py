"""La capa que coloca en un modelo a quien solo esta en la base de datos.

`src/app/proyeccion.py` es el puente entre `src.similitud.foldin` (que sabe
proyectar) y la app (que tiene una BD elegida y un artefacto en disco). Casi todo
lo que hace es negarse cuando no puede hacerlo bien, y eso es justo lo que se
prueba aqui: **una respuesta calculada en el espacio equivocado no se distingue a
simple vista de una buena**.

Se usan modelos construidos con el pipeline real (`dir_modelos_reales`) porque no
hay forma de falsificar lo que la proyeccion necesita: las mu/sd del ajuste, el
estado warm y exactamente las columnas de la capa de features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.app import config as config_app
from src.app.catalogo import ClaveModelo
from src.app.proyeccion import CatalogoProyeccion
from src.similitud import foldin
from src.similitud.modelo import cargar_modelo
from src.tests.app.conftest import CONJUNTO_SLUG, EQUIPO_NUEVO, FICHAJE, JUGADORES


@pytest.fixture
def artefacto(dir_modelos_reales: Path) -> Path:
    return dir_modelos_reales / f"{ClaveModelo('jugador').stem}.npz"


@pytest.fixture
def modelo(dir_modelos_reales: Path):
    clave = ClaveModelo("jugador")
    return cargar_modelo(dir_modelos_reales, clave.formulacion, "jugador",
                         clave.normalizacion, distancia=clave.distancia)


@pytest.fixture
def bd_ampliada(bd: Path) -> Path:
    """La BD del conjunto ampliado: tiene a `FICHAJE`, que el modelo no tiene."""
    return bd.parent / config_app.SUBDIR_CONJUNTOS / CONJUNTO_SLUG / bd.name


class TestProyectar:
    def test_coloca_a_quien_no_esta_en_el_modelo(self, modelo, artefacto, bd_ampliada):
        proyectada = CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0])
        assert proyectada is not None
        assert proyectada.id == FICHAJE[0]
        assert proyectada.nombre == FICHAJE[1]
        assert proyectada.fidelidad == foldin.FIEL
        assert proyectada.n_observaciones == 1

    def test_puntua_contra_las_entidades_del_modelo(self, modelo, artefacto,
                                                    bd_ampliada):
        proyectada = CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0])
        top = proyectada.proyeccion.top(3)
        assert top
        assert {i for i, _ in top} <= {int(e) for e in modelo.entity_ids}

    def test_trae_su_vector_alineado_con_el_modelo(self, modelo, artefacto,
                                                   bd_ampliada):
        """El vector es lo que permite que la ficha, el radar y las coincidencias
        de una entidad proyectada se calculen con el mismo codigo que las demas:
        tiene que tener las mismas columnas."""
        proyectada = CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0])
        assert proyectada.display.shape == (len(modelo.feat_names),)
        assert np.isfinite(proyectada.display).all()

    def test_sabe_de_que_ligas_viene(self, modelo, artefacto, bd_ampliada):
        """Del artefacto no puede salir (no esta en el): sale de sus filas."""
        proyectada = CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0])
        assert proyectada.ligas == ("9-27",)   # `conftest.LIGA_NUEVA`

    def test_una_entidad_del_modelo_se_rechaza(self, modelo, artefacto, bd_ampliada):
        """Tiene su fila en S: proyectarla seria darle una respuesta peor."""
        with pytest.raises(foldin.EntidadNoProyectable, match="ya esta en el modelo"):
            CatalogoProyeccion(bd_ampliada).proyectar(
                modelo, artefacto, JUGADORES[0][0])

    def test_una_entidad_que_no_esta_en_la_bd_se_rechaza(self, modelo, artefacto,
                                                         bd_ampliada):
        with pytest.raises(foldin.EntidadNoProyectable, match="no tiene observaciones"):
            CatalogoProyeccion(bd_ampliada).proyectar(modelo, artefacto, 99999)


class TestCuandoNoSePuede:
    """Devolver None (no se puede con estos datos) y lanzar `EntidadNoProyectable`
    (el modelo no lo admite) son cosas distintas, y la interfaz las explica
    distinto: una se arregla entrenando, la otra no tiene arreglo."""

    def test_sin_base_de_datos(self, modelo, artefacto, tmp_path):
        catalogo = CatalogoProyeccion(tmp_path / "no_existe.db")
        assert catalogo.proyectar(modelo, artefacto, FICHAJE[0]) is None

    def test_sin_las_mu_sd_del_ajuste(self, modelo, artefacto, bd_ampliada):
        """Sin ellas solo se podria reestandarizar con los datos de la consulta,
        que es justo lo que invalidaria la comparacion."""
        modelo.meta.pop("estadisticas_normalizacion")
        assert CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0]) is None

    def test_sin_el_estado_warm_de_un_modelo_que_lo_necesita(
        self, modelo, artefacto, bd_ampliada, tmp_path
    ):
        """La F5 necesita el ancho de kernel del ajuste y NO se estima: con otro,
        la entidad nueva caeria en un espacio distinto del de las demas."""
        copia = tmp_path / artefacto.name
        copia.write_bytes(artefacto.read_bytes())
        copia.with_suffix(".json").write_bytes(
            artefacto.with_suffix(".json").read_bytes())
        assert CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, copia, FICHAJE[0]) is None

    def test_con_otro_espacio_de_features(self, modelo, artefacto, bd_ampliada):
        """Un modelo construido con otro `config` (p. ej. sin el bloque `pos_*`)
        tiene otras columnas: proyectar daria un vector que no es el suyo."""
        modelo.feat_names = list(modelo.feat_names)[:-1]
        assert CatalogoProyeccion(bd_ampliada).proyectar(
            modelo, artefacto, FICHAJE[0]) is None

    def test_una_bd_ilegible_no_tumba_la_busqueda(self, modelo, artefacto, tmp_path):
        """Mismo criterio que en `crudos`: esto es una capacidad extra."""
        rota = tmp_path / "rota.db"
        rota.write_text("esto no es sqlite", encoding="utf-8")
        assert CatalogoProyeccion(rota).proyectar(modelo, artefacto, 1) is None


class TestCache:
    def _contar(self, catalogo, monkeypatch) -> list[int]:
        """Cuenta cuantas veces se monta el espacio (lo caro de todo esto)."""
        veces: list[int] = []
        original = catalogo._preparar

        def espia(*args, **kwargs):
            veces.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(catalogo, "_preparar", espia)
        return veces

    def test_la_segunda_consulta_no_reconstruye_el_espacio(
        self, modelo, artefacto, bd_ampliada, monkeypatch
    ):
        """Derivar la BD entera y mapear sus observaciones cuesta segundos:
        hacerlo por consulta haria la funcion inusable."""
        catalogo = CatalogoProyeccion(bd_ampliada)
        veces = self._contar(catalogo, monkeypatch)
        catalogo.proyectar(modelo, artefacto, FICHAJE[0])
        catalogo.proyectar(modelo, artefacto, FICHAJE[0])
        assert len(veces) == 1

    def test_tocar_la_bd_invalida_lo_cacheado(
        self, modelo, artefacto, bd_ampliada, tmp_path, monkeypatch
    ):
        """Incorporar partidos con el servidor levantado tiene que notarse: la
        huella es (mtime, tamaño) de la BD y del artefacto, como en el resto de
        catalogos."""
        import os

        copia = tmp_path / "copia.db"
        copia.write_bytes(bd_ampliada.read_bytes())
        catalogo = CatalogoProyeccion(copia)
        veces = self._contar(catalogo, monkeypatch)
        catalogo.proyectar(modelo, artefacto, FICHAJE[0])
        estado = copia.stat()
        os.utime(copia, (estado.st_atime, estado.st_mtime + 10))
        assert catalogo.proyectar(modelo, artefacto, FICHAJE[0]) is not None
        assert len(veces) == 2

    def test_no_acumula_espacios_sin_limite(self, dir_modelos_reales, bd_ampliada,
                                            monkeypatch):
        """Cada espacio es la BD entera en float64: el caché es corto a proposito."""
        monkeypatch.setattr("src.app.proyeccion.MAX_EN_CACHE", 1)
        catalogo = CatalogoProyeccion(bd_ampliada)
        for entidad, objetivo in (("jugador", FICHAJE[0]), ("equipo", EQUIPO_NUEVO[0])):
            clave = ClaveModelo(entidad)
            otro = cargar_modelo(
                dir_modelos_reales, clave.formulacion, entidad,
                clave.normalizacion, distancia=clave.distancia)
            ruta = dir_modelos_reales / f"{clave.stem}.npz"
            assert catalogo.proyectar(otro, ruta, objetivo) is not None
        assert len(catalogo._cache) == 1


class TestConjuntoDeLaBD:
    def test_el_universo_trae_nombres_y_no_solo_ids(self, bd_ampliada):
        """El buscador necesita el nombre para poder ofrecer a quien no esta en
        el modelo; la cobertura, solo los ids. Salen de la misma consulta para
        que lo que se puede encontrar sea exactamente lo que se ha contado."""
        from src.app.cobertura import CatalogoUniverso

        universo = CatalogoUniverso(bd_ampliada)
        nombres = universo.nombres("jugador")
        assert nombres[FICHAJE[0]] == FICHAJE[1]
        assert universo.ids("jugador") == frozenset(nombres)

    def test_sin_bd_no_se_inventa_un_universo(self, tmp_path):
        from src.app.cobertura import CatalogoUniverso

        universo = CatalogoUniverso(tmp_path / "no_existe.db")
        assert universo.nombres("jugador") is None
        assert universo.ids("jugador") is None


def test_el_conjunto_ampliado_esta_declarado(bd: Path) -> None:
    """Salvaguarda del fixture: si el conjunto ampliado dejara de tener su
    `conjunto.json`, estas pruebas seguirian pasando midiendo otra cosa."""
    meta = json.loads(
        (bd.parent / config_app.SUBDIR_CONJUNTOS / CONJUNTO_SLUG /
         config_app.FICHERO_CONJUNTO).read_text(encoding="utf-8"))
    assert meta["slug"] == CONJUNTO_SLUG
