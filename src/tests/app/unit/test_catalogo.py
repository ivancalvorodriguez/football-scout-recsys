"""Descubrimiento, resolucion, alta y cacheo de los modelos servibles."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.app import config as config_app
from src.app.catalogo import (
    Catalogo,
    ClaveModelo,
    ModeloNoDisponible,
    NombreInvalido,
    slug,
)
from src.tests.app.conftest import VARIANTE_NOMBRE, VARIANTE_SLUG

# La pareja que sirve la app: jugador con una formulacion, equipo con otra.
STEM_JUGADOR = ClaveModelo("jugador").stem
STEM_EQUIPO = ClaveModelo("equipo").stem


def _copiar(origen: Path, destino: Path, stem: str) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for sufijo in (".npz", ".json"):
        shutil.copy(origen / f"{stem}{sufijo}", destino)


class TestSlug:
    def test_simplifica_a_nombre_de_carpeta(self) -> None:
        assert slug("Con la Bundesliga 15/16") == "con-la-bundesliga-15-16"

    def test_quita_acentos_y_signos(self) -> None:
        assert slug("Modelo ñ ácido!!") == "modelo-n-acido"

    def test_un_nombre_sin_nada_utilizable_queda_vacio(self) -> None:
        assert slug("¿?¡!") == ""


class TestDisponibles:
    def test_encuentra_la_pareja_de_cada_modelo(self, dir_modelos: Path) -> None:
        claves = Catalogo(dir_modelos).disponibles()
        # 2 entidades x 2 modelos (el base y el reentrenado del fixture).
        assert len(claves) == 4
        assert ClaveModelo("jugador") in claves
        assert ClaveModelo("equipo", VARIANTE_SLUG) in claves

    def test_ignora_las_combinaciones_que_no_sirve(self, dir_modelos: Path) -> None:
        """En la raiz estan las 8 que construye `build`; se sirven 2 por modelo."""
        base = [c for c in Catalogo(dir_modelos).disponibles() if c.es_base]
        assert {(c.entidad, c.formulacion, c.normalizacion) for c in base} == {
            ("jugador", *config_app.MODELO_BASE["jugador"]),
            ("equipo", *config_app.MODELO_BASE["equipo"]),
        }

    def test_un_directorio_inexistente_no_rompe(self, tmp_path: Path) -> None:
        assert Catalogo(tmp_path / "no_existe").disponibles() == []

    def test_ignora_un_npz_sin_su_json(self, tmp_path: Path, dir_modelos: Path) -> None:
        """Sin el JSON no hay nombres de entidad: el modelo no es consultable."""
        shutil.copy(dir_modelos / f"{STEM_JUGADOR}.npz", tmp_path)
        assert Catalogo(tmp_path).disponibles() == []

    def test_ignora_ficheros_ajenos(self, tmp_path: Path) -> None:
        (tmp_path / "formulacion9_marciano_global.npz").write_bytes(b"")
        (tmp_path / "formulacion9_marciano_global.json").write_text("{}")
        (tmp_path / "notas.txt").write_text("nada")
        assert Catalogo(tmp_path).disponibles() == []

    def test_entidades_disponibles(self, dir_modelos: Path) -> None:
        assert Catalogo(dir_modelos).entidades_disponibles() == ["jugador", "equipo"]

    def test_opciones_filtra_por_entidad(self, dir_modelos: Path) -> None:
        opciones = Catalogo(dir_modelos).opciones("equipo")
        assert opciones and all(c.entidad == "equipo" for c in opciones)


class TestVariantes:
    def test_el_base_va_primero(self, dir_modelos: Path) -> None:
        """El desplegable tiene que abrir siempre por el modelo de fabrica."""
        variantes = Catalogo(dir_modelos).variantes()
        assert variantes[0].slug == config_app.VARIANTE_BASE
        assert variantes[0].es_base

    def test_la_reentrenada_trae_su_nombre_y_su_fecha(self, dir_modelos: Path) -> None:
        v = Catalogo(dir_modelos).variante(VARIANTE_SLUG)
        assert v.nombre == VARIANTE_NOMBRE
        assert v.origen == config_app.VARIANTE_BASE
        assert v.fecha is not None and v.fecha.year == 2026
        assert v.completo

    def test_sin_metadatos_se_cae_al_nombre_de_la_carpeta(
        self, tmp_path: Path, dir_modelos: Path
    ) -> None:
        """Un `variante.json` ilegible no puede esconder un modelo que existe."""
        carpeta = tmp_path / config_app.SUBDIR_VARIANTES / "a-mano"
        _copiar(dir_modelos, carpeta, STEM_JUGADOR)
        (carpeta / config_app.FICHERO_VARIANTE).write_text("{ roto", encoding="utf-8")
        v = Catalogo(tmp_path).variante("a-mano")
        assert v.nombre == "a-mano" and v.entidades == ("jugador",)

    def test_una_carpeta_sin_artefactos_se_lista_pero_no_se_sirve(
        self, tmp_path: Path
    ) -> None:
        """Es el reentrenamiento en curso (o fallido): la pagina lo enseña, el
        buscador no lo ofrece."""
        (tmp_path / config_app.SUBDIR_VARIANTES / "a-medias").mkdir(parents=True)
        catalogo = Catalogo(tmp_path)
        v = catalogo.variante("a-medias")
        assert v.entidades == () and not v.completo
        assert catalogo.disponibles() == []

    def test_variantes_de_una_entidad(self, tmp_path: Path, dir_modelos: Path) -> None:
        """Un modelo a medio reentrenar solo aparece en la entidad que ya cubre."""
        _copiar(dir_modelos, tmp_path, STEM_JUGADOR)
        _copiar(dir_modelos, tmp_path, STEM_EQUIPO)
        carpeta = tmp_path / config_app.SUBDIR_VARIANTES / "solo-jugador"
        _copiar(dir_modelos, carpeta, STEM_JUGADOR)
        catalogo = Catalogo(tmp_path)
        assert [v.slug for v in catalogo.variantes_de("jugador")] == ["base", "solo-jugador"]
        assert [v.slug for v in catalogo.variantes_de("equipo")] == ["base"]

    def test_un_slug_inventado(self, dir_modelos: Path) -> None:
        with pytest.raises(ModeloNoDisponible):
            Catalogo(dir_modelos).variante("fantasma")


class TestResolver:
    def test_sin_modelo_pedido_sirve_el_base(self, dir_modelos: Path) -> None:
        assert Catalogo(dir_modelos).resolver("jugador") == ClaveModelo("jugador")

    def test_respeta_el_modelo_pedido(self, dir_modelos: Path) -> None:
        clave = Catalogo(dir_modelos).resolver("equipo", variante=VARIANTE_SLUG)
        assert clave == ClaveModelo("equipo", VARIANTE_SLUG)

    def test_la_receta_la_fija_la_entidad(self, dir_modelos: Path) -> None:
        """El usuario elige modelo; con que se construye cada entidad, no."""
        clave = Catalogo(dir_modelos).resolver("jugador", variante=VARIANTE_SLUG)
        assert (clave.formulacion, clave.normalizacion) == config_app.MODELO_BASE["jugador"]

    def test_sin_base_se_sirve_el_primero_que_haya(
        self, tmp_path: Path, dir_modelos: Path
    ) -> None:
        """Con la raiz vacia y un modelo con nombre, la app tiene que servir ese."""
        carpeta = tmp_path / config_app.SUBDIR_VARIANTES / "unico"
        _copiar(dir_modelos, carpeta, STEM_JUGADOR)
        assert Catalogo(tmp_path).resolver("jugador").variante == "unico"

    def test_sin_modelos_de_esa_entidad(self, tmp_path: Path) -> None:
        with pytest.raises(ModeloNoDisponible, match="build"):
            Catalogo(tmp_path).resolver("jugador")

    def test_un_modelo_que_no_cubre_esa_entidad(
        self, tmp_path: Path, dir_modelos: Path
    ) -> None:
        _copiar(dir_modelos, tmp_path, STEM_JUGADOR)
        _copiar(dir_modelos, tmp_path, STEM_EQUIPO)
        carpeta = tmp_path / config_app.SUBDIR_VARIANTES / "solo-jugador"
        _copiar(dir_modelos, carpeta, STEM_JUGADOR)
        with pytest.raises(ModeloNoDisponible, match="artefacto de equipo"):
            Catalogo(tmp_path).resolver("equipo", variante="solo-jugador")


class TestObtener:
    def test_carga_el_modelo_pedido(self, dir_modelos: Path) -> None:
        modelo = Catalogo(dir_modelos).obtener(ClaveModelo("equipo"))
        assert modelo.entidad == "equipo"
        assert modelo.formulacion == config_app.MODELO_BASE["equipo"][0]
        assert modelo.meta["normalizacion"] == config_app.MODELO_BASE["equipo"][1]

    def test_cada_modelo_sirve_su_propio_universo(self, dir_modelos: Path) -> None:
        """Lo que aporta reentrenar: entidades que el base no conoce."""
        catalogo = Catalogo(dir_modelos)
        base = catalogo.obtener(ClaveModelo("jugador"))
        reentrenado = catalogo.obtener(ClaveModelo("jugador", VARIANTE_SLUG))
        assert "Fichaje Reciente" not in base.entity_names
        assert "Fichaje Reciente" in reentrenado.entity_names

    def test_la_segunda_llamada_reutiliza_la_cache(self, dir_modelos: Path) -> None:
        catalogo = Catalogo(dir_modelos)
        clave = ClaveModelo("jugador")
        assert catalogo.obtener(clave) is catalogo.obtener(clave)

    def test_recarga_si_el_artefacto_cambia(self, tmp_path: Path, dir_modelos: Path) -> None:
        """Reentrenar con el servidor levantado debe notarse."""
        _copiar(dir_modelos, tmp_path, STEM_EQUIPO)
        catalogo = Catalogo(tmp_path)
        clave = ClaveModelo("equipo")
        primero = catalogo.obtener(clave)
        npz = tmp_path / f"{STEM_EQUIPO}.npz"
        datos = dict(np.load(npz))
        np.savez_compressed(npz, **datos)
        # Rescribir y volver a leer ocurre dentro de la misma marca de tiempo del
        # sistema de ficheros, asi que se envejece a mano: lo que se prueba es
        # que un mtime distinto invalida la cache, no la resolucion del reloj.
        os.utime(npz, (0, 0))
        assert catalogo.obtener(clave) is not primero

    def test_artefacto_ausente(self, tmp_path: Path) -> None:
        with pytest.raises(ModeloNoDisponible):
            Catalogo(tmp_path).obtener(ClaveModelo("jugador"))


class TestCrearVariante:
    def test_reserva_la_carpeta_y_deja_sus_metadatos(self, tmp_path: Path) -> None:
        catalogo = Catalogo(tmp_path)
        nueva = catalogo.crear_variante("Con la Bundesliga", origen="base")
        assert nueva.slug == "con-la-bundesliga"
        carpeta = catalogo.dir_modelo(nueva.slug)
        assert carpeta.is_dir()
        meta = json.loads((carpeta / config_app.FICHERO_VARIANTE).read_text("utf-8"))
        assert meta["nombre"] == "Con la Bundesliga" and meta["origen"] == "base"

    def test_el_modelo_nuevo_es_el_destino_del_reentrenamiento(self, tmp_path: Path) -> None:
        """La carpeta existe antes de entrenar: es el `--out` del comando."""
        catalogo = Catalogo(tmp_path)
        nueva = catalogo.crear_variante("Prueba")
        assert catalogo.dir_modelo(nueva.slug).exists()
        # Y hasta que el entrenamiento deje artefactos, no se sirve.
        assert catalogo.opciones_de_variante(nueva.slug) == []

    def test_un_nombre_repetido_se_rechaza(self, tmp_path: Path) -> None:
        catalogo = Catalogo(tmp_path)
        catalogo.crear_variante("Prueba")
        with pytest.raises(NombreInvalido, match="Ya hay"):
            catalogo.crear_variante("prueba")     # mismo slug

    @pytest.mark.parametrize("nombre", ["", "   ", "¿?"])
    def test_un_nombre_vacio_se_rechaza(self, tmp_path: Path, nombre: str) -> None:
        with pytest.raises(NombreInvalido):
            Catalogo(tmp_path).crear_variante(nombre)

    def test_no_se_puede_llamar_como_el_de_fabrica(self, tmp_path: Path) -> None:
        with pytest.raises(NombreInvalido, match="fábrica"):
            Catalogo(tmp_path).crear_variante("Base")


class TestClaveModelo:
    def test_la_receta_sale_de_la_entidad(self) -> None:
        clave = ClaveModelo("jugador")
        assert (clave.formulacion, clave.normalizacion) == config_app.MODELO_BASE["jugador"]
        assert clave.es_base

    def test_la_etiqueta_describe_el_artefacto(self) -> None:
        etiqueta = ClaveModelo("jugador").etiqueta
        assert "Jugador" in etiqueta and "F" in etiqueta
