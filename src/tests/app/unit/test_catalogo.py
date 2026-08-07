"""Descubrimiento, resolucion y cacheo de modelos servibles."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.app.catalogo import Catalogo, ClaveModelo, ModeloNoDisponible


class TestDisponibles:
    def test_encuentra_todas_las_combinaciones(self, dir_modelos: Path) -> None:
        claves = Catalogo(dir_modelos).disponibles()
        assert len(claves) == 8
        assert ClaveModelo("jugador", "5", "por_liga") in claves

    def test_un_directorio_inexistente_no_rompe(self, tmp_path: Path) -> None:
        assert Catalogo(tmp_path / "no_existe").disponibles() == []

    def test_ignora_un_npz_sin_su_json(self, tmp_path: Path, dir_modelos: Path) -> None:
        """Sin el JSON no hay nombres de entidad: el modelo no es consultable."""
        shutil.copy(dir_modelos / "formulacion5_jugador_por_liga.npz", tmp_path)
        assert Catalogo(tmp_path).disponibles() == []

    def test_ignora_ficheros_ajenos(self, tmp_path: Path) -> None:
        (tmp_path / "formulacion9_marciano_por_liga.npz").write_bytes(b"")
        (tmp_path / "formulacion9_marciano_por_liga.json").write_text("{}")
        (tmp_path / "notas.txt").write_text("nada")
        assert Catalogo(tmp_path).disponibles() == []

    def test_el_artefacto_sin_sufijo_cuenta_como_por_liga(self, tmp_path: Path) -> None:
        """Formato antiguo: `cargar_modelo` lo sirve como `por_liga` y el
        catalogo debe ofrecer exactamente esa combinacion, no otra."""
        (tmp_path / "formulacion2_equipo.npz").write_bytes(b"")
        (tmp_path / "formulacion2_equipo.json").write_text("{}")
        assert Catalogo(tmp_path).disponibles() == [ClaveModelo("equipo", "2", "por_liga")]

    def test_entidades_disponibles(self, dir_modelos: Path) -> None:
        assert Catalogo(dir_modelos).entidades_disponibles() == ["jugador", "equipo"]

    def test_opciones_filtra_por_entidad(self, dir_modelos: Path) -> None:
        opciones = Catalogo(dir_modelos).opciones("equipo")
        assert opciones and all(c.entidad == "equipo" for c in opciones)


class TestResolver:
    def test_completa_lo_no_especificado_con_la_preferencia(self, dir_modelos: Path) -> None:
        clave = Catalogo(dir_modelos).resolver("jugador")
        assert clave == ClaveModelo("jugador", "5", "por_liga")

    def test_respeta_lo_que_si_se_pide(self, dir_modelos: Path) -> None:
        clave = Catalogo(dir_modelos).resolver(
            "equipo", formulacion="2", normalizacion="global"
        )
        assert clave == ClaveModelo("equipo", "2", "global")

    def test_elige_entre_lo_que_existe_aunque_no_sea_lo_preferido(
        self, tmp_path: Path, dir_modelos: Path
    ) -> None:
        """Con solo la F2 construida, la app debe servir la F2 en vez de fallar."""
        for sufijo in (".npz", ".json"):
            shutil.copy(dir_modelos / f"formulacion2_jugador_global{sufijo}", tmp_path)
        assert Catalogo(tmp_path).resolver("jugador").formulacion == "2"

    def test_sin_modelos_de_esa_entidad(self, tmp_path: Path) -> None:
        with pytest.raises(ModeloNoDisponible, match="build"):
            Catalogo(tmp_path).resolver("jugador")

    def test_combinacion_pedida_inexistente(self, tmp_path: Path, dir_modelos: Path) -> None:
        for sufijo in (".npz", ".json"):
            shutil.copy(dir_modelos / f"formulacion5_jugador_global{sufijo}", tmp_path)
        with pytest.raises(ModeloNoDisponible):
            Catalogo(tmp_path).resolver("jugador", normalizacion="por_liga")


class TestObtener:
    def test_carga_el_modelo_pedido(self, dir_modelos: Path) -> None:
        catalogo = Catalogo(dir_modelos)
        modelo = catalogo.obtener(ClaveModelo("equipo", "5", "global"))
        assert modelo.entidad == "equipo"
        assert modelo.formulacion == "5"
        assert modelo.meta["normalizacion"] == "global"

    def test_la_segunda_llamada_reutiliza_la_cache(self, dir_modelos: Path) -> None:
        catalogo = Catalogo(dir_modelos)
        clave = ClaveModelo("jugador", "5", "por_liga")
        assert catalogo.obtener(clave) is catalogo.obtener(clave)

    def test_recarga_si_el_artefacto_cambia(self, tmp_path: Path, dir_modelos: Path) -> None:
        """Reconstruir los modelos con el servidor levantado debe notarse."""
        for sufijo in (".npz", ".json"):
            shutil.copy(dir_modelos / f"formulacion5_equipo_por_liga{sufijo}", tmp_path)
        catalogo = Catalogo(tmp_path)
        clave = ClaveModelo("equipo", "5", "por_liga")
        primero = catalogo.obtener(clave)
        npz = tmp_path / "formulacion5_equipo_por_liga.npz"
        datos = dict(np.load(npz))
        np.savez_compressed(npz, **datos)
        # Rescribir y volver a leer ocurre dentro de la misma marca de tiempo del
        # sistema de ficheros, asi que se envejece a mano: lo que se prueba es
        # que un mtime distinto invalida la cache, no la resolucion del reloj.
        os.utime(npz, (0, 0))
        assert catalogo.obtener(clave) is not primero

    def test_artefacto_ausente(self, tmp_path: Path) -> None:
        with pytest.raises(ModeloNoDisponible):
            Catalogo(tmp_path).obtener(ClaveModelo("jugador", "5", "por_liga"))


class TestClaveModelo:
    def test_la_etiqueta_describe_las_tres_dimensiones(self) -> None:
        etiqueta = ClaveModelo("jugador", "5", "global").etiqueta
        assert "Jugador" in etiqueta and "F5" in etiqueta and "global" in etiqueta
