"""Datos auxiliares de la BD: rol grueso, minutos y estratos por volumen.

Son el ground truth DEBIL del harness (la posicion no entra como feature en el
modelo, ver `docs/modelo_similitud.md`): la pureza posicional de la Fase 0, el
k-NN de la Fase 5 y la estratificacion de la Fase 1 se apoyan en ellos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluacion import config, datos
from src.tests.evaluacion.conftest import bd_con_posiciones


class TestRolPorJugador:
    def test_agrupa_las_posiciones_finas_en_roles_gruesos(self, bd_roles: Path) -> None:
        """Con las ~24 posiciones finas la pureza seria absurdamente estricta: un
        "Left Center Back" y un "Right Center Back" son el mismo rol.
        """
        assert datos.rol_por_jugador(bd_roles)[1] == "GK"

    def test_elige_el_rol_donde_acumula_mas_minutos(self, bd_roles: Path) -> None:
        """El jugador 2 jugo 80' de central y 20' de extremo: es un defensa."""
        assert datos.rol_por_jugador(bd_roles)[2] == "DEF"

    def test_suma_los_minutos_de_posiciones_que_caen_en_el_mismo_rol(
        self, bd_roles: Path
    ) -> None:
        """El jugador 3 reparte 45'+30' entre dos posiciones, ambas MID."""
        assert datos.rol_por_jugador(bd_roles)[3] == "MID"

    def test_una_posicion_fuera_del_mapeo_queda_como_desconocida(
        self, bd_roles: Path
    ) -> None:
        """No se inventa un rol: las fases descartan estas entidades del calculo."""
        assert datos.rol_por_jugador(bd_roles)[4] == config.ROL_DESCONOCIDO

    def test_devuelve_una_entrada_por_jugador(self, bd_roles: Path) -> None:
        assert set(datos.rol_por_jugador(bd_roles)) == {1, 2, 3, 4}

    def test_las_claves_son_enteros_y_los_valores_texto(self, bd_roles: Path) -> None:
        """Se cruzan con `modelo.entity_ids` (numpy): el tipo tiene que casar."""
        roles = datos.rol_por_jugador(bd_roles)
        assert all(isinstance(k, int) and isinstance(v, str) for k, v in roles.items())

    def test_una_posicion_nula_tambien_cae_en_desconocida(self, tmp_path: Path) -> None:
        """Pasa en la BD real con jugadores sin alineacion utilizable."""
        bd = bd_con_posiciones(tmp_path / "sin_pos.db", [(9, 1, None, 45.0)])
        assert datos.rol_por_jugador(bd)[9] == config.ROL_DESCONOCIDO

    def test_no_escribe_en_la_base_de_datos(self, bd_roles: Path) -> None:
        """Se abre en modo `ro`: el harness nunca puede tocar los datos que evalua."""
        antes = bd_roles.stat().st_mtime_ns
        datos.rol_por_jugador(bd_roles)
        assert bd_roles.stat().st_mtime_ns == antes


class TestMinutosPorJugador:
    def test_suma_los_minutos_de_todos_los_partidos(self, bd_roles: Path) -> None:
        assert datos.minutos_por_jugador(bd_roles)[1] == pytest.approx(180.0)

    def test_suma_tambien_los_repartidos_entre_posiciones(self, bd_roles: Path) -> None:
        assert datos.minutos_por_jugador(bd_roles)[2] == pytest.approx(100.0)

    def test_cubre_a_los_mismos_jugadores_que_el_rol(self, bd_roles: Path) -> None:
        assert set(datos.minutos_por_jugador(bd_roles)) == set(
            datos.rol_por_jugador(bd_roles))

    def test_devuelve_floats(self, bd_roles: Path) -> None:
        assert all(isinstance(v, float) for v in datos.minutos_por_jugador(bd_roles).values())


class TestTerciles:
    def test_reparte_en_cola_torso_y_cabeza(self) -> None:
        etiquetas = datos.terciles(np.arange(9.0))
        assert set(etiquetas.tolist()) == {0, 1, 2}

    def test_los_mayores_caen_en_la_cabeza(self) -> None:
        """La estratificacion existe para no sobreestimar en los de mas volumen
        (sesgo MNAR): la cabeza tiene que ser la cabeza.
        """
        valores = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
        etiquetas = datos.terciles(valores)
        assert etiquetas[-1] == 2
        assert etiquetas[0] == 0

    def test_las_etiquetas_no_decrecen_con_el_valor(self) -> None:
        rng = np.random.default_rng(0)
        valores = np.sort(rng.uniform(0, 1000, size=50))
        etiquetas = datos.terciles(valores)
        assert np.all(np.diff(etiquetas) >= 0)

    def test_un_vector_vacio_devuelve_un_vector_vacio(self) -> None:
        assert datos.terciles(np.array([])).size == 0

    def test_con_todos_los_valores_iguales_no_hay_estratos(self) -> None:
        """Los cortes coinciden y ningun valor los supera: todo queda en cola."""
        assert datos.terciles(np.ones(5)).tolist() == [0] * 5

    def test_devuelve_una_etiqueta_por_valor(self) -> None:
        assert datos.terciles(np.arange(7.0)).shape == (7,)

    def test_los_nombres_cubren_las_tres_etiquetas(self) -> None:
        assert set(datos.NOMBRE_TERCIL) == {0, 1, 2}
        assert datos.NOMBRE_TERCIL[2] == "cabeza"
