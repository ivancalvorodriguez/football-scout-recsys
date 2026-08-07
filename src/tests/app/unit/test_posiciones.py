"""Catálogo de posiciones para el campo de la interfaz."""

from __future__ import annotations

import itertools
import math

from src.app import posiciones
from src.app.contexto import RADIO_MARCA
from src.extraccion.config import POSITIONS_25, position_slug


class TestCatalogo:
    def test_cubre_las_25_posiciones_de_statsbomb(self) -> None:
        """Si faltase una, el campo se dibujaría incompleto en silencio."""
        assert {p.nombre for p in posiciones.POSICIONES} == set(POSITIONS_25.values())

    def test_los_slugs_coinciden_con_las_columnas_del_modelo(self) -> None:
        """`pos_left_back` tiene que ser el mismo slug que genera la extracción."""
        assert all(
            p.slug == position_slug(p.nombre) for p in posiciones.POSICIONES
        )

    def test_cada_posicion_cae_dentro_del_lienzo(self) -> None:
        assert all(
            0.0 <= p.x <= 100.0 and 0.0 <= p.y <= 100.0
            for p in posiciones.POSICIONES
        )

    def test_los_roles_son_los_cuatro_conocidos(self) -> None:
        assert {p.rol for p in posiciones.POSICIONES} <= set(posiciones.ETIQUETA_ROL)

    def test_el_portero_esta_detras_y_el_delantero_delante(self) -> None:
        """El campo se dibuja atacando hacia arriba (y=0 es la portería rival)."""
        portero = posiciones.POR_NOMBRE["Goalkeeper"]
        delantero = posiciones.POR_NOMBRE["Center Forward"]
        assert portero.y > delantero.y

    def test_la_banda_izquierda_va_a_la_izquierda(self) -> None:
        izquierdo = posiciones.POR_NOMBRE["Left Back"]
        derecho = posiciones.POR_NOMBRE["Right Back"]
        assert izquierdo.x < 50.0 < derecho.x

    def test_ninguna_marca_del_campo_se_pisa_con_otra(self) -> None:
        """Las marcas son todas del mismo tamaño, así que basta con la distancia.

        Hay parejas que un mismo jugador combina a menudo (lateral y carrilero,
        delantero centro y segundo delantero): si sus círculos se solapasen, los
        porcentajes de dentro quedarían ilegibles justo en el caso interesante.
        """
        minima = min(
            math.dist((a.x, a.y), (b.x, b.y))
            for a, b in itertools.combinations(posiciones.POSICIONES, 2)
        )
        assert 2 * RADIO_MARCA <= minima


class TestEtiquetas:
    def test_traduce_el_slug_al_espanol(self) -> None:
        assert posiciones.etiqueta_de_slug("pos_left_back") == "Lateral izquierdo"

    def test_un_slug_desconocido_cae_a_algo_legible(self) -> None:
        """Una posición nueva de StatsBomb no puede tumbar la interfaz."""
        assert posiciones.etiqueta_de_slug("pos_sweeper_keeper") == "Sweeper keeper"

    def test_por_nombre_devuelve_none_si_no_existe(self) -> None:
        assert posiciones.por_nombre("Libero") is None
