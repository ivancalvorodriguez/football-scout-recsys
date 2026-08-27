"""Elegir base de datos y recomendar a quien no esta en el modelo.

El caso que se reproduce aqui es el que motiva la funcion: un modelo entrenado
con unos datos y una BD que tiene MAS entidades (porque se han incorporado
partidos, o porque se elige otro conjunto en el desplegable). Antes, preguntar
por una de esas entidades daba «no encontrado» y no habia forma de que dejara de
darlo salvo reentrenar. Ahora se la coloca en el modelo al vuelo.

Estas pruebas van con `dir_modelos_reales` —modelos construidos con el pipeline
de verdad sobre la BD sintetica— y no con los artefactos escritos a mano del
conftest: proyectar exige el mismo espacio de features, las mu/sd del ajuste y el
estado warm, y todo eso solo lo produce el pipeline. Lo que se comprueba con los
artefactos a mano es lo contrario (que se explique por que NO se puede), en
`test_cobertura_vistas.py`.
"""

from __future__ import annotations

import pytest

from src.similitud import foldin
from src.tests.app.conftest import (
    CONJUNTO_SLUG,
    EQUIPO_NUEVO,
    FICHAJE,
    JUGADORES,
)

pytestmark = pytest.mark.integracion


def _texto(respuesta) -> str:
    return respuesta.get_data(as_text=True)


# --- Eleccion de base de datos ------------------------------------------------

class TestElegirDatos:
    def test_el_buscador_ofrece_los_conjuntos_disponibles(self, cliente_real) -> None:
        html = _texto(cliente_real.get("/"))
        assert 'name="datos"' in html
        assert CONJUNTO_SLUG in html

    def test_consultar_sobre_otro_conjunto_lo_deja_dicho(self, cliente_real) -> None:
        """Modelo y datos desparejados es legitimo, pero hay que avisarlo: cambia
        de donde sale cada mitad de la respuesta.

        En una sola frase —que modelo y que datos— y sin la coletilla de que se
        hara con quien solo este en los primeros: eso ya lo dice la respuesta
        misma, rotulada («Calculado fuera del modelo»), cuando toca.
        """
        html = " ".join(_texto(cliente_real.get(f"/?datos={CONJUNTO_SLUG}")).split())
        assert "Consultando sobre los datos" in html
        assert "se coloca en el modelo al vuelo" not in html

    def test_un_conjunto_que_no_existe_es_404(self, cliente_real) -> None:
        assert cliente_real.get("/?datos=inventado").status_code == 404

    def test_sin_pedir_datos_se_usa_el_del_modelo(self, cliente_real) -> None:
        datos = cliente_real.get(
            "/api/similares?entidad=jugador&nombre=Messi&k=1").get_json()
        assert datos["modelo"]["datos_consultados"] == datos["modelo"]["datos"]

    def test_la_api_dice_sobre_que_datos_ha_respondido(self, cliente_real) -> None:
        datos = cliente_real.get(
            f"/api/similares?entidad=jugador&nombre=Messi&k=1&datos={CONJUNTO_SLUG}"
        ).get_json()
        assert datos["modelo"]["datos_consultados"] == CONJUNTO_SLUG


# --- Recomendar a quien no esta en el modelo ----------------------------------

class TestProyectarJugador:
    """`FICHAJE` esta en el conjunto ampliado y no en el modelo (entrenado con la
    base). Es el jugador del ejemplo: se le tiene que poder recomendar."""

    def _pedir(self, cliente, **extra):
        params = {"entidad": "jugador", "id": FICHAJE[0],
                  "datos": CONJUNTO_SLUG, "k": 3, **extra}
        consulta = "&".join(f"{k}={v}" for k, v in params.items())
        return cliente.get(f"/similares?{consulta}")

    def test_devuelve_recomendaciones(self, cliente_real) -> None:
        respuesta = self._pedir(cliente_real)
        assert respuesta.status_code == 200
        html = _texto(respuesta)
        assert FICHAJE[1] in html
        # Y los candidatos son entidades del modelo, no de la nada.
        assert any(nombre in html for _, nombre in JUGADORES)

    def test_se_rotula_como_calculado_fuera_del_modelo(self, cliente_real) -> None:
        """La respuesta es util, pero es de otra clase: el usuario tiene que
        poder distinguirla de un top-k leido de la matriz."""
        html = _texto(self._pedir(cliente_real))
        assert "Calculado fuera del modelo" in html

    def test_tambien_por_nombre(self, cliente_real) -> None:
        respuesta = cliente_real.get(
            f"/similares?entidad=jugador&nombre={FICHAJE[1]}&datos={CONJUNTO_SLUG}&k=2")
        assert respuesta.status_code == 200
        assert "Calculado fuera del modelo" in _texto(respuesta)

    def test_con_la_bd_del_modelo_sigue_sin_existir(self, cliente_real) -> None:
        """No se inventa nadie: sobre la BD base ese jugador no esta en ningun
        sitio, ni en el modelo ni en los datos."""
        respuesta = cliente_real.get(
            f"/similares?entidad=jugador&nombre={FICHAJE[1]}")
        assert respuesta.status_code == 404

    def test_la_api_lo_marca_como_proyectado(self, cliente_real) -> None:
        datos = cliente_real.get(
            f"/api/similares?entidad=jugador&id={FICHAJE[0]}"
            f"&datos={CONJUNTO_SLUG}&k=3").get_json()
        assert datos["proyectada"] == foldin.FIEL
        assert datos["referencia"]["en_modelo"] is False
        assert datos["candidatos"] and all(
            c["en_modelo"] for c in datos["candidatos"])

    def test_trae_su_perfil_y_sus_coincidencias(self, cliente_real) -> None:
        """El radar y las coincidencias no son adornos opcionales: son lo que
        deja leer la recomendacion, y salen del vector derivado de la BD."""
        datos = cliente_real.get(
            f"/api/similares?entidad=jugador&id={FICHAJE[0]}"
            f"&datos={CONJUNTO_SLUG}&k=2").get_json()
        assert datos["perfil"] and any(
            v["percentil"] is not None for v in datos["perfil"])
        assert datos["rasgos"]
        assert all(c["coincidencias"] for c in datos["candidatos"])

    def test_su_ficha_tambien_se_puede_abrir(self, cliente_real) -> None:
        """Los resultados enlazan a la ficha de la referencia: si esa pagina
        fallara, la funcion estaria a medias."""
        respuesta = cliente_real.get(
            f"/jugador/{FICHAJE[0]}?datos={CONJUNTO_SLUG}")
        assert respuesta.status_code == 200
        html = _texto(respuesta)
        assert FICHAJE[1] in html and "Fuera del modelo" in html

    def test_la_ficha_en_json_lo_dice_igual(self, cliente_real) -> None:
        datos = cliente_real.get(
            f"/api/ficha/jugador/{FICHAJE[0]}?datos={CONJUNTO_SLUG}").get_json()
        assert datos["proyectada"] == foldin.FIEL
        assert datos["destacados"] and datos["flojos"]

    def test_un_id_que_no_esta_en_ninguna_parte_sigue_siendo_404(
        self, cliente_real
    ) -> None:
        assert cliente_real.get(
            f"/similares?entidad=jugador&id=99999&datos={CONJUNTO_SLUG}"
        ).status_code == 404


class TestProyectarEquipo:
    """El equipo tambien se proyecta, y desde los modelos v3 con fidelidad FIEL.

    Con la F2 la proyeccion del equipo era APROXIMADA (solo una direccion de la
    agregacion de W). Al promoverse el v238 de `equipo_v3` la entidad pasa a la
    F5, cuya etapa 1 esta definida fuera de muestra igual que la del jugador, asi
    que las dos entidades dan ya la misma clase de respuesta.
    """

    def test_devuelve_recomendaciones_marcadas_como_fieles(
        self, cliente_real
    ) -> None:
        datos = cliente_real.get(
            f"/api/similares?entidad=equipo&id={EQUIPO_NUEVO[0]}"
            f"&datos={CONJUNTO_SLUG}&k=2").get_json()
        assert datos["proyectada"] == foldin.FIEL
        assert datos["candidatos"]

    def test_la_pagina_lo_rotula_igual_que_al_jugador(self, cliente_real) -> None:
        html = _texto(cliente_real.get(
            f"/similares?entidad=equipo&id={EQUIPO_NUEVO[0]}"
            f"&datos={CONJUNTO_SLUG}&k=2"))
        assert "Calculado fuera del modelo" in html
        assert "sin reentrenar nada" in html


# --- Busqueda -----------------------------------------------------------------

class TestBusqueda:
    def test_el_autocompletado_ofrece_a_los_que_no_estan_en_el_modelo(
        self, cliente_real
    ) -> None:
        datos = cliente_real.get(
            f"/api/sugerencias?entidad=jugador&q=Fichaje&datos={CONJUNTO_SLUG}"
        ).get_json()
        nombres = {s["nombre"]: s["en_modelo"] for s in datos["sugerencias"]}
        assert nombres.get(FICHAJE[1]) is False

    def test_las_del_modelo_van_primero(self, cliente_real) -> None:
        """Lo normal es querer las que el modelo tiene ajustadas; las otras se
        ofrecen detras, no mezcladas."""
        datos = cliente_real.get(
            f"/api/sugerencias?entidad=jugador&q=a&datos={CONJUNTO_SLUG}").get_json()
        marcas = [s["en_modelo"] for s in datos["sugerencias"]]
        assert marcas == sorted(marcas, reverse=True)

    def test_sin_ese_conjunto_no_aparece(self, cliente_real) -> None:
        datos = cliente_real.get(
            "/api/sugerencias?entidad=jugador&q=Fichaje").get_json()
        assert datos["sugerencias"] == []
