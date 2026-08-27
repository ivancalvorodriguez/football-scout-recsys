"""La config reproduce el modelo que la app sirve.

No es una prueba de calculo: es un candado. El boton «Entrenar» de `/datos` lanza
`src.incremental.reentrenar`, que ajusta la celda servible de cada entidad con
`config.HIPERPARAMETROS_SERVIBLES`; si esa tabla deja de coincidir con los
hiperparametros del artefacto servido, el usuario acaba con dos modelos de
familias distintas en el mismo desplegable, comparandolos como si fueran lo
mismo. El linaje completo esta en `outputs/modelo/produccion.json`, que no entra
en el repositorio (`outputs/` esta en `.gitignore`), asi que los valores se fijan
aqui.

Cambiarlos es legitimo —el barrido esta para eso—, pero entonces hay que promover
los artefactos correspondientes y actualizar estas cifras a la vez. Que la prueba
rompa es justo el aviso de que falta la otra mitad.
"""

from __future__ import annotations

import pytest

from src.similitud import config

# Combinaciones servidas: v91 de `outputs/evaluacion/jugador_v3` (promovida el
# 26-8-2026) y v349 de `outputs/evaluacion/equipo_v3`, las dos con distancia
# `manhattan`. Son las PRIMERAS del score compuesto de su barrido; ya no hay que
# excluir mahalanobis a mano al leerlo, porque esa geometria se retiro de la
# rejilla el mismo dia (deshace la ponderacion del bloque de posicion). El jugador
# v91 sustituye a v87 (`F5_RFF_DIM` 1024), en `outputs/modelo_historico_20260826/`;
# antes fueron v83 y v238 (euclideas), en `outputs/modelo_historico_20260825/`.
V91 = {"F5_METODO_JUGADOR": "mmd", "F5_RFF_DIM": 1536, "F5_EASE_LAMBDA": 0.0}
V349 = {"F5_METODO_EQUIPO": "mmd", "F5_RFF_DIM": 1216, "F5_EASE_LAMBDA": 0.03}


def test_la_app_sirve_las_dos_combinaciones_promovidas():
    """Jugador F5 + z-score por liga; equipo F5 + z-score global.

    Las dos son F5 (el equipo dejo la F2) y NO comparten normalizacion: la eligio
    la medida, no la simetria, y la interfaz la rotula en cada respuesta.
    """
    assert dict(config.MODELOS_SERVIBLES) == {
        "jugador": ("5", "por_liga"),
        "equipo": ("5", "global"),
    }


def test_la_geometria_servida_es_la_promovida():
    """La distancia tambien es parte de lo promovido, y no es por entidad.

    Importa mas de lo que parece: la geometria pone el SUFIJO del nombre de
    fichero (`modelo.stem_artefacto`), asi que moverla cambia que artefacto busca
    la app en `outputs/modelo/`. La euclidea es la unica que va sin sufijo.
    """
    assert config.DISTANCIA_SERVIBLE == "manhattan"


def test_no_se_sirve_mahalanobis():
    """Candado explicito: esa geometria esta descartada, no solo «no elegida».

    Puntuaba mas alto que la servida, asi que sin esta prueba nada impide que
    vuelva a colarse al promover mirando solo el score. El motivo es que su
    blanqueo deshace la ponderacion del bloque de posicion (`POSITION_SCALING`):
    medido sobre la BD real, el bloque pasa del 2 % al 33 % de la distancia^2.
    """
    assert config.DISTANCIA_SERVIBLE != "mahalanobis"


@pytest.mark.parametrize("entidad, esperado", [("jugador", V91), ("equipo", V349)])
def test_la_tabla_servible_es_la_del_artefacto_promovido(entidad, esperado):
    assert config.HIPERPARAMETROS_SERVIBLES[entidad] == esperado


@pytest.mark.parametrize("entidad, esperado", [("jugador", V91), ("equipo", V349)])
def test_el_context_manager_pone_esos_valores_en_la_celda_servible(entidad, esperado):
    """Es lo que hace `build` y `reentrenar` antes de ajustar la celda servible.

    Un solo `F5_RFF_DIM` de modulo no puede ser 1536 y 1216 a la vez: la tabla
    existe justamente porque las dos entidades comparten formulacion con valores
    distintos, y el context manager es como se aplican sin volverlos globales.
    """
    formulacion, normalizacion = config.MODELOS_SERVIBLES[entidad]
    with config.hiperparametros_servibles(entidad, formulacion, normalizacion):
        vigentes = {nombre: getattr(config, nombre) for nombre in esperado}
    assert vigentes == pytest.approx(esperado)


def test_los_valores_se_restauran_al_salir():
    """Restaurar no es cosmetico: `build` ajusta las cuatro celdas en un proceso.

    Si la celda servible dejase sus valores puestos, las siguientes —el brazo de
    comparacion del TFG— se ajustarian con ellos sin decirlo.
    """
    antes = {n: getattr(config, n) for n in (*V91, *V349)}
    with config.hiperparametros_servibles("equipo", "5", "global"):
        pass
    assert {n: getattr(config, n) for n in (*V91, *V349)} == antes


def test_una_celda_que_no_es_la_servible_no_recibe_nada():
    """La F2 de equipo sigue construyendose con los defaults del modulo."""
    with config.hiperparametros_servibles("equipo", "2", "global") as puestos:
        assert puestos == {}
        assert config.F5_RFF_DIM == 1152        # el default, no el 1216 del v349


def test_la_celda_servible_en_OTRA_geometria_tampoco_recibe_nada():
    """La misma celda medida con otra distancia es otro modelo, no el servido.

    Es lo que evita que un `build --distancia coseno` salga con los
    hiperparametros del artefacto servido y se confunda con el.
    """
    with config.hiperparametros_servibles(
            "jugador", "5", "por_liga", distancia="euclidea") as puestos:
        assert puestos == {}


def test_los_defaults_del_modulo_no_se_mueven_al_promover():
    """Lo que ve todo lo que NO pasa por la tabla: el barrido y el resto de celdas.

    Se quedan en los del v83 (el jugador servido hasta el 16-8-2026) aunque el
    modelo servido ya sea otro. No es descuido: un default entra en la huella de
    todo artefacto construido sin pasar por la tabla, asi que moverlo invalidaria
    las combinaciones ya acumuladas en `outputs/evaluacion/` sin cambiar ni un
    numero de ninguna. Promover un modelo toca `HIPERPARAMETROS_SERVIBLES`, no
    esto.
    """
    assert config.F5_RFF_DIM == 1152
    assert config.F5_EASE_LAMBDA == pytest.approx(0.045)
