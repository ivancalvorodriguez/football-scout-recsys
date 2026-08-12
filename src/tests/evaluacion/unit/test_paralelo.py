"""Reparto de tareas en procesos: numero de trabajadores, log y fallos.

Casi todo se prueba con `trabajos=1`, que ejecuta en este mismo proceso: lo que
interesa aqui es el CONTRATO (que se devuelve, que se imprime, que pasa si una
tarea revienta), y ese es el mismo en serie que en paralelo. El camino con
procesos de verdad —el que puede fallar por serializacion o por `spawn`— se
prueba aparte, marcado como lento, porque arrancar un interprete por trabajador
cuesta segundos.
"""

from __future__ import annotations

import os

import pytest

from src.evaluacion import paralelo


# Funciones de nivel de modulo: con `spawn` las tareas se serializan por NOMBRE,
# asi que una lambda o una funcion local no valdrian como tarea.

def doblar(x: int) -> int:
    return x * 2


def hablar(texto: str) -> str:
    print(texto)
    return texto


def reventar() -> None:
    raise ValueError("boom de la tarea")


def leer_hilos() -> str | None:
    return os.environ.get("OMP_NUM_THREADS")


class TestResolverTrabajos:
    def test_un_entero_se_respeta(self) -> None:
        assert paralelo.resolver_trabajos(4) == 4
        assert paralelo.resolver_trabajos("4") == 4

    def test_auto_es_un_trabajador_por_nucleo(self) -> None:
        assert paralelo.resolver_trabajos(paralelo.AUTO) == (os.cpu_count() or 1)

    def test_sin_valor_es_uno(self) -> None:
        assert paralelo.resolver_trabajos(None) == 1

    def test_un_texto_que_no_es_numero_aborta(self) -> None:
        with pytest.raises(SystemExit, match="se esperaba un entero"):
            paralelo.resolver_trabajos("muchos")

    def test_cero_o_negativo_aborta(self) -> None:
        with pytest.raises(SystemExit, match=">= 1"):
            paralelo.resolver_trabajos(0)


class TestMapearEnSerie:
    def test_devuelve_el_valor_de_cada_tarea_por_su_nombre(self) -> None:
        tareas = [paralelo.Tarea(f"t{i}", doblar, (i,)) for i in range(3)]
        assert paralelo.mapear(tareas, trabajos=1) == {"t0": 0, "t1": 2, "t2": 4}

    def test_sin_tareas_no_hace_nada(self) -> None:
        assert paralelo.mapear([], trabajos=8) == {}

    def test_nombres_repetidos_abortan(self) -> None:
        """El nombre es la clave del resultado: repetirlo perderia una tarea."""
        tareas = [paralelo.Tarea("t", doblar, (1,)), paralelo.Tarea("t", doblar, (2,))]
        with pytest.raises(ValueError, match="nombres unicos"):
            paralelo.mapear(tareas, trabajos=1)

    def test_en_serie_el_log_sale_en_directo(self, capsys) -> None:
        """No se captura: es el modo en que se depura, y una tarea larga tiene que
        poder decir por donde va."""
        paralelo.mapear([paralelo.Tarea("t", hablar, ("hola desde la tarea",))],
                        trabajos=1)
        salida = capsys.readouterr().out
        assert "hola desde la tarea" in salida
        assert "en serie" in salida

    def test_una_tarea_que_falla_propaga(self) -> None:
        """Abortar es deliberado: media tanda acabaria en `barrido_metricas.csv`,
        que es acumulativo, como si fuese buena.
        """
        with pytest.raises(ValueError, match="boom de la tarea"):
            paralelo.mapear([paralelo.Tarea("t", reventar)], trabajos=1)


class TestRepartirHilos:
    def test_reparte_los_nucleos_entre_los_trabajadores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        for var in paralelo._VARS_HILOS:
            monkeypatch.delenv(var, raising=False)
        with paralelo._repartir_hilos(4) as por_proceso:
            assert por_proceso == 2
            assert os.environ["OMP_NUM_THREADS"] == "2"

    def test_deja_el_entorno_como_estaba(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in paralelo._VARS_HILOS:
            monkeypatch.delenv(var, raising=False)
        with paralelo._repartir_hilos(2):
            pass
        assert "OMP_NUM_THREADS" not in os.environ

    def test_no_pisa_lo_que_el_usuario_ya_habia_puesto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Es una decision suya y sabra por que."""
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        with paralelo._repartir_hilos(4):
            assert os.environ["OMP_NUM_THREADS"] == "3"


@pytest.mark.lento
class TestMapearEnProcesos:
    """El camino de verdad: arranca interpretes, serializa y recoge."""

    def test_devuelve_lo_mismo_que_en_serie(self) -> None:
        tareas = [paralelo.Tarea(f"t{i}", doblar, (i,)) for i in range(4)]
        assert (paralelo.mapear(tareas, trabajos=2)
                == paralelo.mapear(tareas, trabajos=1))

    def test_el_log_del_trabajador_llega_al_padre(self, capsys) -> None:
        """Se captura en el hijo y se imprime entero al terminar: sin eso, cuatro
        procesos escribiendo a la vez dejan la consola ilegible."""
        paralelo.mapear([paralelo.Tarea("t", hablar, ("dentro del hijo",))],
                        trabajos=2)
        assert "dentro del hijo" in capsys.readouterr().out

    def test_una_tarea_que_falla_aborta_diciendo_cual(self) -> None:
        tareas = [paralelo.Tarea("buena", doblar, (1,)),
                  paralelo.Tarea("mala", reventar)]
        with pytest.raises(RuntimeError, match="'mala'"):
            paralelo.mapear(tareas, trabajos=2)

    def test_los_hijos_heredan_el_reparto_de_hilos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sin esto cada trabajador lanzaria tantos hilos de BLAS como nucleos y
        el conjunto iria mas lento que con menos trabajadores."""
        for var in paralelo._VARS_HILOS:
            monkeypatch.delenv(var, raising=False)
        valores = paralelo.mapear(
            [paralelo.Tarea(f"t{i}", leer_hilos) for i in range(2)], trabajos=2)
        assert all(v is not None and int(v) >= 1 for v in valores.values())
