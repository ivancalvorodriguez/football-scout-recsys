"""Unidades de trabajo del barrido: celdas, orden y caches del trabajador.

Sin BD ni ajustes: lo que se prueba aqui es la identidad de una celda, el orden en
que se emiten (que es lo que hace util el cache) y las dos reglas del cache de
contextos —cuando vale reutilizar uno y cuando hay que tirar la W que lleva
dentro—. La regla de la W es la que importa de verdad: reutilizar un contexto con
la W ajustada con OTROS hiperparametros seria el fallo mas silencioso posible.
Lo que si toca disco vive en `integracion/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluacion import construccion, trabajo
from src.evaluacion import config as ecfg
from src.similitud import config as scfg


class ContextoFalso:
    """Lo minimo que `trabajo.contexto` toca de un `Contexto` real."""

    def __init__(self, entidad: str, normalizacion: str) -> None:
        self.entidad = entidad
        self.normalizacion = normalizacion
        self._W = ("W ya ajustada",)


@pytest.fixture
def contar_contextos(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Sustituye la construccion del contexto por un contador de llamadas."""
    creados: list[tuple[str, str]] = []

    def falso(db_path, entidad, normalizacion):
        creados.append((entidad, normalizacion))
        return ContextoFalso(entidad, normalizacion)

    monkeypatch.setattr(construccion, "crear_contexto", falso)
    trabajo.limpiar_caches()
    yield creados
    trabajo.limpiar_caches()


class TestCelda:
    def test_el_stem_es_el_nombre_del_artefacto(self) -> None:
        """Tiene que coincidir con lo que escribe `ModeloSimilitud.guardar`, o la
        cache buscaria un fichero que no existe."""
        celda = trabajo.Celda("5", "jugador", "global")
        assert celda.stem == "formulacion5_jugador_global"

    def test_la_etiqueta_identifica_al_modelo_en_el_log(self) -> None:
        assert trabajo.Celda("2", "equipo", "por_liga").etiqueta == "F2_equipo_por_liga"

    def test_es_hashable_y_ordenable(self) -> None:
        """Se usa como clave de diccionario al repartir resultados por celda."""
        assert len({trabajo.Celda("2", "equipo", "global"),
                    trabajo.Celda("2", "equipo", "global")}) == 1


class TestOrdenDeLasCeldas:
    def test_cubre_la_rejilla_pedida(self) -> None:
        celdas = trabajo.celdas(("2", "5"), ("jugador", "equipo"))
        assert len(celdas) == 2 * 2 * len(ecfg.NORMALIZACIONES)
        assert len(set(celdas)) == len(celdas)

    def test_agrupa_las_que_comparten_contexto(self) -> None:
        """Es lo que hace que el cache del trabajador acierte: las celdas de una
        misma (entidad, normalizacion) salen SEGUIDAS, sin volver a ella despues.
        """
        grupos = [(c.entidad, c.normalizacion)
                  for c in trabajo.celdas(("2", "5"), ("jugador", "equipo"))]
        tramos = [g for i, g in enumerate(grupos) if i == 0 or g != grupos[i - 1]]
        assert len(tramos) == len(set(grupos))

    def test_dentro_de_un_contexto_va_primero_la_formulacion_cara(self) -> None:
        """Empezar por lo largo acorta el tiempo total cuando hay varios
        trabajadores (el F2 de jugador tarda >10 min; el F5, segundos)."""
        celdas = trabajo.celdas(("2", "5"), ("jugador",))
        assert [c.formulacion for c in celdas[:2]] == ["2", "5"]

    def test_acotar_la_rejilla_deja_fuera_lo_no_pedido(self) -> None:
        celdas = trabajo.celdas(("5",), ("equipo",))
        assert {c.formulacion for c in celdas} == {"5"}
        assert {c.entidad for c in celdas} == {"equipo"}

    def test_por_defecto_entran_las_dos_normalizaciones(self) -> None:
        """Compararlas es el motivo de que sean un eje de la rejilla: acotarlas
        tiene que ser explicito."""
        celdas = trabajo.celdas(("5",), ("equipo",))
        assert {c.normalizacion for c in celdas} == set(ecfg.NORMALIZACIONES)

    def test_se_puede_acotar_a_una_sola_normalizacion(self) -> None:
        celdas = trabajo.celdas(("5",), ("equipo",), ("global",))
        assert [c.normalizacion for c in celdas] == ["global"]


class TestConfigTemporal:
    def test_fija_y_restaura(self) -> None:
        antes = scfg.F2_L1
        with trabajo.config_temporal({"F2_L1": 0.123}):
            assert scfg.F2_L1 == 0.123
        assert scfg.F2_L1 == antes

    def test_restaura_tambien_si_el_bloque_falla(self) -> None:
        antes = scfg.F5_EASE_LAMBDA
        with pytest.raises(RuntimeError):
            with trabajo.config_temporal({"F5_EASE_LAMBDA": 999.0}):
                raise RuntimeError("boom")
        assert scfg.F5_EASE_LAMBDA == antes


class TestCacheDeContextos:
    def test_dos_celdas_de_la_misma_entidad_y_norma_comparten_contexto(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        """La F2 y la F5 tienen EXACTAMENTE la misma matriz de features: la
        formulacion no entra en `huella.alcance_datos`."""
        bd = tmp_path / "scouting.db"
        trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        assert contar_contextos == [("equipo", "global")]

    def test_otra_normalizacion_es_otro_contexto(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        bd = tmp_path / "scouting.db"
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "por_liga"))
        assert len(contar_contextos) == 2

    def test_cambiar_una_feature_invalida_el_contexto(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        """La matriz X depende de `huella.alcance_datos`: si cambia uno de esos
        atributos, reutilizarla serviria features de otra configuracion."""
        bd = tmp_path / "scouting.db"
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        with trabajo.config_temporal({"F_CLIP_Z": scfg.F_CLIP_Z + 1.0}):
            trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        assert len(contar_contextos) == 2

    def test_el_tope_de_contextos_libera_los_mas_viejos(
        self, contar_contextos: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cada contexto pesa lo que pesa la matriz X entera; sin tope, un
        trabajador acabaria con toda la rejilla en memoria."""
        monkeypatch.setattr(trabajo, "MAX_CONTEXTOS", 1)
        bd = tmp_path / "scouting.db"
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "por_liga"))
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        assert len(contar_contextos) == 3


class TestCacheDeLaW:
    def test_la_w_sobrevive_si_los_hiperparametros_de_la_f2_no_cambian(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        bd = tmp_path / "scouting.db"
        ctx = trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        ctx._W = ("W ajustada",)
        otra = trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        assert otra is ctx and otra._W == ("W ajustada",)

    def test_cambiar_un_hiperparametro_de_la_f2_tira_la_w(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        """El contexto se reutiliza (misma X) pero la W no: depende de los `F2_*`."""
        bd = tmp_path / "scouting.db"
        ctx = trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        ctx._W = ("W ajustada",)
        with trabajo.config_temporal({"F2_L1": scfg.F2_L1 + 0.1}):
            otra = trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        assert otra is ctx and otra._W is None
        assert len(contar_contextos) == 1

    def test_una_celda_f5_no_toca_la_w(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        """La F5 no la usa, asi que descartarla obligaria a reajustarla despues."""
        bd = tmp_path / "scouting.db"
        ctx = trabajo.contexto(bd, trabajo.Celda("2", "equipo", "global"))
        ctx._W = ("W ajustada",)
        trabajo.contexto(bd, trabajo.Celda("5", "equipo", "global"))
        assert ctx._W == ("W ajustada",)
