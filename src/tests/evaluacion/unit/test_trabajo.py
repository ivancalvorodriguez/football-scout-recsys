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

    def falso(db_path, entidad, normalizacion, excluir_ligas=(), **kwargs):
        creados.append((entidad, normalizacion, tuple(excluir_ligas)))
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
        assert (trabajo.Celda("2", "equipo", "por_liga").etiqueta
                == "F2_equipo_por_liga_euclidea")

    def test_el_stem_calla_la_euclidea_pero_la_etiqueta_no(self) -> None:
        """En disco la euclidea no lleva sufijo (para no renombrar lo ya
        construido); en un resultado la geometria se declara siempre."""
        celda = trabajo.Celda("2", "equipo", "por_liga")
        assert celda.stem == "formulacion2_equipo_por_liga"
        assert celda.etiqueta.endswith("_euclidea")

    def test_otra_distancia_si_va_en_el_nombre_del_artefacto(self) -> None:
        """Sin sufijo pisaria el artefacto euclideo con otra geometria."""
        celda = trabajo.Celda("2", "equipo", "por_liga", "manhattan")
        assert celda.stem == "formulacion2_equipo_por_liga_manhattan"

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
        assert contar_contextos == [("equipo", "global", ())]

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


class TestHoldoutEnLaConstruccion:
    """Con `--holdout`, el artefacto se ajusta sin esas ligas — y todo lo demas
    tiene que seguirle.

    Son tres piezas que se pueden desincronizar en silencio: el ajuste, el
    contexto desde el que las fases reconstruyen la S, y la huella con la que la
    cache decide si un artefacto vale. Si el contexto trajera las 6 ligas y el
    artefacto solo 5, las fases compararian matrices de distinto tamaño; si la
    huella no distinguiera, una ejecucion sin hold-out reutilizaria artefactos con
    hold-out (o al reves) y las metricas describirian otro modelo.
    """

    def test_el_contexto_excluye_las_mismas_ligas(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        trabajo.contexto(tmp_path / "scouting.db",
                         trabajo.Celda("5", "equipo", "global"),
                         excluir_ligas=("1238-108",))
        assert contar_contextos == [("equipo", "global", ("1238-108",))]

    def test_es_otro_contexto_que_el_de_la_bd_entera(
        self, contar_contextos: list, tmp_path: Path
    ) -> None:
        """No se puede reutilizar el de la ejecucion sin hold-out: tiene mas
        entidades y otras mu/sd."""
        bd = tmp_path / "scouting.db"
        celda = trabajo.Celda("5", "equipo", "global")
        trabajo.contexto(bd, celda)
        trabajo.contexto(bd, celda, excluir_ligas=("1238-108",))
        assert len(contar_contextos) == 2

    def test_la_huella_distingue_el_hold_out(self, tmp_path: Path) -> None:
        from src.evaluacion import huella

        bd = tmp_path / "scouting.db"
        bd.write_bytes(b"")
        con = huella.calcular("5", "equipo", "global", bd,
                              excluir_ligas=("1238-108",))
        sin = huella.calcular("5", "equipo", "global", bd)
        assert con != sin
        assert con["excluir_ligas"] == ["1238-108"]

    def test_el_orden_de_las_ligas_no_cambia_la_huella(self, tmp_path: Path) -> None:
        """`--holdout a,b` y `--holdout b,a` son el mismo experimento."""
        from src.evaluacion import huella

        bd = tmp_path / "scouting.db"
        bd.write_bytes(b"")
        assert (huella.calcular("5", "equipo", "global", bd, ("a", "b"))
                == huella.calcular("5", "equipo", "global", bd, ("b", "a")))
