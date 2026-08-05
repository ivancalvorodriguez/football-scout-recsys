"""Fidelidad al pipeline real: lo que se evalua tiene que ser el modelo servible.

Es la afirmacion central de `docs/evaluacion.md` ("reconstruir con el indice de
entidades real reproduce el artefacto guardado con max|dS| = 0"). Si dejara de
cumplirse, todas las metricas del harness seguirian saliendo —y describirian otro
modelo. Por eso se comprueba contra el artefacto construido por el pipeline de
produccion, no contra una copia esperada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluacion import construccion, datos, evaluar, fases
from src.evaluacion import config as ecfg
from src.similitud import config as scfg
from src.similitud import formulacion2, formulacion5

pytestmark = [pytest.mark.integracion, pytest.mark.lento]

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}


@pytest.fixture(scope="module")
def ctx_equipo(bd_sintetica: Path) -> construccion.Contexto:
    return construccion.crear_contexto(bd_sintetica, "equipo", "por_liga")


@pytest.fixture(scope="module")
def ctx_jugador(bd_sintetica: Path) -> construccion.Contexto:
    return construccion.crear_contexto(bd_sintetica, "jugador", "por_liga")


class TestCrearContexto:
    def test_los_metadatos_van_alineados_fila_a_fila_con_las_features(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        """`match_id` y `minutos_obs` se usan para dividir y ponderar las mismas
        filas de X: un desalineamiento partiria las entidades por otros partidos.
        """
        M = ctx_equipo.mf.X.shape[0]
        assert ctx_equipo.match_id.shape == (M,)
        assert ctx_equipo.minutos_obs.shape == (M,)
        assert ctx_equipo.idx_real.row_entity.shape == (M,)

    def test_el_equipo_no_pondera_por_minutos(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        """Juega el partido completo: la masa por minutos no distinguiria nada."""
        assert ctx_equipo.minutos_obs.tolist() == [1.0] * len(ctx_equipo.minutos_obs)

    def test_el_jugador_si_lleva_sus_minutos(
        self, ctx_jugador: construccion.Contexto
    ) -> None:
        assert ctx_jugador.minutos_obs.min() > 1.0

    def test_entran_mas_observaciones_que_entidades(
        self, ctx_jugador: construccion.Contexto
    ) -> None:
        """El principio central del modelo: nada se colapsa antes de ajustar."""
        assert ctx_jugador.mf.X.shape[0] > len(ctx_jugador.idx_real.ids)

    def test_no_escribe_en_la_base_de_datos(self, bd_sintetica: Path) -> None:
        antes = bd_sintetica.stat().st_mtime_ns
        construccion.crear_contexto(bd_sintetica, "equipo", "global")
        assert bd_sintetica.stat().st_mtime_ns == antes


class TestFidelidadDeLaReconstruccion:
    """Reconstruir con la agrupacion real == el artefacto que sirve el sistema."""

    @pytest.mark.parametrize("formulacion", ["2", "5"])
    @pytest.mark.parametrize("normalizacion", ["por_liga", "global"])
    def test_el_equipo_se_reproduce_exactamente(
        self, bd_sintetica: Path, formulacion: str, normalizacion: str
    ) -> None:
        ctx = construccion.crear_contexto(bd_sintetica, "equipo", normalizacion)
        artefacto = _FORMULACIONES[formulacion].construir(
            ctx.mf, "equipo", normalizacion)
        S = construccion.reconstruir_S(
            ctx, formulacion, ctx.idx_real.row_entity, len(ctx.idx_real.ids))
        assert np.max(np.abs(S - artefacto.S)) == 0.0

    @pytest.mark.parametrize("formulacion", ["2", "5"])
    def test_el_jugador_se_reproduce_exactamente(
        self, ctx_jugador: construccion.Contexto, formulacion: str
    ) -> None:
        artefacto = _FORMULACIONES[formulacion].construir(
            ctx_jugador.mf, "jugador", "por_liga")
        S = construccion.reconstruir_S(
            ctx_jugador, formulacion, ctx_jugador.idx_real.row_entity,
            len(ctx_jugador.idx_real.ids))
        assert np.max(np.abs(S - artefacto.S)) == 0.0

    def test_el_indice_de_entidades_es_el_mismo_que_el_del_artefacto(
        self, ctx_jugador: construccion.Contexto
    ) -> None:
        """Sin esto, la fila i de la S reconstruida hablaria de otro jugador."""
        artefacto = formulacion5.construir(ctx_jugador.mf, "jugador", "por_liga")
        assert ctx_jugador.idx_real.ids.tolist() == artefacto.entity_ids.tolist()

    def test_la_w_cacheada_sirve_para_cualquier_agrupacion(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        """Depende solo de las observaciones, no de como se agrupen: es lo que
        permite calcularla una vez y reutilizarla para mitades y remuestreos.
        """
        split = construccion.split_par_impar(ctx_equipo)
        S_virtual = construccion.reconstruir_S(
            ctx_equipo, "2", split.row_entity, split.n_virt)
        S_real = construccion.reconstruir_S(
            ctx_equipo, "2", ctx_equipo.idx_real.row_entity,
            len(ctx_equipo.idx_real.ids))
        assert S_virtual.shape[0] > S_real.shape[0]
        assert np.isfinite(S_virtual).all()

    def test_las_dos_normalizaciones_dan_modelos_distintos(
        self, bd_sintetica: Path
    ) -> None:
        """La BD sintetica tiene dos ligas justo para que esto diga algo."""
        matrices = []
        for norm in ("por_liga", "global"):
            ctx = construccion.crear_contexto(bd_sintetica, "equipo", norm)
            matrices.append(construccion.reconstruir_S(
                ctx, "5", ctx.idx_real.row_entity, len(ctx.idx_real.ids)))
        assert not np.allclose(*matrices)


class TestFasesSobreDatosReales:
    """Las fases, corridas sobre la S que produce el pipeline de verdad."""

    @staticmethod
    @pytest.fixture(scope="class")
    def modelo(bd_sintetica: Path):
        ctx = construccion.crear_contexto(bd_sintetica, "equipo", "por_liga")
        return formulacion5.construir(ctx.mf, "equipo", "por_liga")

    def test_la_asimetria_de_la_s_cruda_es_finita_y_no_negativa(
        self, modelo, ctx_equipo: construccion.Contexto, bd_sintetica: Path
    ) -> None:
        roles = datos.rol_por_jugador(bd_sintetica)
        res = fases.fase0_sanity(modelo, roles, ctx_equipo, "5")
        assert res["asimetria"] >= 0.0
        assert np.isfinite(res["asimetria"])

    def test_la_autosimilitud_recupera_mejor_que_el_azar(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        """Es el pilar del protocolo: si un equipo no se reconoce a si mismo entre
        sus propios partidos, el modelo no captura estilo.
        """
        res = fases.fase1_autosimilitud(ctx_equipo, "5")
        assert res["global"]["n"] > 0
        assert res["global"]["mrr"] > res["global"]["azar_top1"]

    def test_los_rangos_de_la_autosimilitud_caben_en_el_pool(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        res = fases.fase1_autosimilitud(ctx_equipo, "2")
        assert 0.0 <= res["A->B"]["top1"] <= 1.0
        assert res["A->B"]["mrr"] >= 1.0 / res["A->B"]["pool"]

    def test_la_estabilidad_devuelve_un_rbo_en_el_intervalo_unidad(
        self, ctx_equipo: construccion.Contexto, modelo
    ) -> None:
        res = fases.fase2_estabilidad(ctx_equipo, "5", modelo.S, B=3)
        assert 0.0 <= res["rbo_medio"] <= 1.0
        assert res["ic_bajo"] <= res["rbo_medio"] <= res["ic_alto"]

    def test_la_triangulacion_compara_las_matrices_reales(
        self, bd_sintetica: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ecfg, "MANTEL_PERMUTACIONES", 19)
        ctxs = {("equipo", n): construccion.crear_contexto(bd_sintetica, "equipo", n)
                for n in ecfg.NORMALIZACIONES}
        modelos = {
            (f, "equipo", n): _FORMULACIONES[f].construir(ctxs[("equipo", n)].mf,
                                                          "equipo", n)
            for f in ("2", "5") for n in ecfg.NORMALIZACIONES
        }
        filas = fases.fase3_triangulacion(modelos, ctxs)
        assert filas
        assert all(-1.0 <= f["r"] <= 1.0 for f in filas)
        assert {f["entidad"] for f in filas} == {"equipo"}

    def test_el_downstream_produce_metricas_finitas(
        self, modelo, bd_sintetica: Path
    ) -> None:
        roles = datos.rol_por_jugador(bd_sintetica)
        minutos = datos.minutos_por_jugador(bd_sintetica)
        res = fases.fase5_downstream(modelo, roles, minutos)
        assert 0.0 <= res["coverage"] <= 1.0
        assert res["diversity"] >= 0.0
        assert res["indegree_medio"] >= 0.0


class TestEvaluacionCompleta:
    """`ejecutar` + los tres entregables, sobre modelos construidos de verdad."""

    @staticmethod
    @pytest.fixture(scope="class")
    def salida(bd_sintetica: Path, tmp_path_factory: pytest.TempPathFactory):
        model_dir = tmp_path_factory.mktemp("modelo_eval")
        ctxs = {}
        for norm in ecfg.NORMALIZACIONES:
            ctx = construccion.crear_contexto(bd_sintetica, "equipo", norm)
            ctxs[("equipo", norm)] = ctx
            for form in ("2", "5"):
                _FORMULACIONES[form].construir(ctx.mf, "equipo", norm).guardar(model_dir)
        modelos = evaluar.cargar_modelos(model_dir, ("2", "5"), ("equipo",))
        roles = datos.rol_por_jugador(bd_sintetica)
        minutos = datos.minutos_por_jugador(bd_sintetica)
        return evaluar.ejecutar(modelos, ctxs, roles, minutos,
                                {"0", "1", "2", "5"}, bootstrap=2)

    def test_evalua_los_cuatro_modelos_de_equipo(self, salida: dict) -> None:
        assert {r["modelo"] for r in salida["f0"]} == {
            f"F{f}_equipo_{n}" for f in ("2", "5") for n in ecfg.NORMALIZACIONES}

    def test_la_fase_4_solo_aparece_para_la_formulacion_5(self, salida: dict) -> None:
        """EASE solo existe en la F5: una fila F2 ahi seria un bug de etiquetado."""
        assert {r["formulacion"] for r in salida["f4"]} == {"5"}

    def test_el_equipo_no_produce_pureza_ni_knn(self, salida: dict) -> None:
        assert all(not np.isfinite(r["pureza_top1"]) for r in salida["f0"])
        assert all(not np.isfinite(r["knn_accuracy"]) for r in salida["f5"])

    def test_escribe_un_csv_por_fase_con_resultados(
        self, salida: dict, tmp_path: Path
    ) -> None:
        evaluar.escribir_csv(salida, tmp_path)
        for nombre in ("fase0_sanity", "fase1_autosimilitud", "fase2_estabilidad",
                       "fase4_denoising", "fase5_downstream"):
            assert (tmp_path / f"{nombre}.csv").exists()

    def test_el_informe_recoge_los_modelos_evaluados(
        self, salida: dict, tmp_path: Path
    ) -> None:
        evaluar.escribir_informe(salida, tmp_path, {
            "fecha": "2026-01-01 10:00", "n_modelos": 4, "bootstrap": 2,
            "features": [],
        })
        texto = (tmp_path / "resumen.md").read_text(encoding="utf-8")
        assert "Fase 1 - Auto-similitud" in texto
        assert "equipo" in texto

    def test_dibuja_las_figuras_clave(self, salida: dict, tmp_path: Path) -> None:
        evaluar.escribir_figuras(salida, tmp_path)
        assert (tmp_path / "figuras" / "fase1_autosimilitud_top1.png").exists()
        assert (tmp_path / "figuras" / "fase2_estabilidad_rbo.png").exists()


class TestJugadorConPosicion:
    def test_el_vector_del_jugador_lleva_el_bloque_de_posicion(
        self, ctx_jugador: construccion.Contexto
    ) -> None:
        """Forma parte del modelo base (`USE_POSITION_FEATURES`), y el harness lo
        declara en el informe porque no cabe en el nombre del artefacto.
        """
        n_pos = sum(1 for c in ctx_jugador.mf.feat_names if str(c).startswith("pos_"))
        assert n_pos == (len(scfg.POSITION_FEATURES) if scfg.USE_POSITION_FEATURES
                         else 0)

    def test_el_vector_del_equipo_no_lo_lleva(
        self, ctx_equipo: construccion.Contexto
    ) -> None:
        assert not [c for c in ctx_equipo.mf.feat_names if str(c).startswith("pos_")]
