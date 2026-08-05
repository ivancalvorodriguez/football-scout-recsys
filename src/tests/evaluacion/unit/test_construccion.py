"""Reconstruccion de S y modelos "virtuales" que solo existen para evaluar.

El split par/impar y las multiplicidades bootstrap son estructuras puras (indices
y conteos), asi que se comprueban contra el resultado exacto sobre contextos
fabricados a mano. La FIDELIDAD al pipeline real (reconstruir == el artefacto
guardado) se prueba en `integracion/`, que es donde hay pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluacion import construccion
from src.similitud import config as scfg
from src.tests.evaluacion.conftest import contexto_sintetico


@pytest.fixture
def ctx() -> construccion.Contexto:
    """Tres entidades: la 0 con 4 partidos, la 1 con 3 y la 2 con uno solo.

    Los partidos NO llegan ordenados: el split tiene que ordenarlos por
    `match_id`, no fiarse del orden de la BD.
    """
    X = np.arange(16.0).reshape(8, 2)
    entity_id = [10, 10, 10, 10, 20, 20, 20, 30]
    match_id = [4, 1, 3, 2, 7, 5, 6, 9]
    return contexto_sintetico(X, entity_id, match_id)


class TestContexto:
    def test_el_jugador_usa_el_metodo_distribucional_del_jugador(
        self, ctx: construccion.Contexto
    ) -> None:
        assert ctx.metodo_f5 == scfg.F5_METODO_JUGADOR

    def test_el_equipo_usa_el_suyo(self) -> None:
        """Se elige por entidad (`formulacion5.py`): MMD en jugador, Sinkhorn en
        equipo. Si el contexto se equivocara, la reconstruccion no seria fiel.
        """
        ctx = contexto_sintetico([[0.0], [1.0]], [1, 2], [1, 2], entidad="equipo")
        assert ctx.metodo_f5 == scfg.F5_METODO_EQUIPO

    def test_la_w_se_calcula_una_sola_vez(self, ctx: construccion.Contexto,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        """Es la pieza cara de la F2 y depende solo de las observaciones: si se
        recalculara por cada agrupacion (real, mitades, cada bootstrap), la Fase 2
        costaria B veces el ajuste completo.
        """
        llamadas = []
        original = construccion.slim.slim_instancia

        def espia(*a, **kw):
            llamadas.append(1)
            return original(*a, **kw)

        monkeypatch.setattr(construccion.slim, "slim_instancia", espia)
        ctx.W()
        ctx.W()
        assert len(llamadas) == 1

    def test_la_w_devuelve_indices_y_pesos_por_observacion(
        self, ctx: construccion.Contexto
    ) -> None:
        cols_idx, cols_val = ctx.W()
        assert len(cols_idx) == len(cols_val) == ctx.mf.X.shape[0]

    def test_el_indice_real_agrupa_las_observaciones_por_entidad(
        self, ctx: construccion.Contexto
    ) -> None:
        assert ctx.idx_real.ids.tolist() == [10, 20, 30]
        assert ctx.idx_real.row_entity.tolist() == [0, 0, 0, 0, 1, 1, 1, 2]


class TestSplitParImpar:
    def test_cada_entidad_con_dos_o_mas_partidos_se_desdobla(
        self, ctx: construccion.Contexto
    ) -> None:
        split = construccion.split_par_impar(ctx)
        # 0 y 1 dan dos mitades cada una; la 2 (un solo partido) solo mitad A.
        assert split.n_virt == 5
        assert len(split.pares_AB) == 2

    def test_las_entidades_de_un_solo_partido_no_son_evaluables(
        self, ctx: construccion.Contexto
    ) -> None:
        """Se quedan en el pool como DISTRACTORAS: suben la dificultad de la
        recuperacion sin poder puntuar (no tienen mitad que recuperar).
        """
        split = construccion.split_par_impar(ctx)
        assert split.real_evaluables.tolist() == [0, 1]
        assert 2 not in split.real_evaluables.tolist()

    def test_reparte_los_partidos_ordenados_por_match_id(
        self, ctx: construccion.Contexto
    ) -> None:
        """Entidad 0: partidos 1,2,3,4 (filas 1,3,2,0). Posiciones pares (1 y 3)
        a la mitad A; impares (2 y 4) a la B. Ordenar importa: con el orden de la
        BD las mitades no serian "partidos alternos".
        """
        split = construccion.split_par_impar(ctx)
        vA, vB = split.pares_AB[0]
        assert sorted(ctx.match_id[split.row_entity == vA].tolist()) == [1, 3]
        assert sorted(ctx.match_id[split.row_entity == vB].tolist()) == [2, 4]

    def test_con_un_numero_impar_de_partidos_la_mitad_a_se_lleva_uno_mas(
        self, ctx: construccion.Contexto
    ) -> None:
        split = construccion.split_par_impar(ctx)
        vA, vB = split.pares_AB[1]           # entidad 20: partidos 5, 6, 7
        assert int((split.row_entity == vA).sum()) == 2
        assert int((split.row_entity == vB).sum()) == 1

    def test_ninguna_observacion_se_queda_sin_asignar(
        self, ctx: construccion.Contexto
    ) -> None:
        """`row_entity` se inicializa a -1: un -1 superviviente seria una
        observacion perdida y la S virtual estaria mal indexada.
        """
        split = construccion.split_par_impar(ctx)
        assert (split.row_entity >= 0).all()
        assert split.row_entity.max() == split.n_virt - 1

    def test_las_dos_mitades_apuntan_a_la_misma_entidad_real(
        self, ctx: construccion.Contexto
    ) -> None:
        split = construccion.split_par_impar(ctx)
        for vA, vB in split.pares_AB:
            assert split.orig_real[vA] == split.orig_real[vB]

    def test_las_mitades_estan_etiquetadas_como_a_y_b(
        self, ctx: construccion.Contexto
    ) -> None:
        """La direccion A->B / B->A de la Fase 1 se construye con esta etiqueta."""
        split = construccion.split_par_impar(ctx)
        for vA, vB in split.pares_AB:
            assert split.half[vA] == 0
            assert split.half[vB] == 1

    def test_una_entidad_con_dos_partidos_reparte_uno_a_cada_mitad(self) -> None:
        ctx = contexto_sintetico([[0.0], [1.0]], [7, 7], [2, 1])
        split = construccion.split_par_impar(ctx)
        assert split.n_virt == 2
        assert split.pares_AB == [(0, 1)]


class TestMultiplicidadesBootstrap:
    def test_conserva_el_numero_de_observaciones_de_cada_entidad(
        self, ctx: construccion.Contexto
    ) -> None:
        """Remuestrear con reemplazo dentro de la entidad: cambia QUE partidos
        pesan, no cuantos. Si no se conservara, el bootstrap mediria tamaño.
        """
        rng = np.random.default_rng(0)
        mult = construccion.multiplicidades_bootstrap(ctx, None, rng)
        for e in range(3):
            obs = ctx.idx_real.row_entity == e
            assert mult[obs].sum() == pytest.approx(obs.sum())

    def test_las_multiplicidades_son_enteros_no_negativos(
        self, ctx: construccion.Contexto
    ) -> None:
        rng = np.random.default_rng(1)
        mult = construccion.multiplicidades_bootstrap(ctx, None, rng)
        assert (mult >= 0).all()
        assert np.all(mult == np.round(mult))

    def test_una_entidad_de_un_solo_partido_siempre_se_remuestrea_a_si_misma(
        self, ctx: construccion.Contexto
    ) -> None:
        rng = np.random.default_rng(2)
        mult = construccion.multiplicidades_bootstrap(ctx, None, rng)
        assert mult[ctx.idx_real.row_entity == 2] == pytest.approx([1.0])

    def test_es_determinista_para_una_misma_secuencia_aleatoria(
        self, ctx: construccion.Contexto
    ) -> None:
        a = construccion.multiplicidades_bootstrap(ctx, None, np.random.default_rng(3))
        b = construccion.multiplicidades_bootstrap(ctx, None, np.random.default_rng(3))
        assert a.tolist() == b.tolist()

    def test_remuestrear_dos_veces_da_composiciones_distintas(
        self, ctx: construccion.Contexto
    ) -> None:
        """Si el remuestreo no perturbara nada, el RBO de la Fase 2 saldria 1 por
        construccion y no mediria estabilidad.
        """
        rng = np.random.default_rng(4)
        muestras = {tuple(construccion.multiplicidades_bootstrap(ctx, None, rng))
                    for _ in range(10)}
        assert len(muestras) > 1

    def test_con_split_remuestrea_dentro_de_cada_mitad(
        self, ctx: construccion.Contexto
    ) -> None:
        split = construccion.split_par_impar(ctx)
        rng = np.random.default_rng(5)
        mult = construccion.multiplicidades_bootstrap(ctx, split, rng)
        for v in range(split.n_virt):
            obs = split.row_entity == v
            assert mult[obs].sum() == pytest.approx(obs.sum())

    def test_devuelve_una_multiplicidad_por_observacion(
        self, ctx: construccion.Contexto
    ) -> None:
        rng = np.random.default_rng(6)
        assert construccion.multiplicidades_bootstrap(ctx, None, rng).shape == (8,)

    def test_una_entidad_virtual_sin_observaciones_se_salta(
        self, ctx: construccion.Contexto
    ) -> None:
        """No deberia darse con el split real, pero una entidad vacia haria que
        `rng.integers(0, 0)` reventara a mitad del bootstrap.
        """
        split = construccion.SplitVirtual(
            row_entity=np.zeros(8, dtype=np.int64), n_virt=2,
            orig_real=np.array([0, 0]), half=np.array([0, 1]),
            pares_AB=[(0, 1)], real_evaluables=np.array([0]))
        mult = construccion.multiplicidades_bootstrap(
            ctx, split, np.random.default_rng(7))
        assert mult.sum() == pytest.approx(8.0)


class TestReconstruirS:
    def test_la_formulacion_2_agrega_la_w_a_entidades(
        self, ctx: construccion.Contexto
    ) -> None:
        S = construccion.reconstruir_S(ctx, "2", ctx.idx_real.row_entity, 3)
        assert S.shape == (3, 3)
        assert S == pytest.approx(S.T)
        assert np.diag(S) == pytest.approx(np.zeros(3))

    def test_la_formulacion_5_devuelve_una_s_simetrica_con_diagonal_cero(
        self, ctx: construccion.Contexto
    ) -> None:
        S = construccion.reconstruir_S(ctx, "5", ctx.idx_real.row_entity, 3)
        assert S == pytest.approx(S.T)
        assert np.diag(S) == pytest.approx(np.zeros(3))

    def test_reconstruye_con_una_agrupacion_arbitraria(
        self, ctx: construccion.Contexto
    ) -> None:
        """Es el punto del modulo: la misma W sirve para las mitades virtuales."""
        split = construccion.split_par_impar(ctx)
        S = construccion.reconstruir_S(ctx, "2", split.row_entity, split.n_virt)
        assert S.shape == (split.n_virt, split.n_virt)

    def test_el_peso_opcional_sustituye_a_la_masa_original(
        self, ctx: construccion.Contexto
    ) -> None:
        """Es como entra el bootstrap: perturbando las masas, sin re-aprender W."""
        w = np.zeros(8)
        S = construccion.reconstruir_S(ctx, "2", ctx.idx_real.row_entity, 3, weight=w)
        assert S == pytest.approx(np.zeros((3, 3)))

    def test_una_formulacion_desconocida_falla_en_vez_de_devolver_algo(
        self, ctx: construccion.Contexto
    ) -> None:
        with pytest.raises(ValueError, match="formulacion desconocida"):
            construccion.reconstruir_S(ctx, "7", ctx.idx_real.row_entity, 3)


class TestReconstruirF5:
    def test_devuelve_la_s_antes_y_despues_de_ease(
        self, ctx: construccion.Contexto
    ) -> None:
        """La Fase 4 compara las dos; la pre no esta en disco, hay que rehacerla."""
        S_post, S_pre = construccion.reconstruir_S_f5(ctx, ctx.idx_real.row_entity, 3)
        assert S_post.shape == S_pre.shape == (3, 3)
        assert not np.allclose(S_post, S_pre)

    def test_la_s_previa_es_la_similitud_distribucional_cruda(
        self, ctx: construccion.Contexto
    ) -> None:
        esperada = construccion.distributional.construir_S(
            ctx.mf, ctx.idx_real.row_entity, 3, ctx.metodo_f5)
        _, S_pre = construccion.reconstruir_S_f5(ctx, ctx.idx_real.row_entity, 3)
        assert S_pre == pytest.approx(esperada)

    def test_la_s_servible_de_la_f5_coincide_con_reconstruir_s(
        self, ctx: construccion.Contexto
    ) -> None:
        S = construccion.reconstruir_S(ctx, "5", ctx.idx_real.row_entity, 3)
        S_post, _ = construccion.reconstruir_S_f5(ctx, ctx.idx_real.row_entity, 3)
        assert S == pytest.approx(S_post)


class TestReconstruirSCruda:
    def test_la_s_cruda_de_la_f2_simetrizada_es_la_servible(
        self, ctx: construccion.Contexto
    ) -> None:
        """Define lo que mide la asimetria de la Fase 0: la señal direccional que
        el pipeline corrige al simetrizar, no un error numerico.
        """
        cruda = construccion.reconstruir_S_cruda(ctx, "2")
        servible = construccion.reconstruir_S(ctx, "2", ctx.idx_real.row_entity, 3)
        assert 0.5 * (cruda + cruda.T) == pytest.approx(servible)

    def test_la_s_cruda_de_la_f5_tiene_la_diagonal_a_cero(
        self, ctx: construccion.Contexto
    ) -> None:
        cruda = construccion.reconstruir_S_cruda(ctx, "5")
        assert np.diag(cruda) == pytest.approx(np.zeros(3))

    def test_la_asimetria_de_la_f5_la_introduce_el_denoising(
        self, ctx: construccion.Contexto
    ) -> None:
        """`S_pre` es simetrica por construccion; `S_pre @ B` ya no tiene por que
        serlo. Es justo lo que la Fase 0 quiere medir.
        """
        _, S_pre = construccion.reconstruir_S_f5(ctx, ctx.idx_real.row_entity, 3)
        assert S_pre == pytest.approx(S_pre.T)
        cruda = construccion.reconstruir_S_cruda(ctx, "5")
        assert cruda.shape == (3, 3)

    def test_una_formulacion_desconocida_falla(self, ctx: construccion.Contexto) -> None:
        with pytest.raises(ValueError, match="formulacion desconocida"):
            construccion.reconstruir_S_cruda(ctx, "3")
