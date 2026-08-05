"""Las fases del protocolo, sobre matrices S escritas a mano.

Cada fase se comprueba contra un resultado CONOCIDO: se fabrica la S (y los roles,
y los minutos) de forma que la respuesta correcta se pueda calcular a mano. Con la
S que salga del pipeline real no se sabria si un 0.62 es el numero correcto o solo
el numero de hoy; eso se prueba en `integracion/`, que valida la fidelidad de la
reconstruccion, no el valor de la metrica.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluacion import config, construccion, fases
from src.tests.evaluacion.conftest import contexto_sintetico, modelo_sintetico


class TestTopKIndices:
    def test_devuelve_los_indices_ordenados_de_mas_a_menos_similar(self) -> None:
        S = np.array([[0.0, 0.3, 0.9, 0.5],
                      [0.3, 0.0, 0.1, 0.2],
                      [0.9, 0.1, 0.0, 0.4],
                      [0.5, 0.2, 0.4, 0.0]])
        assert fases.top_k_indices(S, 0, 3) == [2, 3, 1]

    def test_nunca_se_devuelve_a_si_misma(self) -> None:
        S = np.full((4, 4), 0.5)
        np.fill_diagonal(S, 9.0)            # la diagonal seria siempre la mejor
        for i in range(4):
            assert i not in fases.top_k_indices(S, i, 3)

    def test_descarta_los_ceros_por_defecto(self) -> None:
        """En la F2 un 0 es "sin relacion aprendida", no un candidato valido:
        incluirlos rellenaria el top-k con entidades en orden de indice.
        """
        S = np.array([[0.0, 0.7, 0.0, 0.0],
                      [0.7, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
        assert fases.top_k_indices(S, 0, 3) == [1]

    def test_puede_incluir_los_ceros_si_se_pide(self) -> None:
        S = np.zeros((3, 3))
        S[0, 1] = 0.4
        assert fases.top_k_indices(S, 0, 2, excluir_cero=False) == [1, 2]

    def test_una_fila_sin_candidatos_devuelve_lista_vacia(self) -> None:
        """Comportamiento honesto: la entidad no es consultable, no se rellena."""
        assert fases.top_k_indices(np.zeros((3, 3)), 1, 5) == []

    def test_devuelve_como_mucho_k_elementos(self) -> None:
        rng = np.random.default_rng(0)
        S = np.abs(rng.normal(size=(20, 20)))
        assert len(fases.top_k_indices(S, 0, 4)) == 4

    def test_ignora_los_valores_no_finitos(self) -> None:
        """Un inf ganaria siempre el top-1 y un NaN envenenaria la comparacion."""
        S = np.vstack([np.array([[0.0, np.nan, 0.5, np.inf]]), np.zeros((3, 4))])
        assert fases.top_k_indices(S, 0, 3) == [2]

    def test_coincide_con_ordenar_la_fila_entera(self) -> None:
        """Se usa `argpartition` (O(P)) en vez de `argsort` porque este es el bucle
        mas caliente (la Fase 2 lo llama B x P veces): el resultado tiene que ser
        el mismo, no "parecido".
        """
        rng = np.random.default_rng(1)
        S = np.abs(rng.normal(size=(40, 40)))
        np.fill_diagonal(S, 0.0)
        for i in (0, 7, 39):
            fila = S[i].copy()
            fila[i] = -np.inf
            esperado = [int(j) for j in np.argsort(fila)[::-1][:6]]
            assert fases.top_k_indices(S, i, 6) == esperado

    def test_con_menos_candidatos_que_k_no_particiona(self) -> None:
        S = np.array([[0.0, 0.2, 0.0],
                      [0.2, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        assert fases.top_k_indices(S, 0, 10) == [1]


# --------------------------------------------------------------------------- #
# Fase 0                                                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def ctx_jugador() -> construccion.Contexto:
    """Tres entidades con dos partidos cada una (todas evaluables en el split)."""
    X = np.array([[0.0, 1.0], [0.2, 0.9], [5.0, 5.0],
                  [5.2, 4.8], [-3.0, 2.0], [-3.1, 2.2]])
    return contexto_sintetico(X, [1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 1, 2])


def _cruda_fija(monkeypatch: pytest.MonkeyPatch, S_cruda: np.ndarray) -> None:
    monkeypatch.setattr(construccion, "reconstruir_S_cruda",
                        lambda ctx, form: S_cruda)


class TestFase0Sanity:
    def test_una_s_cruda_simetrica_no_tiene_asimetria(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cruda_fija(monkeypatch, np.array([[0.0, 1.0], [1.0, 0.0]]))
        modelo = modelo_sintetico(np.array([[0.0, 1.0], [1.0, 0.0]]))
        res = fases.fase0_sanity(modelo, {}, ctx_jugador, "2")
        assert res["asimetria"] == pytest.approx(0.0)

    def test_la_asimetria_es_la_norma_relativa_de_la_parte_antisimetrica(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """||S-S^T||/||S|| con S=[[0,1],[0,0]]: ||S||=1, ||S-S^T||=sqrt(2)."""
        _cruda_fija(monkeypatch, np.array([[0.0, 1.0], [0.0, 0.0]]))
        modelo = modelo_sintetico(np.zeros((2, 2)))
        res = fases.fase0_sanity(modelo, {}, ctx_jugador, "2")
        assert res["asimetria"] == pytest.approx(np.sqrt(2.0))

    def test_una_s_cruda_toda_a_cero_no_divide_por_cero(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cruda_fija(monkeypatch, np.zeros((3, 3)))
        modelo = modelo_sintetico(np.zeros((3, 3)))
        assert fases.fase0_sanity(modelo, {}, ctx_jugador, "2")["asimetria"] == 0.0

    def test_la_pureza_es_la_fraccion_de_top1_con_el_mismo_rol(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0 y 1 son defensas y se eligen entre si; 2 (ATT) elige a 0 (DEF): 2/3."""
        _cruda_fija(monkeypatch, np.eye(4))
        S = np.array([[0.0, 0.9, 0.1, 0.0],
                      [0.9, 0.0, 0.2, 0.0],
                      [0.8, 0.1, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
        modelo = modelo_sintetico(S, entity_ids=[10, 11, 12, 13])
        roles = {10: "DEF", 11: "DEF", 12: "ATT", 13: "MID"}
        res = fases.fase0_sanity(modelo, roles, ctx_jugador, "5")
        # La entidad 13 no tiene ningun vecino con score != 0: no puntua.
        assert res["n_pureza"] == 3
        assert res["pureza_top1"] == pytest.approx(2.0 / 3.0)

    def test_las_entidades_con_rol_desconocido_no_cuentan(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cruda_fija(monkeypatch, np.eye(2))
        S = np.array([[0.0, 0.5], [0.5, 0.0]])
        modelo = modelo_sintetico(S, entity_ids=[1, 2])
        roles = {1: config.ROL_DESCONOCIDO, 2: "MID"}
        res = fases.fase0_sanity(modelo, roles, ctx_jugador, "5")
        assert res["n_pureza"] == 1

    def test_sin_ninguna_entidad_evaluable_la_pureza_es_nan(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cruda_fija(monkeypatch, np.eye(2))
        modelo = modelo_sintetico(np.zeros((2, 2)), entity_ids=[1, 2])
        res = fases.fase0_sanity(modelo, {}, ctx_jugador, "5")
        assert np.isnan(res["pureza_top1"])
        assert res["n_pureza"] == 0

    def test_el_equipo_no_tiene_pureza_posicional(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Un equipo no tiene rol: la metrica no aplica, y decirlo con NaN es
        distinto de decirlo con un 0 (que pareceria un modelo malisimo).
        """
        _cruda_fija(monkeypatch, np.eye(2))
        modelo = modelo_sintetico(np.array([[0.0, 1.0], [1.0, 0.0]]), entidad="equipo")
        res = fases.fase0_sanity(modelo, {}, ctx_jugador, "5")
        assert np.isnan(res["pureza_top1"])

    def test_informa_del_avance_por_entidad(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cruda_fija(monkeypatch, np.eye(3))
        modelo = modelo_sintetico(np.zeros((3, 3)))
        vistos: list[tuple[int, int]] = []
        fases.fase0_sanity(modelo, {}, ctx_jugador, "5",
                           lambda h, t: vistos.append((h, t)))
        assert vistos == [(1, 3), (2, 3), (3, 3)]


class TestFaceValidity:
    def test_lista_el_top_k_de_cada_nombre_pedido(self) -> None:
        S = np.array([[0.0, 0.9, 0.2], [0.9, 0.0, 0.1], [0.2, 0.1, 0.0]])
        modelo = modelo_sintetico(S, entity_names=["Ana", "Bea", "Cai"])
        lineas = fases.face_validity(modelo, ["Ana"], k=2)
        assert lineas == ["  Ana -> Bea, Cai"]

    def test_un_nombre_que_no_existe_se_salta_sin_romper(self) -> None:
        """La revision cualitativa no puede tumbar la evaluacion entera."""
        modelo = modelo_sintetico(np.zeros((2, 2)), entity_names=["Ana", "Bea"])
        assert fases.face_validity(modelo, ["Zoe"]) == []

    def test_una_entidad_sin_vecinos_lo_dice(self) -> None:
        modelo = modelo_sintetico(np.zeros((2, 2)), entity_names=["Ana", "Bea"])
        assert fases.face_validity(modelo, ["Ana"]) == ["  Ana -> (sin vecinos)"]


# --------------------------------------------------------------------------- #
# Fase 1 (+ Fase 4)                                                             #
# --------------------------------------------------------------------------- #

class TestRangosDireccion:
    @pytest.fixture
    def split(self) -> construccion.SplitVirtual:
        ctx = contexto_sintetico(np.zeros((4, 1)), [1, 1, 2, 2], [1, 2, 1, 2])
        return construccion.split_par_impar(ctx)

    def test_consulta_con_una_mitad_y_recupera_en_el_pool_de_la_otra(
        self, split: construccion.SplitVirtual
    ) -> None:
        """Protocolo de de-anonimizacion (Decroos & Davis): query anonima contra
        un pool etiquetado. El pool son SOLO las mitades del destino.
        """
        S = np.array([[0.0, 5.0, 0.0, 1.0],
                      [5.0, 0.0, 0.0, 0.0],
                      [0.0, 9.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0]])
        rangos, reales, pool = fases._rangos_direccion(S, split, origen_half=0)
        assert pool == 2
        assert rangos.tolist() == [1.0, 2.0]
        assert reales.tolist() == [0, 1]

    def test_la_direccion_inversa_consulta_desde_la_otra_mitad(
        self, split: construccion.SplitVirtual
    ) -> None:
        S = np.zeros((4, 4))
        S[1, 0] = 7.0        # la mitad B de la entidad 0 recupera a su mitad A
        rangos, _, pool = fases._rangos_direccion(S, split, origen_half=1)
        assert pool == 2
        assert rangos[0] == pytest.approx(1.0)

    def test_devuelve_un_rango_por_entidad_evaluable(
        self, split: construccion.SplitVirtual
    ) -> None:
        rangos, reales, _ = fases._rangos_direccion(np.zeros((4, 4)), split, 0)
        assert rangos.size == reales.size == len(split.pares_AB)


class TestResumenAutosim:
    def test_publica_la_tasa_de_azar_y_el_tamano_del_pool(self) -> None:
        """Sin ellos un top-1 del 30% no significa nada: depende del pool."""
        r = fases._resumen_autosim(np.array([1.0, 1.0, 4.0]), pool=50)
        assert r["pool"] == 50
        assert r["azar_top1"] == pytest.approx(1 / 50)
        assert r["n"] == 3

    def test_reporta_los_top_k_declarados_en_config(self) -> None:
        r = fases._resumen_autosim(np.array([1.0, 6.0]), pool=10)
        for k in config.KS_AUTOSIM:
            assert f"top{k}" in r
        assert r["top1"] == pytest.approx(0.5)
        assert r["top10"] == pytest.approx(1.0)

    def test_el_mrr_sale_de_los_rangos(self) -> None:
        r = fases._resumen_autosim(np.array([1.0, 2.0]), pool=4)
        assert r["mrr"] == pytest.approx(0.75)

    def test_un_pool_vacio_no_divide_por_cero(self) -> None:
        assert np.isnan(fases._resumen_autosim(np.array([]), pool=0)["azar_top1"])


class TestFase1Autosimilitud:
    def test_reporta_las_dos_direcciones_y_su_media(self, ctx_jugador) -> None:
        res = fases.fase1_autosimilitud(ctx_jugador, "2")
        assert set(res) == {"global", "A->B", "B->A", "estratos", "denoising"}
        assert res["global"]["n"] == res["A->B"]["n"] + res["B->A"]["n"]

    def test_estratifica_por_minutos_en_jugador(self, ctx_jugador) -> None:
        """El sesgo MNAR sobreestima en los de alto volumen: hay que poder
        comparar cabeza con cola.
        """
        res = fases.fase1_autosimilitud(ctx_jugador, "2")
        assert set(res["estratos"]) <= {"cola", "torso", "cabeza"}
        assert res["estratos"]

    def test_el_equipo_no_se_estratifica(self) -> None:
        """Juega siempre el partido completo: el estrato no distinguiria nada."""
        ctx = contexto_sintetico(np.arange(8.0).reshape(4, 2), [1, 1, 2, 2],
                                 [1, 2, 1, 2], entidad="equipo")
        res = fases.fase1_autosimilitud(ctx, "2")
        assert res["estratos"] == {}

    def test_la_formulacion_2_no_mide_denoising(self, ctx_jugador) -> None:
        """La Fase 4 es una pregunta sobre EASE, que solo tiene la F5."""
        assert fases.fase1_autosimilitud(ctx_jugador, "2")["denoising"] is None

    def test_la_formulacion_5_compara_antes_y_despues_de_ease(self, ctx_jugador) -> None:
        d = fases.fase1_autosimilitud(ctx_jugador, "5")["denoising"]
        assert set(d) == {"pre", "post", "delta_mrr", "delta_rr_medio", "p_pareado"}
        assert d["delta_mrr"] == pytest.approx(d["post"]["mrr"] - d["pre"]["mrr"])
        assert 0.0 < d["p_pareado"] <= 1.0

    def test_el_pool_es_el_de_las_mitades_no_el_de_las_entidades(
        self, ctx_jugador
    ) -> None:
        res = fases.fase1_autosimilitud(ctx_jugador, "2")
        assert res["A->B"]["pool"] == 3      # tres entidades tienen mitad B

    def test_reenvia_el_progreso_al_bucle_caro(self, ctx_jugador) -> None:
        vistos: list[tuple[int, int]] = []
        fases.fase1_autosimilitud(ctx_jugador, "2", lambda h, t: vistos.append((h, t)))
        assert vistos and vistos[-1][0] == vistos[-1][1]


# --------------------------------------------------------------------------- #
# Fase 2                                                                        #
# --------------------------------------------------------------------------- #

class TestFase2Estabilidad:
    def test_un_modelo_que_no_se_mueve_da_rbo_uno(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fija el limite superior de la metrica: si el remuestreo no cambiara la
        S, el top-k seria identico y el RBO@10 exactamente 1.
        """
        S_base = np.array([[0.0, 0.9, 0.2], [0.9, 0.0, 0.4], [0.2, 0.4, 0.0]])
        monkeypatch.setattr(construccion, "reconstruir_S",
                            lambda *a, **kw: S_base)
        res = fases.fase2_estabilidad(ctx_jugador, "2", S_base, B=3)
        assert res["rbo_medio"] == pytest.approx(1.0)
        assert res["ic_bajo"] == pytest.approx(1.0)
        assert res["B"] == 3

    def test_un_top_k_invertido_baja_el_rbo(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        S_base = np.array([[0.0, 0.9, 0.2], [0.9, 0.0, 0.4], [0.2, 0.4, 0.0]])
        monkeypatch.setattr(construccion, "reconstruir_S",
                            lambda *a, **kw: 1.0 - S_base + np.eye(3) * -1.0)
        res = fases.fase2_estabilidad(ctx_jugador, "2", S_base, B=2)
        assert res["rbo_medio"] < 1.0

    def test_solo_promedia_las_entidades_consultables(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Una entidad sin ningun vecino en el modelo servible no tiene top-k que
        comparar: incluirla como RBO=1 (dos listas vacias) inflaria la estabilidad.
        """
        S_base = np.zeros((3, 3))
        S_base[0, 1] = S_base[1, 0] = 0.5
        monkeypatch.setattr(construccion, "reconstruir_S", lambda *a, **kw: S_base)
        res = fases.fase2_estabilidad(ctx_jugador, "2", S_base, B=1)
        assert res["n_consultables"] == 2

    def test_devuelve_intervalo_de_confianza(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rng = np.random.default_rng(0)
        monkeypatch.setattr(
            construccion, "reconstruir_S",
            lambda *a, **kw: np.abs(rng.normal(size=(3, 3))) * (1 - np.eye(3)))
        res = fases.fase2_estabilidad(ctx_jugador, "2", np.ones((3, 3)) - np.eye(3), B=8)
        assert res["ic_bajo"] <= res["rbo_medio"] <= res["ic_alto"]

    def test_el_tope_de_consultas_submuestrea_las_entidades(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Con P grande, rankear O(B x P) veces es inviable; el tope estima el RBO
        medio sobre una submuestra fija por seed.
        """
        S_base = np.ones((3, 3)) - np.eye(3)
        monkeypatch.setattr(construccion, "reconstruir_S", lambda *a, **kw: S_base)
        monkeypatch.setattr(config, "N_ESTABILIDAD_MAX", 2)
        res = fases.fase2_estabilidad(ctx_jugador, "2", S_base, B=1)
        assert res["n_consultables"] == 2

    def test_informa_de_cada_remuestreo(
        self, ctx_jugador, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        S_base = np.ones((3, 3)) - np.eye(3)
        monkeypatch.setattr(construccion, "reconstruir_S", lambda *a, **kw: S_base)
        vistos: list[tuple[int, int]] = []
        fases.fase2_estabilidad(ctx_jugador, "2", S_base, B=3,
                                reportar=lambda h, t: vistos.append((h, t)))
        assert vistos == [(1, 3), (2, 3), (3, 3)]


# --------------------------------------------------------------------------- #
# Fase 3                                                                        #
# --------------------------------------------------------------------------- #

class TestKendallMedio:
    def test_dos_matrices_iguales_dan_acuerdo_perfecto(self) -> None:
        rng = np.random.default_rng(2)
        A = np.abs(rng.normal(size=(6, 6)))
        assert fases._kendall_medio(A, A, cap=10, seed=0) == pytest.approx(1.0)

    def test_el_orden_inverso_da_desacuerdo_total(self) -> None:
        A = np.arange(16.0).reshape(4, 4)
        assert fases._kendall_medio(A, -A, cap=10, seed=0) == pytest.approx(-1.0)

    def test_submuestrea_cuando_hay_mas_entidades_que_el_tope(self) -> None:
        """Es O(n^2) por consulta: sin tope, la Fase 3 no escalaria."""
        rng = np.random.default_rng(3)
        A = np.abs(rng.normal(size=(40, 40)))
        assert np.isfinite(fases._kendall_medio(A, A, cap=5, seed=0))


class TestFase3Triangulacion:
    @pytest.fixture
    def modelos(self) -> dict:
        rng = np.random.default_rng(4)
        S = np.abs(rng.normal(size=(5, 5)))
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)
        return {
            (form, "jugador", norm): modelo_sintetico(
                S * (1.0 + 0.1 * i), formulacion=form, entity_ids=list(range(5)))
            for i, (form, norm) in enumerate(
                [("2", "por_liga"), ("5", "por_liga"),
                 ("2", "global"), ("5", "global")])
        }

    @pytest.fixture
    def ctxs(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        rng = np.random.default_rng(5)
        S_pre = np.abs(rng.normal(size=(5, 5)))
        S_pre = 0.5 * (S_pre + S_pre.T)
        monkeypatch.setattr(construccion, "reconstruir_S_f5",
                            lambda ctx, row, n, **kw: (S_pre, S_pre))
        ctx = contexto_sintetico(np.zeros((5, 1)), list(range(5)), list(range(5)))
        return {("jugador", "por_liga"): ctx, ("jugador", "global"): ctx}

    def test_cubre_las_cuatro_comparaciones_del_protocolo(
        self, modelos: dict, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 9)
        filas = fases.fase3_triangulacion(modelos, ctxs)
        etiquetas = {f["comparacion"] for f in filas}
        assert "F2 vs F5 (por_liga)" in etiquetas
        assert "F2 vs F5-distrib. cruda (por_liga)" in etiquetas
        assert "F5 pre vs post-EASE (por_liga)" in etiquetas
        assert "por_liga vs global (F2)" in etiquetas

    def test_la_kendall_solo_acompana_a_la_comparacion_f2_vs_f5(
        self, modelos: dict, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Es el acuerdo del RANKING por entidad, complementario al r global; en
        las demas comparaciones no aporta y cuesta O(n^2) por consulta.
        """
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 9)
        filas = fases.fase3_triangulacion(modelos, ctxs)
        con_kendall = {f["comparacion"] for f in filas if np.isfinite(f["kendall"])}
        assert con_kendall == {"F2 vs F5 (por_liga)", "F2 vs F5 (global)"}

    def test_cada_fila_declara_entidad_r_y_p(
        self, modelos: dict, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 9)
        for fila in fases.fase3_triangulacion(modelos, ctxs):
            assert fila["entidad"] == "jugador"
            assert -1.0 <= fila["r"] <= 1.0
            assert 0.0 < fila["p"] <= 1.0
            assert fila["n_entidades"] == 5

    def test_dos_modelos_con_universos_distintos_no_se_comparan(
        self, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Si los ids no coinciden, la celda (i,j) de una S no habla del mismo par
        que la de la otra: el r seria un numero sin significado.
        """
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 9)
        S = np.ones((3, 3)) - np.eye(3)
        modelos = {
            ("2", "jugador", "por_liga"): modelo_sintetico(S, entity_ids=[1, 2, 3]),
            ("5", "jugador", "por_liga"): modelo_sintetico(S, entity_ids=[9, 8, 7]),
        }
        filas = fases.fase3_triangulacion(modelos, ctxs)
        assert not [f for f in filas if f["comparacion"].startswith("F2 vs F5 (")]

    def test_sin_pareja_no_hay_comparacion(self, ctxs: dict) -> None:
        S = np.ones((3, 3)) - np.eye(3)
        solo_f2 = {("2", "jugador", "por_liga"): modelo_sintetico(S)}
        assert fases.fase3_triangulacion(solo_f2, ctxs) == []

    def test_con_muchas_entidades_el_mantel_se_estima_sobre_una_submuestra(
        self, modelos: dict, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Permutar el triangulo superior completo 999 veces es inviable con P
        grande; el n de la fila tiene que declarar sobre cuantas se estimo.
        """
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 9)
        monkeypatch.setattr(config, "N_MANTEL_MAX", 3)
        filas = fases.fase3_triangulacion(modelos, ctxs)
        assert {f["n_entidades"] for f in filas} == {3}

    def test_informa_del_avance_por_comparacion(
        self, modelos: dict, ctxs: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "MANTEL_PERMUTACIONES", 3)
        vistos: list[tuple[int, int]] = []
        filas = fases.fase3_triangulacion(modelos, ctxs,
                                          lambda h, t: vistos.append((h, t)))
        assert len(vistos) == 8            # 2 normalizaciones x 3 + 2 formulaciones
        assert vistos[-1] == (8, 8)
        assert len(filas) <= len(vistos)


# --------------------------------------------------------------------------- #
# Fase 5                                                                        #
# --------------------------------------------------------------------------- #

class TestFase5Downstream:
    @pytest.fixture
    def modelo(self):
        """Seis jugadores en dos bloques (1-3 y 4-6) sin similitud entre bloques.

        El 0 entre bloques importa: `top_k_indices` lo descarta, asi que el top-k
        de un defensa son SOLO defensas. Con una S densa, los 10 vecinos del k-NN
        incluirian a todo el pool y la clasificacion no mediria nada.
        """
        dentro = np.array([[0.0, 0.9, 0.8], [0.9, 0.0, 0.7], [0.8, 0.7, 0.0]])
        S = np.zeros((6, 6))
        S[:3, :3] = dentro
        S[3:, 3:] = dentro
        F = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0],
                      [0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
        return modelo_sintetico(S, entity_ids=[1, 2, 3, 4, 5, 6], feat_display=F)

    ROLES = {1: "DEF", 2: "DEF", 3: "DEF", 4: "ATT", 5: "ATT", 6: "ATT"}
    MINUTOS = {1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0, 5: 500.0, 6: 600.0}

    def test_el_knn_posicional_acierta_cuando_s_captura_el_rol(self, modelo) -> None:
        """Es la metrica OBJETIVA de la fase: no depende de umbrales del PDF."""
        res = fases.fase5_downstream(modelo, self.ROLES, self.MINUTOS)
        assert res["knn_accuracy"] == pytest.approx(1.0)
        assert res["knn_f1_macro"] == pytest.approx(1.0)
        assert res["n_knn"] == 6

    def test_una_s_que_mezcla_los_roles_baja_el_accuracy(self, modelo) -> None:
        """Los bloques de S siguen siendo los mismos; lo que cambia es que ya no
        se corresponden con el rol: la metrica tiene que caer.
        """
        roles = {1: "DEF", 2: "ATT", 3: "ATT", 4: "DEF", 5: "DEF", 6: "ATT"}
        res = fases.fase5_downstream(modelo, roles, self.MINUTOS)
        assert res["knn_accuracy"] < 1.0

    def test_las_entidades_sin_rol_no_entran_en_el_knn(self, modelo) -> None:
        roles = {1: "DEF", 2: "DEF", 3: config.ROL_DESCONOCIDO}
        res = fases.fase5_downstream(modelo, roles, self.MINUTOS)
        assert res["n_knn"] == 2

    def test_una_entidad_cuyos_vecinos_no_tienen_rol_no_puede_votar(
        self, modelo
    ) -> None:
        """Sin votos utiles no hay prediccion: inventar una clase la contaria como
        acierto o como fallo, y ninguna de las dos cosas es cierta.
        """
        roles = {1: "DEF", 2: config.ROL_DESCONOCIDO, 3: config.ROL_DESCONOCIDO}
        assert fases.fase5_downstream(modelo, roles, self.MINUTOS)["n_knn"] == 0

    def test_el_equipo_no_tiene_clasificacion_posicional(self) -> None:
        S = np.array([[0.0, 1.0], [1.0, 0.0]])
        modelo = modelo_sintetico(S, entidad="equipo")
        res = fases.fase5_downstream(modelo, {}, {})
        assert np.isnan(res["knn_accuracy"])
        assert np.isnan(res["popularidad_spearman"])

    def test_la_cobertura_es_la_fraccion_de_entidades_recomendadas(self) -> None:
        """Baja = el sistema recomienda siempre a los mismos (sesgo de popularidad)."""
        S = np.array([[0.0, 0.9, 0.0],
                      [0.9, 0.0, 0.0],
                      [0.5, 0.0, 0.0]])
        modelo = modelo_sintetico(S, entity_ids=[1, 2, 3])
        res = fases.fase5_downstream(modelo, {}, {})
        assert res["coverage"] == pytest.approx(2.0 / 3.0)

    def test_la_diversidad_es_la_distancia_media_entre_los_perfiles_del_top_k(
        self, modelo
    ) -> None:
        """Cada lista tiene 2 vecinos. En cada bloque, dos de las tres listas
        emparejan un perfil (0,0) con uno (3,4) —distancia 5— y la tercera empareja
        los dos (0,0) —distancia 0—: media (5+5+0)*2 / 6 = 10/3.
        """
        res = fases.fase5_downstream(modelo, self.ROLES, self.MINUTOS)
        assert res["diversity"] == pytest.approx(10.0 / 3.0)

    def test_una_lista_de_menos_de_dos_vecinos_no_aporta_diversidad(self) -> None:
        S = np.array([[0.0, 0.5], [0.5, 0.0]])
        modelo = modelo_sintetico(S, feat_display=np.array([[0.0], [1.0]]))
        assert np.isnan(fases.fase5_downstream(modelo, {}, {})["diversity"])

    def test_el_indegree_medio_cuenta_las_veces_recomendado(self, modelo) -> None:
        """Seis listas de 2 vecinos = 12 apariciones repartidas entre 6 entidades."""
        res = fases.fase5_downstream(modelo, self.ROLES, self.MINUTOS)
        assert res["indegree_medio"] == pytest.approx(2.0)

    def test_el_sesgo_de_popularidad_correlaciona_indegree_con_minutos(self) -> None:
        """Si los mas recomendados son siempre los de mas minutos, el sistema
        premia volumen y no estilo. Aqui la entidad 4 es la unica recomendada por
        todas y ademas la de mas minutos: la correlacion tiene que salir positiva.
        """
        S = np.array([[0.0, 0.0, 0.0, 0.9],
                      [0.0, 0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.0, 0.7],
                      [0.9, 0.8, 0.7, 0.0]])
        modelo = modelo_sintetico(S, entity_ids=[1, 2, 3, 4])
        res = fases.fase5_downstream(
            modelo, {}, {1: 10.0, 2: 20.0, 3: 30.0, 4: 900.0})
        assert res["popularidad_spearman"] > 0.0

    def test_informa_del_avance_al_construir_las_listas(self, modelo) -> None:
        vistos: list[tuple[int, int]] = []
        fases.fase5_downstream(modelo, self.ROLES, self.MINUTOS,
                               lambda h, t: vistos.append((h, t)))
        assert vistos == [(i, 6) for i in range(1, 7)]
