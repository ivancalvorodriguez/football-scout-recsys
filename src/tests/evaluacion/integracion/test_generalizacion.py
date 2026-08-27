"""Fase 7: el modelo ajustado SIN una liga, medido sobre esa liga.

Va en `integracion/` porque no se puede falsear: la fase reajusta el modelo con
el pipeline real y despues proyecta, asi que necesita una BD de verdad (la
sintetica de la suite, que trae dos ligas con entidades disjuntas — justo la
forma que pide un hold-out por liga).

Lo que se vigila:

- Que el hold-out sea de VERDAD: ninguna entidad de la liga excluida puede haber
  entrado en el ajuste. Es la propiedad de la que depende todo lo demas; si se
  colara, la fase mediria dentro de muestra dos veces y saldrian numeros
  estupendos.
- Que los dos ambitos se midan igual (mismo pool en la auto-similitud), porque
  la fase no informa de niveles sino de la DIFERENCIA entre ellos.
- Que las DOS formulaciones midan las mismas metricas: cada una con su similitud,
  pero ninguna se queda sin gemela. Si una perdiera `gen_top1`, su score compuesto
  se renormalizaria sobre otras metricas y dejaria de ser comparable con el resto.
- Que una particion imposible se diga en vez de devolver metricas vacias con
  aspecto de normales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluacion import config as ecfg
from src.evaluacion import construccion, datos, generalizacion
from src.similitud import foldin

pytestmark = [pytest.mark.integracion, pytest.mark.lento]

LIGA_FUERA = "2-200"     # `factories.LIGAS_SINTETICAS`: la segunda
LIGA_DENTRO = "1-100"


@pytest.fixture(scope="module")
def roles(bd_sintetica: Path) -> dict[int, str]:
    return datos.rol_por_jugador(bd_sintetica)


@pytest.fixture(scope="module")
def combinacion(bd_sintetica: Path, tmp_path_factory) -> dict:
    """Una combinacion de barrido con la fase 7, por el camino de produccion."""
    from src.evaluacion import barrido

    return barrido.evaluar_combinacion(
        "v01", {}, bd_sintetica, tmp_path_factory.mktemp("barrido_gen"),
        {"0", "7"}, 0, con_figuras=False,
        formulaciones=("5",), entidades=("jugador",),
        normalizaciones=("global",), holdout=(LIGA_FUERA,),
    )


@pytest.fixture(scope="module")
def filas_f2(bd_sintetica: Path) -> list[dict]:
    """La fase entera sobre equipo con la F2: la formulacion que hasta ahora se
    quedaba sin auto-similitud fuera de muestra."""
    return generalizacion.fase7_generalizacion(
        bd_sintetica, "equipo", "global", "2", (LIGA_FUERA,),
        datos.rol_por_jugador(bd_sintetica))


@pytest.fixture(scope="module")
def filas(bd_sintetica: Path) -> list[dict]:
    """La fase entera sobre la BD sintetica (jugador, F5, global)."""
    return generalizacion.fase7_generalizacion(
        bd_sintetica, "jugador", "global", "5", (LIGA_FUERA,),
        datos.rol_por_jugador(bd_sintetica),
    )


class TestElHoldoutEsDeVerdad:
    def test_el_ajuste_no_ve_las_entidades_de_la_liga_excluida(
        self, bd_sintetica: Path
    ) -> None:
        """Lo mismo que hace la fase por dentro, comprobado sobre el contexto:
        excluir la liga tiene que quitar sus filas ANTES de estandarizar."""
        completo = construccion.crear_contexto(bd_sintetica, "jugador", "global")
        train = construccion.crear_contexto(
            bd_sintetica, "jugador", "global", excluir_ligas=(LIGA_FUERA,))
        assert set(np.unique(train.mf.league)) == {LIGA_DENTRO}
        assert len(train.idx_real.ids) < len(completo.idx_real.ids)

    def test_las_mu_sd_tampoco_las_ven(self, bd_sintetica: Path) -> None:
        """Si el z-score se calculase con todo, la liga excluida habria influido
        en la escala de las demas: eso ya es fuga."""
        completo = construccion.crear_contexto(bd_sintetica, "jugador", "global")
        train = construccion.crear_contexto(
            bd_sintetica, "jugador", "global", excluir_ligas=(LIGA_FUERA,))
        mu_todo = completo.mf.estadisticas.mu["__global__"]
        mu_train = train.mf.estadisticas.mu["__global__"]
        assert not np.allclose(mu_todo, mu_train)

    def test_las_entidades_evaluadas_fuera_no_estan_en_el_ajuste(
        self, filas: list[dict]
    ) -> None:
        dentro = next(f for f in filas if f["ambito"] == "dentro")
        fuera = next(f for f in filas if f["ambito"] == "fuera")
        assert fuera["n_holdout"] > 0
        assert dentro["n_train"] == fuera["n_train"]
        # El pool de candidatos es el mismo (las entidades del ajuste) y el
        # numero de consultas, distinto: son dos conjuntos disjuntos.
        assert fuera["n_entidades"] == fuera["n_holdout"]


class TestQueDevuelve:
    def test_una_fila_por_ambito(self, filas: list[dict]) -> None:
        assert [f["ambito"] for f in filas] == ["dentro", "fuera"]

    def test_declara_la_fidelidad_de_la_proyeccion(self, filas: list[dict]) -> None:
        """Sin eso no se puede leer la brecha: no es lo mismo comparar contra una
        proyeccion fiel (F5) que contra media magnitud (F2)."""
        assert all(f["fidelidad"] == foldin.FIEL for f in filas)

    def test_mide_todas_las_metricas_declaradas(self, filas: list[dict]) -> None:
        """La fase existe para dar la GEMELA de cada metrica: si una se quedara
        sin calcular, el score perderia su termino sin que nadie se entere."""
        for f in filas:
            assert set(generalizacion.METRICAS) <= set(f)
        fuera = next(f for f in filas if f["ambito"] == "fuera")
        assert all(np.isfinite(fuera[m]) for m in generalizacion.METRICAS)

    def test_la_estabilidad_solo_se_mide_fuera(self, filas: list[dict]) -> None:
        """Dentro de muestra su gemela es la Fase 2, que remuestrea
        reconstruyendo la S entera: mezclarlas compararia dos protocolos."""
        dentro = next(f for f in filas if f["ambito"] == "dentro")
        fuera = next(f for f in filas if f["ambito"] == "fuera")
        assert not np.isfinite(dentro["rbo_medio"])
        assert np.isfinite(fuera["rbo_medio"])

    def test_los_dos_ambitos_consultan_lo_mismo(self, filas: list[dict]) -> None:
        """La cobertura depende del numero de consultas: con tamaños distintos
        estaria midiendo el tamaño."""
        assert len({f["n_entidades"] for f in filas}) == 1

    def test_la_autosimilitud_usa_el_mismo_pool_en_los_dos(
        self, filas: list[dict]
    ) -> None:
        """El top-1 baja al crecer el pool: con pools distintos, la brecha
        mediria el tamano del pool y no la generalizacion."""
        pools = {f["pool_autosim"] for f in filas}
        assert len(pools) == 1

    def test_las_metricas_estan_en_rango(self, filas: list[dict]) -> None:
        """Todas menos la diversidad son fracciones; la diversidad es una
        distancia y solo tiene que ser no negativa."""
        for f in filas:
            for clave in ("top1", "mrr", "pureza_top1", "knn_accuracy",
                          "rbo_medio", "coverage"):
                valor = f[clave]
                assert not np.isfinite(valor) or 0.0 <= valor <= 1.0
            assert not np.isfinite(f["diversity"]) or f["diversity"] >= 0.0

    def test_recupera_a_alguien_fuera_de_muestra(self, filas: list[dict]) -> None:
        """La comprobacion minima de que la proyeccion hace algo: las entidades
        de la liga excluida reciben candidatos, y no todas el mismo."""
        fuera = next(f for f in filas if f["ambito"] == "fuera")
        assert fuera["distintos_top1"] >= 1
        assert np.isfinite(fuera["score_top1"])


class TestEquipo:
    def test_la_f2_se_mide_igual_pero_marcada_como_aproximada(
        self, filas_f2: list[dict]
    ) -> None:
        """El rotulo es de la PROYECCION contra el modelo, que sigue siendo media
        magnitud. No dice nada de la auto-similitud, que es otra pregunta."""
        assert all(f["fidelidad"] == foldin.APROXIMADA for f in filas_f2)
        fuera = next(f for f in filas_f2 if f["ambito"] == "fuera")
        assert np.isfinite(fuera["coverage"]) and np.isfinite(fuera["diversity"])
        assert np.isfinite(fuera["rbo_medio"])

    def test_la_f2_tambien_tiene_autosimilitud_fuera_de_muestra(
        self, filas_f2: list[dict]
    ) -> None:
        """Entre dos entidades NUEVAS no falta ninguna direccion de la agregacion
        —las dos columnas se resuelven al medir—, asi que la F2 no se queda sin
        `gen_top1` ni `gen_mrr`. Es la mitad del peso de generalizacion del score:
        sin esto, la F2 y la F5 no se puntuaban sobre las mismas metricas."""
        for f in filas_f2:
            assert np.isfinite(f["top1"]) and np.isfinite(f["mrr"])
            assert 0.0 <= f["top1"] <= 1.0 and 0.0 <= f["mrr"] <= 1.0

    def test_los_dos_ambitos_se_miden_con_el_mismo_pool(
        self, filas_f2: list[dict]
    ) -> None:
        """Igual que en la F5: el top-1 baja al crecer el pool, asi que con pools
        distintos la brecha mediria el pool."""
        assert len({f["pool_autosim"] for f in filas_f2}) == 1
        assert next(iter(filas_f2))["pool_autosim"] > 0

    def test_el_rol_no_se_le_aplica_a_los_equipos(
        self, bd_sintetica: Path, roles: dict[int, str]
    ) -> None:
        """Los ids de equipo y de jugador viven en numeraciones distintas: sin
        filtrar, un equipo «acertaria» el rol del futbolista con su mismo id."""
        filas = generalizacion.fase7_generalizacion(
            bd_sintetica, "equipo", "global", "2", (LIGA_FUERA,), roles)
        assert all(f["n_rol"] == 0 for f in filas)


class TestParticionesImposibles:
    def test_sin_holdout_no_se_calcula_nada(
        self, bd_sintetica: Path, roles: dict[int, str]
    ) -> None:
        with pytest.raises(generalizacion.SinHoldout):
            generalizacion.fase7_generalizacion(
                bd_sintetica, "jugador", "global", "5", (), roles)

    def test_una_liga_que_no_deja_a_nadie_fuera_se_avisa(
        self, bd_sintetica: Path, roles: dict[int, str]
    ) -> None:
        """Excluir una liga inexistente deja el ajuste completo y el hold-out
        vacio: sin este control saldrian metricas «de fuera» sin nadie dentro."""
        with pytest.raises(generalizacion.SinHoldout):
            generalizacion.fase7_generalizacion(
                bd_sintetica, "jugador", "global", "5", ("999-999",), roles)

    def test_excluirlo_todo_tampoco(
        self, bd_sintetica: Path, roles: dict[int, str]
    ) -> None:
        with pytest.raises(generalizacion.SinHoldout):
            generalizacion.fase7_generalizacion(
                bd_sintetica, "jugador", "global", "5",
                (LIGA_DENTRO, LIGA_FUERA), roles)


class TestResolverHoldout:
    LIGAS = {"1238-108": "Indian Super league 2021/2022",
             "11-27": "La Liga 2015/2016"}

    def test_por_clave(self) -> None:
        assert generalizacion.resolver_holdout(self.LIGAS, ["1238-108"]) == ("1238-108",)

    def test_por_trozo_del_nombre(self) -> None:
        """La clave es un par de ids que nadie recuerda; el nombre, si."""
        assert generalizacion.resolver_holdout(self.LIGAS, ["india"]) == ("1238-108",)

    def test_sin_duplicados_aunque_se_pida_dos_veces(self) -> None:
        assert generalizacion.resolver_holdout(
            self.LIGAS, ["india", "1238-108"]) == ("1238-108",)

    def test_lo_que_no_casa_es_un_error(self) -> None:
        """Seguir adelante evaluaria un hold-out vacio, que da metricas de
        aspecto normal y no significan nada."""
        with pytest.raises(generalizacion.SinHoldout, match="ninguna liga"):
            generalizacion.resolver_holdout(self.LIGAS, ["bundesliga"])


class TestTopeDeEntidades:
    def test_se_acotan_los_dos_ambitos_por_igual(
        self, bd_sintetica: Path, roles: dict[int, str], monkeypatch
    ) -> None:
        """El tope existe para acotar el coste (una proyeccion por entidad) sin
        romper la comparabilidad: si recorta, recorta los dos lados."""
        monkeypatch.setattr(ecfg, "N_GENERALIZACION_MAX", 2)
        filas = generalizacion.fase7_generalizacion(
            bd_sintetica, "jugador", "global", "5", (LIGA_FUERA,), roles)
        assert all(f["n_entidades"] <= 2 for f in filas)


class TestLlegaHastaElScore:
    """De la fase al score compuesto, por el camino real del barrido.

    La generalizacion no vale de nada si se queda en un CSV que nadie mira: tiene
    que entrar en la comparativa y en el numero con el que se ordenan las
    combinaciones. Este es el unico sitio donde se comprueba entero el cableado
    (fase -> tabla larga -> pesos), que son cuatro modulos y es justo lo que se
    rompe en silencio al añadir una metrica.
    """

    def test_el_barrido_corre_la_fase_7(self, combinacion: dict) -> None:
        assert combinacion["f7"]
        assert {r["ambito"] for r in combinacion["f7"]} == {"dentro", "fuera"}

    def test_todas_las_gemelas_viajan_a_la_tabla_larga(self, combinacion: dict) -> None:
        from src.evaluacion import barrido

        df = barrido._tabla_larga_csv({"v01": combinacion})
        metricas = set(df["metrica"])
        for m in generalizacion.METRICAS:
            assert generalizacion.nombre_metrica(m) in metricas

    def test_cada_gemela_lleva_el_valor_FUERA_de_muestra(
        self, combinacion: dict
    ) -> None:
        """No el de dentro y no una brecha: el nivel sobre la liga que no vio."""
        from src.evaluacion import barrido

        df = barrido._tabla_larga_csv({"v01": combinacion})
        fuera = next(r for r in combinacion["f7"] if r["ambito"] == "fuera")
        for m in generalizacion.METRICAS:
            fila = df[df["metrica"] == generalizacion.nombre_metrica(m)]
            assert fila["valor"].iloc[0] == pytest.approx(fuera[m])

    def test_no_se_publica_ninguna_brecha(self, combinacion: dict) -> None:
        """Se sustituyo por el valor absoluto: una brecha de cero puede ser un
        modelo que generaliza igual de mal que de bien."""
        from src.evaluacion import barrido

        df = barrido._tabla_larga_csv({"v01": combinacion})
        assert not any("brecha" in str(m) for m in df["metrica"])

    def test_entran_en_el_score_compuesto(self, combinacion: dict) -> None:
        from src.evaluacion import barrido, puntuacion

        df = barrido._tabla_larga_csv({"v01": combinacion})
        scores = puntuacion.puntuar(df)
        assert not scores.empty
        for m in generalizacion.METRICAS:
            columna = f"z_{generalizacion.nombre_metrica(m)}"
            assert scores[columna].notna().all(), columna

    def test_la_f2_llega_al_score_con_las_mismas_metricas_que_la_f5(
        self, bd_sintetica: Path, tmp_path_factory
    ) -> None:
        """El motivo de que la F2 mida la auto-similitud fuera de muestra: si le
        faltaran `gen_top1` y `gen_mrr`, su score se renormalizaria sobre un
        conjunto de metricas distinto del de la F5 y comparar las dos
        formulaciones por score dejaria de significar nada."""
        from src.evaluacion import barrido, puntuacion

        combi = barrido.evaluar_combinacion(
            "v01", {}, bd_sintetica, tmp_path_factory.mktemp("barrido_gen_f2"),
            {"0", "7"}, 0, con_figuras=False,
            formulaciones=("2",), entidades=("equipo",),
            normalizaciones=("global",), holdout=(LIGA_FUERA,),
        )
        df = barrido._tabla_larga_csv({"v01": combi})
        medidas = set(df["metrica"])
        for m in ("top1", "mrr"):
            assert generalizacion.nombre_metrica(m) in medidas
        scores = puntuacion.puntuar(df)
        for m in ("top1", "mrr"):
            columna = f"z_{generalizacion.nombre_metrica(m)}"
            assert scores[columna].notna().all(), columna

    def test_cuentan_como_metricas_propias(self, combinacion: dict) -> None:
        """`n_metricas` es lo que audita cuantas han entrado de verdad: con las
        gemelas tiene que subir, no quedarse en las de dentro."""
        from src.evaluacion import barrido, puntuacion

        df = barrido._tabla_larga_csv({"v01": combinacion})
        scores = puntuacion.puntuar(df)
        assert scores["n_metricas"].iloc[0] >= len(generalizacion.METRICAS)


class TestElBarridoAjustaSinLaLiga:
    """Con `--holdout`, el artefacto del barrido NO se ajusta con esa liga.

    Es la diferencia entre el experimento que se quiere hacer («entreno con estas
    ligas y veo que hace con la que dejo fuera») y otro parecido pero distinto
    (entrenar con todo y medir aparte). Se comprueba sobre el artefacto que queda
    en disco, que es el que se promoveria a produccion.
    """

    def test_el_artefacto_no_contiene_entidades_de_la_liga_excluida(
        self, combinacion: dict, bd_sintetica: Path, tmp_path_factory
    ) -> None:
        from src.evaluacion import barrido
        from src.similitud.modelo import cargar_modelo

        destino = tmp_path_factory.mktemp("barrido_sin_liga")
        barrido.evaluar_combinacion(
            "v01", {}, bd_sintetica, destino, {"7"}, 0, con_figuras=False,
            formulaciones=("5",), entidades=("jugador",),
            normalizaciones=("global",), holdout=(LIGA_FUERA,),
        )
        modelo = cargar_modelo(destino / "v01" / "modelo", "5", "jugador", "global")
        ligas = {liga for ligas in modelo.meta["ligas_por_entidad"].values()
                 for liga in ligas}
        assert LIGA_FUERA not in ligas
        assert ligas == {LIGA_DENTRO}

    def test_la_fase_7_no_vuelve_a_ajustar(
        self, bd_sintetica: Path, roles: dict[int, str], monkeypatch
    ) -> None:
        """Si el modelo que se le pasa ya excluye la liga, reajustar seria medir
        la generalizacion de un gemelo del artefacto en vez de la del artefacto."""
        from src.similitud import formulacion5

        ctx = construccion.crear_contexto(
            bd_sintetica, "jugador", "global", excluir_ligas=(LIGA_FUERA,))
        modelo, estado = formulacion5.construir_con_estado(
            ctx.mf, "jugador", "global")

        def no_ajustar(*a, **k):
            pytest.fail("la fase 7 no debe reajustar si le dan el modelo hecho")

        monkeypatch.setattr(formulacion5, "construir_con_estado", no_ajustar)
        filas = generalizacion.fase7_generalizacion(
            bd_sintetica, "jugador", "global", "5", (LIGA_FUERA,), roles,
            modelo=modelo, estado=estado)
        assert len(filas) == 2

    def test_pero_si_el_modelo_SI_vio_la_liga_lo_reajusta(
        self, bd_sintetica: Path, roles: dict[int, str]
    ) -> None:
        """No se fia de quien lo llama: lo comprueba contra la BD. Un modelo con
        la liga dentro no vale como hold-out por mucho que se pase como tal."""
        from src.similitud import formulacion5

        ctx = construccion.crear_contexto(bd_sintetica, "jugador", "global")
        modelo, estado = formulacion5.construir_con_estado(
            ctx.mf, "jugador", "global")
        filas = generalizacion.fase7_generalizacion(
            bd_sintetica, "jugador", "global", "5", (LIGA_FUERA,), roles,
            modelo=modelo, estado=estado)
        # El reajuste interno deja fuera la liga, asi que el modelo evaluado tiene
        # MENOS entidades que el que se le paso.
        assert filas[0]["n_train"] < len(modelo.entity_ids)
