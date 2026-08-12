"""Barrido: producto cartesiano, poda de ejes, cache de artefactos y comparativa.

Aqui no se construye ni se evalua nada (eso cuesta minutos y vive en
`integracion/`): se prueba la logica que decide QUE combinaciones hay, que ejes
tiene sentido barrer, de donde sale cada artefacto y como queda el resumen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluacion import barrido, huella, registro
from src.evaluacion import config as ecfg
from src.similitud import config as scfg
from src.tests.evaluacion.conftest import fila_metrica, modelo_sintetico


@pytest.fixture
def rejilla_pequena(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dos ejes con dos valores: 4 combinaciones en vez de las ~300 reales."""
    monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                        [("F2_L1", [0.25, 0.5]), ("F5_EASE_LAMBDA", [10.0, 50.0])])


class TestValidarHiperparametros:
    def test_la_lista_del_repo_es_valida(self) -> None:
        """Vigila la lista de verdad: un typo en `HIPERPARAMETROS` no sobrescribe
        nada y el barrido acabaria comparando combinaciones identicas en silencio.
        """
        barrido._validar_hiperparametros()

    def test_un_nombre_que_no_existe_en_config_aborta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(barrido, "HIPERPARAMETROS", [("F2_L2_INVENTADO", [1])])
        with pytest.raises(SystemExit, match="hiperparametro desconocido"):
            barrido._validar_hiperparametros()

    def test_un_eje_sin_valores_aborta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(barrido, "HIPERPARAMETROS", [("F2_L1", [])])
        with pytest.raises(SystemExit, match="ningun valor"):
            barrido._validar_hiperparametros()

    def test_un_eje_declarado_dos_veces_aborta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """El segundo pisaria al primero y la rejilla no seria la que se lee."""
        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F2_L1", [0.1]), ("F2_L1", [0.2])])
        with pytest.raises(SystemExit, match="dos veces"):
            barrido._validar_hiperparametros()


class TestConfigTemporal:
    def test_fija_el_valor_mientras_dura_el_bloque(self) -> None:
        """Todo el pipeline lee `src.similitud.config` en tiempo de llamada: fijar
        el atributo es lo que hace efectiva una combinacion.
        """
        with barrido._config_temporal({"F2_L1": 0.987}):
            assert scfg.F2_L1 == 0.987

    def test_restaura_el_valor_anterior(self) -> None:
        antes = scfg.F2_L1
        with barrido._config_temporal({"F2_L1": 0.987}):
            pass
        assert scfg.F2_L1 == antes

    def test_restaura_tambien_si_el_bloque_falla(self) -> None:
        """Si no, una combinacion que peta contaminaria todas las siguientes."""
        antes = scfg.F2_L1
        with pytest.raises(RuntimeError):
            with barrido._config_temporal({"F2_L1": 0.5}):
                raise RuntimeError("boom")
        assert scfg.F2_L1 == antes

    def test_sin_valores_no_cambia_nada(self) -> None:
        antes = scfg.F2_L1
        with barrido._config_temporal({}):
            assert scfg.F2_L1 == antes


class TestEjesYCombinaciones:
    def test_solo_se_barren_los_ejes_que_afectan_a_la_rejilla_pedida(
        self, rejilla_pequena
    ) -> None:
        """La lambda del EASE no toca ningun modelo F2: barrerla con
        `--formulaciones 2` solo duplicaria columnas identicas y el tiempo.
        """
        assert barrido._ejes_relevantes(("2",), ("equipo",)) == ["F2_L1"]
        assert barrido._ejes_relevantes(("5",), ("equipo",)) == ["F5_EASE_LAMBDA"]

    def test_con_la_rejilla_entera_se_barren_todos(self, rejilla_pequena) -> None:
        assert barrido._ejes_relevantes(("2", "5"), ("jugador", "equipo")) == [
            "F2_L1", "F5_EASE_LAMBDA"]

    def test_las_configuraciones_son_el_producto_cartesiano(
        self, rejilla_pequena
    ) -> None:
        combos = barrido.configuraciones(("2", "5"), ("equipo",))
        assert len(combos) == 4
        assert {c["F2_L1"] for c in combos} == {0.25, 0.5}

    def test_un_eje_podado_desaparece_del_producto(self, rejilla_pequena) -> None:
        combos = barrido.configuraciones(("2",), ("equipo",))
        assert combos == [{"F2_L1": 0.25}, {"F2_L1": 0.5}]

    def test_sin_hiperparametros_hay_una_sola_combinacion_vacia(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(barrido, "HIPERPARAMETROS", [])
        assert barrido.configuraciones(("2",), ("equipo",)) == [{}]

    def test_los_nombres_posicionales_son_solo_un_respaldo(
        self, rejilla_pequena
    ) -> None:
        """Los nombres reales de una carpeta viven en su `combinaciones.json`:
        esto solo lo usa `figuras3d` con una carpeta sin registro ni resumen.
        """
        nombres = barrido.combinaciones(("2", "5"), ("equipo",))
        assert list(nombres) == ["v01", "v02", "v03", "v04"]

    def test_los_ejes_publicados_incluyen_los_que_no_se_barren(
        self, rejilla_pequena
    ) -> None:
        """Un eje ausente no significa "sin EASE": significa lo que diga el
        default del repo el dia de la ejecucion.
        """
        assert barrido._ejes_publicados() == ["F2_L1", "F5_EASE_LAMBDA"]

    def test_los_ejes_extra_del_registro_se_siguen_publicando(
        self, rejilla_pequena
    ) -> None:
        """Forman parte de la IDENTIDAD de las combinaciones ya registradas: si
        dejaran de contar, la misma configuracion recibiria un nombre nuevo.
        """
        assert "F2_BETA" in barrido._ejes_publicados(["F2_BETA"])

    def test_un_eje_que_ya_no_existe_en_config_se_descarta(
        self, rejilla_pequena
    ) -> None:
        assert "EJE_BORRADO" not in barrido._ejes_publicados(["EJE_BORRADO"])

    def test_la_configuracion_efectiva_completa_con_los_defaults(self) -> None:
        efectiva = barrido._config_efectiva({"F2_L1": 0.9}, ["F2_L1", "F2_BETA"])
        assert efectiva == {"F2_L1": 0.9, "F2_BETA": scfg.F2_BETA}


class TestCacheDeArtefactos:
    STEM = "formulacion2_equipo_global"

    @pytest.fixture
    def bd(self, tmp_path: Path) -> Path:
        ruta = tmp_path / "scouting.db"
        ruta.write_bytes(b"una base de datos cualquiera")
        return ruta

    def _artefacto(self, model_dir: Path, h: dict) -> None:
        modelo_sintetico(np.zeros((2, 2)), formulacion="2", entidad="equipo",
                         meta={"normalizacion": "global"}).guardar(model_dir)
        huella.escribir(model_dir, self.STEM, h)

    def test_el_nombre_del_artefacto_es_el_que_genera_el_modelo(self) -> None:
        assert barrido._stem("2", "equipo", "global") == self.STEM

    def test_encuentra_un_artefacto_identico_en_otra_combinacion(
        self, tmp_path: Path, bd: Path
    ) -> None:
        """Es lo que hace que una combinacion que solo mueve `F2_L1` copie los
        cuatro modelos F5 de otra en vez de reconstruirlos (>10 min cada F2).
        """
        h = huella.calcular("2", "equipo", "global", bd)
        self._artefacto(tmp_path / "v01" / "modelo", h)
        origen = barrido._origen_reutilizable(
            tmp_path, tmp_path / "v02" / "modelo", self.STEM, h)
        assert origen == tmp_path / "v01" / "modelo"

    def test_no_se_copia_a_si_misma(self, tmp_path: Path, bd: Path) -> None:
        h = huella.calcular("2", "equipo", "global", bd)
        model_dir = tmp_path / "v01" / "modelo"
        self._artefacto(model_dir, h)
        assert barrido._origen_reutilizable(tmp_path, model_dir, self.STEM, h) is None

    def test_una_huella_distinta_no_vale_como_origen(
        self, tmp_path: Path, bd: Path
    ) -> None:
        h = huella.calcular("2", "equipo", "global", bd)
        self._artefacto(tmp_path / "v01" / "modelo", {**h, "codigo": "otro"})
        assert barrido._origen_reutilizable(
            tmp_path, tmp_path / "v02" / "modelo", self.STEM, h) is None

    def test_copiar_lleva_los_tres_ficheros(self, tmp_path: Path, bd: Path) -> None:
        """El .npz, el .json y la huella: sin la huella, la siguiente ejecucion
        volveria a reconstruirlo.
        """
        h = huella.calcular("2", "equipo", "global", bd)
        origen = tmp_path / "v01" / "modelo"
        self._artefacto(origen, h)
        destino = tmp_path / "v02" / "modelo"
        barrido._copiar_artefacto(origen, destino, self.STEM)
        for ext in (".npz", ".json", huella.SUFIJO):
            assert (destino / f"{self.STEM}{ext}").exists()

    def test_la_ruta_del_log_es_relativa_al_directorio_de_trabajo(self) -> None:
        """Todos los comandos se lanzan desde la raiz del repo: la relativa
        identifica el fichero sin llenar la linea con el prefijo de la instalacion.
        """
        p = Path.cwd() / "outputs" / "x.npz"
        assert barrido._ruta_visible(p) == str(Path("outputs") / "x.npz")

    def test_una_ruta_de_fuera_se_imprime_absoluta(self, tmp_path: Path) -> None:
        assert Path(barrido._ruta_visible(tmp_path / "x.npz")).is_absolute()


class TestPlan:
    """El plan de la ejecucion: quien ajusta cada artefacto y quien lo copia.

    Se prueba sin construir nada (los `Paso` se fabrican a mano): lo que decide el
    reparto es la HUELLA, y con celdas de la misma huella basta para comprobar que
    solo una queda como propietaria. El plan calculado contra disco de verdad se
    prueba en `integracion/test_barrido_cache.py`.
    """

    def _paso(self, combinacion: str, celda: tuple, h: dict, via: str = "construye"):
        return barrido.Paso(
            combinacion=combinacion, valores={}, celda=barrido.Celda(*celda),
            model_dir=Path(combinacion) / "modelo", huella=h, via=via)

    def test_solo_una_combinacion_ajusta_cada_artefacto(self) -> None:
        """Sin esto, N trabajadores arrancando a la vez sobre combinaciones
        consecutivas ajustarian N veces el mismo modelo: ninguno veria la huella
        del otro hasta terminar.
        """
        celda = ("2", "equipo", "global")
        pasos = [self._paso("v01", celda, {"h": 1}),
                 self._paso("v02", celda, {"h": 1}),
                 self._paso("v03", celda, {"h": 1})]
        barrido._repartir_construcciones(pasos)
        assert [p.via for p in pasos] == ["construye", "copia", "copia"]
        assert all(p.origen == Path("v01") / "modelo" for p in pasos[1:])

    def test_dos_huellas_distintas_se_ajustan_las_dos(self) -> None:
        celda = ("2", "equipo", "global")
        pasos = [self._paso("v01", celda, {"h": 1}),
                 self._paso("v02", celda, {"h": 2})]
        barrido._repartir_construcciones(pasos)
        assert [p.via for p in pasos] == ["construye", "construye"]

    def test_no_toca_las_celdas_que_ya_salen_de_la_cache(self) -> None:
        celda = ("2", "equipo", "global")
        pasos = [self._paso("v01", celda, {"h": 1}, via="cache"),
                 self._paso("v02", celda, {"h": 1})]
        barrido._repartir_construcciones(pasos)
        assert [p.via for p in pasos] == ["cache", "construye"]

    def test_agrupa_para_evaluar_las_celdas_con_la_misma_huella(self) -> None:
        """Mismo artefacto + fases deterministas = mismas metricas: se evalua una
        vez y el resultado se reparte."""
        celda = ("5", "jugador", "global")
        pasos = [self._paso("v01", celda, {"h": 1}),
                 self._paso("v02", celda, {"h": 1}),
                 self._paso("v03", celda, {"h": 2})]
        grupos = barrido._agrupar(pasos, reutilizar=True)
        assert sorted(len(g) for g in grupos.values()) == [1, 2]

    def test_sin_reutilizar_cada_celda_va_por_su_cuenta(self) -> None:
        celda = ("5", "jugador", "global")
        pasos = [self._paso("v01", celda, {"h": 1}),
                 self._paso("v02", celda, {"h": 1})]
        assert len(barrido._agrupar(pasos, reutilizar=False)) == 2

    def test_una_celda_de_otra_formulacion_nunca_se_agrupa(self) -> None:
        """La huella no lleva la formulacion dentro, pero el stem si: dos modelos
        distintos con hiperparametros que casualmente coincidan no son el mismo.
        """
        pasos = [self._paso("v01", ("2", "equipo", "global"), {"h": 1}),
                 self._paso("v01", ("5", "equipo", "global"), {"h": 1})]
        assert len(barrido._agrupar(pasos, reutilizar=True)) == 2

    def test_el_orden_del_informe_es_el_de_evaluar(self) -> None:
        """Los CSV de una combinacion tienen que salir igual que si se hubiera
        evaluado del tiron, no en el orden en que conviene ejecutarlas."""
        orden = barrido._orden_informe(ecfg.FORMULACIONES, ecfg.ENTIDADES)
        esperado = [(f, e, n) for f in ecfg.FORMULACIONES
                    for e in ecfg.ENTIDADES for n in ecfg.NORMALIZACIONES]
        assert [(c.formulacion, c.entidad, c.normalizacion) for c in orden] == esperado

    def test_el_informe_solo_lleva_las_normalizaciones_pedidas(self) -> None:
        """Con `--normalizaciones global` la otra no se construye: si siguiera en
        el orden del informe, `evaluar_plan` la buscaria en un plan que no la tiene.
        """
        orden = barrido._orden_informe(("5",), ("equipo",), ("global",))
        assert [c.normalizacion for c in orden] == ["global"]


class TestTablaLarga:
    def test_descompone_el_modelo_en_sus_tres_ejes(self) -> None:
        """La etiqueta compacta sola no dice de que eje viene cada palabra."""
        salida = {"f0": [{"formulacion": "5", "entidad": "equipo",
                          "normalizacion": "global", "pureza_top1": 0.5,
                          "asimetria": 0.02}]}
        df = barrido._tabla_larga_csv({"v01": {**salida, "f1": [], "f2": [], "f5": []}})
        fila = df[df["metrica"] == "asimetria"].iloc[0]
        assert fila["modelo"] == "F5_equipo_global"
        assert (fila["formulacion"], fila["entidad"], fila["normalizacion"]) == (
            "5", "equipo", "global")

    def test_la_fase_1_aporta_solo_la_direccion_media(self) -> None:
        """Las tres direcciones en la misma columna darian tres puntos por casilla."""
        base = {"formulacion": "2", "entidad": "equipo", "normalizacion": "global"}
        salida = {"f0": [], "f2": [], "f5": [],
                  "f1": [{**base, "direccion": d, "top1": v, "mrr": v}
                         for d, v in (("global", 0.5), ("A->B", 0.4), ("B->A", 0.6))]}
        df = barrido._tabla_larga_csv({"v01": salida})
        assert df[df["metrica"] == "top1"]["valor"].tolist() == [0.5]

    def test_recoge_una_metrica_por_fase_declarada(self) -> None:
        vacio = {k: [] for k in ("f0", "f1", "f2", "f5")}
        assert barrido._tabla_larga_csv({"v01": vacio}).empty


class TestComparativa:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.1),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.3),
            fila_metrica("v01", "F5_equipo_global", "top1", 0.5),
            fila_metrica("v02", "F5_equipo_global", "top1", 0.4),
        ])

    def test_una_fila_por_modelo_y_una_columna_por_combinacion(self, df) -> None:
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert "| formulacion | entidad | normalizacion | v01 | v02 |" in L

    def test_resalta_el_mejor_valor_de_cada_fila(self, df) -> None:
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert any("| F2 | equipo | global | 0.100 | **0.300** |" == ln for ln in L)

    def test_para_una_metrica_de_menor_es_mejor_resalta_el_minimo(self) -> None:
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "asimetria", 0.9, fase="0"),
            fila_metrica("v02", "F2_equipo_global", "asimetria", 0.1, fase="0"),
        ])
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "asimetria", "Asimetria", ["v01", "v02"])
        assert any("**0.100**" in ln for ln in L)

    def test_no_resalta_nada_si_las_combinaciones_no_discrepan(self) -> None:
        """Una fila con el mismo valor en todas es el MISMO modelo reutilizado
        desde la cache (el equipo no tiene posicion): resaltarlo haria creer que
        gana algo.
        """
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.5),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.5),
        ])
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert not any("**" in ln for ln in L if ln.startswith("| F2"))
        assert any("Ninguna combinacion mueve esta metrica" in ln for ln in L)

    def test_declara_la_combinacion_de_mejor_media(self, df) -> None:
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert any("Mejor media entre combinaciones: **v02**" in ln for ln in L)

    def test_solo_promedia_columnas_con_los_mismos_modelos(self) -> None:
        """Con la carpeta acumulada, una combinacion evaluada sobre media rejilla
        tendria una media que no compara lo mismo que las demas: v03 solo tiene
        uno de los dos modelos, y con el valor mas alto de la tabla. Aun asi no
        puede ganar la media.
        """
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.1),
            fila_metrica("v01", "F5_equipo_global", "top1", 0.2),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.3),
            fila_metrica("v02", "F5_equipo_global", "top1", 0.4),
            fila_metrica("v03", "F2_equipo_global", "top1", 0.99),
        ])
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02", "v03"])
        assert any("Mejor media entre combinaciones: **v02**" in ln for ln in L)

    def test_una_celda_sin_evaluar_se_publica_como_no_disponible(self) -> None:
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.1),
            fila_metrica("v01", "F5_equipo_global", "top1", 0.2),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.9),
        ])
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert any(ln.endswith("| n/a |") for ln in L)

    def test_una_columna_sin_ningun_valor_se_omite_entera(self, df) -> None:
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02", "v99"])
        assert not any("v99" in ln for ln in L)

    def test_una_metrica_que_nadie_calculo_no_genera_tabla(self, df) -> None:
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "rbo_medio", "RBO", ["v01"])
        assert L == []

    def test_sin_ninguna_columna_completa_no_se_declara_media(self) -> None:
        """Cada combinacion trae un modelo distinto: promediar compararia cosas
        distintas, asi que la tabla lo dice en vez de proclamar una ganadora.
        """
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.1),
            fila_metrica("v02", "F5_equipo_global", "top1", 0.9),
        ])
        L: list[str] = []
        barrido._tabla_comparativa(L, df, "top1", "Top-1", ["v01", "v02"])
        assert any("sin media comparable" in ln for ln in L)

    def test_los_modelos_se_agrupan_por_entidad(self) -> None:
        """Jugador y equipo no son comparables entre si: no pueden ir alternados."""
        claves = [("2", "jugador", "global"), ("5", "equipo", "global"),
                  ("2", "equipo", "global")]
        assert [c[1] for c in sorted(claves, key=barrido._orden_modelo)] == [
            "equipo", "equipo", "jugador"]

    def test_los_valores_no_finitos_no_entran_en_la_tabla(self) -> None:
        df = pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", float("nan")),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.3),
        ])
        assert list(barrido._valores_de(df, "top1")) == [
            ("v02", ("2", "equipo", "global"))]

    def test_reconoce_lo_finito(self) -> None:
        assert barrido._es_finito(0.5)
        assert not barrido._es_finito(float("nan"))
        assert not barrido._es_finito(float("inf"))
        assert not barrido._es_finito("0.5")


class TestResumenBarrido:
    @pytest.fixture
    def reg(self) -> dict:
        reg = registro.vacio()
        registro.nombrar(reg, [{"F2_L1": 0.25}, {"F2_L1": 0.5}])
        registro.anotar(reg, ["v01", "v02"],
                        {"ejecucion": "2026-01-01 10:00", "datos": {"bytes": 1},
                         "codigo": {"2": "x", "5": "y"}})
        return reg

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([
            fila_metrica("v01", "F2_equipo_global", "top1", 0.1),
            fila_metrica("v02", "F2_equipo_global", "top1", 0.3),
        ])

    META = {"fecha": "2026-01-01 10:00", "fases": ["0", "1"], "bootstrap": 0,
            "ejes": ["F2_L1"], "evaluadas": ["v01", "v02"],
            "rejilla": "formulaciones ['2'] x entidades ['equipo']"}

    def _texto(self, df, reg, tmp_path: Path, meta=None) -> str:
        barrido.escribir_resumen_barrido(df, reg, tmp_path, meta or self.META)
        return (tmp_path / "resumen_barrido.md").read_text(encoding="utf-8")

    def test_publica_el_indice_de_combinaciones(self, df, reg, tmp_path) -> None:
        texto = self._texto(df, reg, tmp_path)
        assert "| combinacion | F2_L1 | evaluada |" in texto
        assert "| v01 |" in texto

    def test_marca_en_negrita_lo_que_se_aparta_del_default(
        self, df, reg, tmp_path
    ) -> None:
        """Una combinacion puede fijar un eje al valor que ya traia el default:
        eso no es un cambio y no se resalta.
        """
        texto = self._texto(df, reg, tmp_path)
        assert "**`0.25`**" in texto

    def test_declara_que_la_carpeta_es_acumulativa(self, df, reg, tmp_path) -> None:
        texto = self._texto(df, reg, tmp_path)
        assert "ACUMULADAS" in texto
        assert "2 combinaciones" in texto

    def test_explica_los_ejes_que_identifican_cada_modelo(
        self, df, reg, tmp_path
    ) -> None:
        texto = self._texto(df, reg, tmp_path)
        assert all(linea in texto for linea in barrido.evaluar.LEYENDA_EJES)

    def test_avisa_cuando_lo_acumulado_no_es_comparable(
        self, df, reg, tmp_path
    ) -> None:
        """Metricas producidas con otra BD u otro nucleo numerico siguen en la
        misma tabla, pero el informe lo dice.
        """
        registro.anotar(reg, ["v02"], {"datos": {"bytes": 999}, "codigo": {"2": "z"}})
        assert "no se evaluaron todas" in self._texto(df, reg, tmp_path)

    def test_cierra_con_el_ranking_por_score_compuesto(
        self, df, reg, tmp_path
    ) -> None:
        texto = self._texto(df, reg, tmp_path)
        assert "Mejores combinaciones (score compuesto)" in texto
        assert "no es un criterio validado" in texto

    def test_sin_hiperparametros_lo_dice_en_vez_de_publicar_una_tabla_vacia(
        self, df, reg, tmp_path
    ) -> None:
        texto = self._texto(df, reg, tmp_path, {**self.META, "ejes": []})
        assert "No hay hiperparametros declarados" in texto

    def test_una_seccion_sin_ninguna_metrica_no_se_escribe(
        self, df, reg, tmp_path
    ) -> None:
        texto = self._texto(df, reg, tmp_path)
        assert "Estabilidad ante remuestreo" not in texto
        assert "Auto-similitud" in texto

    def test_sin_metricas_puntuables_no_hay_ranking(self, reg, tmp_path) -> None:
        """El score agrega solo las metricas con peso declarado: si el barrido no
        trae ninguna, la seccion no se escribe en vez de salir vacia.
        """
        df = pd.DataFrame([fila_metrica("v01", "F2_equipo_global", "n_pureza", 3.0)])
        assert "score compuesto" not in self._texto(df, reg, tmp_path)


class TestSubconjuntoDeLaRejilla:
    def test_conserva_el_orden_canonico(self) -> None:
        """La rejilla se recorre siempre igual, se escriba el flag como se escriba."""
        assert barrido._subconjunto("5,2", ecfg.FORMULACIONES, "formulaciones") == (
            "2", "5")

    def test_acota_a_lo_pedido(self) -> None:
        assert barrido._subconjunto("equipo", ecfg.ENTIDADES, "entidades") == ("equipo",)

    def test_un_valor_desconocido_aborta_con_las_opciones(self) -> None:
        with pytest.raises(SystemExit, match="Disponibles"):
            barrido._subconjunto("9", ecfg.FORMULACIONES, "formulaciones")

    def test_una_lista_vacia_aborta(self) -> None:
        with pytest.raises(SystemExit, match="al menos una"):
            barrido._subconjunto(" , ", ecfg.ENTIDADES, "entidades")

    def test_acota_tambien_las_normalizaciones(self) -> None:
        """`--normalizaciones` acota igual que los otros dos ejes de la rejilla."""
        assert barrido._subconjunto("global", ecfg.NORMALIZACIONES,
                                    "normalizaciones") == ("global",)


class TestFormatoDeValores:
    def test_describe_los_valores_de_la_combinacion(self) -> None:
        assert barrido._fmt_valores({"F2_L1": 0.5}) == "F2_L1=0.5"

    def test_una_combinacion_sin_ejes_se_declara_como_el_default(self) -> None:
        assert barrido._fmt_valores({}) == "(config por defecto)"
