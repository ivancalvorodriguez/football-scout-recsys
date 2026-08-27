"""Orquestador: contador de pasos, carga de la rejilla, CSV, informe y figuras.

Las fases se sustituyen por dobles que devuelven filas conocidas: lo que se prueba
aqui es el PEGAMENTO (que fase se corre, con que etiqueta acaba cada fila, que
ficheros salen), no el numero que calcula cada fase. Eso ya lo cubre
`test_fases.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluacion import config, evaluar, fases
from src.tests.evaluacion.conftest import modelo_sintetico


class TestFormatoDeDuracion:
    @pytest.mark.parametrize("segundos,esperado", [
        (0.0, "0s"), (45.4, "45s"), (59.9, "60s"),
        (60.0, "1m 00s"), (125.0, "2m 05s"),
        (3600.0, "1h 00m"), (3900.0, "1h 05m"),
    ])
    def test_formatea_segundos_minutos_y_horas(self, segundos, esperado) -> None:
        assert evaluar._dur(segundos) == esperado

    def test_una_duracion_negativa_se_recorta_a_cero(self) -> None:
        """Puede salir de una resta de relojes: no debe imprimirse un '-3s'."""
        assert evaluar._dur(-3.0) == "0s"


class TestProgreso:
    def test_numera_los_pasos_e_informa_del_tiempo(self, capsys) -> None:
        prog = evaluar.Progreso(2)
        with prog.paso("primero"):
            pass
        salida = capsys.readouterr().out
        assert "[1/2] primero" in salida
        assert "hecho en" in salida
        assert "quedan ~" in salida

    def test_el_ultimo_paso_se_marca_como_completado(self, capsys) -> None:
        prog = evaluar.Progreso(1)
        with prog.paso("unico"):
            pass
        assert "completado" in capsys.readouterr().out

    def test_un_total_de_cero_no_divide_por_cero(self) -> None:
        assert evaluar.Progreso(0).total == 1

    def test_cronometra_tambien_si_el_paso_falla(self, capsys) -> None:
        """El contador vive en un `finally`: una fase que peta no puede dejar la
        consola sin la linea de cierre.
        """
        prog = evaluar.Progreso(1)
        with pytest.raises(RuntimeError):
            with prog.paso("revienta"):
                raise RuntimeError("boom")
        assert "hecho en" in capsys.readouterr().out

    def test_sin_contador_solo_se_imprime_la_descripcion(self, capsys) -> None:
        with evaluar._quizas_paso(None, "suelto"):
            pass
        assert capsys.readouterr().out.strip() == "suelto ..."

    def test_con_contador_delega_en_el(self, capsys) -> None:
        prog = evaluar.Progreso(3)
        with evaluar._quizas_paso(prog, "dentro"):
            pass
        assert "[1/3] dentro" in capsys.readouterr().out


class TestSubProgreso:
    def test_la_ultima_iteracion_siempre_se_imprime(self, capsys) -> None:
        rep = evaluar._SubProgreso("bootstrap", cada=1000.0)
        rep(5, 5)
        assert "bootstrap 5/5" in capsys.readouterr().out

    def test_no_satura_la_consola_entre_refrescos(self, capsys) -> None:
        rep = evaluar._SubProgreso("bootstrap", cada=1000.0)
        rep(1, 100)          # el primero pasa (el reloj interno arranca a 0)
        capsys.readouterr()
        rep(2, 100)
        assert capsys.readouterr().out == ""


class TestPasosTotales:
    def test_cuenta_una_fase_por_modelo_mas_la_triangulacion(self) -> None:
        """La Fase 3 es global (compara modelos entre si), no por modelo."""
        assert evaluar._pasos_totales(4, {"0", "1", "3", "5"}, bootstrap=0) == 1 + 4 * 3

    def test_la_estabilidad_no_cuenta_sin_remuestreos(self) -> None:
        assert evaluar._pasos_totales(2, {"2"}, bootstrap=0) == 0
        assert evaluar._pasos_totales(2, {"2"}, bootstrap=10) == 2

    def test_sin_fases_no_hay_pasos(self) -> None:
        assert evaluar._pasos_totales(8, set(), bootstrap=100) == 0


class TestCargarModelos:
    def test_carga_los_artefactos_presentes(self, tmp_path: Path) -> None:
        modelo_sintetico(np.zeros((2, 2)), formulacion="5", entidad="equipo",
                         meta={"normalizacion": "por_liga"}).guardar(tmp_path)
        modelos = evaluar.cargar_modelos(tmp_path, ("5",), ("equipo",))
        assert set(modelos) == {("5", "equipo", "por_liga")}

    def test_avisa_de_los_que_faltan_sin_abortar(self, tmp_path: Path, capsys) -> None:
        """Una rejilla incompleta es normal (`build` puede haberse acotado): el
        harness evalua lo que hay y lo dice.
        """
        assert evaluar.cargar_modelos(tmp_path, ("2",), ("jugador",)) == {}
        assert "falta el modelo F2 jugador" in capsys.readouterr().out

    def test_acotar_la_rejilla_evita_avisos_de_lo_que_no_se_pidio(
        self, tmp_path: Path, capsys
    ) -> None:
        evaluar.cargar_modelos(tmp_path, ("5",), ("equipo",))
        assert "jugador" not in capsys.readouterr().out


class TestEjecutar:
    """`ejecutar` decide QUE se corre y como se etiqueta cada fila."""

    @pytest.fixture
    def modelos(self) -> dict:
        return {("5", "equipo", "por_liga"):
                modelo_sintetico(np.zeros((2, 2)), formulacion="5", entidad="equipo")}

    @pytest.fixture
    def dobles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fases, "fase0_sanity",
                            lambda *a, **kw: {"asimetria": 0.1, "pureza_top1": 0.9,
                                              "n_pureza": 3})
        monkeypatch.setattr(fases, "face_validity", lambda *a, **kw: ["  A -> B"])
        monkeypatch.setattr(fases, "fase1_autosimilitud", lambda *a, **kw: {
            "global": {"top1": 0.5}, "A->B": {"top1": 0.4}, "B->A": {"top1": 0.6},
            "estratos": {"cabeza": {"top1": 0.7}},
            "denoising": {"pre": {"mrr": 0.2, "top1": 0.1},
                          "post": {"mrr": 0.3, "top1": 0.2},
                          "delta_mrr": 0.1, "delta_rr_medio": 0.05,
                          "p_pareado": 0.01},
        })
        monkeypatch.setattr(fases, "fase2_estabilidad",
                            lambda *a, **kw: {"rbo_medio": 0.8})
        monkeypatch.setattr(fases, "fase3_triangulacion",
                            lambda *a, **kw: [{"comparacion": "F2 vs F5", "r": 0.7}])
        monkeypatch.setattr(fases, "fase5_downstream",
                            lambda *a, **kw: {"knn_accuracy": 0.6})

    def test_corre_solo_las_fases_pedidas(self, modelos, dobles) -> None:
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"0"}, bootstrap=0)
        assert salida["f0"] and not salida["f1"] and not salida["f5"]

    def test_cada_fila_lleva_los_cuatro_ejes_del_modelo(self, modelos, dobles) -> None:
        """La etiqueta compacta no dice de que eje viene cada palabra; los CSV se
        cruzan luego por formulacion/entidad/normalizacion/distancia.
        """
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"0"}, bootstrap=0)
        fila = salida["f0"][0]
        assert fila["modelo"] == "F5_equipo_por_liga_euclidea"
        assert (fila["formulacion"], fila["entidad"], fila["normalizacion"],
                fila["distancia"]) == ("5", "equipo", "por_liga", "euclidea")

    def test_la_distancia_evaluada_viaja_a_cada_fila(self, modelos, dobles) -> None:
        """Sin ella, las filas de dos geometrias serian indistinguibles en el CSV."""
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"0"}, bootstrap=0, distancia="manhattan")
        assert salida["f0"][0]["modelo"] == "F5_equipo_por_liga_manhattan"
        assert salida["f0"][0]["distancia"] == "manhattan"

    def test_la_fase_1_emite_las_tres_direcciones_y_los_estratos(
        self, modelos, dobles
    ) -> None:
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"1"}, bootstrap=0)
        assert [r["direccion"] for r in salida["f1"]] == ["global", "A->B", "B->A"]
        assert salida["f1_estratos"][0]["estrato"] == "cabeza"

    def test_el_denoising_de_la_f5_alimenta_la_tabla_de_la_fase_4(
        self, modelos, dobles
    ) -> None:
        """La Fase 4 no es una fase aparte: se calcula dentro de la 1."""
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"1"}, bootstrap=0)
        assert salida["f4"][0]["delta_mrr"] == 0.1
        assert salida["f4"][0]["p_pareado"] == 0.01

    def test_la_estabilidad_se_omite_sin_remuestreos(self, modelos, dobles) -> None:
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"2"}, bootstrap=0)
        assert salida["f2"] == []

    def test_la_triangulacion_se_corre_una_sola_vez(self, modelos, dobles) -> None:
        salida = evaluar.ejecutar(modelos, {("equipo", "por_liga"): None}, {}, {},
                                  {"3"}, bootstrap=0)
        assert len(salida["f3"]) == 1

    def test_la_face_validity_solo_se_recoge_una_vez_por_modelo(
        self, dobles
    ) -> None:
        """Se hace con `por_liga` para no duplicar el mismo bloque cualitativo."""
        modelos = {
            (f, "equipo", n): modelo_sintetico(np.zeros((2, 2)), formulacion=f,
                                               entidad="equipo")
            for f in ("2", "5") for n in ("por_liga", "global")
        }
        salida = evaluar.ejecutar(modelos, {("equipo", n): None
                                            for n in ("por_liga", "global")},
                                  {}, {}, {"0"}, bootstrap=0)
        assert {b["modelo"] for b in salida["face"]} == {
            "F2_equipo_por_liga_euclidea", "F5_equipo_por_liga_euclidea"}

    def test_ignora_las_celdas_de_la_rejilla_sin_artefacto(self, dobles) -> None:
        salida = evaluar.ejecutar({}, {}, {}, {}, {"0", "1", "5"}, bootstrap=5)
        assert all(not filas for clave, filas in salida.items() if clave != "f3")


class TestQueriesFace:
    def test_el_jugador_se_revisa_por_los_de_mas_minutos(self) -> None:
        """Con jugadores de 20 minutos la revision cualitativa no dice nada."""
        modelo = modelo_sintetico(np.zeros((3, 3)), entity_ids=[1, 2, 3],
                                  entity_names=["Ana", "Bea", "Cai"])
        nombres = evaluar._queries_face(modelo, {1: 10.0, 2: 900.0, 3: 500.0}, n=2)
        assert nombres == ["Bea", "Cai"]

    def test_el_equipo_se_revisa_por_orden_de_indice(self) -> None:
        modelo = modelo_sintetico(np.zeros((3, 3)), entidad="equipo",
                                  entity_names=["A", "B", "C"])
        assert evaluar._queries_face(modelo, {}, n=2) == ["A", "B"]

    def test_un_jugador_sin_minutos_registrados_no_rompe_el_orden(self) -> None:
        modelo = modelo_sintetico(np.zeros((2, 2)), entity_ids=[1, 2],
                                  entity_names=["Ana", "Bea"])
        assert evaluar._queries_face(modelo, {}, n=2) == ["Bea", "Ana"]


class TestEscribirCsv:
    @pytest.fixture
    def salida(self) -> dict:
        base = {"modelo": "F5_equipo_global", "formulacion": "5",
                "entidad": "equipo", "normalizacion": "global"}
        vacio = evaluar.tablas_vacias()
        return {**vacio, "f0": [{**base, "asimetria": 0.2}]}

    def test_escribe_un_csv_por_tabla_con_filas(self, salida, tmp_path: Path) -> None:
        evaluar.escribir_csv(salida, tmp_path)
        assert (tmp_path / "fase0_sanity.csv").exists()

    def test_no_escribe_las_tablas_vacias(self, salida, tmp_path: Path) -> None:
        """Un CSV vacio en la carpeta se lee como "la fase corrio y no dio nada"."""
        evaluar.escribir_csv(salida, tmp_path)
        assert not (tmp_path / "fase2_estabilidad.csv").exists()

    def test_las_columnas_extra_van_delante(self, salida, tmp_path: Path) -> None:
        """El barrido mete ahi la combinacion: sin ella los CSV de dos
        combinaciones son indistinguibles salvo por la carpeta que los contiene.
        """
        evaluar.escribir_csv(salida, tmp_path, {"combinacion": "v03"})
        df = pd.read_csv(tmp_path / "fase0_sanity.csv")
        assert list(df.columns)[0] == "combinacion"
        assert df["combinacion"].tolist() == ["v03"]

    def test_no_muta_las_filas_originales(self, salida, tmp_path: Path) -> None:
        evaluar.escribir_csv(salida, tmp_path, {"combinacion": "v03"})
        assert "combinacion" not in salida["f0"][0]


class TestFormatoDelInforme:
    def test_los_numeros_van_con_tres_decimales(self) -> None:
        assert evaluar._fmt(0.123456) == "0.123"

    def test_lo_no_finito_se_declara_como_no_aplicable(self) -> None:
        """Distinto de un 0: la pureza posicional del equipo no existe, no es mala."""
        assert evaluar._fmt(float("nan")) == "n/a"
        assert evaluar._fmt(float("inf")) == "n/a"
        assert evaluar._fmt(None) == "n/a"

    def test_lo_que_no_es_float_se_imprime_tal_cual(self) -> None:
        assert evaluar._fmt(7) == "7"
        assert evaluar._fmt("equipo") == "equipo"

    def test_la_cabecera_abre_por_los_ejes_del_modelo(self) -> None:
        cab, sep = evaluar._cab_id("top-1")
        assert cab == "| formulacion | entidad | normalizacion | distancia | top-1 |"
        assert sep == "|---|---|---|---|---|"

    def test_la_fila_identifica_el_modelo_por_sus_cuatro_ejes(self) -> None:
        fila = evaluar._id({"formulacion": "2", "entidad": "jugador",
                            "normalizacion": "global", "distancia": "coseno"})
        assert fila == "| F2 | jugador | global | coseno |"

    def test_una_fila_sin_distancia_se_lee_como_euclidea(self) -> None:
        """Filas de una evaluacion anterior a que la geometria fuera un eje: era
        la unica que habia, asi que se declara en vez de dejar el hueco."""
        fila = evaluar._id({"formulacion": "2", "entidad": "jugador",
                            "normalizacion": "global"})
        assert fila == "| F2 | jugador | global | euclidea |"


class TestResumenFeatures:
    def test_declara_cuantas_features_y_si_llevan_posicion(self) -> None:
        """Es lo UNICO que distingue dos artefactos homonimos construidos con
        `USE_POSITION_FEATURES` distinto: no cabe en el nombre del fichero.
        """
        modelos = {("5", "jugador", "por_liga"): modelo_sintetico(
            np.zeros((2, 2)), feat_names=["xg", "pos_1", "pos_2"])}
        lineas = evaluar.resumen_features(modelos)
        assert lineas == ["- **jugador**: 3 features, 2 de ellas de posicion (`pos_*`)."]

    def test_lo_dice_tambien_cuando_no_hay_bloque_de_posicion(self) -> None:
        modelos = {("5", "equipo", "por_liga"): modelo_sintetico(
            np.zeros((2, 2)), entidad="equipo", feat_names=["xg", "ppda"])}
        assert evaluar.resumen_features(modelos) == [
            "- **equipo**: 2 features, sin bloque de posicion."]

    def test_una_entidad_sin_modelos_no_aparece(self) -> None:
        assert evaluar.resumen_features({}) == []


class TestEscribirInforme:
    @pytest.fixture
    def salida(self) -> dict:
        base = {"modelo": "F5_equipo_global", "formulacion": "5",
                "entidad": "equipo", "normalizacion": "global"}
        generalizacion = {**base, "holdout": "9-27", "fidelidad": "fiel",
                          "n_train": 20, "n_holdout": 5, "n_entidades": 5,
                          "top1": 0.4, "mrr": 0.5, "pureza_top1": 0.8,
                          "knn_accuracy": 0.75, "rbo_medio": float("nan"),
                          "coverage": 0.6, "diversity": 2.0, "n_rol": 5,
                          "distintos_top1": 4, "score_top1": 0.9,
                          "pool_autosim": 5}
        return {
            **evaluar.tablas_vacias(),
            "f0": [{**base, "asimetria": 0.02, "pureza_top1": 0.95, "n_pureza": 10}],
            "f1": [{**base, "direccion": "global", "pool": 20, "azar_top1": 0.05,
                    "top1": 0.5, "top5": 0.8, "top10": 0.9, "mrr": 0.6}],
            "f1_estratos": [{**base, "estrato": "cabeza", "top1": 0.6, "top5": 0.9,
                             "mrr": 0.7}],
            "f2": [{**base, "rbo_medio": 0.85, "ic_bajo": 0.8, "ic_alto": 0.9,
                    "n_consultables": 20, "B": 100}],
            "f3": [{"comparacion": "F2 vs F5 (global)", "entidad": "equipo",
                    "n_entidades": 20, "r": 0.7, "p": 0.001, "kendall": 0.5}],
            "f4": [{**base, "mrr_pre": 0.5, "top1_pre": 0.4, "mrr_post": 0.6,
                    "top1_post": 0.5, "delta_mrr": 0.1, "delta_rr_medio": 0.05,
                    "p_pareado": 0.01}],
            "f5": [{**base, "knn_accuracy": 0.7, "knn_f1_macro": 0.65,
                    "coverage": 0.9, "diversity": 2.4,
                    "popularidad_spearman": 0.1}],
            "f7": [{**generalizacion, "ambito": "dentro"},
                   {**generalizacion, "ambito": "fuera", "pureza_top1": 0.6,
                    "knn_accuracy": 0.5, "top1": 0.2, "rbo_medio": 0.7}],
            "face": [{"modelo": "F5_equipo_global", "lineas": ["  A -> B, C"]}],
        }

    META = {"fecha": "2026-01-01 10:00", "n_modelos": 1, "bootstrap": 100,
            "features": ["- **equipo**: 2 features, sin bloque de posicion."]}

    def _texto(self, salida: dict, tmp_path: Path) -> str:
        evaluar.escribir_informe(salida, tmp_path, self.META)
        return (tmp_path / "resumen.md").read_text(encoding="utf-8")

    def test_escribe_una_seccion_por_fase(self, salida, tmp_path: Path) -> None:
        texto = self._texto(salida, tmp_path)
        for titulo in ("Fase 0", "Fase 1", "Fase 2", "Fase 3", "Fase 4", "Fase 5",
                       "Fase 6", "Fase 7"):
            assert titulo in texto

    def test_la_fase_7_escribe_cada_metrica_dentro_y_fuera(
        self, salida, tmp_path: Path
    ) -> None:
        """La tabla es lo que deja leer la caida sin calcularla a mano, y lo que
        distingue una metrica que no se pudo medir de una que salio mal."""
        texto = self._texto(salida, tmp_path)
        assert "| dentro | fuera | caida |" in texto
        assert "0.250" in texto        # kNN 0.75 dentro - 0.50 fuera
        assert "el criterio VIAJA" not in texto   # 25 % supera el umbral

    def test_la_estabilidad_solo_sale_fuera_de_muestra(
        self, salida, tmp_path: Path
    ) -> None:
        """Dentro es la Fase 2 sobre el modelo entero: aqui se escribe n/a en vez
        de una cifra que no seria comparable."""
        texto = self._texto(salida, tmp_path)
        assert "| estabilidad RBO@10 | n/a | 0.700 | n/a |" in texto

    def test_explica_los_tres_ejes_que_identifican_al_modelo(
        self, salida, tmp_path: Path
    ) -> None:
        texto = self._texto(salida, tmp_path)
        assert "Que modelo es cada uno" in texto
        assert all(linea in texto for linea in evaluar.LEYENDA_EJES)

    def test_declara_el_espacio_de_features_evaluado(
        self, salida, tmp_path: Path
    ) -> None:
        assert "sin bloque de posicion" in self._texto(salida, tmp_path)

    def test_avisa_de_que_los_umbrales_son_referencias(
        self, salida, tmp_path: Path
    ) -> None:
        """El PDF juzga por convergencia de señales, no por una metrica aislada:
        el informe no puede sugerir lo contrario.
        """
        texto = self._texto(salida, tmp_path)
        assert "**no constantes universales**" in texto
        assert "convergencia" in texto

    def test_el_veredicto_de_pureza_pasa_por_encima_del_umbral(
        self, salida, tmp_path: Path
    ) -> None:
        assert "PASA" in self._texto(salida, tmp_path)

    def test_el_veredicto_de_pureza_avisa_por_debajo_del_umbral(
        self, salida, tmp_path: Path
    ) -> None:
        salida["f0"][0]["pureza_top1"] = 0.2
        assert "REVISAR" in self._texto(salida, tmp_path)

    def test_una_asimetria_alta_se_marca_para_revision(
        self, salida, tmp_path: Path
    ) -> None:
        salida["f0"][0]["asimetria"] = config.ASIMETRIA_ALTA * 10
        texto = self._texto(salida, tmp_path)
        assert "Veredicto asimetria" in texto and "REVISAR" in texto

    def test_publica_el_top_1_frente_al_azar(self, salida, tmp_path: Path) -> None:
        """Un top-1 del 50% no significa nada sin saber el tamaño del pool."""
        texto = self._texto(salida, tmp_path)
        assert "azar top-1" in texto
        assert "10.0x" in texto        # 0.5 / 0.05

    def test_la_fase_6_se_documenta_como_pendiente(
        self, salida, tmp_path: Path
    ) -> None:
        assert "no es automatizable" in self._texto(salida, tmp_path).lower()

    def test_un_informe_sin_resultados_sigue_siendo_valido(self, tmp_path: Path) -> None:
        vacio = evaluar.tablas_vacias()
        texto = self._texto(vacio, tmp_path)
        assert "Fase 6" in texto
        assert "Fase 0" not in texto

    def test_una_pureza_no_finita_no_produce_veredicto(self, tmp_path: Path) -> None:
        """Con solo modelos de equipo la pureza es NaN en todos: no hay minimo."""
        salida = evaluar.tablas_vacias()
        salida["f0"] = [{"modelo": "F5_equipo_global", "formulacion": "5",
                         "entidad": "equipo", "normalizacion": "global",
                         "asimetria": 0.01, "pureza_top1": float("nan"),
                         "n_pureza": 0}]
        texto = self._texto(salida, tmp_path)
        assert "Veredicto pureza" not in texto
        assert "Veredicto asimetria" in texto


class TestEscribirFiguras:
    def test_dibuja_la_autosimilitud_y_la_estabilidad(self, tmp_path: Path) -> None:
        base = {"modelo": "F5_equipo_global"}
        salida = {
            "f1": [{**base, "direccion": "global", "top1": 0.5, "azar_top1": 0.05}],
            "f2": [{**base, "rbo_medio": 0.8}],
        }
        evaluar.escribir_figuras(salida, tmp_path)
        assert (tmp_path / "figuras" / "fase1_autosimilitud_top1.png").exists()
        assert (tmp_path / "figuras" / "fase2_estabilidad_rbo.png").exists()

    def test_sin_datos_no_dibuja_nada(self, tmp_path: Path) -> None:
        evaluar.escribir_figuras({"f1": [], "f2": []}, tmp_path)
        assert list((tmp_path / "figuras").glob("*.png")) == []
