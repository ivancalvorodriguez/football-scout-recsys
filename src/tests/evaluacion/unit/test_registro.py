"""Registro del barrido: nombres estables por configuracion y metricas acumuladas.

Lo que este modulo tiene que garantizar es que relanzar el barrido sobre la misma
carpeta NO sea destructivo: que `v07` siga significando lo mismo y que las
metricas de las combinaciones que no se han vuelto a evaluar sigan ahi. Casi todas
las pruebas describen ese escenario de dos ejecuciones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluacion import registro


class TestNormalizarValores:
    def test_el_mismo_numero_escrito_de_dos_maneras_es_el_mismo_punto(self) -> None:
        """`50` y `50.0` llegan por caminos distintos (config tipada, JSON, tabla
        markdown): si no se unificaran, la carpeta acumularia dos combinaciones
        para el mismo punto de la rejilla.
        """
        assert registro._normalizar(50) == registro._normalizar(50.0)

    def test_los_booleanos_no_se_confunden_con_numeros(self) -> None:
        """`True` es un `int` para Python: sin el caso especial, `USE_X = True`
        y `USE_X = 1` serian la misma configuracion.
        """
        assert registro._normalizar(True) != registro._normalizar(1)

    def test_una_tupla_y_una_lista_con_lo_mismo_son_iguales(self) -> None:
        """Las tuplas de config vuelven de JSON como listas."""
        assert registro._normalizar((1, 2)) == registro._normalizar([1, 2])

    def test_el_texto_se_queda_como_texto(self) -> None:
        assert registro._normalizar("mmd") == "mmd"


class TestClave:
    def test_dos_configuraciones_equivalentes_comparten_clave(self) -> None:
        assert registro.clave({"a": 1}) == registro.clave({"a": 1.0})

    def test_el_orden_de_las_claves_no_cambia_la_identidad(self) -> None:
        assert registro.clave({"a": 1, "b": 2}) == registro.clave({"b": 2, "a": 1})

    def test_los_ejes_de_identidad_recortan_los_dos_lados_igual(self) -> None:
        """Es lo que permite quitar un eje de HIPERPARAMETROS sin que la misma
        configuracion parezca nueva.
        """
        a = {"F2_L1": 0.5, "F5_RFF_DIM": 512}
        b = {"F2_L1": 0.5, "F5_RFF_DIM": 1024}
        assert registro.clave(a, ["F2_L1"]) == registro.clave(b, ["F2_L1"])
        assert registro.clave(a) != registro.clave(b)

    def test_un_eje_ausente_cuenta_como_ausente_no_se_ignora(self) -> None:
        assert registro.clave({}, ["F2_L1"]) != registro.clave({"F2_L1": 0.5}, ["F2_L1"])


class TestOrdenDeNombres:
    def test_ordena_por_numero_y_no_alfabeticamente(self) -> None:
        nombres = ["v10", "v9", "v2"]
        assert sorted(nombres, key=registro.orden_nombre) == ["v2", "v9", "v10"]

    def test_un_nombre_sin_formato_va_al_final(self) -> None:
        assert registro.orden_nombre("otro")[0] > registro.orden_nombre("v99")[0]

    def test_el_siguiente_nombre_libre_continua_la_numeracion(self) -> None:
        assert registro._siguiente_nombre({"v01", "v02"}) == "v03"

    def test_el_primer_nombre_es_v01(self) -> None:
        assert registro._siguiente_nombre(set()) == "v01"

    def test_no_reutiliza_un_hueco_intermedio(self) -> None:
        """Reutilizarlo daria el nombre de una combinacion borrada a otra nueva."""
        assert registro._siguiente_nombre({"v01", "v05"}) == "v06"


class TestLeerMarkdown:
    TABLA = (
        "# Barrido\n\n"
        "| combinacion | F2_L1 | F5_RFF_DIM | evaluada |\n"
        "|---|---|---|---|\n"
        "| v01 | `0.25` | `512` | 2026-01-01 |\n"
        "| v02 | **`0.5`** | `512` | 2026-01-02 |\n"
        "\nTexto posterior.\n"
    )

    def _escribir(self, d: Path) -> Path:
        (d / "resumen_barrido.md").write_text(self.TABLA, encoding="utf-8")
        return d

    def test_recupera_la_configuracion_de_cada_combinacion(self, tmp_path: Path) -> None:
        """Es la unica fuente que tienen los barridos anteriores al registro."""
        combos = registro.leer_markdown(self._escribir(tmp_path))
        assert combos == {"v01": {"F2_L1": 0.25, "F5_RFF_DIM": 512},
                          "v02": {"F2_L1": 0.5, "F5_RFF_DIM": 512}}

    def test_quita_el_marcado_de_las_celdas(self, tmp_path: Path) -> None:
        """La negrita marca "distinto del default", no forma parte del valor."""
        combos = registro.leer_markdown(self._escribir(tmp_path))
        assert combos["v02"]["F2_L1"] == 0.5

    def test_recupera_los_tipos_para_poder_ordenar_los_ejes(self, tmp_path: Path) -> None:
        """Como texto, 1024 iria antes que 256 y la superficie saldria deformada."""
        combos = registro.leer_markdown(self._escribir(tmp_path))
        assert isinstance(combos["v01"]["F5_RFF_DIM"], int)

    def test_la_columna_de_procedencia_no_es_un_eje(self, tmp_path: Path) -> None:
        combos = registro.leer_markdown(self._escribir(tmp_path))
        assert "evaluada" not in combos["v01"]

    def test_convierte_los_booleanos(self, tmp_path: Path) -> None:
        (tmp_path / "resumen_barrido.md").write_text(
            "| combinacion | USE_X |\n|---|---|\n| v01 | `True` |\n", encoding="utf-8")
        assert registro.leer_markdown(tmp_path) == {"v01": {"USE_X": True}}

    def test_un_valor_que_no_es_numero_se_queda_en_texto(self, tmp_path: Path) -> None:
        """`F5_METODO_EQUIPO = "sinkhorn"` es un eje perfectamente barrible."""
        (tmp_path / "resumen_barrido.md").write_text(
            "| combinacion | METODO |\n|---|---|\n| v01 | `sinkhorn` |\n",
            encoding="utf-8")
        assert registro.leer_markdown(tmp_path) == {"v01": {"METODO": "sinkhorn"}}

    def test_la_tabla_termina_en_la_primera_fila_malformada(
        self, tmp_path: Path
    ) -> None:
        """Mejor quedarse con lo leido que asignar valores a los ejes equivocados."""
        (tmp_path / "resumen_barrido.md").write_text(
            "| combinacion | F2_L1 |\n|---|---|\n"
            "| v01 | `0.5` |\n"
            "| v02 | `0.25` | sobra |\n", encoding="utf-8")
        assert registro.leer_markdown(tmp_path) == {"v01": {"F2_L1": 0.5}}

    def test_sin_fichero_devuelve_none(self, tmp_path: Path) -> None:
        assert registro.leer_markdown(tmp_path) is None

    def test_un_resumen_sin_tabla_de_combinaciones_devuelve_none(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "resumen_barrido.md").write_text("# Solo texto\n", encoding="utf-8")
        assert registro.leer_markdown(tmp_path) is None


class TestCargarYGuardar:
    def test_una_carpeta_nueva_arranca_vacia(self, tmp_path: Path) -> None:
        assert registro.cargar(tmp_path) == {"version": registro.VERSION,
                                             "combinaciones": {}}

    def test_migra_un_barrido_anterior_desde_su_resumen(
        self, tmp_path: Path, capsys
    ) -> None:
        """Sin migracion, la numeracion volveria a arrancar en v01 y las carpetas
        ya existentes quedarian asignadas a otras configuraciones.
        """
        (tmp_path / "resumen_barrido.md").write_text(
            "| combinacion | F2_L1 |\n|---|---|\n| v01 | `0.5` |\n", encoding="utf-8")
        reg = registro.cargar(tmp_path)
        assert registro.hiperparametros(reg) == {"v01": {"F2_L1": 0.5}}
        assert "migracion" in capsys.readouterr().out

    def test_un_registro_ilegible_aborta_en_vez_de_empezar_de_cero(
        self, tmp_path: Path
    ) -> None:
        registro.ruta(tmp_path).write_text("{roto", encoding="utf-8")
        with pytest.raises(SystemExit, match="No puedo leer"):
            registro.cargar(tmp_path)

    def test_una_version_distinta_aborta(self, tmp_path: Path) -> None:
        registro.ruta(tmp_path).write_text(
            json.dumps({"version": 99, "combinaciones": {}}), encoding="utf-8")
        with pytest.raises(SystemExit, match="version"):
            registro.cargar(tmp_path)

    def test_lo_guardado_se_relee_igual(self, tmp_path: Path) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"F2_L1": 0.5}])
        registro.guardar(tmp_path, reg)
        assert registro.hiperparametros(registro.cargar(tmp_path)) == {
            "v01": {"F2_L1": 0.5}}

    def test_se_guarda_en_orden_natural(self, tmp_path: Path) -> None:
        reg = {"version": registro.VERSION, "combinaciones": {
            "v10": {"hiperparametros": {}}, "v2": {"hiperparametros": {}}}}
        registro.guardar(tmp_path, reg)
        texto = registro.ruta(tmp_path).read_text(encoding="utf-8")
        assert texto.index('"v2"') < texto.index('"v10"')


class TestEjes:
    def test_los_ejes_variables_son_los_que_de_verdad_se_movieron(self) -> None:
        """Un eje con un solo valor aportaria una unica fila y haria creer que se
        probo algo que no se probo.
        """
        configs = {"v01": {"F2_L1": 0.25, "F2_BETA": 1.0},
                   "v02": {"F2_L1": 0.5, "F2_BETA": 1.0}}
        assert registro.ejes_variables(configs) == {"F2_L1": [0.25, 0.5]}

    def test_los_valores_de_un_eje_numerico_van_ordenados(self) -> None:
        configs = {"v01": {"d": 1024}, "v02": {"d": 256}, "v03": {"d": 512}}
        assert registro.ejes_variables(configs)["d"] == [256, 512, 1024]

    def test_un_eje_no_numerico_se_ordena_alfabeticamente(self) -> None:
        configs = {"v01": {"m": "sinkhorn"}, "v02": {"m": "mmd"}}
        assert registro.ejes_variables(configs)["m"] == ["mmd", "sinkhorn"]

    def test_sin_combinaciones_no_hay_ejes(self) -> None:
        assert registro.ejes_variables({}) == {}

    def test_los_ejes_publicados_son_la_union_con_los_preferidos_delante(self) -> None:
        reg = {"version": 1, "combinaciones": {
            "v01": {"hiperparametros": {"B": 1, "C": 2}}}}
        assert registro.ejes(reg, ["A"]) == ["A", "B", "C"]


class TestNombrar:
    def test_una_configuracion_nueva_recibe_el_siguiente_numero(self) -> None:
        reg = registro.vacio()
        assert list(registro.nombrar(reg, [{"a": 1}, {"a": 2}])) == ["v01", "v02"]

    def test_una_configuracion_ya_vista_recupera_su_nombre(self) -> None:
        """Con el nombre recupera su carpeta y su cache de modelos: es lo que hace
        que relanzar el barrido cueste 0 construcciones.
        """
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}, {"a": 2}])
        assert list(registro.nombrar(reg, [{"a": 2}])) == ["v02"]

    def test_ampliar_la_rejilla_no_renumera_lo_anterior(self) -> None:
        """El escenario que motiva el modulo: editar HIPERPARAMETROS y relanzar."""
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}, {"a": 2}])
        seleccion = registro.nombrar(reg, [{"a": 1}, {"a": 2}, {"a": 3}])
        assert seleccion == {"v01": {"a": 1}, "v02": {"a": 2}, "v03": {"a": 3}}

    def test_muta_el_registro_con_las_entradas_nuevas(self) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}])
        assert "v01" in reg["combinaciones"]

    def test_devuelve_la_seleccion_en_orden_natural(self) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": i} for i in range(12)])
        seleccion = registro.nombrar(reg, [{"a": 11}, {"a": 2}])
        assert list(seleccion) == ["v03", "v12"]

    def test_una_entrada_migrada_recupera_los_tipos_sin_cambiar_de_identidad(
        self,
    ) -> None:
        """Del markdown se lee lo que se pudo reconstruir del texto; esta es la
        unica ocasion de mejorarlo.
        """
        reg = {"version": 1, "combinaciones": {
            "v01": {"hiperparametros": {"a": 1}}}}
        registro.nombrar(reg, [{"a": 1.0}])
        assert reg["combinaciones"]["v01"]["hiperparametros"] == {"a": 1.0}
        assert list(reg["combinaciones"]) == ["v01"]


class TestProcedencia:
    def test_anotar_marca_las_combinaciones_evaluadas(self) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}])
        registro.anotar(reg, ["v01"],
                        {registro.CAMPO_FECHA: "2026-01-01", "datos": {}, "codigo": {}})
        assert registro.procedencias(reg)["v01"][registro.CAMPO_FECHA] == "2026-01-01"

    def test_una_carpeta_con_el_campo_de_fecha_antiguo_conserva_la_fecha(
        self, tmp_path: Path
    ) -> None:
        """El campo se llamaba antes de otra manera. Al leer la carpeta se
        renombra: nadie va a reevaluar cientos de combinaciones por eso, y la
        columna `evaluada` del resumen tiene que seguir diciendo cuando fue.
        """
        ruta = registro.ruta(tmp_path)
        ruta.write_text(json.dumps({"version": registro.VERSION, "combinaciones": {
            "v01": {"hiperparametros": {"a": 1}, "corrida": "2026-01-01 10:00"}}}),
            encoding="utf-8")
        reg = registro.cargar(tmp_path)
        assert reg["combinaciones"]["v01"][registro.CAMPO_FECHA] == "2026-01-01 10:00"
        assert "corrida" not in reg["combinaciones"]["v01"]

    def test_la_procedencia_no_incluye_los_hiperparametros(self) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}])
        registro.anotar(reg, ["v01"], {registro.CAMPO_FECHA: "hoy"})
        assert "hiperparametros" not in registro.procedencias(reg)["v01"]

    def test_una_carpeta_de_una_sola_ejecucion_es_homogenea(self) -> None:
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}, {"a": 2}])
        registro.anotar(reg, ["v01", "v02"],
                        {registro.CAMPO_FECHA: "hoy", "datos": {"bytes": 1},
                         "codigo": {"2": "x"}})
        assert registro.homogeneo(reg)

    def test_avisa_cuando_se_acumulan_metricas_de_mundos_distintos(self) -> None:
        """Reextraer la BD o tocar el nucleo numerico hace que los numeros de la
        misma tabla no sean comparables. No se borra nada: se avisa.
        """
        reg = registro.vacio()
        registro.nombrar(reg, [{"a": 1}, {"a": 2}])
        registro.anotar(reg, ["v01"], {"datos": {"bytes": 1}, "codigo": {"2": "x"}})
        registro.anotar(reg, ["v02"], {"datos": {"bytes": 2}, "codigo": {"2": "x"}})
        assert not registro.homogeneo(reg)

    def test_una_carpeta_sin_procedencias_se_considera_homogenea(self) -> None:
        assert registro.homogeneo(registro.vacio())


class TestCompletarEjes:
    def test_rellena_un_eje_nuevo_desde_las_huellas_de_sus_modelos(
        self, tmp_path: Path
    ) -> None:
        """La huella de cada artefacto registra con que hiperparametros se
        construyo: es la fuente autoritativa, mejor que el default de hoy.
        """
        modelos = tmp_path / "v01" / "modelo"
        modelos.mkdir(parents=True)
        (modelos / f"formulacion2_equipo_global{registro.SUFIJO_HUELLA}").write_text(
            json.dumps({"hiperparametros": {"F2_L1": 0.125}}), encoding="utf-8")
        reg = {"version": 1, "combinaciones": {"v01": {"hiperparametros": {}}}}
        avisos = registro.completar_ejes(tmp_path, reg, ["F2_L1"], {"F2_L1": 0.5})
        assert reg["combinaciones"]["v01"]["hiperparametros"]["F2_L1"] == 0.125
        assert avisos == []

    def test_si_no_consta_en_ningun_sitio_asume_el_default_avisando(
        self, tmp_path: Path
    ) -> None:
        """Un default que haya cambiado desde entonces etiquetaria mal esa
        combinacion: por eso se avisa en vez de rellenar en silencio.
        """
        reg = {"version": 1, "combinaciones": {"v01": {"hiperparametros": {}}}}
        avisos = registro.completar_ejes(tmp_path, reg, ["F2_L1"], {"F2_L1": 0.5})
        assert reg["combinaciones"]["v01"]["hiperparametros"]["F2_L1"] == 0.5
        assert avisos and "asumido del default" in avisos[0]

    def test_no_toca_los_ejes_ya_registrados(self, tmp_path: Path) -> None:
        reg = {"version": 1, "combinaciones": {
            "v01": {"hiperparametros": {"F2_L1": 0.9}}}}
        registro.completar_ejes(tmp_path, reg, ["F2_L1"], {"F2_L1": 0.5})
        assert reg["combinaciones"]["v01"]["hiperparametros"]["F2_L1"] == 0.9

    def test_dos_huellas_que_discrepan_no_deciden(self, tmp_path: Path) -> None:
        """El alcance por celda hace que muchas huellas ni mencionen el eje; si las
        que lo mencionan no coinciden, no hay valor autoritativo.
        """
        modelos = tmp_path / "v01" / "modelo"
        modelos.mkdir(parents=True)
        for stem, valor in (("a", 0.1), ("b", 0.2)):
            (modelos / f"{stem}{registro.SUFIJO_HUELLA}").write_text(
                json.dumps({"hiperparametros": {"F2_L1": valor}}), encoding="utf-8")
        reg = {"version": 1, "combinaciones": {"v01": {"hiperparametros": {}}}}
        avisos = registro.completar_ejes(tmp_path, reg, ["F2_L1"], {"F2_L1": 0.5})
        assert avisos

    def test_una_huella_corrupta_se_ignora(self, tmp_path: Path) -> None:
        modelos = tmp_path / "v01" / "modelo"
        modelos.mkdir(parents=True)
        (modelos / f"x{registro.SUFIJO_HUELLA}").write_text("{roto", encoding="utf-8")
        reg = {"version": 1, "combinaciones": {"v01": {"hiperparametros": {}}}}
        registro.completar_ejes(tmp_path, reg, ["F2_L1"], {"F2_L1": 0.5})
        assert reg["combinaciones"]["v01"]["hiperparametros"]["F2_L1"] == 0.5


class TestMetricasAcumuladas:
    @staticmethod
    def _fila(combinacion: str, modelo: str, metrica: str, valor: float) -> dict:
        return {"combinacion": combinacion, "modelo": modelo, "formulacion": "2",
                "entidad": "equipo", "normalizacion": "global",
                "fase": "1", "metrica": metrica, "valor": valor}

    def test_una_carpeta_sin_metricas_devuelve_una_tabla_vacia(
        self, tmp_path: Path
    ) -> None:
        assert registro.leer_metricas(tmp_path).empty

    def test_lo_escrito_se_relee(self, tmp_path: Path) -> None:
        df = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.5)])
        registro.escribir_metricas(tmp_path, df)
        assert registro.leer_metricas(tmp_path)["valor"].tolist() == [0.5]

    def test_las_claves_se_releen_como_texto(self, tmp_path: Path) -> None:
        """`formulacion` y `fase` viajan como texto pero `read_csv` los devuelve
        enteros: sin homogeneizar, las filas de disco y las nuevas no se
        reconocerian como la misma celda al fundirlas.
        """
        registro.escribir_metricas(tmp_path, pd.DataFrame(
            [self._fila("v01", "F2_equipo_global", "top1", 0.5)]))
        df = registro.leer_metricas(tmp_path)
        assert df["formulacion"].tolist() == ["2"]
        assert df["fase"].tolist() == ["1"]

    def test_lo_recalculado_sustituye_a_lo_guardado(self) -> None:
        previa = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.1)])
        nueva = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.9)])
        fundida = registro.acumular(previa, nueva)
        assert fundida["valor"].tolist() == [0.9]

    def test_lo_que_esta_ejecucion_no_toca_se_conserva(self) -> None:
        """Es lo que permite barrer en tandas: una ejecucion acotada con
        `--formulaciones 5` no puede borrar las filas F2 de esa combinacion.
        """
        previa = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.1)])
        nueva = pd.DataFrame([self._fila("v01", "F5_equipo_global", "top1", 0.9)])
        fundida = registro.acumular(previa, nueva)
        assert set(fundida["modelo"]) == {"F2_equipo_global", "F5_equipo_global"}

    def test_la_sustitucion_es_por_metrica_no_por_combinacion_entera(self) -> None:
        previa = pd.DataFrame([self._fila("v01", "F2_equipo_global", "mrr", 0.3)])
        nueva = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.9)])
        fundida = registro.acumular(previa, nueva)
        assert set(fundida["metrica"]) == {"mrr", "top1"}

    def test_acumular_sobre_una_tabla_vacia_devuelve_la_nueva(self) -> None:
        nueva = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.9)])
        assert registro.acumular(pd.DataFrame(), nueva).equals(registro._tipos(nueva))

    def test_una_ejecucion_sin_resultados_no_borra_lo_acumulado(self) -> None:
        previa = pd.DataFrame([self._fila("v01", "F2_equipo_global", "top1", 0.1)])
        assert len(registro.acumular(previa, pd.DataFrame())) == 1

    def test_el_resultado_va_ordenado_en_orden_natural_de_combinacion(self) -> None:
        filas = [self._fila(c, "F2_equipo_global", "top1", 0.1)
                 for c in ("v10", "v2", "v1")]
        fundida = registro.acumular(pd.DataFrame(), pd.DataFrame(filas))
        fundida = registro.acumular(fundida, pd.DataFrame(filas))
        assert fundida["combinacion"].tolist() == ["v1", "v2", "v10"]
