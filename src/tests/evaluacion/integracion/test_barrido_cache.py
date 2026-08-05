"""Barrido de verdad: cache de artefactos y carpeta acumulativa entre corridas.

Se construyen y evaluan modelos reales (equipo, que es la entidad barata), porque
lo que se prueba es justo lo que no se puede simular: que la segunda corrida NO
reconstruya nada, que una combinacion copie los artefactos de otra cuando su
huella coincide, y que relanzar el barrido sobre la misma carpeta conserve los
nombres `vNN` y las metricas anteriores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluacion import barrido, datos, huella, registro
from src.similitud import config as scfg

pytestmark = [pytest.mark.integracion, pytest.mark.lento]

# Rejilla minima: solo la F5 de equipo, cuyo ajuste (Sinkhorn sobre 6 entidades)
# cuesta segundos. Barrer la F2 de jugador aqui costaria minutos sin probar nada
# nuevo sobre la cache.
FORMULACIONES = ("5",)
ENTIDADES = ("equipo",)


@pytest.fixture
def rejilla_pequena(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list]]:
    """Dos ejes de dos valores = 4 combinaciones, en vez de las ~300 del repo."""
    ejes = [("F5_EASE_LAMBDA", [10.0, 50.0]), ("F5_RFF_DIM", [512, 1024])]
    monkeypatch.setattr(barrido, "HIPERPARAMETROS", ejes)
    return ejes


class TestConstruirGrid:
    def test_la_primera_corrida_construye_todo(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        conteo = barrido._construir_grid(
            bd_sintetica, tmp_path / "v01" / "modelo", tmp_path,
            formulaciones=FORMULACIONES, entidades=ENTIDADES)
        assert conteo == {"en_cache": 0, "copiados": 0, "adoptados": 0,
                          "construidos": 2}

    def test_la_segunda_corrida_no_reconstruye_nada(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """Relanzar el barrido sin cambios tiene que costar 0 construcciones."""
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        conteo = barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                         formulaciones=FORMULACIONES,
                                         entidades=ENTIDADES)
        assert conteo["en_cache"] == 2
        assert conteo["construidos"] == 0

    def test_una_combinacion_copia_los_artefactos_identicos_de_otra(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """La lambda del EASE no entra en el alcance de ningun modelo F2... y aqui,
        con la config sin tocar, dos combinaciones distintas comparten huella: la
        segunda copia en segundos lo que la primera tardo en ajustar.
        """
        barrido._construir_grid(bd_sintetica, tmp_path / "v01" / "modelo", tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        conteo = barrido._construir_grid(
            bd_sintetica, tmp_path / "v02" / "modelo", tmp_path,
            formulaciones=FORMULACIONES, entidades=ENTIDADES)
        assert conteo["copiados"] == 2
        assert conteo["construidos"] == 0

    def test_un_hiperparametro_distinto_obliga_a_reconstruir(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        barrido._construir_grid(bd_sintetica, tmp_path / "v01" / "modelo", tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        with barrido._config_temporal({"F5_EASE_LAMBDA": scfg.F5_EASE_LAMBDA + 7.0}):
            conteo = barrido._construir_grid(
                bd_sintetica, tmp_path / "v02" / "modelo", tmp_path,
                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        assert conteo["construidos"] == 2

    def test_rehacer_ignora_la_cache(self, bd_sintetica: Path, tmp_path: Path) -> None:
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        conteo = barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                         rehacer=True, formulaciones=FORMULACIONES,
                                         entidades=ENTIDADES)
        assert conteo["construidos"] == 2

    def test_cada_artefacto_queda_con_su_huella(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        for norm in ("por_liga", "global"):
            stem = barrido._stem("5", "equipo", norm)
            assert huella.leer(model_dir, stem) is not None

    def test_el_log_dice_de_donde_sale_cada_modelo(
        self, bd_sintetica: Path, tmp_path: Path, capsys
    ) -> None:
        """Es lo que separa los minutos de un ajuste de los segundos de una copia."""
        barrido._construir_grid(bd_sintetica, tmp_path / "v01" / "modelo", tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        salida = capsys.readouterr().out
        assert "[construye]" in salida
        assert "NO se carga de disco, se construye en:" in salida
        barrido._construir_grid(bd_sintetica, tmp_path / "v01" / "modelo", tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        assert "carga de disco:" in capsys.readouterr().out

    def test_adopta_un_artefacto_sin_huella_si_se_pide(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """Los modelos anteriores a la cache no tienen huella. Adoptarlos es opt-in
        porque afirma algo no verificable a posteriori: que ni el nucleo numerico
        ni la BD han cambiado desde que se construyeron.
        """
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        for p in model_dir.glob(f"*{huella.SUFIJO}"):
            p.unlink()
        conteo = barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                         adoptar=True, formulaciones=FORMULACIONES,
                                         entidades=ENTIDADES)
        assert conteo["adoptados"] == 2
        assert conteo["construidos"] == 0

    def test_sin_adoptar_esos_mismos_artefactos_se_reconstruyen(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=FORMULACIONES, entidades=ENTIDADES)
        for p in model_dir.glob(f"*{huella.SUFIJO}"):
            p.unlink()
        conteo = barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                         formulaciones=FORMULACIONES,
                                         entidades=ENTIDADES)
        assert conteo["construidos"] == 2

    def test_no_construye_lo_que_queda_fuera_del_subconjunto(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "v01" / "modelo"
        barrido._construir_grid(bd_sintetica, model_dir, tmp_path,
                                formulaciones=("5",), entidades=("equipo",))
        assert not list(model_dir.glob("formulacion2_*"))
        assert not list(model_dir.glob("*_jugador_*"))


class TestEvaluarCombinacion:
    def test_construye_evalua_y_escribe_los_csv_de_la_combinacion(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        roles = datos.rol_por_jugador(bd_sintetica)
        minutos = datos.minutos_por_jugador(bd_sintetica)
        barrido.evaluar_combinacion(
            "v01", {"F5_EASE_LAMBDA": 25.0}, bd_sintetica, tmp_path,
            {"0", "1"}, 0, roles, minutos, con_figuras=False,
            formulaciones=FORMULACIONES, entidades=ENTIDADES)
        csv = tmp_path / "v01" / "fase0_sanity.csv"
        assert csv.exists()
        assert pd.read_csv(csv)["combinacion"].tolist() == ["v01"] * 2

    def test_dibuja_las_figuras_de_la_combinacion_si_se_piden(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        roles = datos.rol_por_jugador(bd_sintetica)
        minutos = datos.minutos_por_jugador(bd_sintetica)
        barrido.evaluar_combinacion(
            "v01", {}, bd_sintetica, tmp_path, {"1"}, 0, roles, minutos,
            con_figuras=True, formulaciones=FORMULACIONES, entidades=ENTIDADES)
        assert (tmp_path / "v01" / "figuras" /
                "fase1_autosimilitud_top1.png").exists()

    def test_la_combinacion_fija_de_verdad_el_hiperparametro(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """Si no lo fijara, todas las combinaciones darian el mismo modelo y el
        barrido compararia columnas identicas sin decirlo.
        """
        roles = datos.rol_por_jugador(bd_sintetica)
        minutos = datos.minutos_por_jugador(bd_sintetica)
        barrido.evaluar_combinacion(
            "v01", {"F5_EASE_LAMBDA": 1234.0}, bd_sintetica, tmp_path,
            {"0"}, 0, roles, minutos, con_figuras=False,
            formulaciones=FORMULACIONES, entidades=ENTIDADES)
        h = huella.leer(tmp_path / "v01" / "modelo",
                        barrido._stem("5", "equipo", "por_liga"))
        assert h["hiperparametros"]["F5_EASE_LAMBDA"] == 1234.0
        assert scfg.F5_EASE_LAMBDA != 1234.0     # y se restaura al salir

    def test_sin_ningun_modelo_construido_aborta(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit, match="No se construyo ningun modelo"):
            barrido.evaluar_combinacion(
                "v01", {}, bd_sintetica, tmp_path, {"0"}, 0, {}, {},
                con_figuras=False, formulaciones=(), entidades=())


class TestCarpetaAcumulativa:
    """Dos corridas sobre la misma carpeta, que es el escenario que motiva todo."""

    def _correr(self, bd: Path, out: Path, *args: str) -> None:
        barrido.main(["--db", str(bd), "--out", str(out),
                      "--formulaciones", ",".join(FORMULACIONES),
                      "--entidades", ",".join(ENTIDADES),
                      "--fases", "0", "--bootstrap", "0", "--sin-figuras", *args])

    def test_una_corrida_escribe_los_tres_entregables(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena
    ) -> None:
        self._correr(bd_sintetica, tmp_path)
        assert (tmp_path / "resumen_barrido.md").exists()
        assert (tmp_path / registro.ARCHIVO_METRICAS).exists()
        assert (tmp_path / registro.ARCHIVO).exists()

    def test_hay_una_carpeta_por_combinacion(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena
    ) -> None:
        self._correr(bd_sintetica, tmp_path)
        assert sorted(p.name for p in tmp_path.glob("v*")) == [
            "v01", "v02", "v03", "v04"]

    def test_relanzar_sin_cambios_conserva_los_nombres(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena
    ) -> None:
        """Cada `vNN` tiene que seguir designando la misma configuracion (la fecha
        de la corrida si cambia: es lo unico que se reescribe).
        """
        self._correr(bd_sintetica, tmp_path)
        antes = registro.hiperparametros(registro.cargar(tmp_path))
        self._correr(bd_sintetica, tmp_path)
        assert registro.hiperparametros(registro.cargar(tmp_path)) == antes

    def test_ampliar_la_rejilla_suma_puntos_sin_pisar_los_anteriores(
        self, bd_sintetica: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Es la forma prevista de barrer: en tandas, acabando con una rejilla
        completa en las superficies 3D.
        """
        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [10.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)
        primera = registro.leer_metricas(tmp_path)

        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [10.0, 50.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)
        segunda = registro.leer_metricas(tmp_path)

        assert set(primera["combinacion"]) == {"v01"}
        assert set(segunda["combinacion"]) == {"v01", "v02"}

    def test_una_configuracion_ya_vista_recupera_su_nombre(
        self, bd_sintetica: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Antes el `vNN` salia de la POSICION en el producto cartesiano, asi que
        añadir un valor por delante cambiaba el significado de cada nombre.
        """
        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [50.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)

        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [10.0, 50.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)

        combos = registro.hiperparametros(registro.cargar(tmp_path))
        assert combos["v01"]["F5_EASE_LAMBDA"] == 50.0
        assert combos["v02"]["F5_EASE_LAMBDA"] == 10.0

    def test_el_resumen_compara_todas_las_combinaciones_acumuladas(
        self, bd_sintetica: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [10.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)
        monkeypatch.setattr(barrido, "HIPERPARAMETROS",
                            [("F5_EASE_LAMBDA", [50.0]), ("F5_RFF_DIM", [512])])
        self._correr(bd_sintetica, tmp_path)

        texto = (tmp_path / "resumen_barrido.md").read_text(encoding="utf-8")
        assert "| v01 |" in texto and "| v02 |" in texto
        assert "2 combinaciones" in texto

    def test_la_segunda_corrida_no_vuelve_a_ajustar_los_modelos(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena, capsys
    ) -> None:
        self._correr(bd_sintetica, tmp_path)
        capsys.readouterr()
        self._correr(bd_sintetica, tmp_path)
        assert "[construye]" not in capsys.readouterr().out

    def test_avisa_de_un_eje_que_una_combinacion_anterior_no_registro(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena, capsys
    ) -> None:
        """Pasa al AÑADIR un eje a HIPERPARAMETROS: las combinaciones ya
        registradas no lo declaran. Si tampoco consta en sus huellas se asume el
        default vigente, y eso hay que decirlo: un default que haya cambiado desde
        entonces etiquetaria mal esa combinacion.
        """
        reg = registro.vacio()
        reg["combinaciones"]["v09"] = {"hiperparametros": {"F5_EASE_LAMBDA": 10.0}}
        registro.guardar(tmp_path, reg)
        self._correr(bd_sintetica, tmp_path)
        assert "eje sin registrar en una combinacion anterior" in capsys.readouterr().out

    def test_anota_la_procedencia_de_cada_combinacion(
        self, bd_sintetica: Path, tmp_path: Path, rejilla_pequena
    ) -> None:
        """Acumular numeros producidos con otra BD u otro nucleo numerico los haria
        incomparables sin avisar.
        """
        self._correr(bd_sintetica, tmp_path)
        proc = registro.procedencias(registro.cargar(tmp_path))
        assert all({"corrida", "datos", "codigo"} <= set(p) for p in proc.values())
        assert registro.homogeneo(registro.cargar(tmp_path))
