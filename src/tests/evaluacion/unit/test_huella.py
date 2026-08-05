"""Huella de un modelo: la cache que decide si un artefacto se reutiliza.

Es el modulo con el fallo mas caro posible del proyecto: servir en silencio una S
vieja despues de cambiar un hiperparametro o el nucleo numerico. Por eso casi todas
las pruebas son NEGATIVAS —comprueban que la huella DEJA de coincidir cuando algo
cambia—, y no solo que coincide cuando no cambia nada.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluacion import huella
from src.similitud import config as scfg
from src.tests.evaluacion.conftest import modelo_sintetico

STEM = "formulacion2_equipo_global"


@pytest.fixture
def bd(tmp_path: Path) -> Path:
    ruta = tmp_path / "scouting.db"
    ruta.write_bytes(b"no importa el contenido, si el tamano y el mtime")
    return ruta


@pytest.fixture
def h(bd: Path) -> dict:
    return huella.calcular("2", "equipo", "global", bd)


def _guardar_artefacto(model_dir: Path, stem: str = STEM, **kw) -> None:
    """Escribe el par .npz + .json que `coincide` exige que exista."""
    form, entidad, norm = stem.replace("formulacion", "").split("_", 2)
    modelo_sintetico(np.zeros((2, 2)), formulacion=form, entidad=entidad,
                     meta={"normalizacion": norm, **kw.pop("meta", {})},
                     **kw).guardar(model_dir)


class TestAlcance:
    def test_una_celda_f2_no_depende_de_los_hiperparametros_de_la_f5(self) -> None:
        """Es lo que permite que una combinacion que solo mueve `F2_L1` copie los
        cuatro modelos F5 de otra en vez de reconstruirlos.
        """
        attrs = huella.alcance("2", "jugador")
        assert "F2_L1" in attrs
        assert not [a for a in attrs if a.startswith("F5_")]

    def test_una_celda_de_equipo_no_depende_de_la_posicion(self) -> None:
        """El equipo no recibe el bloque `pos_*`: mover la posicion no puede
        invalidar sus modelos.
        """
        attrs = huella.alcance("5", "equipo")
        assert "USE_POSITION_FEATURES" not in attrs
        assert "POSITION_SCALING" not in attrs

    def test_la_celda_de_jugador_si_depende_de_la_posicion(self) -> None:
        assert "USE_POSITION_FEATURES" in huella.alcance("2", "jugador")

    def test_la_f5_incluye_el_metodo_distribucional_de_su_entidad(self) -> None:
        """MMD en jugador, Sinkhorn en equipo: cambiarlo cambia el artefacto."""
        assert "F5_METODO_JUGADOR" in huella.alcance("5", "jugador")
        assert "F5_METODO_EQUIPO" in huella.alcance("5", "equipo")
        assert "F5_METODO_EQUIPO" not in huella.alcance("5", "jugador")

    def test_el_alcance_va_ordenado(self) -> None:
        """La huella se compara clave a clave, pero el orden estable hace que dos
        huellas equivalentes se lean igual en disco.
        """
        attrs = huella.alcance("2", "equipo")
        assert attrs == sorted(attrs)


class TestCoberturaDeHiperparametros:
    def test_pasa_con_la_configuracion_actual(self) -> None:
        huella._comprobar_cobertura()          # no debe lanzar

    def test_un_hiperparametro_nuevo_sin_clasificar_rompe_el_barrido(
        self, monkeypatch: pytest.MonkeyPatch, bd: Path
    ) -> None:
        """Preferimos romper a generar huellas incompletas: un atributo fuera de la
        huella haria que la cache sirviera un modelo construido con otro valor.
        """
        monkeypatch.setattr(scfg, "UN_PARAMETRO_NUEVO", 1.0, raising=False)
        with pytest.raises(RuntimeError, match="sin clasificar"):
            huella.calcular("2", "equipo", "global", bd)

    def test_los_atributos_en_minuscula_no_cuentan_como_hiperparametros(
        self, monkeypatch: pytest.MonkeyPatch, bd: Path
    ) -> None:
        monkeypatch.setattr(scfg, "algo_privado", 1.0, raising=False)
        huella.calcular("2", "equipo", "global", bd)


class TestCalcular:
    def test_recoge_la_celda_los_datos_y_el_codigo(self, h: dict) -> None:
        assert h["celda"] == {"formulacion": "2", "entidad": "equipo",
                              "normalizacion": "global"}
        assert set(h["datos"]) == {"bytes", "mtime_ns"}
        assert len(h["codigo"]) == 64          # sha256 en hexadecimal

    def test_recoge_los_hiperparametros_del_alcance_de_la_celda(self, h: dict) -> None:
        assert set(h["hiperparametros"]) == set(huella.alcance("2", "equipo"))

    def test_recoge_el_valor_vigente_de_config(
        self, bd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Se llama dentro del `_config_temporal` de la combinacion: si leyera el
        default, todas las combinaciones tendrian la misma huella.
        """
        monkeypatch.setattr(scfg, "F2_L1", 0.987)
        assert huella.calcular("2", "equipo", "global", bd)["hiperparametros"]["F2_L1"] == 0.987

    def test_queda_en_forma_canonica_json(self, h: dict) -> None:
        """Varios hiperparametros son tuplas y JSON las relee como listas: sin
        canonizar, una huella recien calculada nunca igualaria a la de disco y la
        cache no acertaria jamas.
        """
        assert h == json.loads(json.dumps(h))

    def test_el_hash_de_codigo_es_distinto_por_formulacion(self, bd: Path) -> None:
        """Tocar `slim.py` no puede invalidar los F5 que no lo usan... y tocar
        `distributional.py` no puede invalidar los F2.
        """
        h2 = huella.calcular("2", "equipo", "global", bd)
        h5 = huella.calcular("5", "equipo", "global", bd)
        assert h2["codigo"] != h5["codigo"]

    def test_el_hash_de_codigo_es_estable_entre_llamadas(self, bd: Path) -> None:
        assert (huella.calcular("2", "equipo", "global", bd)["codigo"]
                == huella.calcular("2", "jugador", "por_liga", bd)["codigo"])


class TestPersistencia:
    def test_lo_escrito_se_relee_igual(self, tmp_path: Path, h: dict) -> None:
        huella.escribir(tmp_path, STEM, h)
        assert huella.leer(tmp_path, STEM) == h

    def test_la_huella_vive_junto_al_artefacto(self, tmp_path: Path, h: dict) -> None:
        ruta = huella.escribir(tmp_path, STEM, h)
        assert ruta == tmp_path / f"{STEM}{huella.SUFIJO}"

    def test_leer_lo_que_no_existe_devuelve_none(self, tmp_path: Path) -> None:
        assert huella.leer(tmp_path, STEM) is None

    def test_una_huella_corrupta_se_trata_como_ausente(self, tmp_path: Path) -> None:
        """Reconstruir es caro pero correcto; leer basura seria incorrecto."""
        huella.ruta(tmp_path, STEM).write_text("{no es json", encoding="utf-8")
        assert huella.leer(tmp_path, STEM) is None

    def test_invalidar_borra_la_huella(self, tmp_path: Path, h: dict) -> None:
        """Se llama ANTES de reconstruir: si la construccion peta, el artefacto a
        medias queda sin huella y nadie lo reutiliza.
        """
        huella.escribir(tmp_path, STEM, h)
        huella.invalidar(tmp_path, STEM)
        assert not huella.ruta(tmp_path, STEM).exists()

    def test_invalidar_lo_que_no_existe_no_falla(self, tmp_path: Path) -> None:
        huella.invalidar(tmp_path, STEM)

    def test_el_artefacto_esta_completo_solo_con_los_dos_ficheros(
        self, tmp_path: Path
    ) -> None:
        assert not huella.artefacto_completo(tmp_path, STEM)
        (tmp_path / f"{STEM}.npz").write_bytes(b"")
        assert not huella.artefacto_completo(tmp_path, STEM)
        (tmp_path / f"{STEM}.json").write_text("{}", encoding="utf-8")
        assert huella.artefacto_completo(tmp_path, STEM)


class TestHiperparametrosIguales:
    def test_los_mismos_valores_son_iguales(self) -> None:
        a = {"hiperparametros": {"F2_L1": 0.5, "F2_BETA": 1.0}}
        assert huella.hiperparametros_iguales(a, dict(a))

    def test_un_valor_distinto_ya_no_es_igual(self) -> None:
        a = {"hiperparametros": {"F2_L1": 0.5}}
        b = {"hiperparametros": {"F2_L1": 0.6}}
        assert not huella.hiperparametros_iguales(a, b)

    def test_una_huella_con_menos_atributos_no_vale(self) -> None:
        """De los que le faltan no se sabe con que valor se construyo: darla por
        buena es exactamente el fallo que la cache tiene que evitar.
        """
        guardada = {"hiperparametros": {"F2_L1": 0.5}}
        pedida = {"hiperparametros": {"F2_L1": 0.5, "F2_BETA": 1.0}}
        assert not huella.hiperparametros_iguales(guardada, pedida)

    def test_una_huella_con_atributos_de_mas_tampoco(self) -> None:
        guardada = {"hiperparametros": {"F2_L1": 0.5, "F2_BETA": 1.0}}
        pedida = {"hiperparametros": {"F2_L1": 0.5}}
        assert not huella.hiperparametros_iguales(guardada, pedida)

    def test_sin_bloque_de_hiperparametros_no_es_igual(self) -> None:
        assert not huella.hiperparametros_iguales({}, {"hiperparametros": {}})


class TestCoincide:
    def test_reconoce_el_artefacto_construido_con_esa_huella(
        self, tmp_path: Path, h: dict
    ) -> None:
        _guardar_artefacto(tmp_path)
        huella.escribir(tmp_path, STEM, h)
        assert huella.coincide(tmp_path, STEM, h)

    def test_una_huella_sin_artefacto_no_sirve(self, tmp_path: Path, h: dict) -> None:
        huella.escribir(tmp_path, STEM, h)
        assert not huella.coincide(tmp_path, STEM, h)

    def test_un_artefacto_sin_huella_no_se_reutiliza(
        self, tmp_path: Path, h: dict
    ) -> None:
        """Es el caso de los modelos anteriores a la cache: solo `--adoptar-existentes`
        los recupera, y con reservas.
        """
        _guardar_artefacto(tmp_path)
        assert not huella.coincide(tmp_path, STEM, h)

    @pytest.mark.parametrize("campo,valor", [
        ("version", 99),
        ("celda", {"formulacion": "5", "entidad": "equipo", "normalizacion": "global"}),
        ("codigo", "otro-hash"),
        ("datos", {"bytes": 1, "mtime_ns": 2}),
    ])
    def test_cualquier_parte_distinta_obliga_a_reconstruir(
        self, tmp_path: Path, h: dict, campo: str, valor
    ) -> None:
        _guardar_artefacto(tmp_path)
        huella.escribir(tmp_path, STEM, {**h, campo: valor})
        assert not huella.coincide(tmp_path, STEM, h)

    def test_un_hiperparametro_distinto_obliga_a_reconstruir(
        self, tmp_path: Path, h: dict
    ) -> None:
        _guardar_artefacto(tmp_path)
        otra = {**h, "hiperparametros": {**h["hiperparametros"], "F2_L1": 999.0}}
        huella.escribir(tmp_path, STEM, otra)
        assert not huella.coincide(tmp_path, STEM, h)

    def test_reextraer_la_bd_invalida_la_cache(self, tmp_path: Path, bd: Path) -> None:
        """La huella de datos son tamaño y mtime: reextraer cambia los dos."""
        h_antes = huella.calcular("2", "equipo", "global", bd)
        _guardar_artefacto(tmp_path)
        huella.escribir(tmp_path, STEM, h_antes)
        bd.write_bytes(b"otra base de datos, de otro tamano por completo")
        h_despues = huella.calcular("2", "equipo", "global", bd)
        assert not huella.coincide(tmp_path, STEM, h_despues)


class TestProcedencia:
    def test_declara_los_datos_y_el_codigo_de_las_dos_formulaciones(
        self, bd: Path
    ) -> None:
        """Se hashean SIEMPRE las dos, tambien la que la corrida no construya: si
        dependiera de `--formulaciones`, dos corridas de la misma carpeta
        parecerian discrepar solo por haberse acotado distinto.
        """
        p = huella.procedencia(bd)
        assert set(p) == {"datos", "codigo"}
        assert set(p["codigo"]) == {"2", "5"}

    def test_dos_corridas_seguidas_declaran_la_misma_procedencia(self, bd: Path) -> None:
        assert huella.procedencia(bd) == huella.procedencia(bd)


class TestAdoptar:
    """Recuperacion opt-in de artefactos anteriores a la cache."""

    @pytest.fixture
    def h_equipo(self, bd: Path) -> dict:
        return huella.calcular("2", "equipo", "global", bd)

    def test_adopta_un_artefacto_cuyo_meta_concuerda(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        _guardar_artefacto(tmp_path, meta={"n_neighbors": scfg.F2_N_NEIGHBORS,
                                           "beta": scfg.F2_BETA, "l1": scfg.F2_L1})
        assert huella.adoptar(tmp_path, STEM, "equipo", h_equipo)
        assert huella.coincide(tmp_path, STEM, h_equipo)

    def test_no_adopta_lo_que_no_existe(self, tmp_path: Path, h_equipo: dict) -> None:
        assert not huella.adoptar(tmp_path, STEM, "equipo", h_equipo)

    def test_no_adopta_si_un_hiperparametro_del_meta_discrepa(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        _guardar_artefacto(tmp_path, meta={"l1": scfg.F2_L1 + 1.0})
        assert not huella.adoptar(tmp_path, STEM, "equipo", h_equipo)

    def test_un_campo_ausente_del_meta_no_bloquea_la_adopcion(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        """`None` = no aplica a ese metodo (sinkhorn_reg con mmd): no informa de
        nada, asi que no puede decidir.
        """
        _guardar_artefacto(tmp_path)
        assert huella.adoptar(tmp_path, STEM, "equipo", h_equipo)

    def test_no_adopta_si_la_normalizacion_no_es_la_de_la_celda(
        self, tmp_path: Path, bd: Path
    ) -> None:
        _guardar_artefacto(tmp_path)
        otra_celda = huella.calcular("2", "equipo", "por_liga", bd)
        assert not huella.adoptar(tmp_path, STEM, "equipo", otra_celda)

    def test_no_adopta_si_la_formulacion_no_es_la_de_la_celda(
        self, tmp_path: Path, bd: Path
    ) -> None:
        _guardar_artefacto(tmp_path)
        assert not huella.adoptar(tmp_path, STEM, "equipo",
                                  huella.calcular("5", "equipo", "global", bd))

    def test_no_adopta_si_la_entidad_no_coincide(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        _guardar_artefacto(tmp_path)
        assert not huella.adoptar(tmp_path, STEM, "jugador", h_equipo)

    def test_nunca_adopta_un_artefacto_con_bloque_de_posicion(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        """`feat_names` delata SI hay columnas `pos_*` pero no con que escalado
        entraron: los tres modos de `POSITION_SCALING` son indistinguibles a
        posteriori.
        """
        _guardar_artefacto(tmp_path, feat_names=["xg", "pos_1"])
        assert not huella.adoptar(tmp_path, STEM, "equipo", h_equipo)

    def test_nunca_adopta_si_la_configuracion_pide_posicion(
        self, tmp_path: Path, bd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Con el modelo base actual (con posicion), en la practica solo se
        adoptan modelos de equipo.
        """
        monkeypatch.setattr(scfg, "USE_POSITION_FEATURES", True)
        stem = "formulacion2_jugador_global"
        _guardar_artefacto(tmp_path, stem)
        h = huella.calcular("2", "jugador", "global", bd)
        assert not huella.adoptar(tmp_path, stem, "jugador", h)

    def test_no_escribe_huella_cuando_rechaza(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        _guardar_artefacto(tmp_path, meta={"beta": scfg.F2_BETA + 5.0})
        huella.adoptar(tmp_path, STEM, "equipo", h_equipo)
        assert huella.leer(tmp_path, STEM) is None

    def test_un_json_de_artefacto_corrupto_no_se_adopta(
        self, tmp_path: Path, h_equipo: dict
    ) -> None:
        (tmp_path / f"{STEM}.npz").write_bytes(b"")
        (tmp_path / f"{STEM}.json").write_text("{roto", encoding="utf-8")
        assert not huella.adoptar(tmp_path, STEM, "equipo", h_equipo)
