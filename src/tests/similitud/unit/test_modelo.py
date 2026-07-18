"""Artefacto de modelo: indexado, features de display, guardado/carga y top-k."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.similitud.features import MatrizFeatures
from src.similitud.modelo import (
    ModeloSimilitud,
    cargar_modelo,
    features_display,
    indexar_entidades,
    ligas_por_entidad,
    top_k,
)


def _mf(entity_id, entity_name=None, X=None, weight=None, league=None) -> MatrizFeatures:
    n = len(entity_id)
    if X is None:
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
    return MatrizFeatures(
        X=X,
        feat_names=[f"f{i}" for i in range(X.shape[1])],
        entity_id=np.array(entity_id),
        entity_name=np.array(entity_name if entity_name is not None
                            else [f"Ent {i}" for i in entity_id]),
        weight=np.ones(n) if weight is None else np.array(weight, dtype=float),
        league=np.array(league if league is not None else ["L1"] * n),
    )


def _modelo(S: np.ndarray, **cambios) -> ModeloSimilitud:
    n = len(S)
    base = dict(
        formulacion="2",
        entidad="jugador",
        S=S,
        entity_ids=np.arange(n),
        entity_names=[f"Ent {i}" for i in range(n)],
        feat_names=["f0", "f1"],
        feat_display=np.zeros((n, 2)),
        meta={"normalizacion": "por_liga"},
    )
    base.update(cambios)
    return ModeloSimilitud(**base)


class TestIndexarEntidades:
    def test_asigna_un_indice_por_entidad(self) -> None:
        idx = indexar_entidades(_mf([7, 7, 3, 3, 3]))
        assert idx.ids.tolist() == [3, 7]
        assert len(idx.names) == 2

    def test_el_orden_es_por_id_y_estable_entre_ejecuciones(self) -> None:
        """Las filas de S se alinean con `ids`: el orden debe ser reproducible."""
        a = indexar_entidades(_mf([9, 1, 5, 1]))
        b = indexar_entidades(_mf([1, 5, 9, 1]))
        assert a.ids.tolist() == b.ids.tolist() == [1, 5, 9]

    def test_row_entity_mapea_cada_observacion_a_su_entidad(self) -> None:
        idx = indexar_entidades(_mf([7, 3, 7]))
        # ids ordenados = [3, 7] -> la entidad 7 es el indice 1.
        assert idx.row_entity.tolist() == [1, 0, 1]

    def test_los_nombres_van_alineados_con_los_ids(self) -> None:
        idx = indexar_entidades(_mf([7, 3], ["Messi", "Pedri"]))
        assert idx.names == ["Pedri", "Messi"]   # ordenados por id: 3, 7

    def test_una_sola_entidad(self) -> None:
        idx = indexar_entidades(_mf([4, 4, 4]))
        assert idx.ids.tolist() == [4]
        assert idx.row_entity.tolist() == [0, 0, 0]


class TestFeaturesDisplay:
    def test_promedia_las_observaciones_de_cada_entidad(self) -> None:
        X = np.array([[0.0, 1.0], [2.0, 3.0], [10.0, 10.0]])
        mf = _mf([1, 1, 2], X=X)
        display = features_display(mf, indexar_entidades(mf))
        assert display == pytest.approx(np.array([[1.0, 2.0], [10.0, 10.0]]))

    def test_pondera_por_los_minutos(self) -> None:
        X = np.array([[0.0], [10.0]])
        mf = _mf([1, 1], X=X, weight=[1.0, 0.1])
        display = features_display(mf, indexar_entidades(mf))
        assert display == pytest.approx(np.array([[10.0 * 0.1 / 1.1]]))

    def test_una_entidad_sin_masa_no_divide_por_cero(self) -> None:
        mf = _mf([1, 1], X=np.array([[1.0], [2.0]]), weight=[0.0, 0.0])
        assert np.isfinite(features_display(mf, indexar_entidades(mf))).all()

    def test_tiene_una_fila_por_entidad(self) -> None:
        mf = _mf([1, 1, 2, 3])
        assert features_display(mf, indexar_entidades(mf)).shape == (3, 2)


class TestLigasPorEntidad:
    def test_recoge_las_ligas_de_cada_entidad(self) -> None:
        mf = _mf([1, 1, 2], league=["11-90", "16-90", "11-90"])
        ligas = ligas_por_entidad(mf, indexar_entidades(mf))
        assert ligas == {"1": ["11-90", "16-90"], "2": ["11-90"]}

    def test_no_repite_ligas(self) -> None:
        mf = _mf([1, 1, 1], league=["11-90", "11-90", "11-90"])
        assert ligas_por_entidad(mf, indexar_entidades(mf)) == {"1": ["11-90"]}

    def test_se_indexa_por_id_no_por_nombre(self) -> None:
        """Dos jugadores pueden llamarse igual; agrupar por nombre fusionaria sus
        ligas y etiquetaria mal a los candidatos.
        """
        mf = _mf([1, 2], ["Danilo", "Danilo"], league=["11-90", "16-90"])
        ligas = ligas_por_entidad(mf, indexar_entidades(mf))
        assert ligas == {"1": ["11-90"], "2": ["16-90"]}

    def test_las_claves_son_texto_para_poder_serializarse_en_json(self) -> None:
        mf = _mf([1, 2])
        ligas = ligas_por_entidad(mf, indexar_entidades(mf))
        assert all(isinstance(k, str) for k in ligas)
        json.dumps(ligas)   # no debe lanzar


class TestGuardarYCargar:
    def test_ida_y_vuelta_conserva_el_modelo(self, tmp_path: Path) -> None:
        S = np.array([[0.0, 0.5], [0.5, 0.0]])
        original = _modelo(S, entity_names=["Messi", "Pedri"],
                           feat_display=np.array([[1.0, 2.0], [3.0, 4.0]]),
                           meta={"normalizacion": "por_liga", "descripcion": "x"})
        original.guardar(tmp_path)
        cargado = cargar_modelo(tmp_path, "2", "jugador", normalizacion="por_liga")
        assert cargado.S == pytest.approx(original.S)
        assert cargado.entity_ids.tolist() == original.entity_ids.tolist()
        assert cargado.entity_names == ["Messi", "Pedri"]
        assert cargado.feat_names == original.feat_names
        assert cargado.feat_display == pytest.approx(original.feat_display)
        assert cargado.meta["descripcion"] == "x"

    def test_el_nombre_del_artefacto_lleva_la_normalizacion(self, tmp_path: Path) -> None:
        """Cada modo genera un artefacto independiente; sin sufijo, construir los
        dos sobrescribiria el primero.
        """
        _modelo(np.zeros((2, 2)), meta={"normalizacion": "global"}).guardar(tmp_path)
        assert (tmp_path / "formulacion2_jugador_global.npz").exists()
        assert (tmp_path / "formulacion2_jugador_global.json").exists()

    def test_las_dos_normalizaciones_conviven(self, tmp_path: Path) -> None:
        _modelo(np.zeros((2, 2)), meta={"normalizacion": "por_liga"}).guardar(tmp_path)
        _modelo(np.ones((2, 2)), meta={"normalizacion": "global"}).guardar(tmp_path)
        por_liga = cargar_modelo(tmp_path, "2", "jugador", normalizacion="por_liga")
        glob = cargar_modelo(tmp_path, "2", "jugador", normalizacion="global")
        assert por_liga.S.sum() == 0.0
        assert glob.S.sum() == 4.0

    def test_crea_el_directorio_de_salida(self, tmp_path: Path) -> None:
        destino = tmp_path / "no" / "existe"
        _modelo(np.zeros((2, 2))).guardar(destino)
        assert destino.exists()

    def test_el_json_es_legible_y_va_en_utf8(self, tmp_path: Path) -> None:
        _modelo(np.zeros((2, 2)), entity_names=["Söyüncü", "Pedri"]).guardar(tmp_path)
        meta = json.loads(
            (tmp_path / "formulacion2_jugador_por_liga.json").read_text(encoding="utf-8"))
        assert meta["entity_names"] == ["Söyüncü", "Pedri"]

    def test_un_modelo_antiguo_sin_sufijo_vale_como_por_liga(self, tmp_path: Path) -> None:
        """Compatibilidad: los artefactos previos a la opcion `--normalizacion`
        eran z-score por liga y no llevaban sufijo.
        """
        modelo = _modelo(np.array([[0.0, 0.3], [0.3, 0.0]]), meta={})
        modelo.guardar(tmp_path)
        assert (tmp_path / "formulacion2_jugador.npz").exists()
        cargado = cargar_modelo(tmp_path, "2", "jugador", normalizacion="por_liga")
        assert cargado.S[0, 1] == pytest.approx(0.3)

    def test_pedir_global_no_cae_al_artefacto_antiguo(self, tmp_path: Path) -> None:
        """Un modelo antiguo NO lleva normalizacion global: servirlo en silencio
        daria recomendaciones de otro modelo sin que nadie se entere.
        """
        _modelo(np.zeros((2, 2)), meta={}).guardar(tmp_path)
        with pytest.raises(FileNotFoundError):
            cargar_modelo(tmp_path, "2", "jugador", normalizacion="global")

    def test_sin_modelo_el_error_dice_que_se_ha_probado(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="formulacion5_equipo"):
            cargar_modelo(tmp_path, "5", "equipo", normalizacion="por_liga")


class TestTopK:
    def test_devuelve_las_mas_similares_ordenadas(self) -> None:
        S = np.array([
            [0.0, 0.1, 0.9, 0.5],
            [0.1, 0.0, 0.2, 0.3],
            [0.9, 0.2, 0.0, 0.4],
            [0.5, 0.3, 0.4, 0.0],
        ])
        resultado = top_k(_modelo(S), 0, 3)
        assert [j for j, _ in resultado] == [2, 3, 1]
        assert [round(s, 4) for _, s in resultado] == [0.9, 0.5, 0.1]

    def test_nunca_se_recomienda_a_si_misma(self) -> None:
        S = np.array([[0.9, 0.5], [0.5, 0.9]])   # diagonal alta a proposito
        assert [j for j, _ in top_k(_modelo(S), 0, 2)] == [1]

    def test_respeta_el_limite_k(self) -> None:
        rng = np.random.default_rng(0)
        S = np.abs(rng.normal(size=(10, 10))) + 0.01
        assert len(top_k(_modelo(S), 0, 3)) == 3

    def test_descarta_los_candidatos_con_score_cero(self) -> None:
        """En la Formulacion 2, S es muy dispersa: un 0 significa "sin relacion
        aprendida", no "similar". Incluirlos rellenaria el top-k con entidades
        arbitrarias en orden de indice.
        """
        S = np.array([
            [0.0, 0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        assert [j for j, _ in top_k(_modelo(S), 0, 3)] == [1]

    def test_puede_devolver_menos_de_k(self) -> None:
        """Comportamiento honesto para una entidad poco conectada."""
        S = np.zeros((5, 5))
        S[0, 1] = S[1, 0] = 0.7
        assert len(top_k(_modelo(S), 0, 4)) == 1

    def test_una_entidad_aislada_no_devuelve_nada(self) -> None:
        assert top_k(_modelo(np.zeros((4, 4))), 0, 3) == []

    def test_admite_scores_negativos(self) -> None:
        """EASE puede dar negativos: -0.1 es informacion, no "sin relacion"."""
        S = np.array([[0.0, -0.1, -0.5], [-0.1, 0.0, 0.2], [-0.5, 0.2, 0.0]])
        assert [j for j, _ in top_k(_modelo(S), 0, 2)] == [1, 2]

    def test_ignora_los_no_finitos(self) -> None:
        S = np.array([[0.0, np.nan, 0.3], [np.nan, 0.0, 0.1], [0.3, 0.1, 0.0]])
        assert [j for j, _ in top_k(_modelo(S), 0, 2)] == [2]

    def test_no_modifica_la_matriz_del_modelo(self) -> None:
        """`top_k` pone -inf en la diagonal: debe hacerlo sobre una copia."""
        S = np.array([[0.5, 0.4], [0.4, 0.5]])
        modelo = _modelo(S)
        top_k(modelo, 0, 1)
        assert modelo.S[0, 0] == 0.5
