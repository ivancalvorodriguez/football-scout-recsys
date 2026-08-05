"""Score compuesto: agregacion de las metricas de un barrido en un unico numero.

El score es una HEURISTICA de lectura declarada como tal en `docs/evaluacion.md`,
no un criterio validado. Estas pruebas fijan lo que si tiene que ser exacto: la
orientacion de cada metrica, que las z se calculen dentro de la entidad (jugador y
equipo nunca se mezclan) y que una metrica ausente no penalice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluacion import puntuacion


def _fila(modelo: str, combinacion: str, metrica: str, valor: float,
          entidad: str = "equipo") -> dict:
    form, _ent, norm = modelo.lstrip("F").split("_", 2)
    return {"combinacion": combinacion, "modelo": modelo, "formulacion": form,
            "entidad": entidad, "normalizacion": norm,
            "fase": "1", "metrica": metrica, "valor": valor}


class TestOrientacion:
    def test_la_asimetria_es_menor_es_mejor(self) -> None:
        """Es la unica: mide cuanta señal direccional corrige el pipeline."""
        assert puntuacion.orientacion("asimetria") == "min"

    def test_el_resto_de_metricas_es_mayor_es_mejor(self) -> None:
        for metrica in ("top1", "mrr", "rbo_medio", "knn_accuracy", "coverage"):
            assert puntuacion.orientacion(metrica) == "max"

    def test_una_metrica_desconocida_asume_mayor_es_mejor(self) -> None:
        assert puntuacion.orientacion("inventada") == puntuacion.MEJOR_POR_DEFECTO


class TestZ:
    def test_centra_y_escala(self) -> None:
        z = puntuacion._z([1.0, 2.0, 3.0])
        assert sum(z) == pytest.approx(0.0)
        assert np.std(z) == pytest.approx(1.0)

    def test_una_metrica_constante_no_discrimina(self) -> None:
        """Meterla como NaN dejaria sin score a modelos perfectamente evaluados."""
        assert puntuacion._z([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_conserva_el_orden(self) -> None:
        z = puntuacion._z([3.0, 1.0, 2.0])
        assert z[1] < z[2] < z[0]


class TestPuntuar:
    def test_un_barrido_sin_metricas_puntuables_da_una_tabla_vacia(self) -> None:
        df = pd.DataFrame([_fila("F2_equipo_global", "v01", "n_pureza", 3.0)])
        assert puntuacion.puntuar(df).empty

    def test_ordena_mejor_primero_dentro_de_cada_entidad(self) -> None:
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.9),
        ])
        scores = puntuacion.puntuar(df)
        assert scores["combinacion"].tolist() == ["v02", "v01"]

    def test_la_asimetria_entra_con_el_signo_cambiado(self) -> None:
        """Sin orientar, una S mas asimetrica puntuaria mejor."""
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "asimetria", 0.01),
            _fila("F2_equipo_global", "v02", "asimetria", 0.90),
        ])
        scores = puntuacion.puntuar(df).set_index("combinacion")
        assert scores.loc["v01", puntuacion.NOMBRE] > scores.loc["v02", puntuacion.NOMBRE]

    def test_las_z_se_calculan_dentro_de_la_entidad(self) -> None:
        """Jugador y equipo tienen distintas features, distinto pool y distinto
        metodo en la F5: mezclarlos compararia cosas que no son comparables.
        """
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1, entidad="equipo"),
            _fila("F2_equipo_global", "v02", "top1", 0.2, entidad="equipo"),
            _fila("F2_jugador_global", "v01", "top1", 10.0, entidad="jugador"),
            _fila("F2_jugador_global", "v02", "top1", 20.0, entidad="jugador"),
        ])
        scores = puntuacion.puntuar(df)
        por_entidad = scores.groupby("entidad")[puntuacion.NOMBRE].max()
        # La escala absoluta desaparece: el mejor de cada entidad tiene la misma z.
        assert por_entidad["equipo"] == pytest.approx(por_entidad["jugador"])

    def test_una_metrica_que_no_existe_para_ese_modelo_no_penaliza(self) -> None:
        """La pureza posicional del equipo es NaN: renormalizar los pesos sobre lo
        disponible evita que un modelo de equipo puntue peor por no tener rol.
        """
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.5),
            _fila("F2_equipo_global", "v01", "pureza_top1", float("nan")),
            _fila("F5_equipo_global", "v01", "top1", 0.5),
        ])
        scores = puntuacion.puntuar(df)
        assert scores[puntuacion.NOMBRE].tolist() == pytest.approx([0.0, 0.0])
        assert scores["n_metricas"].tolist() == [1, 1]

    def test_publica_las_z_que_componen_el_score(self) -> None:
        """El numero tiene que ser auditable, no un veredicto opaco."""
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.9),
        ])
        scores = puntuacion.puntuar(df)
        assert "z_top1" in scores.columns
        assert all(f"z_{m}" in scores.columns for m in puntuacion.PESOS)

    def test_pondera_segun_los_pesos_declarados(self) -> None:
        """La validez pesa mas que la higiene: una S casi aleatoria da cobertura y
        diversidad altisimas, y no puede ganar por eso.
        """
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.9),
            _fila("F2_equipo_global", "v01", "coverage", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "coverage", 0.9),
        ])
        scores = puntuacion.puntuar(df).set_index("combinacion")
        assert scores.loc["v01", puntuacion.NOMBRE] > scores.loc["v02", puntuacion.NOMBRE]

    def test_unos_pesos_a_medida_cambian_el_ganador(self) -> None:
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.9),
            _fila("F2_equipo_global", "v01", "coverage", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "coverage", 0.9),
        ])
        scores = puntuacion.puntuar(df, {"top1": 0.1, "coverage": 1.0})
        assert scores["combinacion"].tolist()[0] == "v02"

    def test_descarta_las_filas_sin_valor(self) -> None:
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", float("nan")),
            _fila("F2_equipo_global", "v02", "top1", 0.5),
        ])
        scores = puntuacion.puntuar(df)
        assert scores["combinacion"].tolist() == ["v02"]


class TestComoMetrica:
    def test_el_score_viaja_como_una_metrica_mas(self) -> None:
        """Asi entra en la rejilla, las figuras y los CSV sin duplicar codigo."""
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.9),
        ])
        filas = puntuacion.como_metrica(puntuacion.puntuar(df), df)
        assert set(filas["metrica"]) == {puntuacion.NOMBRE}
        assert set(filas.columns) >= set(df.columns)

    def test_recupera_los_tres_ejes_del_modelo(self) -> None:
        df = pd.DataFrame([_fila("F5_jugador_por_liga", "v01", "top1", 0.5,
                                 entidad="jugador")])
        fila = puntuacion.como_metrica(puntuacion.puntuar(df), df).iloc[0]
        assert (fila["formulacion"], fila["entidad"], fila["normalizacion"]) == (
            "5", "jugador", "por_liga")

    def test_un_modelo_que_no_esta_en_la_tabla_se_ignora(self) -> None:
        df = pd.DataFrame([_fila("F2_equipo_global", "v01", "top1", 0.5)])
        scores = pd.DataFrame([{"entidad": "equipo", "modelo": "F9_fantasma",
                                "combinacion": "v01", puntuacion.NOMBRE: 1.0}])
        assert puntuacion.como_metrica(scores, df).empty


class TestRanking:
    def test_lista_las_mejores_por_entidad(self) -> None:
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.9),
        ])
        lineas = puntuacion.ranking(puntuacion.puntuar(df), n=1)
        assert lineas[0] == "  equipo:"
        assert "v02" in lineas[1]

    def test_el_score_se_imprime_con_signo(self) -> None:
        """Es una z: sin el signo no se distingue "por encima" de "por debajo"."""
        df = pd.DataFrame([
            _fila("F2_equipo_global", "v01", "top1", 0.1),
            _fila("F2_equipo_global", "v02", "top1", 0.9),
        ])
        lineas = puntuacion.ranking(puntuacion.puntuar(df))
        assert any("+" in ln for ln in lineas[1:])
        assert any("-" in ln for ln in lineas[1:])
