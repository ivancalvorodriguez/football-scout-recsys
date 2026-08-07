"""Dominio de la app: búsqueda por nombre, top-k y su explicación."""

from __future__ import annotations

import numpy as np
import pytest

from src.app import fases, servicio
from src.similitud.modelo import ModeloSimilitud, top_k


class TestEtiquetaFeature:
    def test_una_metrica_del_catalogo_se_traduce(self) -> None:
        assert servicio.etiqueta_feature("pass_completion_pct") == "% de acierto en el pase"

    def test_una_metrica_desconocida_cae_a_algo_legible(self) -> None:
        """Añadir una feature al pipeline no puede tumbar la interfaz."""
        assert servicio.etiqueta_feature("metrica_nueva_x") == "Metrica nueva x"

    def test_las_columnas_de_posicion_se_marcan_como_tales(self) -> None:
        """`pos_left_center_back` en una tabla de scouting no se entiende."""
        assert servicio.etiqueta_feature("pos_left_center_back") == "Posición: Central izquierdo"


class TestEntidad:
    def test_devuelve_id_nombre_y_ligas(self, modelo_jugador: ModeloSimilitud) -> None:
        e = servicio.entidad(modelo_jugador, 0)
        assert (e.indice, e.id, e.nombre) == (0, 10, "Lionel Messi")
        assert e.ligas == ("11-27",)

    def test_indice_fuera_de_rango(self, modelo_jugador: ModeloSimilitud) -> None:
        with pytest.raises(servicio.EntidadDesconocida):
            servicio.entidad(modelo_jugador, 99)

    def test_sin_ligas_en_meta_no_rompe(self, modelo_jugador: ModeloSimilitud) -> None:
        modelo_jugador.meta.pop("ligas_por_entidad")
        assert servicio.entidad(modelo_jugador, 0).ligas == ()

    def test_indice_por_id(self, modelo_jugador: ModeloSimilitud) -> None:
        assert servicio.indice_por_id(modelo_jugador, 30) == 2

    def test_indice_por_id_inexistente(self, modelo_jugador: ModeloSimilitud) -> None:
        with pytest.raises(servicio.EntidadDesconocida, match="999"):
            servicio.indice_por_id(modelo_jugador, 999)


class TestBuscar:
    def test_coincidencia_parcial_sin_acentos_ni_mayusculas(
        self, modelo_jugador: ModeloSimilitud
    ) -> None:
        encontrados = servicio.buscar(modelo_jugador, "MESSI")
        assert [e.nombre for e in encontrados] == ["Lionel Messi"]

    def test_devuelve_todas_las_coincidencias_sin_fallar(
        self, modelo_jugador: ModeloSimilitud
    ) -> None:
        """Al contrario que `consulta.resolver`, aquí la ambigüedad es normal:
        el usuario está escribiendo y elegirá después."""
        nombres = [e.nombre for e in servicio.buscar(modelo_jugador, "Garcia")]
        assert sorted(nombres) == ["Marc Garcia", "Sergio Garcia"]

    def test_la_coincidencia_exacta_va_primero(self) -> None:
        modelo = _modelo(["Messi Cuccittini", "Otro Messi", "Messi"])
        assert servicio.buscar(modelo, "Messi")[0].nombre == "Messi"

    def test_el_prefijo_gana_a_la_coincidencia_interior(self) -> None:
        modelo = _modelo(["Lionel Messi", "Messi Cuccittini"])
        assert servicio.buscar(modelo, "Messi")[0].nombre == "Messi Cuccittini"

    def test_respeta_el_limite(self, modelo_jugador: ModeloSimilitud) -> None:
        assert len(servicio.buscar(modelo_jugador, "a", limite=2)) == 2

    def test_texto_vacio_no_devuelve_nada(self, modelo_jugador: ModeloSimilitud) -> None:
        assert servicio.buscar(modelo_jugador, "   ") == []

    def test_los_homonimos_se_distinguen_por_id(
        self, modelo_jugador: ModeloSimilitud
    ) -> None:
        encontrados = servicio.buscar(modelo_jugador, "Nombre Repetido")
        assert sorted(e.id for e in encontrados) == [50, 60]


class TestExplicacion:
    def test_coincidencias_priorizan_lo_notable_y_parecido(self) -> None:
        modelo = _modelo(["A", "B"], feat_display=np.array([
            [3.0, 0.1, 3.0],   # A: destaca en f0 y f2
            [3.0, 0.1, -3.0],  # B: acompaña en f0, opuesto en f2
        ]))
        co = servicio.coincidencias(modelo, 0, 1, n=1)
        assert [c.feature for c in co] == ["f0"]
        assert (co[0].referencia, co[0].candidato) == (3.0, 3.0)

    def test_se_puede_pedir_ninguna(self, modelo_jugador: ModeloSimilitud) -> None:
        assert servicio.coincidencias(modelo_jugador, 0, 1, n=0) == ()

    def test_rasgos_son_las_features_mas_alejadas_de_la_media(self) -> None:
        modelo = _modelo(["A", "B"], feat_display=np.array([
            [0.2, -2.5, 1.0],
            [0.0, 0.0, 0.0],
        ]))
        rasgos = servicio.rasgos(modelo, 0, n=2)
        assert [r.feature for r in rasgos] == ["f1", "f2"]
        assert rasgos[0].valor == -2.5


class TestRecomendar:
    def test_el_ranking_es_el_del_modelo(self, modelo_jugador: ModeloSimilitud) -> None:
        """La app no reordena: solo viste la salida de `modelo.top_k`."""
        rec = servicio.recomendar(modelo_jugador, 0, k=3)
        esperado = [j for j, _ in top_k(modelo_jugador, 0, 3)]
        assert [c.entidad.indice for c in rec.candidatos] == esperado
        assert [c.rango for c in rec.candidatos] == [1, 2, 3]

    def test_nunca_se_recomienda_a_si_misma(self, modelo_jugador: ModeloSimilitud) -> None:
        rec = servicio.recomendar(modelo_jugador, 1, k=10)
        assert all(c.entidad.indice != 1 for c in rec.candidatos)

    def test_incompleto_cuando_el_modelo_da_menos_de_k(self) -> None:
        """S dispersa (típica de la F2): mejor menos candidatos que rellenar."""
        modelo = _modelo(["A", "B", "C"], S=np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))
        rec = servicio.recomendar(modelo, 0, k=2)
        assert len(rec.candidatos) == 1
        assert rec.incompleto

    def test_completo_cuando_hay_suficientes(self, modelo_jugador: ModeloSimilitud) -> None:
        assert not servicio.recomendar(modelo_jugador, 0, k=2).incompleto

    def test_la_referencia_va_en_el_resultado(self, modelo_jugador: ModeloSimilitud) -> None:
        rec = servicio.recomendar(modelo_jugador, 2, k=1)
        assert rec.referencia.nombre == "Sergio Garcia"
        assert rec.rasgos


class TestPosicionFueraDeLaExplicacion:
    """La posición dice QUÉ es una entidad, no cómo juega.

    Se busca que jueguen parecido, así que coincidir en jugar de extremo no es
    una coincidencia de juego, ni destacar en la propia posición es destacar.
    """

    def _modelo(self) -> ModeloSimilitud:
        return _modelo(
            ["A", "B"],
            feat_names=["pos_left_wing", "passes"],
            # La posición domina en |z| a propósito: sin filtro se colaría.
            feat_display=np.array([[9.0, 1.0], [9.0, 1.0]]),
        )

    def test_no_aparece_entre_las_coincidencias(self) -> None:
        co = servicio.coincidencias(self._modelo(), 0, 1, n=2)
        assert [c.feature for c in co] == ["passes"]

    def test_no_aparece_entre_los_rasgos(self) -> None:
        assert [r.feature for r in servicio.rasgos(self._modelo(), 0, n=2)] == ["passes"]

    def test_no_aparece_entre_los_destacados(self) -> None:
        mejores, peores = servicio.destacados_y_flojos(self._modelo(), 0, n=1)
        assert not any(
            r.feature.startswith("pos_") for r in mejores + peores
        )

    def test_un_modelo_solo_de_posiciones_no_explica_nada(self) -> None:
        modelo = _modelo(["A", "B"], feat_names=["pos_left_wing"],
                         feat_display=np.array([[1.0], [1.0]]))
        assert servicio.coincidencias(modelo, 0, 1, n=1) == ()
        assert servicio.rasgos(modelo, 0, n=1) == ()


class TestUnidades:
    """Cada métrica dice en qué está expresada: 5,07 y 0,69 no son lo mismo."""

    def _rasgo(self, feature: str, entidad: str) -> servicio.Rasgo:
        modelo = _modelo(["A"], entidad=entidad, feat_names=[feature],
                         feat_display=np.array([[2.0]]))
        return servicio.rasgos(modelo, 0, n=1, crudos=np.array([[1.0]]))[0]

    def test_un_conteo_de_jugador_va_por_90(self) -> None:
        assert self._rasgo("progressive_passes", "jugador").unidad == "por 90 min"

    def test_el_mismo_conteo_en_equipo_va_por_partido(self) -> None:
        """El equipo juega el partido completo; no hay minutos de equipo."""
        assert self._rasgo("progressive_passes", "equipo").unidad == "por partido"

    def test_un_porcentaje_no_repite_la_unidad(self) -> None:
        """El propio valor ya lleva su «%»."""
        assert self._rasgo("pass_completion_pct", "jugador").unidad == ""

    def test_la_calidad_de_tiro_es_por_tiro_no_por_90(self) -> None:
        assert self._rasgo("npxg_per_shot", "jugador").unidad == "por tiro"

    def test_las_medias_por_secuencia_lo_dicen(self) -> None:
        assert self._rasgo("passes_per_sequence", "equipo").unidad == "media por secuencia"

    def test_ppda_es_una_media_del_partido(self) -> None:
        assert self._rasgo("ppda", "equipo").unidad == "media por partido"

    def test_la_coincidencia_tambien_la_lleva(self) -> None:
        modelo = _modelo(["A", "B"], feat_names=["passes"],
                         feat_display=np.array([[2.0], [1.0]]))
        co = servicio.coincidencias(modelo, 0, 1, n=1)
        assert co[0].unidad == "por 90 min"


class TestValoresReales:
    """El z-score ELIGE la métrica; el valor real es lo que se MUESTRA."""

    def _modelo_dos(self) -> ModeloSimilitud:
        return _modelo(["A", "B"], feat_names=["passes", "xa"],
                       feat_display=np.array([[3.0, -1.0], [2.0, 0.0]]))

    def test_el_rasgo_lleva_el_valor_real(self) -> None:
        crudos = np.array([[60.0, 0.25], [30.0, 0.10]])
        rasgos = servicio.rasgos(self._modelo_dos(), 0, n=1, crudos=crudos)
        assert rasgos[0].valor == 3.0        # z: por esto sale en la lista
        assert rasgos[0].crudo == 60.0       # real: esto es lo que se lee
        assert rasgos[0].texto == "60.0"

    def test_sin_crudos_el_rasgo_no_inventa_un_valor(self) -> None:
        """Sin BD la interfaz vuelve al z-score, no a un 0 falso."""
        rasgos = servicio.rasgos(self._modelo_dos(), 0, n=1)
        assert rasgos[0].crudo is None and rasgos[0].texto == "—"

    def test_un_nan_en_la_tabla_es_sin_dato(self) -> None:
        crudos = np.array([[np.nan, 0.25], [30.0, 0.10]])
        rasgos = servicio.rasgos(self._modelo_dos(), 0, n=1, crudos=crudos)
        assert rasgos[0].crudo is None and rasgos[0].texto == "—"

    def test_el_orden_no_depende_del_valor_real(self) -> None:
        """Ordenar por valor real pondría siempre delante lo de más volumen."""
        crudos = np.array([[1.0, 999.0], [1.0, 999.0]])
        rasgos = servicio.rasgos(self._modelo_dos(), 0, n=1, crudos=crudos)
        assert rasgos[0].feature == "passes"  # el de mayor |z|, no el de mayor valor

    def test_la_coincidencia_lleva_los_dos_valores_reales(self) -> None:
        crudos = np.array([[60.0, 0.25], [55.0, 0.10]])
        co = servicio.coincidencias(self._modelo_dos(), 0, 1, n=1, crudos=crudos)
        assert (co[0].referencia_cruda, co[0].candidato_cruda) == (60.0, 55.0)
        assert (co[0].texto_referencia, co[0].texto_candidato) == ("60.0", "55.0")

    def test_las_unidades_se_respetan_por_metrica(self) -> None:
        modelo = _modelo(["A"], feat_names=["pass_completion_pct"],
                         feat_display=np.array([[2.0]]))
        rasgos = servicio.rasgos(modelo, 0, n=1, crudos=np.array([[0.83]]))
        assert rasgos[0].texto == "83 %"

    def test_destacados_y_flojos_tambien_los_llevan(self) -> None:
        modelo = _modelo(["A"], feat_names=["passes", "xa"],
                         feat_display=np.array([[2.0, -2.0]]))
        mejores, peores = servicio.destacados_y_flojos(
            modelo, 0, n=1, crudos=np.array([[60.0, 0.05]])
        )
        assert mejores[0].texto == "60.0" and peores[0].texto == "0.05"

    def test_recomendar_los_propaga_a_los_candidatos(self) -> None:
        crudos = np.array([[60.0, 0.25], [55.0, 0.10]])
        rec = servicio.recomendar(self._modelo_dos(), 0, k=1, crudos=crudos)
        assert rec.rasgos[0].crudo == 60.0
        assert rec.candidatos[0].coincidencias[0].candidato_cruda == 55.0

    def test_ficha_los_propaga(self) -> None:
        modelo = _modelo(["A"], feat_names=["passes", "xa"],
                         feat_display=np.array([[2.0, -2.0]]))
        ficha = servicio.ficha(modelo, 0, n_rasgos=1, crudos=np.array([[60.0, 0.05]]))
        assert ficha.destacados[0].crudo == 60.0


class TestDestacadosYFlojos:
    def test_separa_los_dos_extremos(self) -> None:
        modelo = _modelo(["A"], feat_names=["passes", "xa", "npxg", "duels_won"],
                         feat_display=np.array([[2.0, 1.0, -1.0, -2.0]]))
        mejores, peores = servicio.destacados_y_flojos(modelo, 0, n=2)
        assert [r.feature for r in mejores] == ["passes", "xa"]
        assert [r.feature for r in peores] == ["duels_won", "npxg"]

    def test_una_metrica_invertida_alta_es_un_punto_debil(self) -> None:
        """Que te regateen mucho no es destacar, por muy alto que sea el z."""
        modelo = _modelo(["A"], feat_names=["dribbled_past", "passes"],
                         feat_display=np.array([[3.0, 0.0]]))
        mejores, peores = servicio.destacados_y_flojos(modelo, 0, n=1)
        assert [r.feature for r in peores] == ["dribbled_past"]
        assert peores[0].invertida
        assert [r.feature for r in mejores] == ["passes"]

    def test_el_valor_mostrado_es_el_z_sin_tocar(self) -> None:
        """Se invierte para ordenar, no para mentir sobre el dato."""
        modelo = _modelo(["A"], feat_names=["dribbled_past", "passes"],
                         feat_display=np.array([[3.0, 0.0]]))
        _, peores = servicio.destacados_y_flojos(modelo, 0, n=1)
        assert peores[0].valor == 3.0

    def test_etiqueta_la_fase_de_cada_metrica(self) -> None:
        modelo = _modelo(["A"], feat_names=["npxg", "passes"],
                         feat_display=np.array([[2.0, 0.0]]))
        mejores, _ = servicio.destacados_y_flojos(modelo, 0, n=1)
        assert mejores[0].fase == "Finalización"

    def test_la_posicion_no_es_un_punto_fuerte(self) -> None:
        """Un extremo no «destaca» por jugar de extremo."""
        modelo = _modelo(["A"], feat_names=["pos_left_wing", "passes", "xa"],
                         feat_display=np.array([[9.0, 1.0, -1.0]]))
        mejores, peores = servicio.destacados_y_flojos(modelo, 0, n=1)
        assert "pos_left_wing" not in [r.feature for r in mejores + peores]

    def test_con_pocas_metricas_las_listas_no_se_solapan(self) -> None:
        """Con 3 métricas, pedir 4 y 4 repetiría la misma en ambas listas."""
        modelo = _modelo(["A"], feat_names=["passes", "xa", "npxg"],
                         feat_display=np.array([[2.0, 1.0, 0.0]]))
        mejores, peores = servicio.destacados_y_flojos(modelo, 0, n=4)
        assert not {r.feature for r in mejores} & {r.feature for r in peores}

    def test_un_modelo_sin_metricas_no_rompe(self) -> None:
        modelo = _modelo(["A"], feat_names=["pos_left_wing"],
                         feat_display=np.array([[1.0]]))
        assert servicio.destacados_y_flojos(modelo, 0, n=2) == ((), ())

    def test_se_puede_pedir_ninguna(self, modelo_jugador: ModeloSimilitud) -> None:
        assert servicio.destacados_y_flojos(modelo_jugador, 0, n=0) == ((), ())


class TestPerfilYFicha:
    def test_sin_matriz_no_hay_radar(self) -> None:
        """La figura es opcional: sin ella la interfaz omite el bloque."""
        assert servicio.perfil(None, 0) == ()

    def test_el_perfil_trae_una_fase_por_vertice(self, fases_jugador) -> None:
        assert len(servicio.perfil(fases_jugador, 0)) == len(fases.FASES_JUGADOR)

    def test_la_ficha_junta_perfil_y_extremos(
        self, modelo_jugador: ModeloSimilitud, fases_jugador
    ) -> None:
        ficha = servicio.ficha(modelo_jugador, 0, matriz=fases_jugador, n_rasgos=2)
        assert ficha.entidad.nombre == "Lionel Messi"
        assert len(ficha.perfil) == len(fases.FASES_JUGADOR)
        assert len(ficha.destacados) == len(ficha.flojos) == 2

    def test_recomendar_perfila_tambien_a_los_candidatos(
        self, modelo_jugador: ModeloSimilitud, fases_jugador
    ) -> None:
        """Sin el perfil del candidato no se puede superponer el radar."""
        rec = servicio.recomendar(modelo_jugador, 0, k=2, matriz=fases_jugador)
        n = len(fases.FASES_JUGADOR)
        assert all(len(c.perfil) == n for c in rec.candidatos)
        assert len(rec.perfil) == n

    def test_sin_matriz_recomendar_sigue_funcionando(
        self, modelo_jugador: ModeloSimilitud
    ) -> None:
        rec = servicio.recomendar(modelo_jugador, 0, k=2)
        assert len(rec.candidatos) == 2
        assert rec.perfil == ()


def _modelo(nombres: list[str], **cambios) -> ModeloSimilitud:
    """Modelo mínimo en memoria (S densa por defecto, sin empates)."""
    n = len(nombres)
    base = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    S = (base + base.T) / 2.0
    np.fill_diagonal(S, 0.0)
    argumentos = dict(
        formulacion="5",
        entidad="jugador",
        S=S,
        entity_ids=np.arange(n),
        entity_names=list(nombres),
        feat_names=["f0", "f1", "f2"],
        feat_display=np.zeros((n, 3)),
        meta={"normalizacion": "por_liga"},
    )
    argumentos.update(cambios)
    return ModeloSimilitud(**argumentos)
