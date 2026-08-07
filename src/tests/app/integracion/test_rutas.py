"""Vistas HTML y API JSON, a través del cliente de pruebas de Flask."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.app.factoria import crear_app

pytestmark = pytest.mark.integracion


def _texto(respuesta) -> str:
    return respuesta.get_data(as_text=True)


class TestInicio:
    def test_sirve_el_formulario(self, cliente) -> None:
        r = cliente.get("/")
        assert r.status_code == 200
        assert "Buscar similares" in _texto(r)

    def test_ofrece_las_dos_entidades(self, cliente) -> None:
        html = _texto(cliente.get("/"))
        assert "Jugador" in html and "Equipo" in html

    def test_sin_modelos_explica_como_construirlos(self, tmp_path: Path) -> None:
        cliente = crear_app(tmp_path / "vacio", testing=True).test_client()
        r = cliente.get("/")
        assert r.status_code == 503
        assert "src.similitud.build" in _texto(r)


class TestSimilares:
    def test_top_k_por_nombre(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Messi&k=3")
        assert r.status_code == 200
        assert "Lionel Messi" in _texto(r)

    def test_respeta_el_k_pedido(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=2"))
        assert "Top 2 más similares" in html

    def test_acota_un_k_desmesurado(self, cliente) -> None:
        """Un k enorme no es un error del usuario, es una petición cara."""
        r = cliente.get("/similares?entidad=jugador&nombre=Messi&k=9999")
        assert r.status_code == 200

    def test_k_no_numerico_es_peticion_invalida(self, cliente) -> None:
        assert cliente.get("/similares?entidad=jugador&nombre=Messi&k=x").status_code == 400

    def test_por_id_es_la_via_univoca(self, cliente) -> None:
        """Dos entidades comparten nombre: el id decide cuál es la referencia."""
        r = cliente.get("/similares?entidad=jugador&id=60")
        assert r.status_code == 200
        assert "id 60" in _texto(r)

    def test_id_inexistente(self, cliente) -> None:
        assert cliente.get("/similares?entidad=jugador&id=999").status_code == 404

    def test_equipos(self, cliente) -> None:
        r = cliente.get("/similares?entidad=equipo&nombre=Barcelona")
        assert r.status_code == 200
        assert "Real Madrid" in _texto(r) or "Sevilla" in _texto(r)

    def test_un_nombre_ambiguo_lista_candidatos_en_vez_de_fallar(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Garcia")
        assert r.status_code == 200
        html = _texto(r)
        assert "Sergio Garcia" in html and "Marc Garcia" in html

    def test_un_nombre_inexistente_da_404_con_explicacion(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Cristiano")
        assert r.status_code == 404
        assert "Sin resultados" in _texto(r)

    def test_sin_nombre_ni_id(self, cliente) -> None:
        assert cliente.get("/similares?entidad=jugador").status_code == 400

    def test_entidad_desconocida(self, cliente) -> None:
        assert cliente.get("/similares?entidad=arbitro&nombre=x").status_code == 400

    def test_formulacion_desconocida(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Messi&formulacion=9")
        assert r.status_code == 400

    def test_elige_el_modelo_pedido(self, cliente) -> None:
        html = _texto(
            cliente.get("/similares?entidad=jugador&nombre=Messi&formulacion=2&normalizacion=global")
        )
        assert "Formulación 2" in html and "global" in html

    def test_una_combinacion_sin_artefacto_no_finge_que_no_hay_nada(
        self, dir_modelos: Path, tmp_path: Path
    ) -> None:
        """Con la F5 construida y la F2 no, pedir la F2 informa de ESO."""
        destino = tmp_path / "solo_f5"
        destino.mkdir()
        for sufijo in (".npz", ".json"):
            origen = dir_modelos / f"formulacion5_jugador_por_liga{sufijo}"
            (destino / origen.name).write_bytes(origen.read_bytes())
        cliente = crear_app(destino, testing=True).test_client()
        r = cliente.get("/similares?entidad=jugador&nombre=Messi&formulacion=2")
        assert r.status_code == 503
        assert "No hay modelo para" in _texto(r)

    def test_las_ligas_se_muestran_con_su_nombre(self, cliente) -> None:
        assert "La Liga 2015/2016" in _texto(
            cliente.get("/similares?entidad=jugador&nombre=Messi")
        )

    def test_el_candidato_enlaza_a_su_ficha(self, cliente) -> None:
        """Desde una recomendación se entra al detalle de esa entidad."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        assert "/jugador/" in html


class TestContextoEnLosListados:
    def test_el_candidato_lleva_su_equipo_y_su_liga(self, cliente) -> None:
        """El nombre solo no dice nada: el club es el contexto que falta."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=3"))
        assert "Barcelona (La Liga 2015/2016)" in html

    def test_cada_equipo_va_con_la_liga_que_le_toca(self, cliente) -> None:
        """El 20 juega en dos competiciones distintas, en orden cronologico."""
        html = _texto(cliente.get("/similares?entidad=jugador&id=20&k=1"))
        i = html.index("Barcelona (La Liga 2015/2016)")
        j = html.index("Real Madrid (Ligue 1 2015/2016)")
        assert i < j

    def test_dos_equipos_de_la_misma_liga_no_repiten_la_liga(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&id=60&k=1"))
        assert "Barcelona, Sevilla (La Liga 2015/2016)" in html

    def test_ya_no_se_resume_con_un_contador(self, cliente) -> None:
        """El «(+1)» ocultaba en que liga jugo cada equipo."""
        assert "(+1)" not in _texto(cliente.get("/similares?entidad=jugador&id=20&k=1"))

    def test_el_radar_de_equipo_va_ancho_y_centrado(self, cliente) -> None:
        """Sin campo de posiciones al lado, el radar se queda solo en la fila."""
        html = _texto(cliente.get("/similares?entidad=equipo&nombre=Barcelona&k=1"))
        assert "radar-amplio" in html

    def test_el_radar_del_jugador_comparte_sitio_con_el_campo(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        # El de la referencia si va solo; el del candidato, no.
        assert html.count("radar-amplio") == 1

    def test_sin_bd_el_equipo_se_omite_sin_romper(self, cliente_sin_bd) -> None:
        r = cliente_sin_bd.get("/similares?entidad=jugador&nombre=Messi&k=2")
        assert r.status_code == 200
        assert "Barcelona" not in _texto(r)

    def test_la_explicacion_incluye_el_heptagono(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        # Un poligono por serie: referencia + candidato.
        assert html.count("radar-area radar-referencia") == 2
        assert html.count("radar-area radar-candidato") == 1

    def test_la_explicacion_compara_las_posiciones(self, cliente) -> None:
        """Un circulo por posicion de la union, con los dos porcentajes dentro."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        assert "campo-juego" in html
        assert "marca-pct-a" in html and "marca-pct-b" in html

    def test_la_marca_dice_al_raton_de_quien_es_cada_cifra(self, cliente) -> None:
        """Dentro del circulo solo caben las cifras; el resto va en el tooltip."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        assert "Extremo derecho — Lionel Messi: 100 % ·" in html

    def test_la_marca_lleva_sus_datos_para_el_globo(self, cliente) -> None:
        """`campo.js` pinta el globo con esto, sin releer el SVG."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        assert 'data-posicion="Extremo derecho"' in html
        assert 'data-pct-a="100 %"' in html and 'data-pct-b=' in html
        # Los nombres son de la figura entera, no de cada marca.
        assert 'data-nombre-a="Lionel Messi"' in html and "data-nombre-b=" in html
        assert 'class="campo-globo"' in html

    def test_todos_los_circulos_del_campo_miden_igual(self, cliente) -> None:
        """El circulo dice DONDE se juega; el cuanto lo dice el numero."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        radios = set(re.findall(r'class="marca" cx="[^"]+" cy="[^"]+"\s+r="([^"]+)"', html))
        assert len(radios) == 1

    def test_la_posicion_no_es_una_coincidencia(self, cliente) -> None:
        """Coincidir en jugar de extremo no es coincidir en como se juega."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=3"))
        assert "Posición:" not in html

    def test_los_equipos_no_llevan_campo_de_posiciones(self, cliente) -> None:
        """Un equipo no tiene reparto de minutos por posicion."""
        html = _texto(cliente.get("/similares?entidad=equipo&nombre=Barcelona&k=1"))
        assert "campo-juego" not in html
        assert "radar-area" in html


class TestValoresReales:
    """Lo que se enseña es la métrica en sus unidades, no el z-score."""

    def test_la_ficha_muestra_el_valor_real(self, cliente) -> None:
        """El 10 promedia 60 pases por 90'; su z-score sintetico es otra cosa."""
        html = _texto(cliente.get("/jugador/10"))
        assert "60.0" in html

    def test_el_equipo_va_por_partido(self, cliente) -> None:
        html = _texto(cliente.get("/equipo/1"))
        assert "500.0" in html and "por partido" in html

    def test_el_jugador_va_por_90_minutos(self, cliente) -> None:
        assert "por 90 minutos" in _texto(cliente.get("/jugador/10"))

    def test_cada_metrica_dice_en_que_unidad_esta(self, cliente) -> None:
        """5,07 y 0,69 no son la misma clase de numero."""
        assert "por 90 min" in _texto(cliente.get("/jugador/10"))

    def test_el_equipo_no_dice_por_90(self, cliente) -> None:
        html = _texto(cliente.get("/equipo/1"))
        assert "por 90 min" not in html and "por partido" in html

    def test_la_columna_del_z_sigue_estando(self, cliente) -> None:
        """Es lo que justifica que esa metrica salga en la lista."""
        assert "z-secundario" in _texto(cliente.get("/jugador/10"))

    def test_las_coincidencias_llevan_valor_real(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=1"))
        assert "valor-real" in html

    def test_sin_bd_se_vuelve_al_z_score(self, cliente_sin_bd) -> None:
        """Los valores reales se reconstruyen de la BD; sin ella no los hay."""
        html = _texto(cliente_sin_bd.get("/jugador/10"))
        assert "z-secundario" not in html
        assert "Medidas en z-scores frente a la media de su liga y temporada" in html

    def test_sin_bd_la_ficha_sigue_completa(self, cliente_sin_bd) -> None:
        html = _texto(cliente_sin_bd.get("/jugador/10"))
        assert "Destaca en" in html and "radar-area" in html


class TestReferenciaDelZScore:
    """Contra QUE se mide el z que se enseña depende de la normalizacion.

    Con `global` la media es la de todo el dataset, mezclando ligas: decir «la
    media de su liga» ahi es falso, y ademas contradecia a la etiqueta del
    modelo que sale al lado ("z-score global").
    """

    def test_el_modelo_global_no_habla_de_ligas(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=1"
                                  "&formulacion=5&normalizacion=global"))
        assert "la media global (todas las ligas)" in html
        assert "media de su liga" not in html

    def test_el_modelo_por_liga_si(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=1"
                                  "&formulacion=5&normalizacion=por_liga"))
        assert "la media de su liga y temporada" in html
        assert "media global" not in html

    def test_la_ficha_usa_la_misma_referencia(self, cliente) -> None:
        html = _texto(cliente.get("/jugador/10?formulacion=5&normalizacion=global"))
        assert "la media global (todas las ligas)" in html
        assert "media de su liga" not in html

    def test_la_tabla_de_rasgos_tambien(self, cliente) -> None:
        """El macro se importa `with context`: sin eso el rotulo se quedaba fijo."""
        html = _texto(cliente.get("/jugador/10?formulacion=5&normalizacion=global"))
        assert (
            'title="Desviaciones típicas respecto a la media global '
            '(todas las ligas)"' in html
        )


class TestApiValoresReales:
    def test_el_rasgo_trae_z_y_valor_real(self, cliente) -> None:
        ficha = cliente.get("/api/ficha/jugador/10").get_json()
        rasgo = next(
            r for r in ficha["destacados"] + ficha["flojos"]
            if r["feature"] == "passes"
        )
        assert rasgo["crudo"] == pytest.approx(60.0)
        assert rasgo["texto"] == "60.0"
        assert rasgo["valor"] != rasgo["crudo"]  # el z es otra cosa

    def test_una_metrica_sin_dato_en_la_bd_es_null(self, cliente) -> None:
        """`interceptions` esta a NULL en la BD sintetica."""
        ficha = cliente.get("/api/ficha/jugador/10").get_json()
        rasgo = next(
            r for r in ficha["destacados"] + ficha["flojos"]
            if r["feature"] == "interceptions"
        )
        assert rasgo["crudo"] is None and rasgo["texto"] == "—"

    def test_las_coincidencias_traen_los_dos_valores(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=jugador&id=10&k=1").get_json()
        co = datos["candidatos"][0]["coincidencias"][0]
        assert {"referencia_cruda", "candidato_cruda",
                "texto_referencia", "texto_candidato"} <= set(co)

    def test_sin_bd_los_valores_reales_son_null(self, app_sin_bd) -> None:
        rasgo = app_sin_bd.test_client().get(
            "/api/ficha/jugador/10"
        ).get_json()["destacados"][0]
        assert rasgo["crudo"] is None
        assert rasgo["valor"] is not None  # el z-score siempre esta


class TestFicha:
    def test_la_ficha_de_un_jugador(self, cliente) -> None:
        r = cliente.get("/jugador/10")
        assert r.status_code == 200
        html = _texto(r)
        assert "Lionel Messi" in html and "Barcelona" in html

    def test_muestra_el_heptagono_y_los_dos_extremos(self, cliente) -> None:
        html = _texto(cliente.get("/jugador/10"))
        assert "radar-area" in html
        assert "Destaca en" in html and "Flaquea en" in html

    def test_muestra_el_campo_con_sus_posiciones(self, cliente) -> None:
        html = _texto(cliente.get("/jugador/10"))
        assert "campo-juego" in html and "Extremo derecho" in html

    def test_la_marca_del_campo_se_explica_al_pasar_el_raton(self, cliente) -> None:
        """Sin comparacion no hay leyenda de colores: el tooltip lo dice entero."""
        html = _texto(cliente.get("/jugador/10"))
        assert "<title>Extremo derecho — 100 % de sus minutos</title>" in html

    def test_la_ficha_no_declara_un_segundo_jugador(self, cliente) -> None:
        """Con una sola serie, el globo escribe «de sus minutos», no un nombre."""
        html = _texto(cliente.get("/jugador/10"))
        assert 'data-pct-a="100 %"' in html
        assert "data-pct-b=" not in html and "data-nombre-b=" not in html

    def test_detalla_la_trayectoria_cuando_hay_varias_etapas(self, cliente) -> None:
        html = _texto(cliente.get("/jugador/20"))
        assert "Barcelona" in html and "Real Madrid" in html
        assert "La Liga 2015/2016" in html and "Ligue 1 2015/2016" in html

    def test_el_radar_de_la_ficha_va_ancho(self, cliente) -> None:
        assert "radar-amplio" in _texto(cliente.get("/equipo/1"))

    def test_la_ficha_de_un_equipo(self, cliente) -> None:
        r = cliente.get("/equipo/1")
        assert r.status_code == 200
        assert "Barcelona" in _texto(r)

    def test_un_equipo_no_pinta_campo_de_posiciones(self, cliente) -> None:
        assert "campo-juego" not in _texto(cliente.get("/equipo/1"))

    def test_conserva_el_modelo_de_la_recomendacion(self, cliente) -> None:
        """La ficha se lee sobre el mismo artefacto que produjo el ranking."""
        html = _texto(cliente.get("/jugador/10?formulacion=2&normalizacion=global"))
        assert "Formulación 2" in html and "z-score global" in html

    def test_un_id_inexistente_da_404(self, cliente) -> None:
        assert cliente.get("/jugador/999").status_code == 404

    def test_enlaza_de_vuelta_a_los_similares(self, cliente) -> None:
        assert "/similares?" in _texto(cliente.get("/jugador/10"))

    def test_sin_bd_la_ficha_sigue_sirviendose(self, cliente_sin_bd) -> None:
        """Sin BD se pierden el equipo y el campo, no la recomendación."""
        r = cliente_sin_bd.get("/jugador/10")
        assert r.status_code == 200
        html = _texto(r)
        assert "campo-juego" not in html
        assert "radar-area" in html


class TestApiFicha:
    def test_estructura_de_la_respuesta(self, cliente) -> None:
        datos = cliente.get("/api/ficha/jugador/10").get_json()
        assert datos["entidad"]["nombre"] == "Lionel Messi"
        assert datos["entidad"]["equipo"] == "Barcelona"
        assert datos["destacados"] and datos["flojos"]

    def test_el_jugador_tiene_seis_fases_y_el_equipo_siete(self, cliente) -> None:
        """La 7.a de jugador (Perfil/Valor global) no es calculable."""
        assert len(cliente.get("/api/ficha/jugador/10").get_json()["perfil"]) == 6
        assert len(cliente.get("/api/ficha/equipo/1").get_json()["perfil"]) == 7

    def test_ninguna_fase_es_la_posicion(self, cliente) -> None:
        claves = {f["fase"] for f in cliente.get("/api/ficha/jugador/10").get_json()["perfil"]}
        assert "perfil" not in claves

    def test_el_perfil_trae_percentil_y_z_por_fase(self, cliente) -> None:
        fases = cliente.get("/api/ficha/jugador/10").get_json()["perfil"]
        assert {"fase", "etiqueta", "percentil", "z"} <= set(fases[0])

    def test_las_posiciones_van_en_fracciones(self, cliente) -> None:
        posiciones = cliente.get("/api/ficha/jugador/10").get_json()["posiciones"]
        assert posiciones[0]["etiqueta"] == "Extremo derecho"
        assert posiciones[0]["fraccion"] == 1.0

    def test_el_equipo_no_lleva_posiciones(self, cliente) -> None:
        assert "posiciones" not in cliente.get("/api/ficha/equipo/1").get_json()

    def test_entidad_desconocida_es_400(self, cliente) -> None:
        assert cliente.get("/api/ficha/arbitro/1").status_code == 400

    def test_id_inexistente_es_404(self, cliente) -> None:
        r = cliente.get("/api/ficha/jugador/999")
        assert r.status_code == 404
        assert r.get_json()["error"]


class TestApiSugerencias:
    def test_devuelve_las_coincidencias(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=garcia").get_json()
        assert {s["nombre"] for s in datos["sugerencias"]} == {"Sergio Garcia", "Marc Garcia"}

    def test_incluye_id_y_ligas_legibles(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=messi").get_json()
        sugerencia = datos["sugerencias"][0]
        assert sugerencia["id"] == 10
        assert sugerencia["ligas"] == ["11-27"]
        assert sugerencia["ligas_nombre"] == ["La Liga 2015/2016"]

    def test_el_equipo_desambigua_a_los_homonimos(self, cliente) -> None:
        """Con dos "Garcia" en pantalla, el club es lo que los distingue."""
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=garcia").get_json()
        equipos = {s["nombre"]: s["equipo"] for s in datos["sugerencias"]}
        assert equipos == {"Sergio Garcia": "Sevilla", "Marc Garcia": None}

    def test_la_sugerencia_trae_la_trayectoria_compuesta(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=suarez").get_json()
        assert datos["sugerencias"][0]["contexto"] == (
            "Barcelona (La Liga 2015/2016) · Real Madrid (Ligue 1 2015/2016)"
        )

    def test_la_trayectoria_va_estructurada_y_agrupada(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=repetido").get_json()
        sesenta = next(s for s in datos["sugerencias"] if s["id"] == 60)
        assert len(sesenta["trayectoria"]) == 1
        assert [e["nombre"] for e in sesenta["trayectoria"][0]["equipos"]] == [
            "Barcelona", "Sevilla"
        ]

    def test_los_equipos_no_llevan_campo_equipo(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=equipo&q=barcelona").get_json()
        assert "equipo" not in datos["sugerencias"][0]

    def test_consulta_vacia_no_devuelve_nada(self, cliente) -> None:
        assert cliente.get("/api/sugerencias?entidad=jugador&q=").get_json()["sugerencias"] == []

    def test_entidad_desconocida_da_json_de_error(self, cliente) -> None:
        r = cliente.get("/api/sugerencias?entidad=arbitro&q=x")
        assert r.status_code == 400
        assert "error" in r.get_json()


class TestApiSimilares:
    def test_estructura_de_la_respuesta(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=jugador&nombre=Messi&k=2").get_json()
        assert datos["referencia"]["nombre"] == "Lionel Messi"
        assert datos["modelo"] == {
            "entidad": "jugador", "formulacion": "5", "normalizacion": "por_liga",
        }
        assert len(datos["candidatos"]) == 2
        assert datos["candidatos"][0]["rango"] == 1
        assert datos["candidatos"][0]["coincidencias"]

    def test_los_scores_van_en_orden_decreciente(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=equipo&nombre=Barcelona").get_json()
        scores = [c["score"] for c in datos["candidatos"]]
        assert scores == sorted(scores, reverse=True)

    def test_la_ambiguedad_es_300_con_los_candidatos(self, cliente) -> None:
        r = cliente.get("/api/similares?entidad=jugador&nombre=Garcia")
        assert r.status_code == 300
        assert len(r.get_json()["candidatos"]) == 2

    def test_lo_inexistente_es_404(self, cliente) -> None:
        r = cliente.get("/api/similares?entidad=jugador&nombre=Cristiano")
        assert r.status_code == 404
        assert r.get_json()["candidatos"] == []

    def test_sin_referencia_es_400(self, cliente) -> None:
        assert cliente.get("/api/similares?entidad=jugador").status_code == 400

    def test_una_ruta_de_api_inexistente_responde_json(self, cliente) -> None:
        r = cliente.get("/api/no_existe")
        assert r.status_code == 404
        assert r.get_json()["error"]

    def test_los_acentos_no_se_escapan(self, cliente) -> None:
        """Los nombres de StatsBomb llevan caracteres fuera de latin-1."""
        r = cliente.get("/api/sugerencias?entidad=jugador&q=soyuncu")
        assert "Çağlar Söyüncü" in r.get_data(as_text=True)
        assert "\\u" not in r.get_data(as_text=True)


class TestPaginasDeError:
    def test_una_ruta_web_inexistente_devuelve_html(self, cliente) -> None:
        r = cliente.get("/no_existe")
        assert r.status_code == 404
        assert "no existe" in _texto(r)
