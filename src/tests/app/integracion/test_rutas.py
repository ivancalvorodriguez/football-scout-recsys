"""Vistas HTML y API JSON, a través del cliente de pruebas de Flask."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app import glosario
from src.app.catalogo import ClaveModelo
from src.app.factoria import crear_app
from src.tests.app.conftest import (
    CONJUNTO_NOMBRE,
    CONJUNTO_SLUG,
    VARIANTE_NOMBRE,
    VARIANTE_SLUG,
    cliente_autenticado,
)

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

    def test_sin_modelos_explica_como_construirlos(
        self, tmp_path: Path, fichero_usuarios: Path
    ) -> None:
        cliente = cliente_autenticado(crear_app(
            tmp_path / "vacio", ruta_usuarios=fichero_usuarios, testing=True))
        r = cliente.get("/")
        assert r.status_code == 503
        html = _texto(r)
        # La pagina explica que faltan los modelos y que hay que construirlos
        # antes, pero sin nombrar modulos ni ficheros del codigo: la interfaz no
        # cita rutas del repositorio.
        assert "construir" in html.lower()
        assert "src." not in html and "docs/" not in html


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
        """Hay dos "Repetido" en el modelo Y en la BD: el usuario elige."""
        r = cliente.get("/similares?entidad=jugador&nombre=Repetido")
        assert r.status_code == 200
        html = _texto(r)
        assert html.count("Nombre Repetido") >= 2

    def test_quien_no_esta_en_la_bd_no_vuelve_ambigua_la_busqueda(
        self, cliente
    ) -> None:
        """«Marc Garcia» está en el modelo y no en la BD: no cuenta como opción.

        Antes «Garcia» era ambiguo por él y salía una lista de dos; ahora resuelve
        derecho al único García del que hay datos.

        (Marc sí puede aparecer más abajo, entre los CANDIDATOS de Sergio: eso es
        otra cosa y se comprueba en `test_como_candidato_de_otro_...`.)
        """
        r = cliente.get("/similares?entidad=jugador&nombre=Garcia")
        assert r.status_code == 200
        html = _texto(r)
        assert "Varias coincidencias" not in html    # no es la página de elegir
        assert "id 30" in html                       # la referencia es Sergio

    def test_pedir_a_quien_la_bd_no_tiene_explica_por_que(self, cliente) -> None:
        """404, pero diciendo que el modelo sí lo conoce y lo que falta son datos."""
        r = cliente.get("/similares?entidad=jugador&nombre=Marc Garcia")
        assert r.status_code == 404
        assert "no tiene a este jugador" in _texto(r)
        r = cliente.get("/similares?entidad=jugador&id=40")
        assert r.status_code == 404
        assert "no tiene a este jugador" in _texto(r)

    def test_como_candidato_de_otro_si_sale_pero_sin_enlace(self, cliente) -> None:
        """El modelo sabe cuánto se parece: esconderlo tiraría esa información.

        Lo que no tiene es ficha —ni similares propios—, así que sale rotulado y
        su nombre NO es un enlace: no hay opción que pulsar que acabe en un 404.
        """
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=50"))
        assert "Marc Garcia" in html
        assert "fuera de estos datos" in html
        assert "/jugador/40" not in html
        # El resto de candidatos sí se enlazan (el 30 está en la BD).
        assert "/jugador/30" in html

    def test_la_api_marca_al_candidato_que_no_esta_en_los_datos(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=jugador&id=10&k=50").get_json()
        marcas = {c["id"]: c["en_datos"] for c in datos["candidatos"]}
        assert marcas[40] is False
        assert marcas[30] is True

    def test_un_nombre_inexistente_da_404_con_explicacion(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Cristiano")
        assert r.status_code == 404
        assert "Sin resultados" in _texto(r)

    def test_sin_nombre_ni_id(self, cliente) -> None:
        assert cliente.get("/similares?entidad=jugador").status_code == 400

    def test_entidad_desconocida(self, cliente) -> None:
        assert cliente.get("/similares?entidad=arbitro&nombre=x").status_code == 400

    def test_modelo_desconocido(self, cliente) -> None:
        """404 y no 400: un slug que no le corresponde a esta cuenta no se
        confirma ni se desmiente (ver `_modelo_pedido`)."""
        r = cliente.get("/similares?entidad=jugador&nombre=Messi&modelo=fantasma")
        assert r.status_code == 404

    def test_sin_pedir_modelo_responde_el_base(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi"))
        assert config_app.NOMBRE_BASE in html

    def test_elige_el_modelo_pedido(self, cliente) -> None:
        """Y responde con SU universo: el reentrenado conoce al fichaje nuevo."""
        html = _texto(cliente.get(
            f"/similares?entidad=jugador&nombre=Fichaje&modelo={VARIANTE_SLUG}"))
        assert VARIANTE_NOMBRE in html and "Fichaje Reciente" in html

    def test_ese_nombre_no_existe_en_el_modelo_base(self, cliente) -> None:
        r = cliente.get("/similares?entidad=jugador&nombre=Fichaje")
        assert r.status_code == 404

    def test_un_modelo_que_no_cubre_la_entidad_no_finge_que_no_hay_nada(
        self, dir_modelos: Path, tmp_path: Path, fichero_usuarios: Path
    ) -> None:
        """Un reentrenamiento a medias (solo jugador) pedido para equipo informa de ESO."""
        destino = tmp_path / "a_medias"
        raiz = destino / config_app.SUBDIR_VARIANTES / "solo-jugador"
        raiz.mkdir(parents=True)
        for sufijo in (".npz", ".json"):
            origen = dir_modelos / f"{ClaveModelo('jugador').stem}{sufijo}"
            (raiz / origen.name).write_bytes(origen.read_bytes())
            base = dir_modelos / f"{ClaveModelo('equipo').stem}{sufijo}"
            (destino / base.name).write_bytes(base.read_bytes())
        cliente = cliente_autenticado(crear_app(
            destino, ruta_usuarios=fichero_usuarios, testing=True))
        r = cliente.get("/similares?entidad=equipo&nombre=Barcelona&modelo=solo-jugador")
        assert r.status_code == 503
        assert "no tiene artefacto de equipo" in _texto(r)

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


class TestFaseEnLasCoincidencias:
    """Cada métrica de «coinciden en» dice a qué fase de juego pertenece."""

    def test_la_tabla_tiene_columna_de_fase(self, cliente) -> None:
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=1"))
        assert "<th>Fase</th>" in html and "celda-fase" in html

    def test_escribe_la_fase_de_cada_metrica(self, cliente) -> None:
        """Las features del modelo de prueba son una por fase de jugador."""
        html = _texto(cliente.get("/similares?entidad=jugador&id=10&k=1"))
        for fase in ("Progresión", "Creación", "Juego aéreo", "Defensa"):
            assert fase in html

    def test_tambien_sin_base_de_datos(self, cliente_sin_bd) -> None:
        """La fase sale de la taxonomía, no de la BD: no depende de ella."""
        html = _texto(cliente_sin_bd.get("/similares?entidad=jugador&id=10&k=1"))
        assert "celda-fase" in html

    def test_la_api_la_expone(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=jugador&id=10&k=1").get_json()
        coincidencias = datos["candidatos"][0]["coincidencias"]
        assert all("fase" in co for co in coincidencias)
        assert any(co["fase"] for co in coincidencias)

    def test_sin_bd_se_vuelve_al_z_score(self, cliente_sin_bd) -> None:
        """Los valores reales se reconstruyen de la BD; sin ella no los hay."""
        html = _texto(cliente_sin_bd.get("/jugador/10"))
        assert "z-secundario" not in html
        assert ("Medidas en z-scores frente a "
                + config_app.REFERENCIA_Z["por_liga"]) in html

    def test_sin_bd_la_ficha_sigue_completa(self, cliente_sin_bd) -> None:
        html = _texto(cliente_sin_bd.get("/jugador/10"))
        assert "Destaca en" in html and "radar-area" in html


class TestConjuntoDeDatosDelModelo:
    """Cada modelo se sirve con los adornos de la BD sobre la que se entrenó.

    El fixture monta el caso real: el modelo reentrenado conoce a un jugador y a
    un equipo que solo existen en el conjunto de datos ampliado. Servirlo con la
    BD del conjunto base los dejaria sin equipo, sin posiciones, sin valores
    reales y con la liga en crudo — justo a las entidades por las que se amplio.
    """

    def test_el_equipo_del_fichaje_sale_de_su_conjunto(self, cliente) -> None:
        html = _texto(cliente.get(f"/jugador/80?modelo={VARIANTE_SLUG}"))
        assert "Fichaje Reciente" in html
        assert "Recien Ascendido" in html          # su equipo, solo en la ampliada

    def test_y_tambien_su_liga_y_sus_posiciones(self, cliente) -> None:
        html = _texto(cliente.get(f"/jugador/80?modelo={VARIANTE_SLUG}"))
        assert "Liga Recien Llegada" in html       # traducida, no la clave "9-27"
        assert "campo-juego" in html               # minutos por posicion

    def test_la_api_dice_con_que_datos_responde(self, cliente) -> None:
        datos = cliente.get(
            f"/api/ficha/jugador/80?modelo={VARIANTE_SLUG}").get_json()
        assert datos["modelo"]["datos"] == CONJUNTO_SLUG
        assert datos["entidad"]["equipo"] == "Recien Ascendido"

    def test_el_modelo_base_sigue_leyendo_el_conjunto_base(self, cliente) -> None:
        """Y no se contamina: el fichaje no existe en su universo."""
        assert cliente.get("/jugador/80").status_code == 404
        datos = cliente.get("/api/ficha/jugador/10").get_json()
        assert datos["modelo"]["datos"] == config_app.VARIANTE_BASE

    def test_la_interfaz_rotula_el_conjunto(self, cliente) -> None:
        html = _texto(cliente.get(f"/similares?entidad=jugador&id=10&k=1"
                                  f"&modelo={VARIANTE_SLUG}"))
        assert f"datos: {CONJUNTO_NOMBRE}" in html

    def test_un_conjunto_que_ya_no_esta_no_deja_de_servir(
        self, dir_modelos: Path, tmp_path: Path, fichero_usuarios: Path
    ) -> None:
        """Se pierde algun adorno; el modelo se sigue sirviendo."""
        cliente = cliente_autenticado(crear_app(
            dir_modelos, db_path=tmp_path / "otra" / "scouting.db",
            ruta_usuarios=fichero_usuarios, testing=True))
        r = cliente.get(f"/similares?entidad=jugador&id=10&k=1&modelo={VARIANTE_SLUG}")
        assert r.status_code == 200


class TestReferenciaDelZScore:
    """Contra QUE se mide el z que se enseña depende de la normalizacion.

    Decir «la media global» en la ficha de un jugador servido `por_liga` seria
    falso y contradiria ademas a la etiqueta del modelo que sale al lado. El
    rotulo no esta escrito en la plantilla: sale de `config.REFERENCIA_Z` segun la
    normalizacion del modelo, y esta clase es lo que comprueba que de verdad la
    sigue.

    Las expectativas se DERIVAN de `config.MODELO_BASE` en vez de fijarse a mano:
    que normalizacion sirve cada entidad ha cambiado ya dos veces al promover
    modelos nuevos (las dos `global`, luego jugador `por_liga` + equipo `global`,
    y desde el 25-8-2026 las dos `por_liga`), y lo que se quiere garantizar es la
    correspondencia, no una pareja concreta.
    """

    POR_LIGA = config_app.REFERENCIA_Z["por_liga"]
    GLOBAL = config_app.REFERENCIA_Z["global"]

    @staticmethod
    def _esperado(entidad: str) -> tuple[str, str]:
        """(rotulo que debe salir, rotulo que NO debe salir) para esa entidad."""
        _, normalizacion = config_app.MODELO_BASE[entidad]
        otra = "global" if normalizacion == "por_liga" else "por_liga"
        return config_app.REFERENCIA_Z[normalizacion], config_app.REFERENCIA_Z[otra]

    def test_el_rotulo_depende_de_la_normalizacion(self) -> None:
        """Las dos redacciones existen; la app sirve la que le toque al modelo."""
        assert "liga" in self.POR_LIGA
        assert "global" in self.GLOBAL
        assert self.POR_LIGA != self.GLOBAL

    @pytest.mark.parametrize("entidad, id_", [("jugador", 10), ("equipo", 1)])
    def test_cada_entidad_usa_la_de_su_modelo(self, cliente, entidad, id_) -> None:
        """Mismo codigo, dos modelos: cada uno con el rotulo de SU normalizacion."""
        toca, no_toca = self._esperado(entidad)
        html = _texto(cliente.get(f"/similares?entidad={entidad}&id={id_}&k=1"))
        assert toca in html
        assert no_toca not in html

    def test_la_ficha_usa_la_misma_referencia(self, cliente) -> None:
        toca, no_toca = self._esperado("jugador")
        html = _texto(cliente.get("/jugador/10"))
        assert toca in html
        assert no_toca not in html

    def test_la_tabla_de_rasgos_tambien(self, cliente) -> None:
        """El macro se importa `with context`: sin eso el rotulo se quedaba fijo."""
        html = _texto(cliente.get("/jugador/10"))
        assert f'title="Desviaciones típicas respecto a {self.POR_LIGA}"' in html


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
        rasgo = cliente_autenticado(app_sin_bd).get(
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

    def test_el_campo_de_la_ficha_tambien(self, cliente) -> None:
        """Va solo en su bloque, igual que el radar: mismo ancho y centrado."""
        html = _texto(cliente.get("/jugador/10"))
        assert "campo-amplio" in html and "radar-amplio" in html

    def test_el_campo_de_la_comparacion_no(self, cliente) -> None:
        """Ahi comparte fila con el radar del candidato, no se centra."""
        html = _texto(cliente.get("/similares?entidad=jugador&nombre=Messi&k=1"))
        assert "campo-amplio" not in html

    def test_la_ficha_de_un_equipo(self, cliente) -> None:
        r = cliente.get("/equipo/1")
        assert r.status_code == 200
        assert "Barcelona" in _texto(r)

    def test_un_equipo_no_pinta_campo_de_posiciones(self, cliente) -> None:
        assert "campo-juego" not in _texto(cliente.get("/equipo/1"))

    def test_conserva_el_modelo_de_la_recomendacion(self, cliente) -> None:
        """La ficha se lee sobre el mismo artefacto que produjo el ranking."""
        html = _texto(cliente.get(f"/jugador/10?modelo={VARIANTE_SLUG}"))
        assert VARIANTE_NOMBRE in html
        # Y el enlace de vuelta a los similares no pierde el modelo elegido.
        assert f"modelo={VARIANTE_SLUG}" in html

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
        """«Marc Garcia» está en el modelo y no en la BD: no se ofrece."""
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=garcia").get_json()
        assert {s["nombre"] for s in datos["sugerencias"]} == {"Sergio Garcia"}

    def test_incluye_id_y_ligas_legibles(self, cliente) -> None:
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=messi").get_json()
        sugerencia = datos["sugerencias"][0]
        assert sugerencia["id"] == 10
        assert sugerencia["ligas"] == ["11-27"]
        assert sugerencia["ligas_nombre"] == ["La Liga 2015/2016"]

    def test_el_equipo_desambigua_a_los_homonimos(self, cliente) -> None:
        """Con dos "Nombre Repetido" en pantalla, el club es lo que los distingue."""
        datos = cliente.get("/api/sugerencias?entidad=jugador&q=repetido").get_json()
        equipos = {s["id"]: s["equipo"] for s in datos["sugerencias"]}
        assert equipos == {50: "Real Madrid", 60: "Sevilla"}

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
            "entidad": "jugador",
            "variante": config_app.VARIANTE_BASE,
            "nombre": config_app.NOMBRE_BASE,
            "datos": config_app.VARIANTE_BASE,
            # Con qué datos se entrenó y sobre cuáles se ha consultado: sin
            # `?datos=`, los mismos.
            "datos_consultados": config_app.VARIANTE_BASE,
            "formulacion": config_app.MODELO_BASE["jugador"][0],
            "normalizacion": config_app.MODELO_BASE["jugador"][1],
        }
        assert len(datos["candidatos"]) == 2
        assert datos["candidatos"][0]["rango"] == 1
        assert datos["candidatos"][0]["coincidencias"]

    def test_los_scores_van_en_orden_decreciente(self, cliente) -> None:
        datos = cliente.get("/api/similares?entidad=equipo&nombre=Barcelona").get_json()
        scores = [c["score"] for c in datos["candidatos"]]
        assert scores == sorted(scores, reverse=True)

    def test_la_ambiguedad_es_300_con_los_candidatos(self, cliente) -> None:
        r = cliente.get("/api/similares?entidad=jugador&nombre=Repetido")
        assert r.status_code == 300
        assert len(r.get_json()["candidatos"]) == 2

    def test_quien_no_esta_en_la_bd_es_404_explicado(self, cliente) -> None:
        r = cliente.get("/api/similares?entidad=jugador&id=40")
        assert r.status_code == 404
        assert "no tiene a este jugador" in r.get_json()["error"]

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


class TestGlosario:
    """Qué mide cada métrica, por entidad y fase."""

    def test_se_sirve(self, cliente) -> None:
        r = cliente.get("/glosario")
        assert r.status_code == 200
        assert "Glosario de métricas" in _texto(r)

    def test_separa_jugador_de_equipo(self, cliente) -> None:
        """Los dos vectores no comparten catálogo de métricas."""
        html = _texto(cliente.get("/glosario"))
        assert "Métricas de jugadores" in html and "Métricas de equipos" in html

    def test_agrupa_por_fase(self, cliente) -> None:
        html = _texto(cliente.get("/glosario"))
        for fase in ("Progresión", "Juego aéreo", "Territorio", "Estilo"):
            assert fase in html

    def test_explica_las_metricas_y_no_solo_las_nombra(self, cliente) -> None:
        html = _texto(cliente.get("/glosario"))
        assert glosario.DEFINICIONES["deep_completions"] in html

    def test_no_necesita_ningun_modelo(self, tmp_path: Path) -> None:
        """El catálogo de métricas no depende de qué haya construido el usuario.

        Es justo con la app recién instalada cuando más falta hace saber qué
        significa cada cosa, así que la página no puede exigir un artefacto.
        """
        cliente = crear_app(tmp_path / "vacio", testing=True).test_client()
        r = cliente.get("/glosario")
        assert r.status_code == 200
        assert "Goles esperados" in _texto(r)

    def test_se_llega_desde_cualquier_pagina(self, cliente) -> None:
        assert 'href="/glosario"' in _texto(cliente.get("/"))


class TestPaginasDeError:
    def test_una_ruta_web_inexistente_devuelve_html(self, cliente) -> None:
        r = cliente.get("/no_existe")
        assert r.status_code == 404
        assert "no existe" in _texto(r)
