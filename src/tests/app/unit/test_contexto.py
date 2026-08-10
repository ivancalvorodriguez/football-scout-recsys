"""Contexto de jugador leído de la BD: trayectoria y minutos por posición."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from src.app.contexto import (
    CatalogoContexto,
    MarcaCampo,
    Militancia,
    PasoPorLiga,
    UsoPosicion,
    marcas_campo,
    reparto_entero,
    texto_trayectoria,
)
from src.app.posiciones import POR_NOMBRE


@pytest.fixture
def contexto(bd: Path) -> CatalogoContexto:
    return CatalogoContexto(bd)


class TestTrayectoria:
    def test_lee_el_equipo_y_la_liga(self, contexto: CatalogoContexto) -> None:
        pasos = contexto.trayectoria(10)
        assert [p.nombres for p in pasos] == ["Barcelona"]
        assert [p.liga for p in pasos] == ["11-27"]

    def test_acumula_partidos_y_minutos_por_etapa(
        self, contexto: CatalogoContexto
    ) -> None:
        militancia = contexto.jugador(10).militancias[0]
        assert (militancia.partidos, militancia.minutos) == (3, 270.0)

    def test_va_en_orden_cronologico(self, contexto: CatalogoContexto) -> None:
        """La carrera se lee hacia delante, no por quien acumula mas minutos."""
        pasos = contexto.trayectoria(20)
        assert [p.nombres for p in pasos] == ["Barcelona", "Real Madrid"]
        assert [p.desde for p in pasos] == ["2015-09-01", "2016-02-01"]

    def test_cada_etapa_lleva_su_liga(self, contexto: CatalogoContexto) -> None:
        """El 20 cambia de equipo Y de competicion."""
        assert [p.liga for p in contexto.trayectoria(20)] == ["11-27", "7-27"]

    def test_dos_equipos_de_la_misma_liga_van_juntos(
        self, contexto: CatalogoContexto
    ) -> None:
        """Repetir la liga por cada club seria ruido."""
        pasos = contexto.trayectoria(60)
        assert len(pasos) == 1
        assert pasos[0].nombres == "Barcelona, Sevilla"
        assert pasos[0].liga == "11-27"

    def test_un_jugador_sin_partidos_no_tiene_trayectoria(
        self, contexto: CatalogoContexto
    ) -> None:
        assert contexto.trayectoria(40) == ()
        assert contexto.jugador(40).equipo_actual is None

    def test_el_equipo_actual_es_el_ultimo(self, contexto: CatalogoContexto) -> None:
        assert contexto.jugador(20).equipo_actual.nombre == "Real Madrid"


class TestTextoTrayectoria:
    def test_pone_el_equipo_junto_a_su_liga(self, contexto: CatalogoContexto) -> None:
        texto = texto_trayectoria(
            contexto.trayectoria(20), {"11-27": "La Liga", "7-27": "Ligue 1"}.get
        )
        assert texto == "Barcelona (La Liga) · Real Madrid (Ligue 1)"

    def test_agrupa_los_de_la_misma_liga(self, contexto: CatalogoContexto) -> None:
        texto = texto_trayectoria(contexto.trayectoria(60), lambda c: "La Liga")
        assert texto == "Barcelona, Sevilla (La Liga)"

    def test_sin_liga_queda_solo_el_equipo(self) -> None:
        """BD incompleta: el nombre del club ya es mejor que nada."""
        paso = PasoPorLiga(
            liga="",
            equipos=(Militancia(1, "Barcelona", "", 90.0, 1, ""),),
            desde="",
        )
        assert texto_trayectoria((paso,), lambda c: "") == "Barcelona"

    def test_sin_trayectoria_no_hay_texto(self, contexto: CatalogoContexto) -> None:
        assert texto_trayectoria(contexto.trayectoria(40), lambda c: "x") == ""


class TestPosiciones:
    def test_calcula_la_fraccion_de_minutos(self, contexto: CatalogoContexto) -> None:
        usos = contexto.jugador(20).posiciones
        assert [u.posicion.nombre for u in usos] == ["Center Forward", "Left Wing"]
        assert usos[0].fraccion == pytest.approx(135.0 / 225.0)

    def test_las_fracciones_suman_uno(self, contexto: CatalogoContexto) -> None:
        for jugador in (10, 20, 30):
            usos = contexto.jugador(jugador).posiciones
            assert sum(u.fraccion for u in usos) == pytest.approx(1.0)

    def test_ordena_de_mas_a_menos_minutos(self, contexto: CatalogoContexto) -> None:
        fracciones = [u.fraccion for u in contexto.jugador(20).posiciones]
        assert fracciones == sorted(fracciones, reverse=True)

    def test_una_posicion_fuera_del_catalogo_se_ignora(self, tmp_path: Path) -> None:
        ruta = _bd_posiciones(tmp_path, [(1, 1, "Libero", 90.0), (1, 2, "Center Back", 90.0)])
        usos = CatalogoContexto(ruta).jugador(1).posiciones
        assert [u.posicion.nombre for u in usos] == ["Center Back"]


class TestDegradacion:
    def test_sin_bd_todo_queda_vacio(self, tmp_path: Path) -> None:
        """La BD es un adorno: la app sirve recomendaciones igual sin ella."""
        contexto = CatalogoContexto(tmp_path / "no_existe.db")
        assert not contexto.disponible
        assert contexto.jugador(10).militancias == ()
        assert contexto.trayectoria(10) == ()

    def test_una_bd_sin_las_tablas_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "vacia.db"
        with sqlite3.connect(ruta) as conn:
            conn.execute("CREATE TABLE otra (x INTEGER)")
        assert CatalogoContexto(ruta).jugador(10).posiciones == ()

    def test_un_fichero_que_no_es_sqlite_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "basura.db"
        ruta.write_text("esto no es una base de datos")
        assert not CatalogoContexto(ruta).disponible


class TestCache:
    def test_no_reabre_la_bd_en_cada_consulta(
        self, bd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contexto = CatalogoContexto(bd)
        contexto.jugador(10)
        monkeypatch.setattr(
            contexto, "_cargar", lambda: (_ for _ in ()).throw(AssertionError("recargo"))
        )
        assert contexto.jugador(20).militancias

    def test_recarga_si_la_bd_cambia(self, tmp_path: Path) -> None:
        ruta = _bd_posiciones(tmp_path, [(1, 1, "Center Back", 90.0)])
        contexto = CatalogoContexto(ruta)
        assert len(contexto.jugador(1).posiciones) == 1
        with sqlite3.connect(ruta) as conn:
            conn.execute(
                "INSERT INTO player_match_positions VALUES (1, 2, 'Left Back', 90.0)"
            )
        # Escritura y relectura caen en la misma marca de tiempo del sistema de
        # ficheros: se envejece a mano para probar la invalidación en sí.
        os.utime(ruta, (0, 0))
        assert len(contexto.jugador(1).posiciones) == 2


class TestMarcasCampo:
    def _uso(self, nombre: str, fraccion: float) -> UsoPosicion:
        return UsoPosicion(
            posicion=POR_NOMBRE[nombre], minutos=fraccion * 90.0, fraccion=fraccion
        )

    def test_el_porcentaje_se_escribe_como_tal(self) -> None:
        """El circulo no codifica el cuanto: lo dice el numero de dentro."""
        marca = MarcaCampo(posicion=POR_NOMBRE["Left Wing"], fraccion=0.921)
        assert marca.porcentaje == "92 %"

    def test_el_cero_de_uno_de_los_dos_se_escribe(self) -> None:
        """Que el candidato no juegue ahi es un dato, no una ausencia."""
        marca = MarcaCampo(
            posicion=POR_NOMBRE["Left Wing"], fraccion=1.0, fraccion_otro=0.0
        )
        assert marca.porcentaje_otro == "0 %"

    def test_une_las_posiciones_de_los_dos_jugadores(self) -> None:
        """Que el candidato juegue donde la referencia no es justo lo que hay que ver."""
        marcas = marcas_campo(
            (self._uso("Left Wing", 1.0),), (self._uso("Right Wing", 1.0),)
        )
        assert {m.posicion.nombre for m in marcas} == {"Left Wing", "Right Wing"}

    def test_la_posicion_ausente_en_uno_vale_cero_para_ese(self) -> None:
        marcas = marcas_campo(
            (self._uso("Left Wing", 1.0),), (self._uso("Right Wing", 1.0),)
        )
        izquierda = next(m for m in marcas if m.posicion.nombre == "Left Wing")
        assert (izquierda.fraccion, izquierda.fraccion_otro) == (1.0, 0.0)

    def test_sin_comparacion_el_otro_es_none(self) -> None:
        marcas = marcas_campo((self._uso("Left Wing", 1.0),))
        assert marcas[0].fraccion_otro is None
        assert marcas[0].porcentaje_otro == "—"

    def test_las_burbujas_grandes_van_primero(self) -> None:
        """Se dibujan en ese orden para que las pequeñas no queden tapadas."""
        marcas = marcas_campo(
            (self._uso("Left Wing", 0.2), self._uso("Center Forward", 0.8))
        )
        assert marcas[0].posicion.nombre == "Center Forward"

    def test_dos_jugadores_de_la_misma_ficha(self, contexto: CatalogoContexto) -> None:
        marcas = contexto.marcas(20, 10)
        assert {m.posicion.nombre for m in marcas} == {
            "Center Forward", "Left Wing", "Right Wing"
        }


class TestOrdenDeLasMarcas:
    """Comparando dos jugadores manda la REFERENCIA, no el maximo de los dos.

    La lista de debajo del campo se recorta a las primeras posiciones: tiene que
    empezar por donde juega aquel de quien se buscan parecidos.
    """

    def _uso(self, nombre: str, fraccion: float) -> UsoPosicion:
        return UsoPosicion(
            posicion=POR_NOMBRE[nombre], minutos=fraccion * 90.0, fraccion=fraccion
        )

    def test_ordena_por_el_primer_jugador(self) -> None:
        marcas = marcas_campo(
            (self._uso("Right Wing", 0.7), self._uso("Center Forward", 0.3)),
            (self._uso("Center Forward", 0.9), self._uso("Right Wing", 0.1)),
        )
        assert [m.posicion.nombre for m in marcas] == ["Right Wing", "Center Forward"]

    def test_una_posicion_solo_del_candidato_va_detras(self) -> None:
        """Aunque el candidato juegue ahi el 100 % de sus minutos."""
        marcas = marcas_campo(
            (self._uso("Right Wing", 1.0),), (self._uso("Left Back", 1.0),)
        )
        assert [m.posicion.nombre for m in marcas] == ["Right Wing", "Left Back"]

    def test_el_candidato_desempata(self) -> None:
        """Entre dos posiciones que la referencia no usa, manda el candidato."""
        marcas = marcas_campo(
            (self._uso("Right Wing", 1.0),),
            (self._uso("Left Back", 0.3), self._uso("Center Back", 0.7)),
        )
        assert [m.posicion.nombre for m in marcas] == [
            "Right Wing", "Center Back", "Left Back"
        ]

    def test_con_un_solo_jugador_sigue_siendo_de_mas_a_menos(self) -> None:
        marcas = marcas_campo(
            (self._uso("Left Wing", 0.2), self._uso("Center Forward", 0.8))
        )
        assert [m.fraccion for m in marcas] == [0.8, 0.2]


class TestRepartoDeLosPorcentajes:
    """Las cifras escritas en el campo tienen que sumar 100 tambien redondeadas.

    Redondear cada una por su cuenta no cuadra: tres tercios dan 99 y tres
    sextos, 101. Se reparte por el metodo del resto mayor.
    """

    def _uso(self, nombre: str, fraccion: float) -> UsoPosicion:
        return UsoPosicion(
            posicion=POR_NOMBRE[nombre], minutos=fraccion * 90.0, fraccion=fraccion
        )

    def test_tres_tercios_suman_cien(self) -> None:
        marcas = marcas_campo((
            self._uso("Left Wing", 1 / 3), self._uso("Center Forward", 1 / 3),
            self._uso("Right Wing", 1 / 3),
        ))
        assert sum(m.entero for m in marcas) == 100
        assert sorted(m.porcentaje for m in marcas) == ["33 %", "33 %", "34 %"]

    def test_las_dos_columnas_de_una_comparacion_suman_cien(self) -> None:
        marcas = marcas_campo(
            (self._uso("Right Wing", 2 / 3), self._uso("Center Forward", 1 / 3)),
            (self._uso("Center Forward", 1 / 6), self._uso("Left Wing", 5 / 6)),
        )
        assert sum(m.entero for m in marcas) == 100
        assert sum(m.entero_otro for m in marcas) == 100

    def test_el_punto_que_sobra_va_al_resto_mayor(self) -> None:
        """No al primero de la lista: la cifra que menos se aleja de su valor."""
        marcas = marcas_campo((
            self._uso("Center Back", 0.5), self._uso("Left Back", 1 / 6),
            self._uso("Right Back", 1 / 6), self._uso("Center Midfield", 1 / 6),
        ))
        porcentajes = {m.posicion.nombre: m.porcentaje for m in marcas}
        assert porcentajes["Center Back"] == "50 %"
        assert sorted(porcentajes.values()) == ["16 %", "17 %", "17 %", "50 %"]

    def test_una_marca_suelta_redondea_por_su_cuenta(self) -> None:
        """Fuera de `marcas_campo` no hay conjunto entre el que repartir."""
        marca = MarcaCampo(posicion=POR_NOMBRE["Left Wing"], fraccion=0.921)
        assert marca.entero is None
        assert marca.porcentaje == "92 %"

    def test_sin_minutos_no_se_reparte_nada(self) -> None:
        assert reparto_entero([]) == ()
        assert reparto_entero([0.0, 0.0]) == (0, 0)

    def test_el_reparto_no_se_pasa_ni_se_queda_corto(self) -> None:
        assert sum(reparto_entero([1 / 7] * 7)) == 100
        assert sum(reparto_entero([0.999, 0.001])) == 100

    def test_una_posicion_no_jugada_se_queda_en_cero(self) -> None:
        """El 0 % del candidato es un dato; el reparto no puede inventarle un 1 %."""
        marcas = marcas_campo(
            (self._uso("Right Wing", 1.0),), (self._uso("Left Back", 1.0),)
        )
        derecha = next(m for m in marcas if m.posicion.nombre == "Right Wing")
        assert derecha.porcentaje_otro == "0 %"


class TestRotuloDeLaMarca:
    """Lo que se lee al dejar el raton sobre una marca del campo.

    Dentro del circulo solo caben las cifras: el tooltip es el unico sitio donde
    se dice de que posicion son.
    """

    def test_un_jugador_dice_posicion_y_porcentaje(self) -> None:
        marca = MarcaCampo(posicion=POR_NOMBRE["Left Wing"], fraccion=0.45)
        assert marca.rotulo() == "Extremo izquierdo — 45 % de sus minutos"

    def test_comparando_pone_nombre_a_cada_porcentaje(self) -> None:
        """El color lo dice en la figura; el tooltip es texto plano."""
        marca = MarcaCampo(
            posicion=POR_NOMBRE["Left Wing"], fraccion=0.45, fraccion_otro=0.1
        )
        assert marca.rotulo("Messi", "Pedri") == (
            "Extremo izquierdo — Messi: 45 % · Pedri: 10 %"
        )

    def test_comparando_sin_nombres_deja_los_dos_porcentajes(self) -> None:
        """El `_campo.html` de una ficha no pasa el segundo nombre."""
        marca = MarcaCampo(
            posicion=POR_NOMBRE["Left Wing"], fraccion=0.45, fraccion_otro=0.1
        )
        assert marca.rotulo() == "Extremo izquierdo — 45 % · 10 %"

    def test_el_cero_del_otro_tambien_se_dice(self) -> None:
        """No jugar nunca ahi es la diferencia que hay que leer."""
        marca = MarcaCampo(
            posicion=POR_NOMBRE["Right Wing"], fraccion=0.8, fraccion_otro=0.0
        )
        assert marca.rotulo("Messi", "Pedri").endswith("Pedri: 0 %")


def _bd_posiciones(tmp_path: Path, filas: list[tuple]) -> Path:
    """BD con solo la tabla de posiciones (para probar casos aislados)."""
    ruta = tmp_path / "posiciones.db"
    with sqlite3.connect(ruta) as conn:
        conn.execute(
            "CREATE TABLE player_match_positions (player_id INTEGER, match_id INTEGER, "
            "position_name TEXT, minutes REAL)"
        )
        conn.executemany("INSERT INTO player_match_positions VALUES (?, ?, ?, ?)", filas)
    return ruta
