"""Contexto de cada jugador leído de la BD (equipo y minutos por posición).

Los artefactos de `outputs/modelo/` traen la S, los nombres y las ligas, pero NO
el equipo de cada jugador ni su reparto de minutos por posición:

- el equipo vive en `player_match_stats.team_id`, y además por partido (un
  jugador puede cambiar de club entre temporadas);
- las columnas `pos_*` del modelo sí existen, pero llegan z-scoreadas y divididas
  por sqrt(25) (`similitud.features._reescalar_bloque_zscoreado`), así que de
  ellas ya no se puede recuperar el «45 % de sus minutos como lateral
  izquierdo». Eso solo está en `player_match_positions`.

Por eso este módulo va a la BD, **en solo lectura** y con el mismo contrato que
`ligas`: es un adorno. Si la BD no está, todo devuelve vacío y la interfaz omite
el equipo y el campo de posiciones, sin romperse. La tabla entera cabe
holgadamente en memoria (unos miles de jugadores) y se cachea con la misma huella
(mtime, tamaño) que usan `ligas` y `catalogo`, de modo que volver a extraer con el
servidor levantado se refleja sin reiniciarlo.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from . import posiciones as cat_posiciones
from .posiciones import Posicion


@dataclass(frozen=True)
class Militancia:
    """Paso de un jugador por un equipo EN UNA LIGA concreta.

    El grano es (equipo, competición-temporada) y no solo el equipo: un jugador
    puede jugar la liga y la copa con el mismo club, o cambiar de club a mitad de
    temporada, y lo que contextualiza su nombre es el par «dónde y en qué
    competición», no cada mitad por separado.
    """

    equipo_id: int
    nombre: str
    liga: str            # clave `competition_id-season_id`, como en el modelo
    minutos: float
    partidos: int
    desde: str           # fecha del primer partido; "" si la BD no la tiene


@dataclass(frozen=True)
class PasoPorLiga:
    """Los equipos de un jugador dentro de una misma liga.

    Se agrupa porque repetir la liga por cada club es ruido: quien juega media
    temporada en dos equipos de la misma competición se lee mejor como «Sevilla,
    Betis · La Liga 2015/2016» que como dos entradas con la liga duplicada.
    """

    liga: str
    equipos: tuple[Militancia, ...]   # en orden cronológico
    desde: str

    @property
    def nombres(self) -> str:
        return ", ".join(m.nombre for m in self.equipos)


def texto_trayectoria(
    pasos: tuple[PasoPorLiga, ...], nombre_liga: Callable[[str], str]
) -> str:
    """Trayectoria en una línea: «Sevilla, Betis (La Liga 2015/2016) · ...».

    `nombre_liga` traduce la clave `"11-27"`; se recibe como argumento porque ese
    catálogo vive en `ligas` y este módulo no tiene por qué conocerlo. Sin fecha
    ni liga (BD incompleta) queda solo el nombre del equipo, que ya es mejor que
    nada.
    """
    return " · ".join(
        f"{p.nombres} ({nombre_liga(p.liga)})" if p.liga else p.nombres
        for p in pasos
    )


@dataclass(frozen=True)
class UsoPosicion:
    """Minutos de un jugador en una posición concreta."""

    posicion: Posicion
    minutos: float
    fraccion: float  # sobre el total de minutos con posición registrada


@dataclass(frozen=True)
class Jugador:
    """Todo lo que la BD aporta sobre un jugador, ya agregado."""

    # En orden CRONOLÓGICO (por el primer partido), no por minutos: la carrera se
    # lee hacia delante, y con varios clubes lo relevante es el recorrido.
    militancias: tuple[Militancia, ...] = ()
    posiciones: tuple[UsoPosicion, ...] = ()      # de más a menos minutos

    @property
    def trayectoria(self) -> tuple[PasoPorLiga, ...]:
        """Militancias agrupadas por liga, en orden cronológico."""
        por_liga: dict[str, list[Militancia]] = {}
        for m in self.militancias:
            por_liga.setdefault(m.liga, []).append(m)
        pasos = [
            PasoPorLiga(liga=liga, equipos=tuple(ms), desde=ms[0].desde)
            for liga, ms in por_liga.items()
        ]
        # `militancias` ya viene ordenada, así que el primer elemento de cada
        # grupo marca cuándo empieza ese paso.
        pasos.sort(key=lambda p: (p.desde, p.liga))
        return tuple(pasos)

    @property
    def equipo_actual(self) -> Militancia | None:
        """Último equipo del que hay registro (el más reciente)."""
        return self.militancias[-1] if self.militancias else None


_SIN_DATOS = Jugador()


# --- Campo de fútbol ----------------------------------------------------------

# Radio de la marca, en las unidades del lienzo de `posiciones` (100x100). Es el
# MISMO para todas: el círculo solo dice DÓNDE se juega, y el cuánto lo dicen los
# porcentajes escritos dentro. Codificarlo también en el tamaño haría la figura
# más difícil de leer, no más informativa —comparar áreas a ojo es impreciso— y
# encogería justo las marcas cuyo número hay que poder leer.
#
# El tope es la mitad de la distancia entre las dos posiciones más juntas del
# catálogo (Mediapunta y Segundo delantero, a 11): por encima de 5,5 los círculos
# de posiciones vecinas se pisarían, y hay parejas que un mismo jugador combina a
# menudo (lateral y carrilero, delantero centro y segundo delantero). Lo fija
# `test_posiciones.py`, para que mover una posición del catálogo o subir el radio
# rompa en vez de degradar el dibujo en silencio.
RADIO_MARCA = 5.4


def reparto_entero(fracciones: Sequence[float], total: int = 100) -> tuple[int, ...]:
    """Reparte `total` puntos enteros entre `fracciones` sin perder ninguno.

    Redondear cada porcentaje por su cuenta no suma 100: tres posiciones a
    33,33 % se escriben «33 % · 33 % · 33 %» y quien lee el campo echa en falta
    un punto; tres al 16,67 % dan 17+17+17 = 101. Se usa el método del **resto
    mayor** (Hare-Niemeyer): a cada marca le toca su parte entera y los puntos
    que sobran van a las que tenían el decimal más alto, así que la suma es
    exactamente `total` y ninguna cifra se aleja más de un punto de su valor.

    Las fracciones llegan ya normalizadas (cada jugador reparte sus minutos entre
    sus posiciones), pero se vuelven a normalizar por su suma para que el reparto
    cuadre también si el conjunto que se pasa no cubre el 100 % del jugador. Sin
    minutos en ninguna posición no hay nada que repartir y todo queda a 0.
    """
    valores = [max(0.0, float(f)) for f in fracciones]
    suma = sum(valores)
    if suma <= 0.0:
        return tuple(0 for _ in valores)
    objetivo = [v * total / suma for v in valores]
    enteros = [math.floor(v) for v in objetivo]
    # Los empates los decide el orden de llegada, que es el de la figura: si dos
    # posiciones tienen el mismo resto, el punto va a la que se pinta antes.
    sobran = total - sum(enteros)
    orden = sorted(
        range(len(valores)), key=lambda i: (-(objetivo[i] - enteros[i]), i)
    )
    for i in orden[:sobran]:
        enteros[i] += 1
    return tuple(enteros)


def _porcentaje(fraccion: float, entero: int | None) -> str:
    """Cifra de una marca: la repartida si la hay, y si no el redondeo suelto."""
    if entero is None:
        return f"{fraccion * 100:.0f} %"
    return f"{entero} %"


@dataclass(frozen=True)
class MarcaCampo:
    """Una posición dibujada sobre el campo, con la carga de uno o dos jugadores.

    `fraccion_otro` es `None` cuando se dibuja a un solo jugador, y 0.0 cuando se
    comparan dos y este no ha jugado nunca ahí (que es un dato, no una ausencia).

    `entero`/`entero_otro` son el porcentaje YA repartido por `reparto_entero`,
    que es cosa del conjunto de marcas y no de cada una por separado. Van a
    `None` cuando la marca se construye suelta (fuera de `marcas_campo`): sin
    conjunto no hay reparto que hacer y la cifra sale de redondear su fracción.
    """

    posicion: Posicion
    fraccion: float
    fraccion_otro: float | None = None
    entero: int | None = None
    entero_otro: int | None = None

    @property
    def porcentaje(self) -> str:
        return _porcentaje(self.fraccion, self.entero)

    @property
    def porcentaje_otro(self) -> str:
        if self.fraccion_otro is None:
            return "—"
        return _porcentaje(self.fraccion_otro, self.entero_otro)

    def rotulo(self, nombre_a: str = "", nombre_b: str = "") -> str:
        """Texto del tooltip de la marca: qué posición es y cuánto se juega ahí.

        Dentro del círculo solo caben los porcentajes (r=5,4 da para dos líneas
        de 2,9px), así que el nombre de la posición y de quién es cada número
        viven aquí. Comparando dos jugadores se antepone su nombre: en la figura
        eso lo dice el color, pero el tooltip es texto plano —lo lee también un
        lector de pantalla— y ahí el color no existe.
        """
        if self.fraccion_otro is None:
            return f"{self.posicion.etiqueta} — {self.porcentaje} de sus minutos"
        if nombre_a and nombre_b:
            return (
                f"{self.posicion.etiqueta} — {nombre_a}: {self.porcentaje} · "
                f"{nombre_b}: {self.porcentaje_otro}"
            )
        return (
            f"{self.posicion.etiqueta} — {self.porcentaje} · {self.porcentaje_otro}"
        )


def marcas_campo(
    usos: tuple[UsoPosicion, ...],
    otros: tuple[UsoPosicion, ...] | None = None,
) -> tuple[MarcaCampo, ...]:
    """Une el reparto de minutos de uno o dos jugadores en marcas del campo.

    Se incluyen las posiciones en las que juega CUALQUIERA de los dos: si el
    candidato juega en una banda donde la referencia no aparece, esa diferencia
    es justo lo que hay que ver.

    El orden lo marca **el primer jugador**, que es la referencia de la búsqueda:
    la lista de debajo del campo se recorta a las primeras posiciones, así que
    tiene que empezar por donde juega aquel de quien se buscan parecidos, no por
    donde juega el candidato. Ordenar por el máximo de los dos metía delante
    posiciones en las que la referencia no aparece. Con un solo jugador el
    criterio es el mismo de siempre (de más a menos minutos).

    Los porcentajes se reparten al final (`reparto_entero`) y no marca a marca,
    para que las cifras escritas sumen 100 en cada columna aun redondeadas.
    """
    propias = {u.posicion.slug: u for u in usos}
    ajenas = {u.posicion.slug: u for u in (otros or ())}
    crudas = [
        (
            (propias.get(slug) or ajenas[slug]).posicion,
            propias[slug].fraccion if slug in propias else 0.0,
            (
                None if otros is None
                else (ajenas[slug].fraccion if slug in ajenas else 0.0)
            ),
        )
        for slug in propias.keys() | ajenas.keys()
    ]
    # El segundo jugador solo desempata: entre dos posiciones que la referencia
    # usa lo mismo (a menudo, ninguna de las dos) manda el candidato.
    crudas.sort(key=lambda c: (-c[1], -(c[2] or 0.0), c[0].etiqueta))

    enteros = reparto_entero([c[1] for c in crudas])
    comparando = otros is not None
    enteros_otro = (
        reparto_entero([c[2] or 0.0 for c in crudas]) if comparando
        else [None] * len(crudas)
    )
    return tuple(
        MarcaCampo(
            posicion=posicion,
            fraccion=fraccion,
            fraccion_otro=fraccion_otro,
            entero=entero,
            entero_otro=entero_otro,
        )
        for (posicion, fraccion, fraccion_otro), entero, entero_otro
        in zip(crudas, enteros, enteros_otro)
    )


@dataclass
class _Tabla:
    """Todo lo cargado de la BD de una vez."""

    jugadores: dict[int, Jugador] = field(default_factory=dict)


class CatalogoContexto:
    """Contexto de jugador leído de la BD, cacheado en memoria."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._tabla_cache: _Tabla | None = None
        self._huella: tuple[float, int] | None = None
        self._lock = threading.Lock()

    # --- Carga ---------------------------------------------------------------

    def _consultar(self, sql: str) -> list[tuple]:
        """Ejecuta una consulta en solo lectura; [] ante cualquier problema.

        Igual que `ligas`: una BD ausente, con esquema antiguo o corrupta deja a
        la app sin adorno, no sin servicio.
        """
        if not self.db_path.is_file():
            return []
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                return conn.execute(sql).fetchall()
        except sqlite3.Error:
            return []

    def _cargar(self) -> _Tabla:
        militancias = self._cargar_militancias()
        posiciones = self._cargar_posiciones()
        return _Tabla(jugadores={
            jugador_id: Jugador(
                militancias=militancias.get(jugador_id, ()),
                posiciones=posiciones.get(jugador_id, ()),
            )
            for jugador_id in set(militancias) | set(posiciones)
        })

    def _cargar_militancias(self) -> dict[int, tuple[Militancia, ...]]:
        """Paso de cada jugador por cada (equipo, liga), en orden cronológico.

        Se agrupa por (jugador, equipo, competición-temporada) sobre
        `player_match_stats`: su grano es jugador-partido, así que un traspaso o
        un cambio de competición aparecen como grupos distintos y todos se
        conservan. La liga se compone igual que en el pipeline
        (`similitud.data.LEAGUE_KEY`), para que la clave sea la misma que trae el
        artefacto y se pueda traducir con el catálogo de ligas.

        El `LEFT JOIN` con `matches` es deliberado: si faltase el partido, la
        militancia sigue apareciendo (sin liga ni fecha) en vez de desaparecer.
        `minutes_played` y `match_date` pueden ser NULL, de ahí los COALESCE.
        """
        filas = self._consultar(
            """
            SELECT p.player_id, p.team_id, t.team_name,
                   m.competition_id, m.season_id,
                   SUM(COALESCE(p.minutes_played, 0.0)), COUNT(*),
                   MIN(COALESCE(m.match_date, ''))
            FROM player_match_stats p
            LEFT JOIN teams t ON t.team_id = p.team_id
            LEFT JOIN matches m ON m.match_id = p.match_id
            GROUP BY p.player_id, p.team_id, m.competition_id, m.season_id
            """
        )
        por_jugador: dict[int, list[Militancia]] = {}
        for jugador, equipo, nombre, comp, temporada, minutos, partidos, desde in filas:
            if equipo is None:
                continue
            liga = "" if comp is None or temporada is None else f"{comp}-{temporada}"
            por_jugador.setdefault(int(jugador), []).append(Militancia(
                equipo_id=int(equipo),
                nombre=str(nombre) if nombre else f"Equipo {equipo}",
                liga=liga,
                minutos=float(minutos or 0.0),
                partidos=int(partidos),
                desde=str(desde or ""),
            ))
        # Cronológico por el primer partido. Sin fecha en la BD el orden lo
        # decide la liga y luego el nombre, que al menos es estable entre
        # ejecuciones (un orden arbitrario cambiaría de una recarga a otra).
        return {
            j: tuple(sorted(v, key=lambda m: (m.desde, m.liga, m.nombre)))
            for j, v in por_jugador.items()
        }

    def _cargar_posiciones(self) -> dict[int, tuple[UsoPosicion, ...]]:
        """Minutos por posición de cada jugador, como fracción de su total."""
        filas = self._consultar(
            """
            SELECT player_id, position_name, SUM(COALESCE(minutes, 0.0))
            FROM player_match_positions
            GROUP BY player_id, position_name
            """
        )
        crudo: dict[int, list[tuple[Posicion, float]]] = {}
        for jugador_id, nombre, minutos in filas:
            posicion = cat_posiciones.por_nombre(str(nombre))
            if posicion is None or not minutos:
                continue
            crudo.setdefault(int(jugador_id), []).append((posicion, float(minutos)))

        usos: dict[int, tuple[UsoPosicion, ...]] = {}
        for jugador_id, pares in crudo.items():
            total = sum(m for _, m in pares)
            if total <= 0.0:
                continue
            lista = [
                UsoPosicion(posicion=p, minutos=m, fraccion=m / total)
                for p, m in pares
            ]
            lista.sort(key=lambda u: (-u.fraccion, u.posicion.etiqueta))
            usos[jugador_id] = tuple(lista)
        return usos

    def _tabla(self) -> _Tabla:
        """Tabla cacheada, recargada si la BD ha cambiado."""
        huella: tuple[float, int] | None = None
        if self.db_path.is_file():
            estado = self.db_path.stat()
            huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            if self._tabla_cache is None or self._huella != huella:
                self._tabla_cache = self._cargar()
                self._huella = huella
            return self._tabla_cache

    # --- Consulta ------------------------------------------------------------

    @property
    def disponible(self) -> bool:
        """True si la BD ha aportado algo (si no, la interfaz omite los bloques)."""
        return bool(self._tabla().jugadores)

    def jugador(self, jugador_id: int) -> Jugador:
        """Contexto de un jugador; vacío si no está en la BD."""
        return self._tabla().jugadores.get(int(jugador_id), _SIN_DATOS)

    def marcas(
        self, jugador_id: int, otro_id: int | None = None
    ) -> tuple[MarcaCampo, ...]:
        """Marcas del campo de un jugador, o de dos para compararlos."""
        propias = self.jugador(jugador_id).posiciones
        otras = self.jugador(otro_id).posiciones if otro_id is not None else None
        return marcas_campo(propias, otras)

    def trayectoria(self, jugador_id: int) -> tuple[PasoPorLiga, ...]:
        """Equipos del jugador agrupados por liga, en orden cronológico."""
        return self.jugador(jugador_id).trayectoria
