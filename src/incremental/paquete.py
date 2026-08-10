"""Formato de entrada de partidos nuevos: descubrimiento y validacion.

Un **paquete de partidos** es un directorio con la MISMA estructura que
`open-data/data` (StatsBomb Open Data). Se eligio ese formato y no uno propio
porque el motor de extraccion (`src.extraccion`) lee exactamente eso: asi los
partidos nuevos pasan por el mismo codigo que los que ya estan en la BD y las
metricas son comparables por construccion. Un formato intermedio (CSV de
metricas ya calculadas, por ejemplo) obligaria a mantener dos definiciones de
cada metrica y romperia esa garantia.

    <paquete>/
        competitions.json                            (1 fichero)
        matches/<competition_id>/<season_id>.json     (1 por competicion-temporada)
        events/<match_id>.json                        (1 por partido)
        lineups/<match_id>.json                       (1 por partido)

`open-data/data` es, literalmente, un paquete valido; y un paquete puede llevar
un solo partido. Los campos exigidos de cada fichero estan declarados abajo en
`CAMPOS_*` y son EXACTAMENTE los que consume el motor: esas constantes son la
fuente de verdad del formato, y `docs/incremental.md` su explicacion.

La validacion separa `error` (el partido no se puede procesar o produciria
metricas mudas) de `aviso` (se procesa, pero hay algo sospechoso: coordenadas
fuera del campo 120x80, tiros sin xG, posiciones fuera del catalogo de 25...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.extraccion.config import FIELD_LENGTH, FIELD_WIDTH, POSITIONS_25

# --- Contrato de campos (fuente de verdad del formato) -----------------------
# Cada entrada: (ruta_del_campo, obligatorio). La ruta usa "." para descender en
# sub-objetos. Se comprueban con `_falta`.

# Fila de `competitions.json`. Identifica la competicion-temporada; los nombres
# son los que veran el usuario y los CLI (que seleccionan SIEMPRE por nombre).
CAMPOS_COMPETICION = [
    ("competition_id", True),
    ("season_id", True),
    ("competition_name", True),
    ("season_name", True),
    ("country_name", False),
    ("competition_gender", False),
]

# Objeto de `matches/<c>/<s>.json`.
CAMPOS_PARTIDO = [
    ("match_id", True),
    ("home_team.home_team_id", True),
    ("home_team.home_team_name", True),
    ("away_team.away_team_id", True),
    ("away_team.away_team_name", True),
    ("home_score", False),
    ("away_score", False),
    ("match_date", False),
    ("match_week", False),          # sin el, `--pct` cae a % de partidos por fecha
    ("competition_stage.name", False),
]

# Evento de `events/<match_id>.json`. `minute`/`second` son obligatorios aunque
# el codigo tenga default 0: sin ellos el reloj del partido se queda en 0, nadie
# acumula minutos y el partido entero se guardaria vacio en silencio.
CAMPOS_EVENTO = [
    ("id", True),
    ("type.name", True),
    ("team.id", True),
    ("team.name", True),
    ("period", True),
    ("minute", True),
    ("second", True),
    ("possession", False),          # sin el, el evento no entra en ninguna secuencia
    ("possession_team.id", False),
    ("possession_team.name", False),
    ("play_pattern.name", False),
    ("player.id", False),           # solo los eventos con ejecutante
    ("player.name", False),
    ("location", False),
]

# Equipo de `lineups/<match_id>.json`.
CAMPOS_ALINEACION = [
    ("team_id", True),
    ("team_name", True),
    ("lineup", True),
]

# Jugador dentro de `lineup`. `positions` puede venir vacia (convocado que no
# jugo): esos jugadores simplemente no generan fila.
CAMPOS_JUGADOR = [
    ("player_id", True),
    ("player_name", True),
    ("positions", True),
    ("player_nickname", False),     # si viene, es el nombre que se guarda
]

# Spell de `positions`: tramo continuo en una posicion, con reloj "MM:SS".
# `to` vacio/None significa "hasta el final del partido".
CAMPOS_SPELL = [
    ("position_id", True),
    ("position", True),             # NOMBRE de la posicion (no `position_name`)
    ("from", True),
    ("to", False),
]

NIVEL_ERROR = "error"
NIVEL_AVISO = "aviso"


@dataclass(frozen=True)
class Problema:
    """Un hallazgo de la validacion, con su nivel y donde aparecio."""

    nivel: str
    donde: str
    mensaje: str

    def __str__(self) -> str:
        return f"[{self.nivel}] {self.donde}: {self.mensaje}"


@dataclass
class PartidoPaquete:
    """Un partido del paquete, ya resuelto a su competicion-temporada."""

    match_id: int
    competicion: dict[str, Any]     # fila de competitions.json (comp + season)
    partido: dict[str, Any]         # objeto del matches/<c>/<s>.json

    @property
    def etiqueta(self) -> str:
        home = (self.partido.get("home_team") or {}).get("home_team_name", "?")
        away = (self.partido.get("away_team") or {}).get("away_team_name", "?")
        return f"{home} vs {away}"


@dataclass
class Paquete:
    """Contenido de un paquete ya descubierto (sin leer eventos)."""

    raiz: Path
    partidos: list[PartidoPaquete] = field(default_factory=list)
    problemas: list[Problema] = field(default_factory=list)

    @property
    def errores(self) -> list[Problema]:
        return [p for p in self.problemas if p.nivel == NIVEL_ERROR]

    @property
    def avisos(self) -> list[Problema]:
        return [p for p in self.problemas if p.nivel == NIVEL_AVISO]

    def competiciones(self) -> list[dict[str, Any]]:
        """Competicion-temporada distintas presentes, en orden de aparicion."""
        vistas: dict[tuple[int, int], dict[str, Any]] = {}
        for p in self.partidos:
            clave = (int(p.competicion["competition_id"]), int(p.competicion["season_id"]))
            vistas.setdefault(clave, p.competicion)
        return list(vistas.values())


# --- Utilidades de acceso ----------------------------------------------------
def _bajar(obj: Any, ruta: str) -> Any:
    """Valor de `obj` en una ruta con puntos ('team.name'), o None si falta."""
    actual = obj
    for parte in ruta.split("."):
        if not isinstance(actual, dict):
            return None
        actual = actual.get(parte)
        if actual is None:
            return None
    return actual


def _faltantes(obj: Any, campos: list[tuple[str, bool]]) -> list[str]:
    """Rutas obligatorias ausentes en `obj`."""
    return [ruta for ruta, obligatorio in campos if obligatorio and _bajar(obj, ruta) is None]


def _leer_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --- Descubrimiento ----------------------------------------------------------
def _leer_competiciones(raiz: Path, problemas: list[Problema]) -> dict[tuple[int, int], dict]:
    """Indexa `competitions.json` por (competition_id, season_id)."""
    path = raiz / "competitions.json"
    if not path.exists():
        problemas.append(Problema(
            NIVEL_ERROR, "competitions.json",
            "no existe; es obligatorio y describe la competicion y temporada de "
            "cada fichero de matches/",
        ))
        return {}
    try:
        filas = _leer_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        problemas.append(Problema(NIVEL_ERROR, "competitions.json", f"JSON ilegible: {exc}"))
        return {}
    if not isinstance(filas, list):
        problemas.append(Problema(
            NIVEL_ERROR, "competitions.json",
            f"debe ser una LISTA de competicion-temporada, no {type(filas).__name__}",
        ))
        return {}

    indice: dict[tuple[int, int], dict] = {}
    for i, fila in enumerate(filas):
        faltan = _faltantes(fila, CAMPOS_COMPETICION)
        if faltan:
            problemas.append(Problema(
                NIVEL_ERROR, f"competitions.json[{i}]",
                f"faltan campos obligatorios: {', '.join(faltan)}",
            ))
            continue
        indice[(int(fila["competition_id"]), int(fila["season_id"]))] = fila
    return indice


def _ficheros_de_partidos(raiz: Path) -> list[tuple[int, int, Path]]:
    """(competition_id, season_id, ruta) de cada `matches/<c>/<s>.json` legible."""
    base = raiz / "matches"
    if not base.is_dir():
        return []
    salida: list[tuple[int, int, Path]] = []
    for dir_comp in sorted(base.iterdir()):
        if not dir_comp.is_dir() or not dir_comp.name.isdigit():
            continue
        for fichero in sorted(dir_comp.glob("*.json")):
            if fichero.stem.isdigit():
                salida.append((int(dir_comp.name), int(fichero.stem), fichero))
    return salida


def _coincide(valor: Any, buscado: str | None) -> bool:
    """Compara nombres sin distinguir mayusculas ni espacios sobrantes."""
    if buscado is None:
        return True
    return str(valor).strip().casefold() == buscado.strip().casefold()


def leer(
    raiz: Path,
    ids: Iterable[int] | None = None,
    competicion: str | None = None,
    temporada: str | None = None,
) -> Paquete:
    """Descubre los partidos de un paquete (sin abrir events/ ni lineups/).

    Tres filtros, combinables, todos opcionales (sin ninguno se traen todos los
    partidos del paquete):

    - `ids`: lista de `match_id` concretos.
    - `competicion` / `temporada`: por NOMBRE, como en el resto del proyecto
      (`"1. Bundesliga"`, `"2015/2016"`), resuelto contra `competitions.json`.

    Los problemas ESTRUCTURALES (falta competitions.json, un matches/ que apunta
    a una competicion no declarada, un partido sin equipos...) quedan en
    `paquete.problemas`; los del contenido de cada partido los da `validar`.
    """
    raiz = Path(raiz)
    paquete = Paquete(raiz=raiz)
    if not raiz.is_dir():
        paquete.problemas.append(Problema(
            NIVEL_ERROR, str(raiz), "no existe o no es un directorio"))
        return paquete

    competiciones = _leer_competiciones(raiz, paquete.problemas)
    ficheros = _ficheros_de_partidos(raiz)
    if not ficheros:
        paquete.problemas.append(Problema(
            NIVEL_ERROR, "matches/",
            "no hay ningun matches/<competition_id>/<season_id>.json; sin el no "
            "se sabe a que competicion-temporada pertenecen los eventos",
        ))
        return paquete

    filtro = {int(i) for i in ids} if ids is not None else None
    vistos: dict[int, str] = {}
    descartados_por_nombre = 0
    for cid, sid, path in ficheros:
        rel = f"matches/{cid}/{sid}.json"
        clave = (cid, sid)
        fila = competiciones.get(clave)
        if fila is None:
            # Se reporta aunque haya filtro por nombre: un matches/ sin fila en
            # competitions.json son datos que nadie sabe de donde salen, y
            # saltarselo en silencio es peor que molestar.
            paquete.problemas.append(Problema(
                NIVEL_ERROR, rel,
                f"competition_id={cid}, season_id={sid} no esta declarado en "
                "competitions.json",
            ))
            continue
        if not (_coincide(fila.get("competition_name"), competicion)
                and _coincide(fila.get("season_name"), temporada)):
            descartados_por_nombre += 1
            continue
        try:
            partidos = _leer_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            paquete.problemas.append(Problema(NIVEL_ERROR, rel, f"JSON ilegible: {exc}"))
            continue
        if not isinstance(partidos, list):
            paquete.problemas.append(Problema(
                NIVEL_ERROR, rel, "debe ser una LISTA de partidos"))
            continue

        for j, partido in enumerate(partidos):
            faltan = _faltantes(partido, CAMPOS_PARTIDO)
            if faltan:
                paquete.problemas.append(Problema(
                    NIVEL_ERROR, f"{rel}[{j}]",
                    f"faltan campos obligatorios: {', '.join(faltan)}",
                ))
                continue
            match_id = int(partido["match_id"])
            if filtro is not None and match_id not in filtro:
                continue
            if match_id in vistos:
                paquete.problemas.append(Problema(
                    NIVEL_ERROR, rel,
                    f"match_id {match_id} duplicado (ya aparecia en {vistos[match_id]}); "
                    "un match_id identifica un unico partido",
                ))
                continue
            vistos[match_id] = rel
            paquete.partidos.append(PartidoPaquete(
                match_id=match_id, competicion=competiciones[clave], partido=partido))

    if filtro is not None:
        for perdido in sorted(filtro - set(vistos)):
            paquete.problemas.append(Problema(
                NIVEL_ERROR, "matches/",
                f"se pidio el partido {perdido} pero no aparece en ningun "
                "matches/<competition_id>/<season_id>.json del paquete",
            ))
    if not paquete.partidos and descartados_por_nombre and not paquete.errores:
        pedido = " / ".join(x for x in (competicion, temporada) if x)
        paquete.problemas.append(Problema(
            NIVEL_ERROR, "competitions.json",
            f"ninguna competicion-temporada del paquete se llama {pedido!r}; "
            "los nombres deben coincidir con los de competitions.json",
        ))
    return paquete


# --- Validacion del contenido de un partido ----------------------------------
def _validar_eventos(raiz: Path, p: PartidoPaquete) -> list[Problema]:
    rel = f"events/{p.match_id}.json"
    path = raiz / "events" / f"{p.match_id}.json"
    if not path.exists():
        return [Problema(NIVEL_ERROR, rel, "no existe (obligatorio: es la fuente de las metricas)")]
    try:
        eventos = _leer_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [Problema(NIVEL_ERROR, rel, f"JSON ilegible: {exc}")]
    if not isinstance(eventos, list) or not eventos:
        return [Problema(NIVEL_ERROR, rel, "debe ser una LISTA no vacia de eventos")]

    problemas: list[Problema] = []
    faltan_por_campo: dict[str, int] = {}
    ids_evento: set[str] = set()
    duplicados = 0
    fuera_campo = 0
    sin_posesion = 0
    tiros_sin_xg = 0
    equipos_evento: set[Any] = set()
    minuto_max = 0

    for ev in eventos:
        if not isinstance(ev, dict):
            problemas.append(Problema(NIVEL_ERROR, rel, "hay entradas que no son objetos JSON"))
            break
        for ruta in _faltantes(ev, CAMPOS_EVENTO):
            faltan_por_campo[ruta] = faltan_por_campo.get(ruta, 0) + 1
        ident = ev.get("id")
        if ident is not None:
            if ident in ids_evento:
                duplicados += 1
            ids_evento.add(ident)
        equipo = _bajar(ev, "team.name")
        if equipo is not None:
            equipos_evento.add(equipo)
        if isinstance(ev.get("minute"), (int, float)):
            minuto_max = max(minuto_max, int(ev["minute"]))
        loc = ev.get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            try:
                x, y = float(loc[0]), float(loc[1])
            except (TypeError, ValueError):
                fuera_campo += 1
            else:
                if not (0.0 <= x <= FIELD_LENGTH and 0.0 <= y <= FIELD_WIDTH):
                    fuera_campo += 1
        if ev.get("possession") is None:
            sin_posesion += 1
        if _bajar(ev, "type.name") == "Shot" and _bajar(ev, "shot.statsbomb_xg") is None:
            tiros_sin_xg += 1

    for ruta, cuantos in sorted(faltan_por_campo.items()):
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            f"{cuantos} evento(s) sin el campo obligatorio '{ruta}'",
        ))
    if duplicados:
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            f"{duplicados} evento(s) con 'id' repetido; el id enlaza el pase clave "
            "con su tiro (xA) y debe ser unico dentro del partido",
        ))
    if minuto_max == 0:
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            "ningun evento supera el minuto 0: el reloj del partido se quedaria a "
            "cero y no se acumularian minutos jugados",
        ))
    if len(equipos_evento) != 2:
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            f"los eventos mencionan {len(equipos_evento)} equipos distintos en "
            "'team.name'; deben ser exactamente 2",
        ))
    if fuera_campo:
        problemas.append(Problema(
            NIVEL_AVISO, rel,
            f"{fuera_campo} evento(s) con 'location' fuera del campo "
            f"{FIELD_LENGTH:.0f}x{FIELD_WIDTH:.0f}; las metricas por zona (xT, "
            "tercio final, area) asumen ese sistema de coordenadas",
        ))
    if sin_posesion:
        problemas.append(Problema(
            NIVEL_AVISO, rel,
            f"{sin_posesion} evento(s) sin 'possession': quedan fuera de las "
            "metricas de secuencia del equipo (PPDA, pases por secuencia, 10+...)",
        ))
    if tiros_sin_xg:
        problemas.append(Problema(
            NIVEL_AVISO, rel,
            f"{tiros_sin_xg} tiro(s) sin 'shot.statsbomb_xg': contaran como xG 0 "
            "y hundiran xG/npxG/xA del partido",
        ))
    return problemas


def _validar_alineaciones(raiz: Path, p: PartidoPaquete) -> list[Problema]:
    rel = f"lineups/{p.match_id}.json"
    path = raiz / "lineups" / f"{p.match_id}.json"
    if not path.exists():
        return [Problema(
            NIVEL_ERROR, rel,
            "no existe (obligatorio: de aqui salen los minutos y las posiciones)")]
    try:
        equipos = _leer_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [Problema(NIVEL_ERROR, rel, f"JSON ilegible: {exc}")]
    if not isinstance(equipos, list):
        return [Problema(NIVEL_ERROR, rel, "debe ser una LISTA con los dos equipos")]

    problemas: list[Problema] = []
    if len(equipos) != 2:
        problemas.append(Problema(
            NIVEL_ERROR, rel, f"tiene {len(equipos)} equipos; deben ser exactamente 2"))

    del_partido = {
        int(_bajar(p.partido, "home_team.home_team_id") or -1),
        int(_bajar(p.partido, "away_team.away_team_id") or -2),
    }
    ids_alineacion: set[int] = set()
    con_minutos = 0
    posiciones_raras: set[str] = set()

    for k, equipo in enumerate(equipos):
        faltan = _faltantes(equipo, CAMPOS_ALINEACION)
        if faltan:
            problemas.append(Problema(
                NIVEL_ERROR, f"{rel}[{k}]", f"faltan campos obligatorios: {', '.join(faltan)}"))
            continue
        ids_alineacion.add(int(equipo["team_id"]))
        jugadores = equipo.get("lineup") or []
        if not isinstance(jugadores, list):
            problemas.append(Problema(NIVEL_ERROR, f"{rel}[{k}].lineup", "debe ser una lista"))
            continue
        for jugador in jugadores:
            faltan = _faltantes(jugador, CAMPOS_JUGADOR)
            if faltan:
                problemas.append(Problema(
                    NIVEL_ERROR, f"{rel}[{k}].lineup",
                    f"jugador sin campos obligatorios: {', '.join(faltan)}"))
                continue
            spells = jugador.get("positions") or []
            if spells:
                con_minutos += 1
            for spell in spells:
                faltan = _faltantes(spell, CAMPOS_SPELL)
                if faltan:
                    problemas.append(Problema(
                        NIVEL_ERROR, f"{rel}[{k}].lineup",
                        f"spell de posicion sin campos obligatorios: {', '.join(faltan)} "
                        f"(jugador {jugador.get('player_id')})"))
                    continue
                nombre_pos = str(spell["position"])
                if nombre_pos not in POSITIONS_25.values():
                    posiciones_raras.add(nombre_pos)

    if ids_alineacion and ids_alineacion != del_partido:
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            f"los team_id de las alineaciones {sorted(ids_alineacion)} no coinciden "
            f"con los del partido {sorted(del_partido)}",
        ))
    if con_minutos == 0:
        problemas.append(Problema(
            NIVEL_ERROR, rel,
            "ningun jugador tiene tramos en 'positions': el partido no generaria "
            "ninguna fila jugador-partido",
        ))
    if posiciones_raras:
        problemas.append(Problema(
            NIVEL_AVISO, rel,
            "posiciones fuera del catalogo de 25 de StatsBomb "
            f"({', '.join(sorted(posiciones_raras))}): esos minutos no tendran "
            "columna 'pos_*' en el vector del jugador",
        ))
    return problemas


def validar(paquete: Paquete, partido: PartidoPaquete) -> list[Problema]:
    """Valida los eventos y las alineaciones de UN partido del paquete."""
    return _validar_eventos(paquete.raiz, partido) + _validar_alineaciones(paquete.raiz, partido)


def validar_todo(paquete: Paquete) -> dict[int, list[Problema]]:
    """Valida todos los partidos descubiertos. {match_id: problemas}."""
    return {p.match_id: validar(paquete, p) for p in paquete.partidos}
