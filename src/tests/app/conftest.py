"""Fixtures de la app web.

Los artefactos se escriben a mano con `ModeloSimilitud.guardar` en vez de correr
`src.similitud.build`: la app solo consume la capa servible, así que entrenar de
verdad haría las pruebas lentas sin cubrir nada más. El ajuste ya se prueba en
`src/tests/similitud/`.

La BD sintética replica solo las tablas que la app lee (nunca escribe), con el
esquema real de la extracción: ligas, equipos, partidos con su fecha, minutos por
posición y las métricas con las que se reconstruyen los valores reales.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from src.app.factoria import crear_app
from src.extraccion import database
from src.extraccion.config import POSITIONS_25
from src.similitud.modelo import ModeloSimilitud

# Universo sintetico. Hay dos "Garcia" a proposito (para ejercitar la ambiguedad),
# dos entidades con el MISMO nombre (la identidad es el id, no el nombre) y un
# nombre con caracteres fuera de latin-1, como los hay en StatsBomb.
JUGADORES = [
    (10, "Lionel Messi"),
    (20, "Luis Suarez"),
    (30, "Sergio Garcia"),
    (40, "Marc Garcia"),
    (50, "Nombre Repetido"),
    (60, "Nombre Repetido"),
    (70, "Çağlar Söyüncü"),
]
EQUIPOS = [(1, "Barcelona"), (2, "Real Madrid"), (3, "Sevilla")]

# Una metrica por fase de jugador mas una columna de posicion, para que el radar
# tenga todos sus vertices medibles y la posicion se pueda filtrar.
FEATURES = [
    "passes",         # Progresion
    "xa",             # Creacion
    "npxg",           # Finalizacion
    "duels_won",      # Duelos
    "aerial_won",     # Juego aereo
    "interceptions",  # Defensa
    "pos_left_wing",  # bloque de posicion (fuera de las fases)
]
LIGAS = {"jugador": "11-27", "equipo": "7-27"}

# (player_id, match_id, team_id, minutes_played, passes). El 20 cambia de equipo
# a proposito y el 40 no aparece: sin equipo registrado, la interfaz lo omite.
#
# Los `passes` estan elegidos para que el per-90 sea exacto y la media ponderada
# de cada jugador salga redonda (el 20 juega un partido de 45' con la mitad de
# pases, asi que su per-90 es el mismo en los tres): las pruebas de valores
# reales comprueban numeros concretos, no aproximaciones.
MINUTOS_POR_PARTIDO = [
    (10, 1, 1, 90.0, 50.0), (10, 2, 1, 90.0, 60.0), (10, 3, 1, 90.0, 70.0),
    # El 20 se va a otro equipo Y a otra competicion: dos etapas distintas.
    (20, 1, 1, 90.0, 30.0), (20, 2, 1, 90.0, 30.0), (20, 4, 2, 45.0, 15.0),
    (30, 1, 3, 90.0, 20.0),
    (50, 2, 2, 90.0, 10.0),
    # El 60 cambia de equipo DENTRO de la misma competicion: se agrupan juntos.
    (60, 1, 1, 90.0, 12.0), (60, 3, 3, 90.0, 15.0),
    (70, 1, 2, 90.0, 40.0), (70, 2, 2, 90.0, 40.0),
]

# Resto de metricas del modelo sintetico, iguales en todos los partidos de un
# jugador para que su media sea el propio valor. `interceptions` se deja a NULL a
# proposito: es el caso «la BD no tiene ese dato», que la interfaz escribe «—».
METRICAS_JUGADOR = {
    "xa": 0.25, "npxg": 0.5, "duels_won": 4.0, "aerial_won": 2.0,
    "passes_completed": 20.0, "duels_total": 8.0, "take_ons": 2.0,
    "take_ons_won": 1.0, "aerial_lost": 1.0, "np_shots": 2.0,
    "shots_on_target": 1.0, "np_goals": 1.0,
}

# (team_id, match_id, passes, npxg).
PARTIDOS_EQUIPO = [
    (1, 1, 500.0, 1.5), (1, 2, 500.0, 1.5),
    (2, 1, 400.0, 1.0), (2, 4, 400.0, 1.0),
    (3, 2, 300.0, 0.5), (3, 3, 300.0, 0.5),
]

# (match_id, competition_id, season_id, match_date). Las fechas fijan el orden
# cronologico de la trayectoria, y el partido 4 es de OTRA competicion.
PARTIDOS = [
    (1, 11, 27, "2015-09-01"),
    (2, 11, 27, "2015-10-01"),
    (3, 11, 27, "2015-11-01"),
    (4, 7, 27, "2016-02-01"),
]

# (player_id, match_id, position_name, minutes). El 10 es especialista puro y el
# 30 reparte sus minutos, que es lo que separa los extremos del eje de perfil.
POSICIONES = [
    (10, 1, "Right Wing", 90.0), (10, 2, "Right Wing", 90.0),
    (10, 3, "Right Wing", 90.0),
    (20, 1, "Center Forward", 90.0), (20, 2, "Center Forward", 45.0),
    (20, 2, "Left Wing", 45.0), (20, 4, "Left Wing", 45.0),
    (30, 1, "Left Back", 45.0), (30, 1, "Left Wing Back", 45.0),
    (50, 2, "Goalkeeper", 90.0),
    (70, 1, "Center Back", 90.0), (70, 2, "Center Back", 90.0),
]


def _S(n: int) -> np.ndarray:
    """Matriz de similitud simetrica, con diagonal a 0 y sin empates."""
    base = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    S = (base + base.T) / (2.0 * n * n)
    np.fill_diagonal(S, 0.0)
    return S


def _modelo(entidad: str, formulacion: str, normalizacion: str) -> ModeloSimilitud:
    filas = JUGADORES if entidad == "jugador" else EQUIPOS
    ids = np.array([i for i, _ in filas])
    nombres = [n for _, n in filas]
    n = len(filas)
    # feat_display distinto por formulacion para poder distinguir los artefactos.
    display = np.linspace(-2.0, 2.0, n * len(FEATURES)).reshape(n, len(FEATURES))
    return ModeloSimilitud(
        formulacion=formulacion,
        entidad=entidad,
        S=_S(n),
        entity_ids=ids,
        entity_names=nombres,
        feat_names=list(FEATURES),
        feat_display=display * (1.0 if formulacion == "5" else -1.0),
        meta={
            "normalizacion": normalizacion,
            "descripcion": f"modelo sintetico F{formulacion}",
            "ligas_por_entidad": {str(i): [LIGAS[entidad]] for i in ids},
        },
    )


@pytest.fixture(scope="session")
def dir_modelos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directorio con las 8 combinaciones (2 formulaciones x 2 entidades x 2 norm.)."""
    destino = tmp_path_factory.mktemp("modelos")
    for entidad in ("jugador", "equipo"):
        for formulacion in ("2", "5"):
            for normalizacion in ("por_liga", "global"):
                _modelo(entidad, formulacion, normalizacion).guardar(destino)
    return destino


@pytest.fixture(scope="session")
def bd(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BD sintetica con el esquema REAL de la extraccion.

    El esquema lo crea `src.extraccion.database.init_schema` en vez de escribirlo
    a mano: la app reconstruye los valores reales de las metricas con la misma
    cadena que el pipeline (`similitud.data` -> `similitud.features.derivar`), asi
    que necesita la BD tal cual es. Una copia a mano se desincronizaria en cuanto
    cambiase una columna, y la prueba pasaria comprobando otra cosa.
    """
    ruta = tmp_path_factory.mktemp("bd") / "scouting.db"
    with closing(database.connect(ruta)) as conn:
        database.init_schema(conn)
        for comp in (
            {"competition_id": 11, "competition_name": "La Liga",
             "country_name": "Spain", "competition_gender": "male"},
            {"competition_id": 7, "competition_name": "Ligue 1",
             "country_name": "France", "competition_gender": "male"},
        ):
            database.upsert_competition(conn, comp)
        database.upsert_season(conn, {"season_id": 27, "season_name": "2015/2016"})
        for team_id, nombre in EQUIPOS:
            database.upsert_team(conn, team_id, nombre)
        for player_id, nombre in JUGADORES:
            database.upsert_player(conn, player_id, nombre)
        for match_id, competition_id, season_id, fecha in PARTIDOS:
            database.upsert_match(conn, {
                "match_id": match_id, "competition_id": competition_id,
                "season_id": season_id, "match_date": fecha,
            })
        for player_id, match_id, team_id, minutos, passes in MINUTOS_POR_PARTIDO:
            database.upsert_player_stats(conn, {
                "player_id": player_id, "match_id": match_id, "team_id": team_id,
                "minutes_played": minutos, "passes": passes,
                # Escalado a los minutos jugados, para que el per-90 sea estable.
                **{k: v * minutos / 90.0 for k, v in METRICAS_JUGADOR.items()},
            })
        for team_id, match_id, passes, npxg in PARTIDOS_EQUIPO:
            database.upsert_team_stats(conn, {
                "team_id": team_id, "match_id": match_id,
                "passes": passes, "npxg": npxg,
            })
        # `position_id` es parte de la clave primaria, asi que tiene que ser el
        # del catalogo: cualquier invento colisionaria entre dos posiciones del
        # mismo (jugador, partido) y tumbaria la insercion.
        id_por_posicion = {n: i for i, n in POSITIONS_25.items()}
        for player_id, match_id, posicion, minutos in POSICIONES:
            conn.execute(
                "INSERT INTO player_match_positions "
                "(player_id, match_id, position_id, position_name, minutes) "
                "VALUES (?, ?, ?, ?, ?)",
                (player_id, match_id, id_por_posicion[posicion], posicion, minutos),
            )
        conn.commit()
    return ruta


@pytest.fixture(scope="session")
def bd_ligas(bd: Path) -> Path:
    """Alias historico para las pruebas que solo miran la traduccion de ligas."""
    return bd


@pytest.fixture
def app(dir_modelos: Path, bd: Path):
    return crear_app(dir_modelos, db_path=bd, testing=True)


@pytest.fixture
def cliente(app):
    return app.test_client()


@pytest.fixture
def app_sin_bd(dir_modelos: Path, tmp_path: Path):
    """La BD es opcional: la app tiene que servir igual sin ella."""
    return crear_app(dir_modelos, db_path=tmp_path / "no_existe.db", testing=True)


@pytest.fixture
def cliente_sin_bd(app_sin_bd):
    return app_sin_bd.test_client()


@pytest.fixture
def modelo_jugador():
    """Modelo de jugadores en memoria, para probar `servicio` sin disco."""
    return _modelo("jugador", "5", "por_liga")


@pytest.fixture
def fases_jugador(modelo_jugador):
    """Perfil por fases del modelo de jugadores en memoria."""
    from src.app.fases import construir_matriz

    return construir_matriz(
        modelo_jugador.entidad, modelo_jugador.feat_names, modelo_jugador.feat_display
    )
