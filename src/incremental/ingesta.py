"""Punto de entrada: incorpora un paquete de partidos nuevos a la BD.

    python -m src.incremental.ingesta --paquete DIR [--db RUTA]
                                      [--partidos 3890561,3890505]
                                      [--solo-validar] [--omitir-invalidos]
                                      [--sin-copia]

El paquete es un directorio con formato StatsBomb (ver `paquete.py` y
`docs/incremental.md`). El calculo de metricas NO se reimplementa aqui: se llama
a `src.extraccion.extract.process_match`, el mismo que produjo lo que ya hay en
la BD, para que las filas nuevas sean comparables con las viejas.

Antes de escribir se hace una copia de seguridad de la BD (salvo `--sin-copia`)
y se avisa de los choques de identidad: un `player_id` o `team_id` que ya existe
con OTRO nombre casi siempre significa que el paquete inventa ids en vez de usar
los de StatsBomb, y el UPSERT sobreescribiria a la entidad equivocada.

Deja un `ingesta.json` junto a la BD con las entidades nuevas; `reentrenar` lo
usa solo para informar (no depende de el: recalcula lo que hay en la BD).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from src.extraccion import database as db
from src.extraccion.extract import process_match
from src.similitud.consulta import configurar_consola

from . import config, paquete as pq

configurar_consola()


# --- Fotografia de la BD (para saber que es nuevo) ---------------------------
def _foto(db_path: Path) -> dict[str, Any]:
    """Ids ya presentes en la BD. BD inexistente -> todo vacio."""
    vacia = {"jugadores": {}, "equipos": {}, "partidos": set(), "ligas": set()}
    if not Path(db_path).exists():
        return vacia
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        try:
            jugadores = {int(i): str(n) for i, n in
                         conn.execute("SELECT player_id, player_name FROM players")}
            equipos = {int(i): str(n) for i, n in
                       conn.execute("SELECT team_id, team_name FROM teams")}
            partidos = {int(i) for (i,) in conn.execute("SELECT match_id FROM matches")}
            ligas = {(int(c), int(s)) for c, s in
                     conn.execute("SELECT DISTINCT competition_id, season_id FROM matches")}
        except sqlite3.OperationalError:
            return vacia  # BD sin esquema todavia
    return {"jugadores": jugadores, "equipos": equipos, "partidos": partidos, "ligas": ligas}


def _choques(antes: dict[str, Any], paquete: pq.Paquete) -> list[str]:
    """Ids de equipo del paquete que en la BD tienen otro nombre.

    Solo se puede comprobar con los equipos: los jugadores no aparecen en
    `matches/` (habria que abrir cada lineup, que es justo lo que se hace luego
    al procesar). Es la senal barata de "estos ids no son los de StatsBomb".
    """
    avisos: list[str] = []
    for p in paquete.partidos:
        for lado in ("home", "away"):
            equipo = p.partido.get(f"{lado}_team") or {}
            tid = equipo.get(f"{lado}_team_id")
            nombre = equipo.get(f"{lado}_team_name")
            if tid is None:
                continue
            previo = antes["equipos"].get(int(tid))
            if previo is not None and previo != nombre:
                avisos.append(
                    f"team_id {tid} ya existe en la BD como {previo!r} y el paquete "
                    f"lo llama {nombre!r}"
                )
    return sorted(set(avisos))


# --- Copia de seguridad -------------------------------------------------------
def copia_seguridad(db_path: Path, dir_copias: Path = config.DIR_COPIAS) -> Path | None:
    """Copia la BD antes de tocarla. Devuelve la ruta de la copia (None si no habia BD)."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    dir_copias = Path(dir_copias)
    dir_copias.mkdir(parents=True, exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d-%H%M%S")
    destino = dir_copias / f"{db_path.stem}.bak-{sello}{db_path.suffix}"
    shutil.copy2(db_path, destino)
    return destino


# --- Ingesta ------------------------------------------------------------------
def ingerir(
    paquete: pq.Paquete,
    partidos: list[pq.PartidoPaquete],
    db_path: Path,
) -> dict[str, Any]:
    """Procesa `partidos` contra la BD y devuelve el informe de lo incorporado."""
    antes = _foto(db_path)
    t0 = time.perf_counter()

    conn = db.connect(Path(db_path))
    db.init_schema(conn)
    for comp in paquete.competiciones():
        db.upsert_competition(conn, {
            "competition_id": int(comp["competition_id"]),
            "competition_name": comp["competition_name"],
            "country_name": comp.get("country_name"),
            "competition_gender": comp.get("competition_gender"),
        })
        db.upsert_season(conn, {
            "season_id": int(comp["season_id"]),
            "season_name": comp["season_name"],
        })
    conn.commit()

    filas_jugador = filas_equipo = 0
    for i, p in enumerate(partidos, start=1):
        nj, ne = process_match(
            conn, p.partido,
            int(p.competicion["competition_id"]), int(p.competicion["season_id"]),
            paquete.raiz,
        )
        filas_jugador += nj
        filas_equipo += ne
        estado = "actualiza" if p.match_id in antes["partidos"] else "nuevo"
        print(f"  [{i}/{len(partidos)}] {p.match_id} {p.etiqueta} — {nj} jugadores ({estado})")
    conn.close()

    despues = _foto(db_path)
    nuevos_jugadores = sorted(set(despues["jugadores"]) - set(antes["jugadores"]))
    nuevos_equipos = sorted(set(despues["equipos"]) - set(antes["equipos"]))
    nuevas_ligas = sorted(despues["ligas"] - antes["ligas"])
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "paquete": str(paquete.raiz),
        "db": str(db_path),
        "segundos": round(time.perf_counter() - t0, 1),
        "partidos_procesados": len(partidos),
        "partidos_nuevos": sorted(set(despues["partidos"]) - antes["partidos"]),
        "partidos_actualizados": sorted(
            p.match_id for p in partidos if p.match_id in antes["partidos"]),
        "filas_jugador_partido": filas_jugador,
        "filas_equipo_partido": filas_equipo,
        "jugadores_nuevos": [
            {"player_id": i, "player_name": despues["jugadores"][i]} for i in nuevos_jugadores],
        "equipos_nuevos": [
            {"team_id": i, "team_name": despues["equipos"][i]} for i in nuevos_equipos],
        "ligas_nuevas": [
            {"competition_id": c, "season_id": s} for c, s in nuevas_ligas],
        "totales_bd": {
            "jugadores": len(despues["jugadores"]),
            "equipos": len(despues["equipos"]),
            "partidos": len(despues["partidos"]),
        },
    }


# --- CLI ----------------------------------------------------------------------
def _imprimir_problemas(titulo: str, problemas: list[pq.Problema], limite: int = 20) -> None:
    if not problemas:
        return
    print(f"\n{titulo} ({len(problemas)}):")
    for p in problemas[:limite]:
        print(f"  {p}")
    if len(problemas) > limite:
        print(f"  ... y {len(problemas) - limite} mas")


def _ids(texto: str | None) -> list[int] | None:
    if not texto:
        return None
    return [int(t) for t in texto.replace(";", ",").split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Incorpora partidos nuevos a la BD de scouting.")
    p.add_argument("--paquete", type=Path, required=True,
                   help="Directorio con formato StatsBomb (competitions.json, matches/, "
                        "events/, lineups/).")
    p.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH,
                   help="BD SQLite destino (se crea si no existe).")
    p.add_argument("--partidos", default=None,
                   help="Lista de match_id separados por comas; por defecto, todos los "
                        "del paquete.")
    p.add_argument("--competicion", default=None,
                   help='Filtra por nombre de competicion (p. ej. "1. Bundesliga").')
    p.add_argument("--temporada", default=None,
                   help='Filtra por nombre de temporada (p. ej. "2015/2016").')
    p.add_argument("--solo-validar", action="store_true",
                   help="Valida el paquete y sale sin escribir nada.")
    p.add_argument("--omitir-invalidos", action="store_true",
                   help="Procesa los partidos validos en vez de abortar por los que "
                        "tienen errores.")
    p.add_argument("--sin-copia", action="store_true",
                   help="No hace copia de seguridad de la BD antes de escribir.")
    args = p.parse_args(argv)

    paquete = pq.leer(args.paquete, ids=_ids(args.partidos),
                      competicion=args.competicion, temporada=args.temporada)
    _imprimir_problemas("Problemas de estructura", paquete.problemas)
    if paquete.errores:
        print("\nEl paquete no se puede leer. Corrige los errores de estructura.")
        return 2
    if not paquete.partidos:
        print("\nEl paquete no contiene ningun partido.")
        return 2

    print(f"\nPaquete: {paquete.raiz} — {len(paquete.partidos)} partido(s) en "
          + ", ".join(f'{c["competition_name"]} {c["season_name"]}'
                      for c in paquete.competiciones()))

    print("Validando eventos y alineaciones...")
    por_partido = pq.validar_todo(paquete)
    invalidos = {mid for mid, probs in por_partido.items()
                 if any(x.nivel == pq.NIVEL_ERROR for x in probs)}
    todos = [x for probs in por_partido.values() for x in probs]
    _imprimir_problemas("Problemas de contenido", todos)

    antes = _foto(args.db)
    for aviso in _choques(antes, paquete):
        print(f"  [aviso] {aviso}")
    ya_estaban = [p.match_id for p in paquete.partidos if p.match_id in antes["partidos"]]
    if ya_estaban:
        print(f"  [aviso] {len(ya_estaban)} partido(s) ya estaban en la BD: se "
              "reprocesan y se sobreescriben (UPSERT), no se duplican")

    if args.solo_validar:
        print(f"\nValidacion: {len(paquete.partidos) - len(invalidos)} partido(s) "
              f"procesables, {len(invalidos)} con errores.")
        return 1 if invalidos else 0

    procesables = [p for p in paquete.partidos if p.match_id not in invalidos]
    if invalidos and not args.omitir_invalidos:
        print(f"\n{len(invalidos)} partido(s) con errores. Corrigelos o repite con "
              "--omitir-invalidos para procesar solo los validos.")
        return 2
    if not procesables:
        print("\nNingun partido procesable.")
        return 2

    if not args.sin_copia:
        copia = copia_seguridad(args.db)
        print(f"\nCopia de seguridad: {copia}" if copia
              else "\nNo habia BD previa: no hay copia que hacer.")

    print(f"\nIncorporando {len(procesables)} partido(s) a {args.db}")
    informe = ingerir(paquete, procesables, args.db)

    destino = Path(args.db).parent / config.NOMBRE_INFORME
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nHecho en {informe['segundos']}s: "
          f"{informe['filas_jugador_partido']} filas jugador-partido, "
          f"{informe['filas_equipo_partido']} equipo-partido.")
    print(f"  partidos nuevos: {len(informe['partidos_nuevos'])} "
          f"(actualizados: {len(informe['partidos_actualizados'])})")
    print(f"  jugadores nuevos: {len(informe['jugadores_nuevos'])}")
    print(f"  equipos nuevos: {len(informe['equipos_nuevos'])}")
    print(f"  ligas nuevas: {len(informe['ligas_nuevas'])}")
    print(f"  informe: {destino}")
    print("\nSiguiente paso: python -m src.incremental.reentrenar "
          f"--db {args.db}   (incluye lo nuevo en las recomendaciones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
