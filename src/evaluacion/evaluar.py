"""Punto de entrada: evalua TODOS los modelos de `outputs/modelo/` segun el PDF.

    python -m src.evaluacion.evaluar [--db RUTA] [--model-dir DIR] [--out DIR]
                                     [--fases 0,1,2,3,5] [--bootstrap B]

Recorre la rejilla (formulacion 2/5 x jugador/equipo x por_liga/global), carga
cada artefacto servible y ejecuta el protocolo priorizado de
`docs/Como_evaluar.pdf`. Para la auto-similitud y la estabilidad construye los
modelos "que no existen" (mitades pares/impares y remuestreos bootstrap) sobre la
marcha, reutilizando el pipeline real de `src.similitud` (reconstruye los
artefactos guardados de forma exacta, verificado). Escribe CSV + figuras + un
informe en `outputs/evaluacion/`.

Fase 6 (validacion humana con scouts) no es automatizable: se documenta en el
informe como paso pendiente segun el propio protocolo.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.similitud.consulta import configurar_consola
from src.similitud.modelo import cargar_modelo

from . import config, construccion, datos, fases

configurar_consola()

matplotlib = None  # se importa perezosamente solo si se generan figuras


# --------------------------------------------------------------------------- #
# Progreso y tiempos                                                            #
# --------------------------------------------------------------------------- #

def _dur(segundos: float) -> str:
    """Formatea una duracion en segundos como '45s', '2m 03s' o '1h 05m'."""
    segundos = max(0.0, float(segundos))
    if segundos < 60:
        return f"{segundos:.0f}s"
    m, s = divmod(int(round(segundos)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Progreso:
    """Contador de pasos que imprime, por cada paso, cuanto tardo, cuanto se
    lleva acumulado y una estimacion (ETA) del tiempo restante.

    La ETA es una extrapolacion simple (ritmo medio por paso x pasos que
    faltan): orientativa, no exacta, porque los pasos no cuestan lo mismo (el
    bootstrap de la Fase 2 domina). Sirve para saber si quedan segundos o
    minutos, que es lo que se pide.
    """

    def __init__(self, total: int):
        self.total = max(int(total), 1)
        self.hecho = 0
        self.t0 = time.perf_counter()

    @contextmanager
    def paso(self, desc: str):
        self.hecho += 1
        idx = self.hecho
        print(f"[{idx}/{self.total}] {desc} ...", flush=True)
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            transcurrido = time.perf_counter() - self.t0
            restantes = self.total - self.hecho
            linea = f"      hecho en {_dur(dt)} | transcurrido {_dur(transcurrido)}"
            if restantes > 0:
                eta = (transcurrido / self.hecho) * restantes
                linea += f" | quedan ~{_dur(eta)} ({restantes} pasos)"
            else:
                linea += " | completado"
            print(linea, flush=True)


@contextmanager
def _quizas_paso(prog: "Progreso | None", desc: str):
    """Usa el contador si hay uno; si no (p. ej. llamadas sueltas), solo imprime."""
    if prog is None:
        print(f"{desc} ...", flush=True)
        yield
    else:
        with prog.paso(desc):
            yield


class _SubProgreso:
    """Reportero del bucle interno de un paso (p. ej. las B iteraciones del
    bootstrap de la Fase 2). Refresca una unica linea en sitio con ``\\r``,
    mostrando el numero de iteracion intermedia, su tiempo y su ETA propia.

    Se refresca como mucho cada ``cada`` segundos para no saturar la consola;
    la ultima iteracion siempre se imprime y cierra la linea con un salto.
    """

    def __init__(self, etiqueta: str, cada: float = 0.5):
        self.etiqueta = etiqueta
        self.cada = cada
        self.t0 = time.perf_counter()
        self._ultimo = 0.0

    def __call__(self, hecho: int, total: int) -> None:
        ahora = time.perf_counter()
        if hecho < total and (ahora - self._ultimo) < self.cada:
            return
        self._ultimo = ahora
        transcurrido = ahora - self.t0
        eta = (transcurrido / hecho) * (total - hecho) if hecho else 0.0
        fin = "\n" if hecho >= total else ""
        print(f"\r        - {self.etiqueta} {hecho}/{total} | "
              f"{_dur(transcurrido)} | ETA ~{_dur(eta)}      ",
              end=fin, flush=True)


# --------------------------------------------------------------------------- #
# Carga de modelos y contextos                                                 #
# --------------------------------------------------------------------------- #

def cargar_modelos(
    model_dir: Path,
    formulaciones: tuple[str, ...] = config.FORMULACIONES,
    entidades: tuple[str, ...] = config.ENTIDADES,
) -> dict:
    """Carga los artefactos presentes en la rejilla (clave = tripleta).

    `formulaciones`/`entidades` acotan la rejilla; por defecto, entera. Acotarla
    evita avisar de modelos que no se pidieron.
    """
    modelos: dict = {}
    for form in formulaciones:
        for entidad in entidades:
            for norm in config.NORMALIZACIONES:
                try:
                    modelos[(form, entidad, norm)] = cargar_modelo(
                        model_dir, form, entidad, norm
                    )
                except FileNotFoundError:
                    print(f"  [aviso] falta el modelo F{form} {entidad} {norm}")
    return modelos


def crear_contextos(
    db_path: Path,
    prog: "Progreso | None" = None,
    entidades: tuple[str, ...] = config.ENTIDADES,
) -> dict:
    """Un contexto por (entidad, normalizacion); reutilizado por F2 y F5.

    Es el arranque mas caro (reconstruye la parte pesada del pipeline: la W de
    la F2 y el mapa RFF de la F5), por eso se cronometra por contexto y por eso
    `entidades` permite no montar el de una entidad que no se va a evaluar.
    """
    ctxs: dict = {}
    for entidad in entidades:
        for norm in config.NORMALIZACIONES:
            with _quizas_paso(prog, f"[setup] contexto {entidad}/{norm}"):
                ctxs[(entidad, norm)] = construccion.crear_contexto(db_path, entidad, norm)
    return ctxs


# --------------------------------------------------------------------------- #
# Ejecucion de las fases                                                        #
# --------------------------------------------------------------------------- #

def _clave(form, entidad, norm):
    return f"F{form}_{entidad}_{norm}"


def _pasos_totales(n_modelos: int, fases_sel: set[str], bootstrap: int) -> int:
    """Cuantos pasos cronometrados hara `ejecutar` (para el contador y la ETA)."""
    por_combo = (
        ("0" in fases_sel)
        + ("1" in fases_sel)
        + ("2" in fases_sel and bootstrap > 0)
        + ("5" in fases_sel)
    )
    return (1 if "3" in fases_sel else 0) + n_modelos * por_combo


def ejecutar(
    modelos: dict, ctxs: dict, roles: dict, minutos: dict,
    fases_sel: set[str], bootstrap: int, prog: "Progreso | None" = None,
) -> dict:
    """Corre las fases seleccionadas y devuelve tablas (listas de dicts)."""
    salida = {k: [] for k in
              ("f0", "f1", "f1_estratos", "f2", "f3", "f4", "f5", "face")}

    combos = [(f, e, n) for f in config.FORMULACIONES
              for e in config.ENTIDADES for n in config.NORMALIZACIONES
              if (f, e, n) in modelos]

    if "3" in fases_sel:
        with _quizas_paso(prog, "[Fase 3] triangulacion (test de Mantel)"):
            rep = _SubProgreso("comparacion") if prog is not None else None
            salida["f3"] = fases.fase3_triangulacion(modelos, ctxs, rep)

    for i, (form, entidad, norm) in enumerate(combos, start=1):
        modelo = modelos[(form, entidad, norm)]
        ctx = ctxs[(entidad, norm)]
        etiqueta = _clave(form, entidad, norm)
        suf = f"{etiqueta} (modelo {i}/{len(combos)})"
        base = {"modelo": etiqueta, "formulacion": form,
                "entidad": entidad, "normalizacion": norm}

        if "0" in fases_sel:
            with _quizas_paso(prog, f"[Fase 0] sanity {suf}"):
                rep = _SubProgreso("entidad") if prog is not None else None
                salida["f0"].append(
                    {**base, **fases.fase0_sanity(modelo, roles, ctx, form, rep)})
                if norm == "por_liga":
                    nombres = _queries_face(modelo, minutos)
                    salida["face"].append(
                        {"modelo": etiqueta,
                         "lineas": fases.face_validity(modelo, nombres)}
                    )

        if "1" in fases_sel:
            with _quizas_paso(prog, f"[Fase 1] auto-similitud {suf}"):
                etiq_sub = "SLIM obs" if form == "2" else "OT nube"
                rep = _SubProgreso(etiq_sub) if prog is not None else None
                res = fases.fase1_autosimilitud(ctx, form, rep)
                salida["f1"].append({**base, "direccion": "global", **res["global"]})
                salida["f1"].append({**base, "direccion": "A->B", **res["A->B"]})
                salida["f1"].append({**base, "direccion": "B->A", **res["B->A"]})
                for nombre, m in res["estratos"].items():
                    salida["f1_estratos"].append({**base, "estrato": nombre, **m})
                if res["denoising"] is not None:
                    d = res["denoising"]
                    salida["f4"].append({
                        **base,
                        "mrr_pre": d["pre"]["mrr"], "top1_pre": d["pre"]["top1"],
                        "mrr_post": d["post"]["mrr"], "top1_post": d["post"]["top1"],
                        "delta_mrr": d["delta_mrr"],
                        "delta_rr_medio": d["delta_rr_medio"], "p_pareado": d["p_pareado"],
                    })

        if "2" in fases_sel and bootstrap > 0:
            with _quizas_paso(prog, f"[Fase 2] estabilidad {suf} (B={bootstrap})"):
                reportar = _SubProgreso("bootstrap") if prog is not None else None
                salida["f2"].append({
                    **base,
                    **fases.fase2_estabilidad(ctx, form, modelo.S, bootstrap, reportar),
                })

        if "5" in fases_sel:
            with _quizas_paso(prog, f"[Fase 5] downstream {suf}"):
                rep = _SubProgreso("entidad") if prog is not None else None
                salida["f5"].append(
                    {**base, **fases.fase5_downstream(modelo, roles, minutos, rep)}
                )

    return salida


def _queries_face(modelo, minutos: dict, n: int = 6) -> list[str]:
    """Nombres para la face validity: entidades con mas volumen (jugador)."""
    if modelo.entidad == "jugador":
        vol = [(minutos.get(int(i), 0.0), nm)
               for i, nm in zip(modelo.entity_ids, modelo.entity_names)]
        vol.sort(reverse=True)
        return [nm for _, nm in vol[:n]]
    return list(modelo.entity_names[:n])


# --------------------------------------------------------------------------- #
# Escritura de resultados                                                       #
# --------------------------------------------------------------------------- #

def escribir_csv(salida: dict, out_dir: Path, extra: dict | None = None) -> None:
    """Vuelca cada fase a su CSV. `extra` añade columnas fijas al principio de
    todas las filas (el barrido mete ahi la variante: sin eso, los CSV de dos
    variantes son indistinguibles salvo por la carpeta que los contiene).
    """
    tablas = {
        "fase0_sanity": salida["f0"],
        "fase1_autosimilitud": salida["f1"],
        "fase1_estratos_minutos": salida["f1_estratos"],
        "fase2_estabilidad": salida["f2"],
        "fase3_triangulacion": salida["f3"],
        "fase4_denoising": salida["f4"],
        "fase5_downstream": salida["f5"],
    }
    for nombre, filas in tablas.items():
        if filas:
            if extra:
                filas = [{**extra, **fila} for fila in filas]
            pd.DataFrame(filas).to_csv(out_dir / f"{nombre}.csv", index=False)


# --------------------------------------------------------------------------- #
# Identificacion del modelo en los informes                                     #
# --------------------------------------------------------------------------- #
# La etiqueta compacta (`F2_jugador_global`) no dice de que eje viene cada
# palabra, y la MISMA etiqueta designa modelos distintos si cambia el espacio de
# features (con o sin posicion). Los informes abren cada fila con los tres ejes y
# declaran el espacio de features de los artefactos que se cargaron.

COLUMNAS_ID = ("formulacion", "entidad", "normalizacion")

LEYENDA_EJES = [
    "- **formulacion** — `F2`: SLIM instancia-instancia sobre las observaciones "
    "+ agregacion de la W a nivel entidad. `F5`: similitud distribucional entre "
    "las nubes de observaciones (MMD en jugador, Sinkhorn en equipo) + EASE de "
    "re-ranking.",
    "- **entidad** — `jugador` o `equipo`. No son comparables entre si: distintas "
    "features (el jugador va per-90 y ponderado por minutos), distinto pool y "
    "distinto metodo en la F5.",
    "- **normalizacion** — `por_liga`: z-score dentro de cada "
    "competicion-temporada. `global`: z-score sobre todo el dataset (habilita "
    "comparar entre ligas).",
]


def _cab_id(*extra: str) -> tuple[str, str]:
    """Cabecera y separador de una tabla que empieza por los ejes del modelo."""
    cols = [*COLUMNAS_ID, *extra]
    return "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)


def _id(r: dict) -> str:
    """Celdas de identificacion (ya con la barra final) de una fila-modelo."""
    return f"| F{r['formulacion']} | {r['entidad']} | {r['normalizacion']} |"


def resumen_features(modelos: dict) -> list[str]:
    """Una linea por entidad con la dimension del vector y si lleva posicion.

    Es lo unico que distingue a dos artefactos homonimos construidos con
    `USE_POSITION_FEATURES` distinto, asi que el informe lo declara en la
    cabecera en vez de dejarlo implicito en el nombre del fichero.
    """
    lineas = []
    for entidad in config.ENTIDADES:
        modelo = next(
            (m for clave, m in sorted(modelos.items()) if clave[1] == entidad), None
        )
        if modelo is None:
            continue
        n_pos = sum(1 for c in modelo.feat_names if str(c).startswith("pos_"))
        detalle = f"{len(modelo.feat_names)} features"
        detalle += (f", {n_pos} de ellas de posicion (`pos_*`)" if n_pos
                    else ", sin bloque de posicion")
        lineas.append(f"- **{entidad}**: {detalle}.")
    return lineas


def _fmt(x, dec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{dec}f}"
    return str(x)


def escribir_informe(salida: dict, out_dir: Path, meta: dict) -> None:
    """Genera `resumen.md`: tablas + veredicto orientativo por fase."""
    L: list[str] = []
    L.append("# Evaluacion de los modelos de similitud (sin ground truth)\n")
    L.append(f"Generado el {meta['fecha']}. Protocolo: `docs/Como_evaluar.pdf`.\n")
    L.append(f"Modelos evaluados: {meta['n_modelos']}. "
             f"Bootstrap B={meta['bootstrap']}. "
             f"Split de auto-similitud: partidos **pares vs impares** por entidad.\n")
    L.append("> Los umbrales citados son referencias de la literatura "
             "(Decroos & Davis 2019, SoccerMix 2020, Webber et al. 2010), **no "
             "constantes universales**: dependen del tamano del pool, criterio de "
             "minutos y liga. La validez se juzga por **convergencia** de senales, "
             "no por una metrica aislada.\n")

    L.append("## Que modelo es cada uno\n")
    L.append("Las tablas abren cada fila con los tres ejes que identifican al "
             "modelo (la etiqueta compacta seria `F<formulacion>_<entidad>_"
             "<normalizacion>`):\n")
    L.extend(LEYENDA_EJES)
    if meta.get("features"):
        L.append("\nEspacio de features de los artefactos evaluados (no va en el "
                 "nombre del modelo, pero forma parte de su identidad):\n")
        L.extend(meta["features"])
    L.append("")

    # Fase 0.
    if salida["f0"]:
        L.append("## Fase 0 - Sanity checks\n")
        L.append(f"Umbrales: pureza posicional del top-1 > {config.PUREZA_MIN:.0%}; "
                 f"asimetria de la S **cruda** (antes de simetrizar) < "
                 f"{config.ASIMETRIA_ALTA}. La asimetria se mide sobre la S "
                 "direccional que el pipeline corrige (la W de la F2 / el denoising "
                 "de EASE de la F5); el artefacto servible ya se simetriza, asi que "
                 "su asimetria seria ~0 y no diagnosticaria nada.\n")
        L.extend(_cab_id("asimetria cruda \\|\\|S-S^T\\|\\|/\\|\\|S\\|\\|",
                         "pureza top-1", "n"))
        for r in salida["f0"]:
            L.append(f"{_id(r)} {_fmt(r['asimetria'])} | "
                     f"{_fmt(r['pureza_top1'])} | {r['n_pureza']} |")
        L.append("")
        peor = min((r for r in salida["f0"] if np.isfinite(r["pureza_top1"])),
                   key=lambda r: r["pureza_top1"], default=None)
        if peor:
            ok = peor["pureza_top1"] >= config.PUREZA_MIN
            L.append(f"**Veredicto pureza:** minima {peor['pureza_top1']:.1%} "
                     f"({peor['modelo']}) -> "
                     f"{'PASA' if ok else 'REVISAR'} el umbral de "
                     f"{config.PUREZA_MIN:.0%}.")
        peor_a = max(salida["f0"], key=lambda r: r["asimetria"], default=None)
        if peor_a:
            ok_a = peor_a["asimetria"] < config.ASIMETRIA_ALTA
            L.append(f"**Veredicto asimetria:** maxima {_fmt(peor_a['asimetria'])} "
                     f"({peor_a['modelo']}) -> "
                     f"{'PASA' if ok_a else 'REVISAR'} el umbral de "
                     f"{config.ASIMETRIA_ALTA} (simetrizar es razonable si es baja).\n")

    # Fase 1.
    f1g = [r for r in salida["f1"] if r["direccion"] == "global"]
    if f1g:
        L.append("## Fase 1 - Auto-similitud (el pilar)\n")
        L.append("Cada entidad se desdobla en mitades por partidos pares/impares; "
                 "se mide si una mitad recupera a la otra (media de A->B y B->A). "
                 "Se reporta la tasa de azar (1/pool). El split cronologico t vs "
                 "t+1 del PDF no aplica: la BD es de una unica temporada, asi que "
                 "el split intra-temporada pares/impares es la unica variante "
                 "posible (el PDF lo contempla para pocas temporadas).\n")
        L.extend(_cab_id("pool", "azar top-1", "top-1", "top-5", "top-10",
                         "MRR", "x azar"))
        for r in f1g:
            veces = r["top1"] / r["azar_top1"] if r["azar_top1"] else float("nan")
            L.append(f"{_id(r)} {r['pool']} | {_fmt(r['azar_top1'],4)} | "
                     f"{_fmt(r['top1'])} | {_fmt(r['top5'])} | {_fmt(r['top10'])} | "
                     f"{_fmt(r['mrr'])} | {_fmt(veces,1)}x |")
        L.append("")
        L.append("Referencias del PDF (distinto pool, no comparables directamente): "
                 "Player Vectors top-1 ~38% (pool ~741), SoccerMix ~48% (pool ~193). "
                 "Cualquier top-1 > 20-30x el azar es senal fuerte.\n")

        if salida["f1_estratos"]:
            L.append("### Estratificado por minutos (jugador)\n")
            L.extend(_cab_id("estrato", "top-1", "top-5", "MRR"))
            for r in salida["f1_estratos"]:
                L.append(f"{_id(r)} {r['estrato']} | {_fmt(r['top1'])} | "
                         f"{_fmt(r['top5'])} | {_fmt(r['mrr'])} |")
            L.append("\nEl sesgo de popularidad/MNAR sobreestima el desempeno en "
                     "jugadores de alto volumen (cabeza): comparar estratos.\n")

    # Fase 4.
    if salida["f4"]:
        L.append("## Fase 4 - Aporte del denoising (F5: EASE)\n")
        L.append("Auto-similitud PRE-EASE (similitud distribucional cruda) vs "
                 "POST-EASE. Regla: adoptar EASE solo si mejora significativamente "
                 "(p<0.05) sin degradar.\n")
        L.extend(_cab_id("MRR pre", "MRR post", "delta MRR", "delta 1/rango", "p"))
        for r in salida["f4"]:
            L.append(f"{_id(r)} {_fmt(r['mrr_pre'])} | {_fmt(r['mrr_post'])} | "
                     f"{_fmt(r['delta_mrr'])} | {_fmt(r['delta_rr_medio'])} | "
                     f"{_fmt(r['p_pareado'])} |")
        L.append("")

    # Fase 2.
    if salida["f2"]:
        L.append("## Fase 2 - Estabilidad (bootstrap + RBO@10)\n")
        L.append(f"RBO@10 (p={config.RBO_P}) medio del top-k ante remuestreo de "
                 f"partidos. Umbral: >{config.RBO_ESTABLE} estable, "
                 f"<{config.RBO_PREOCUPANTE} preocupante. El bootstrap reutiliza la "
                 "estructura aprendida (W/embedding) y perturba la composicion de "
                 "partidos (ver limitaciones en el modulo).\n")
        L.extend(_cab_id("RBO@10 medio", "IC 95%", "n consultables", "B"))
        for r in salida["f2"]:
            L.append(f"{_id(r)} {_fmt(r['rbo_medio'])} | "
                     f"[{_fmt(r['ic_bajo'])}, {_fmt(r['ic_alto'])}] | "
                     f"{r['n_consultables']} | {r['B']} |")
        L.append("")

    # Fase 3.
    if salida["f3"]:
        L.append("## Fase 3 - Triangulacion (test de Mantel)\n")
        L.append(f"r de Spearman entre matrices S (>{config.MANTEL_R_CONVERGE} "
                 "sugiere misma senal). Se prioriza la magnitud de r sobre p "
                 "(el test infla la significancia ante autocorrelacion posicional). "
                 "Incluye las dos validaciones internas del PDF: la agregacion de "
                 "la F2 vs la S distribucional cruda (¿preserva la distribucion?) y "
                 "la F5 pre vs post-EASE (¿el denoising conserva la estructura?). La "
                 "Kendall tau (solo F2 vs F5) es el acuerdo medio del ranking de "
                 "vecinos POR entidad, complementario al acuerdo global de r.\n")
        L.append("| comparacion | entidad | n entidades | r | p | Kendall t |")
        L.append("|---|---|---|---|---|---|")
        for r in salida["f3"]:
            L.append(f"| {r['comparacion']} | {r['entidad']} | "
                     f"{r.get('n_entidades','')} | {_fmt(r['r'])} | {_fmt(r['p'],4)} | "
                     f"{_fmt(r.get('kendall', float('nan')))} |")
        L.append("")

    # Fase 5.
    if salida["f5"]:
        L.append("## Fase 5 - Downstream + beyond-accuracy\n")
        L.append("Clasificacion posicional por k-NN (metrica objetiva de si S "
                 "captura rol/estilo) + cobertura, diversidad y sesgo de popularidad.\n")
        L.extend(_cab_id("kNN acc", "kNN F1", "coverage", "diversity",
                         "pop. Spearman"))
        for r in salida["f5"]:
            L.append(f"{_id(r)} {_fmt(r['knn_accuracy'])} | "
                     f"{_fmt(r['knn_f1_macro'])} | {_fmt(r['coverage'])} | "
                     f"{_fmt(r['diversity'])} | {_fmt(r['popularidad_spearman'])} |")
        L.append("\nCoverage baja o popularidad alta = el sistema recomienda "
                 "siempre a los mismos (cabeza). Diversidad muy baja = "
                 "recomendaciones redundantes.\n")

    # Face validity.
    if salida["face"]:
        L.append("## Face validity (revision cualitativa)\n")
        for bloque in salida["face"]:
            L.append(f"**{bloque['modelo']}**\n")
            L.append("```")
            L.extend(bloque["lineas"] or ["  (sin resultados)"])
            L.append("```\n")

    # Fase 6.
    L.append("## Fase 6 - Validacion humana (pendiente)\n")
    L.append("El juez final segun el PDF es un user study con >=3 scouts (estilo "
             "PlayeRank): presentar pares (query, candidato) del top-k vs. "
             "aleatorios y medir concordancia + acuerdo inter-anotador (Fleiss k). "
             "No es automatizable; queda como trabajo futuro de la memoria.\n")

    (out_dir / "resumen.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figuras                                                                       #
# --------------------------------------------------------------------------- #

def escribir_figuras(salida: dict, out_dir: Path) -> None:
    """Un par de figuras clave (auto-similitud y estabilidad) por modelo."""
    global matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [aviso] sin figuras (matplotlib no disponible): {e}")
        return

    fig_dir = out_dir / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1g = [r for r in salida["f1"] if r["direccion"] == "global"]
    if f1g:
        etiquetas = [r["modelo"] for r in f1g]
        top1 = [r["top1"] for r in f1g]
        azar = [r["azar_top1"] for r in f1g]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(etiquetas))
        ax.bar(x, top1, color="#3b7dd8", label="top-1")
        ax.plot(x, azar, "r--o", label="azar (1/pool)")
        ax.set_xticks(x); ax.set_xticklabels(etiquetas, rotation=45, ha="right")
        ax.set_ylabel("top-1 accuracy"); ax.set_title("Auto-similitud (pares/impares)")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "fase1_autosimilitud_top1.png", dpi=120)
        plt.close(fig)

    if salida["f2"]:
        etiquetas = [r["modelo"] for r in salida["f2"]]
        rbo = [r["rbo_medio"] for r in salida["f2"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(etiquetas))
        ax.bar(x, rbo, color="#5aa469")
        ax.axhline(config.RBO_ESTABLE, color="green", ls="--", label="estable (0.7)")
        ax.axhline(config.RBO_PREOCUPANTE, color="red", ls="--", label="preocupante (0.5)")
        ax.set_xticks(x); ax.set_xticklabels(etiquetas, rotation=45, ha="right")
        ax.set_ylabel("RBO@10 medio"); ax.set_ylim(0, 1)
        ax.set_title("Estabilidad ante bootstrap"); ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "fase2_estabilidad_rbo.png", dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalua los modelos de similitud.")
    p.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH)
    p.add_argument("--model-dir", type=Path, default=config.DEFAULT_MODEL_DIR)
    p.add_argument("--out", type=Path, default=config.DEFAULT_OUT_DIR)
    p.add_argument("--fases", type=str, default="0,1,2,3,5",
                   help="Subconjunto de fases a ejecutar (coma-separado).")
    p.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_B,
                   help="Remuestreos bootstrap de la Fase 2 (0 para omitir).")
    p.add_argument("--sin-figuras", action="store_true")
    args = p.parse_args(argv)

    fases_sel = {s.strip() for s in args.fases.split(",") if s.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("Cargando modelos y datos...")
    modelos = cargar_modelos(args.model_dir)
    if not modelos:
        raise SystemExit(f"No hay modelos en {args.model_dir}")
    roles = datos.rol_por_jugador(args.db)
    minutos = datos.minutos_por_jugador(args.db)

    n_ctx = len(config.ENTIDADES) * len(config.NORMALIZACIONES)
    total = n_ctx + _pasos_totales(len(modelos), fases_sel, args.bootstrap)
    prog = Progreso(total)
    print(f"{len(modelos)} modelos | fases {sorted(fases_sel)} | "
          f"bootstrap B={args.bootstrap} | {total} pasos cronometrados.\n")

    ctxs = crear_contextos(args.db, prog)
    salida = ejecutar(modelos, ctxs, roles, minutos, fases_sel, args.bootstrap, prog)

    escribir_csv(salida, args.out)
    if not args.sin_figuras:
        escribir_figuras(salida, args.out)
    escribir_informe(salida, args.out, {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "n_modelos": len(modelos),
        "bootstrap": args.bootstrap if "2" in fases_sel else 0,
        "features": resumen_features(modelos),
    })
    dt = time.perf_counter() - t0
    print(f"\nListo en {dt:.0f}s. Resultados en {args.out}/ (ver resumen.md).")


if __name__ == "__main__":
    main()
