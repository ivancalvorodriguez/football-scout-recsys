"""Comparador de recomendaciones entre modos de normalizacion (por_liga vs global).

    python -m src.similitud.comparar --entidad jugador --formulacion 2 \\
        --jugadores "Messi,Memphis Depay,Pedri" --k 10
    python -m src.similitud.comparar --entidad equipo --formulacion 5 \\
        --equipos "Barcelona,Real Madrid,Manchester City" --k 10

Carga los modelos ya entrenados (no reentrena) para cada modo de normalizacion
indicado en `--normalizaciones`, resuelve las entidades de referencia por nombre
y compara los top-k. Genera:

- ``comparativa_<form>_<entidad>.csv`` con los rankings paralelos.
- ``heatmap_<form>_<entidad>.png`` con la interseccion top-k entre modos.
- ``jaccard_<form>_<entidad>.png`` con el Jaccard por referencia.
- ``ranking_<form>_<entidad>_<ref>.png`` (uno por referencia) con los
  rankings paralelos y los scores.

Las figuras se escriben en `--out-dir` (default ``outputs/comparativa``).
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display: solo PNG a disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import config
from .modelo import cargar_modelo, top_k


# --- Resolucion de nombres ---------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().strip()


def _resolver(nombre: str, entity_names: list[str]) -> int:
    objetivo = _norm(nombre)
    nombres_norm = [_norm(n) for n in entity_names]
    for i, n in enumerate(nombres_norm):
        if n == objetivo:
            return i
    candidatos = [i for i, n in enumerate(nombres_norm) if objetivo in n]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise SystemExit(f"No se encontro ninguna entidad que contenga {nombre!r}.")
    opciones = ", ".join(entity_names[i] for i in candidatos[:10])
    raise SystemExit(f"Ambiguo {nombre!r}; coincide con: {opciones} ...")


# --- Calculo -----------------------------------------------------------------

def _rankings(modelo, ref_idx: int, k: int) -> list[tuple[str, float]]:
    """Top-k como [(nombre, score), ...] segun el modelo."""
    return [(modelo.entity_names[j], float(s)) for j, s in top_k(modelo, ref_idx, k)]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _rank_comun(top_a: list[str], top_b: list[str]) -> dict[str, tuple[int, int]]:
    """Mapa nombre -> (rank_en_a, rank_en_b). Solo candidatos en ambos tops."""
    pos_a = {n: r for r, n in enumerate(top_a, 1)}
    pos_b = {n: r for r, n in enumerate(top_b, 1)}
    comunes = set(pos_a) & set(pos_b)
    return {n: (pos_a[n], pos_b[n]) for n in comunes}


# --- Salidas -----------------------------------------------------------------

def _escribir_csv(
    path: Path,
    filas: list[dict],
) -> None:
    pd.DataFrame(filas).to_csv(path, index=False, encoding="utf-8")


def _plot_heatmap(
    path: Path,
    referencias: list[str],
    top_a_por_ref: dict[str, list[str]],
    top_b_por_ref: dict[str, list[str]],
    modo_a: str,
    modo_b: str,
    k: int,
) -> None:
    """Para cada candidato del top-k en A, marca en que posicion aparece en B.

    Matriz (referencia x k): cada columna es una posicion del top-k de A; el
    numero de la celda es la posicion en B (o 0 si no aparece). Asi se ve
    visualmente que candidatos "se mueven" entre modos.
    """
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * k + 2), max(4, 0.5 * len(referencias) + 2)))
    M = np.zeros((len(referencias), k), dtype=int)
    anot = [[""] * k for _ in range(len(referencias))]
    for i, ref in enumerate(referencias):
        pos_b = {n: r for r, n in enumerate(top_b_por_ref[ref], 1)}
        for j, cand in enumerate(top_a_por_ref[ref]):
            r = pos_b.get(cand, 0)
            M[i, j] = r
            anot[i][j] = f"#{r}" if r else "—"
    # Celda 0 = no aparece en B; la pintamos con escala separada.
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=k)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"#{j+1}" for j in range(k)])
    ax.set_yticks(range(len(referencias)))
    ax.set_yticklabels(referencias)
    ax.set_xlabel(f"Posicion en top-{k} de A ({modo_a})")
    ax.set_ylabel("Referencia")
    ax.set_title(f"Posicion del candidato en B ({modo_b})\n0 = no aparece en top-{k}")
    for i in range(len(referencias)):
        for j in range(k):
            ax.text(j, i, anot[i][j], ha="center", va="center",
                    color="white" if M[i, j] <= k // 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"Rank en {modo_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ranking_paralelo(
    path: Path,
    referencia: str,
    top_a: list[tuple[str, float]],
    top_b: list[tuple[str, float]],
    modo_a: str,
    modo_b: str,
    k: int,
) -> None:
    """Rankings paralelos: para cada candidato, posicion en A y en B, con su score."""
    nombres = sorted({n for n, _ in top_a} | {n for n, _ in top_b})
    pos_a = {n: r for r, (n, _) in enumerate(top_a, 1)}
    pos_b = {n: r for r, (n, _) in enumerate(top_b, 1)}
    sc_a = {n: s for n, s in top_a}
    sc_b = {n: s for n, s in top_b}

    fig, (ax_rank, ax_score) = plt.subplots(1, 2, figsize=(13, max(4, 0.4 * len(nombres) + 2)))
    y = np.arange(len(nombres))
    ax_rank.hlines(y, [pos_a.get(n, 0) for n in nombres],
                        [pos_b.get(n, 0) for n in nombres],
                   color="lightgray", linewidth=2, zorder=1)
    ax_rank.scatter([pos_a.get(n, np.nan) for n in nombres], y,
                    label=modo_a, s=70, zorder=3)
    ax_rank.scatter([pos_b.get(n, np.nan) for n in nombres], y,
                    label=modo_b, s=70, marker="x", zorder=3)
    for i, n in enumerate(nombres):
        ax_rank.text(k + 0.5, i, n, va="center", fontsize=9)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([""] * len(nombres))
    ax_rank.set_xlim(0.5, k + 0.5)
    ax_rank.set_xlabel(f"Rank en top-{k}")
    ax_rank.set_title(f"Ranking: {referencia}")
    ax_rank.invert_xaxis()  # rank 1 a la izquierda
    ax_rank.legend(loc="lower right")
    ax_rank.grid(True, axis="x", linestyle=":", alpha=0.4)

    ax_score.scatter([sc_a.get(n, np.nan) for n in nombres], y,
                     label=modo_a, s=70, zorder=3)
    ax_score.scatter([sc_b.get(n, np.nan) for n in nombres], y,
                     label=modo_b, s=70, marker="x", zorder=3)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(nombres, fontsize=9)
    ax_score.set_xlabel("Score (S)")
    ax_score.set_title("Score por candidato")
    ax_score.legend(loc="lower right")
    ax_score.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax_score.axvline(0.0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_jaccard(
    path: Path,
    referencias: list[str],
    jaccards: list[float],
    modo_a: str,
    modo_b: str,
    k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(referencias) + 2), 4))
    bars = ax.bar(range(len(referencias)), jaccards, color="#4c72b0")
    ax.set_xticks(range(len(referencias)))
    ax.set_xticklabels(referencias, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Jaccard")
    ax.set_title(
        f"Jaccard del top-{k}: {modo_a} vs {modo_b}  (media = {np.mean(jaccards):.2f})"
    )
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{jaccards[i]:.2f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Orquestacion ------------------------------------------------------------

def _comparar_para(
    *,
    entidad: str,
    formulacion: str,
    normalizaciones: list[str],
    referencias: list[str],
    k: int,
    model_dir: Path,
    out_dir: Path,
) -> None:
    if len(normalizaciones) < 2:
        raise SystemExit("Hace falta al menos 2 normalizaciones para comparar.")

    # Carga ambos modelos. Ambos comparten el mismo entity_names (mismo universo).
    modelos = {
        m: cargar_modelo(model_dir, formulacion, entidad, normalizacion=m)
        for m in normalizaciones
    }
    nombres_ref = modelos[normalizaciones[0]].entity_names

    out_dir.mkdir(parents=True, exist_ok=True)
    filas_csv: list[dict] = []
    top_por_ref_modo: dict[str, dict[str, list[str]]] = {}
    jaccards: list[float] = []

    print(f"\n=== Comparador | {entidad} | formulacion {formulacion} ===")
    print(f"Modos: {' vs '.join(normalizaciones)}    k={k}    refs={len(referencias)}")
    for ref in referencias:
        # Resolver en el primer modelo; los nombres son los mismos en todos.
        try:
            i = _resolver(ref, nombres_ref)
        except SystemExit as e:
            print(f"  [aviso] referencia omitida: {e}")
            continue
        ref_nombre = nombres_ref[i]
        top_por_ref_modo[ref_nombre] = {}
        per_ref: dict[str, list[tuple[str, float]]] = {}
        for modo in normalizaciones:
            per_ref[modo] = _rankings(modelos[modo], i, k)
            top_por_ref_modo[ref_nombre][modo] = [n for n, _ in per_ref[modo]]
            for rango, (cand, score) in enumerate(per_ref[modo], 1):
                filas_csv.append({
                    "referencia": ref_nombre,
                    "modo": modo,
                    "rank": rango,
                    "candidato": cand,
                    "score": score,
                })
        j = _jaccard(top_por_ref_modo[ref_nombre][normalizaciones[0]],
                     top_por_ref_modo[ref_nombre][normalizaciones[1]])
        comunes = _rank_comun(
            top_por_ref_modo[ref_nombre][normalizaciones[0]],
            top_por_ref_modo[ref_nombre][normalizaciones[1]],
        )
        jaccards.append(j)
        print(
            f"  {ref_nombre:<40s}  Jaccard={j:.2f}  "
            f"interseccion={len(comunes)}/{k}"
        )

    if not jaccards:
        print("  (sin referencias validas, nada que escribir)")
        return

    # CSV
    csv_path = out_dir / f"comparativa_form{formulacion}_{entidad}.csv"
    _escribir_csv(csv_path, filas_csv)
    print(f"  CSV  -> {csv_path}")

    modo_a, modo_b = normalizaciones[0], normalizaciones[1]
    refs = list(top_por_ref_modo.keys())

    # Heatmap
    heat_path = out_dir / f"heatmap_form{formulacion}_{entidad}.png"
    _plot_heatmap(
        heat_path, refs,
        {r: top_por_ref_modo[r][modo_a] for r in refs},
        {r: top_por_ref_modo[r][modo_b] for r in refs},
        modo_a, modo_b, k,
    )
    print(f"  Heat -> {heat_path}")

    # Barras de Jaccard
    jacc_path = out_dir / f"jaccard_form{formulacion}_{entidad}.png"
    _plot_jaccard(jacc_path, refs, jaccards, modo_a, modo_b, k)
    print(f"  Jacc -> {jacc_path}")

    # Ranking paralelo: uno por referencia
    # Recomponemos (nombre, score) por modo desde filas_csv para reutilizar todo
    # el calculo previo sin recalcular.
    scores_por_ref_modo: dict[str, dict[str, list[tuple[str, float]]]] = {
        r: {m: [] for m in normalizaciones} for r in refs
    }
    for fila in filas_csv:
        r = fila["referencia"]
        m = fila["modo"]
        if r in scores_por_ref_modo and m in scores_por_ref_modo[r]:
            scores_por_ref_modo[r][m].append((fila["candidato"], fila["score"]))
    for ref in refs:
        safe = "".join(c if c.isalnum() else "_" for c in ref)[:60]
        path = out_dir / f"ranking_form{formulacion}_{entidad}_{safe}.png"
        _plot_ranking_paralelo(
            path, ref,
            scores_por_ref_modo[ref][modo_a],
            scores_por_ref_modo[ref][modo_b],
            modo_a, modo_b, k,
        )
    print(f"  Rank -> {len(refs)} figuras en {out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Compara recomendaciones entre modos de normalizacion."
    )
    p.add_argument("--entidad", choices=["jugador", "equipo"], required=True)
    p.add_argument("--formulacion", choices=["2", "5"], default="2")
    p.add_argument("--normalizaciones", default="por_liga,global",
                   help="Lista separada por comas (minimo 2).")
    p.add_argument("--jugadores", default=None,
                   help="Lista de jugadores de referencia (separada por comas).")
    p.add_argument("--equipos", default=None,
                   help="Lista de equipos de referencia (separada por comas).")
    p.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--modelo-dir", type=Path, default=config.DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/comparativa"))
    args = p.parse_args(argv)

    normalizaciones = [m.strip() for m in args.normalizaciones.split(",") if m.strip()]
    if args.entidad == "jugador":
        if not args.jugadores:
            raise SystemExit("Para --entidad jugador hay que pasar --jugadores.")
        referencias = [s.strip() for s in args.jugadores.split(",") if s.strip()]
    else:
        if not args.equipos:
            raise SystemExit("Para --entidad equipo hay que pasar --equipos.")
        referencias = [s.strip() for s in args.equipos.split(",") if s.strip()]

    _comparar_para(
        entidad=args.entidad,
        formulacion=args.formulacion,
        normalizaciones=normalizaciones,
        referencias=referencias,
        k=args.k,
        model_dir=args.modelo_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
