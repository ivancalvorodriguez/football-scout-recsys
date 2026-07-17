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
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display: solo PNG a disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .consulta import ErrorResolucion, configurar_consola, resolver
from .features import NORMALIZACIONES_VALIDAS
from .modelo import cargar_modelo, top_k

configurar_consola()


# --- Calculo -----------------------------------------------------------------

@dataclass
class _Comparacion:
    """Resultado de comparar los modos de normalizacion para UNA referencia.

    La identidad es `idx` (indice de entidad), nunca el nombre: dos entidades
    pueden compartir nombre, y dos consultas distintas ("Messi", "Lionel Messi")
    pueden resolver a la misma. El nombre es solo etiqueta de presentacion.
    """

    idx: int
    nombre: str
    tops: dict[str, list[tuple[str, float]]]  # modo -> [(candidato, score), ...]
    jaccard: float

    def candidatos(self, modo: str) -> list[str]:
        return [n for n, _ in self.tops[modo]]


def _rankings(modelo, ref_idx: int, k: int) -> list[tuple[str, float]]:
    """Top-k como [(nombre, score), ...] segun el modelo."""
    return [(modelo.entity_names[j], float(s)) for j, s in top_k(modelo, ref_idx, k)]


def _ligas_del_modelo(modelo) -> set[str]:
    """Ligas (competition-season) que cubre el modelo, segun su `meta`."""
    return {
        liga
        for ligas in modelo.meta.get("ligas_por_entidad", {}).values()
        for liga in ligas
    }


def _avisar_si_una_sola_liga(modelo) -> None:
    """Avisa si comparar `por_liga` contra `global` no puede dar informacion.

    Con una unica liga-temporada en la BD, el z-score por liga y el global se
    calculan sobre las mismas filas: son la MISMA transformacion, los modelos
    salen identicos y el Jaccard es 1.00 siempre. El resultado no es un hallazgo
    ("las normalizaciones coinciden"), es una comparacion vacia.
    """
    ligas = _ligas_del_modelo(modelo)
    if len(ligas) > 1:
        return
    unica = next(iter(ligas), "ninguna")
    print(
        f"  [AVISO] El modelo cubre una sola liga-temporada ({unica}): 'por_liga' y\n"
        f"          'global' son la misma transformacion, asi que el Jaccard sera 1.00\n"
        f"          por construccion. Para que la comparacion diga algo, extrae al\n"
        f"          menos 2 competicion-temporada a la misma BD."
    )


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

def _etiquetas(comparaciones: list[_Comparacion]) -> list[str]:
    """Etiqueta de cada referencia para los ejes de las figuras.

    Dos entidades distintas pueden compartir nombre; en ese caso se anexa el
    indice para que cada barra/fila sea identificable.
    """
    repetidos = {
        c.nombre for c in comparaciones
        if sum(o.nombre == c.nombre for o in comparaciones) > 1
    }
    return [
        f"{c.nombre} [#{c.idx}]" if c.nombre in repetidos else c.nombre
        for c in comparaciones
    ]


def _escribir_csv(
    path: Path,
    filas: list[dict],
) -> None:
    pd.DataFrame(filas).to_csv(path, index=False, encoding="utf-8")


def _plot_heatmap(
    path: Path,
    comparaciones: list[_Comparacion],
    modo_a: str,
    modo_b: str,
    k: int,
) -> None:
    """Para cada candidato del top-k en A, marca en que posicion aparece en B.

    Matriz (referencia x k): cada columna es una posicion del top-k de A; el
    numero de la celda es la posicion en B (o 0 si no aparece). Asi se ve
    visualmente que candidatos "se mueven" entre modos.
    """
    etiquetas = _etiquetas(comparaciones)
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * k + 2), max(4, 0.5 * len(comparaciones) + 2))
    )
    M = np.zeros((len(comparaciones), k), dtype=int)
    # "—" por defecto: cubre tanto "no aparece en B" como las posiciones que el
    # top-k de A no llega a llenar (`top_k` puede devolver menos de k).
    anot = [["—"] * k for _ in range(len(comparaciones))]
    for i, c in enumerate(comparaciones):
        pos_b = {n: r for r, n in enumerate(c.candidatos(modo_b), 1)}
        for j, cand in enumerate(c.candidatos(modo_a)):
            r = pos_b.get(cand, 0)
            M[i, j] = r
            anot[i][j] = f"#{r}" if r else "—"
    # Celda 0 = no aparece en B; la pintamos con escala separada.
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=k)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"#{j+1}" for j in range(k)])
    ax.set_yticks(range(len(comparaciones)))
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel(f"Posicion en top-{k} de A ({modo_a})")
    ax.set_ylabel("Referencia")
    ax.set_title(f"Posicion del candidato en B ({modo_b})\n0 = no aparece en top-{k}")
    for i in range(len(comparaciones)):
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
    comparaciones: list[_Comparacion],
    modo_a: str,
    modo_b: str,
    k: int,
) -> None:
    etiquetas = _etiquetas(comparaciones)
    jaccards = [c.jaccard for c in comparaciones]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(comparaciones) + 2), 4))
    bars = ax.bar(range(len(comparaciones)), jaccards, color="#4c72b0")
    ax.set_xticks(range(len(comparaciones)))
    ax.set_xticklabels(etiquetas, rotation=30, ha="right")
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
    desconocidas = [m for m in normalizaciones if m not in NORMALIZACIONES_VALIDAS]
    if desconocidas:
        raise SystemExit(
            f"Normalizacion(es) desconocida(s): {', '.join(desconocidas)}. "
            f"Validas: {', '.join(NORMALIZACIONES_VALIDAS)}."
        )

    # Carga ambos modelos. Ambos comparten el mismo entity_names (mismo universo).
    modelos = {
        m: cargar_modelo(model_dir, formulacion, entidad, normalizacion=m)
        for m in normalizaciones
    }
    nombres_ref = modelos[normalizaciones[0]].entity_names
    modo_a, modo_b = normalizaciones[0], normalizaciones[1]

    print(f"\n=== Comparador | {entidad} | formulacion {formulacion} ===")
    print(f"Modos: {' vs '.join(normalizaciones)}    k={k}    refs={len(referencias)}")
    if set(normalizaciones) == set(NORMALIZACIONES_VALIDAS):
        _avisar_si_una_sola_liga(modelos[modo_a])

    comparaciones: list[_Comparacion] = []
    vistas: set[int] = set()
    for ref in referencias:
        # Resolver en el primer modelo; los nombres son los mismos en todos.
        try:
            i = resolver(ref, nombres_ref)
        except ErrorResolucion as e:
            print(f"  [aviso] referencia omitida: {e}")
            continue
        if i in vistas:
            print(f"  [aviso] referencia duplicada omitida: {ref!r} -> {nombres_ref[i]}")
            continue
        vistas.add(i)

        tops = {modo: _rankings(modelos[modo], i, k) for modo in normalizaciones}
        cand_a = [n for n, _ in tops[modo_a]]
        cand_b = [n for n, _ in tops[modo_b]]
        comparaciones.append(_Comparacion(
            idx=i, nombre=nombres_ref[i], tops=tops, jaccard=_jaccard(cand_a, cand_b),
        ))
        comunes = _rank_comun(cand_a, cand_b)
        print(
            f"  {nombres_ref[i]:<40s}  Jaccard={comparaciones[-1].jaccard:.2f}  "
            f"interseccion={len(comunes)}/{k}"
        )

    if not comparaciones:
        print("  (sin referencias validas, nada que escribir)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    filas_csv = [
        {"referencia": c.nombre, "modo": modo, "rank": rango,
         "candidato": cand, "score": score}
        for c in comparaciones
        for modo in normalizaciones
        for rango, (cand, score) in enumerate(c.tops[modo], 1)
    ]
    csv_path = out_dir / f"comparativa_form{formulacion}_{entidad}.csv"
    _escribir_csv(csv_path, filas_csv)
    print(f"  CSV  -> {csv_path}")

    # Heatmap
    heat_path = out_dir / f"heatmap_form{formulacion}_{entidad}.png"
    _plot_heatmap(heat_path, comparaciones, modo_a, modo_b, k)
    print(f"  Heat -> {heat_path}")

    # Barras de Jaccard
    jacc_path = out_dir / f"jaccard_form{formulacion}_{entidad}.png"
    _plot_jaccard(jacc_path, comparaciones, modo_a, modo_b, k)
    print(f"  Jacc -> {jacc_path}")

    # Ranking paralelo: uno por referencia.
    for c in comparaciones:
        safe = "".join(ch if ch.isalnum() else "_" for ch in c.nombre)[:60]
        path = out_dir / f"ranking_form{formulacion}_{entidad}_{safe}.png"
        _plot_ranking_paralelo(
            path, c.nombre, c.tops[modo_a], c.tops[modo_b], modo_a, modo_b, k,
        )
    print(f"  Rank -> {len(comparaciones)} figuras en {out_dir}")


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
