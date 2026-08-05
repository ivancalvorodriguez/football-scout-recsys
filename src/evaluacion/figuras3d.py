"""Superficies 3D de un barrido ya generado: metrica sobre el plano de dos hiperparametros.

    python -m src.evaluacion.figuras3d [--barrido DIR] [--out DIR]
                                       [--metricas m1,m2] [--modelos F2_equipo_global,...]
                                       [--reduccion max|media] [--elev E] [--azim A]
                                       [--escala auto|lineal|log|ordinal] [--densidad N]
                                       [--mostrar] [--sin-score] [--sin-html]

Este script NO construye ni evalua nada: LEE la salida de
`python -m src.evaluacion.barrido` (`barrido_metricas.csv` +
`combinaciones.json`) y dibuja, por cada (metrica, modelo), una superficie con
un hiperparametro en X, otro en Y y el valor de la metrica en Z. Como la carpeta
del barrido es ACUMULATIVA, esos dos ficheros traen TODOS los puntos evaluados
en ella —vengan de una corrida o de cinco—, asi que ampliar la rejilla es
relanzar el barrido con mas valores y volver a dibujar. Las tablas del
`resumen_barrido.md` dicen que combinacion gana; la superficie enseña la FORMA
del optimo: si hay una meseta, una cresta estrecha o un maximo en el borde de la
rejilla (que es la señal de que el barrido se quedo corto y hay que ampliar el
rango de valores, no de que ese valor sea el bueno).

Tres decisiones de lectura, importantes para no malinterpretar las figuras:

- **Los ejes van en su ESCALA REAL** (`src.evaluacion.malla`), no en posiciones
  ordinales: los valores barridos no estan equiespaciados (`F2_BETA` va
  2.5 -> 5 -> 7.5, `F5_EASE_LAMBDA` puede ir 10 -> 50 -> 200 -> 500) y darle a
  todos la misma casilla deforma la superficie —una meseta ancha puede parecer
  una cresta estrecha solo por como se eligieron los valores a probar. Cada eje
  elige ademas su escala (`--escala auto`): logaritmica si sus valores son
  claramente multiplicativos (512/1024/2048), lineal si no. La etiqueta del tick
  sigue siendo el valor real, y el pie declara la escala usada. Con `--escala
  ordinal` se recupera el reparto uniforme de antes (y es lo que se usa a la
  fuerza en un eje cuyos valores no sean numeros).
- **La superficie entre los puntos es una ESTIMACION**, no un dato. Se interpola
  con PCHIP monotono (ver `malla.interpolar`), que no sobrepasa: la superficie
  nunca sale del rango de los puntos que la rodean ni crea maximos que no se
  hayan medido. Los puntos realmente evaluados van marcados en negro y son los
  unicos numeros que salen en `rejilla_3d.csv`. Con `--densidad 0` se dibuja la
  rejilla cruda, sin interpolar.
- **Con mas de dos ejes barridos** se dibuja una figura por PAREJA de ejes, y las
  celdas se reducen sobre los ejes restantes (`--reduccion`, por defecto el mejor
  valor segun la orientacion de la metrica). Es una PROYECCION, no un corte: la
  celda (x, y) enseña lo mejor alcanzable ahi, no el valor de una combinacion
  concreta. Se declara en el pie de cada figura.

Ademas de las metricas del barrido se dibuja el **score compuesto** de
`src.evaluacion.puntuacion` (una metrica mas, `score_compuesto`): la media ponderada
de todas las metricas llevadas a z-score dentro de la entidad. Con el se genera una
figura extra por entidad, `comparativa_<entidad>__score_compuesto__*.png`, que
superpone la superficie de CADA modelo en los mismos ejes para ver **cual gana y en
que zona de hiperparametros**. Es la unica figura que superpone modelos, porque el
score es lo unico comparable entre ellos; lee las reservas de `puntuacion` antes de
citarlo (es una heuristica de lectura, no un criterio validado).

Salida en `<barrido>/figuras3d/`:

- un PNG por (metrica, modelo, pareja de ejes) + las comparativas por entidad;
- `superficies.html`, un visor **interactivo autocontenido** con todas las
  superficies embebidas: se gira con el raton, sin servidor ni dependencias (ver
  `figuras3d_html`). Es lo que permite mirar una cresta desde el angulo que haga
  falta, cosa que un PNG con la camara fija no da;
- `rejilla_3d.csv` (los valores MEDIDOS de cada casilla, en formato largo, con la
  escala usada en cada eje; la superficie estimada no sale ahi: no es un
  resultado) y `score.csv` (el score con las z que lo componen, para que sea
  auditable).

Los CSV y el HTML recogen exactamente lo que se dibujo en esa corrida, asi que
`--metricas`/`--modelos` los deja acotados a ese filtro. El score, en cambio, se
calcula siempre sobre TODAS las metricas del barrido: si dependiera del filtro,
seria un numero distinto en cada corrida.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import barrido, figuras3d_html, malla, puntuacion, registro
from . import config as ecfg

# Orientacion de cada metrica (mayor/menor es mejor): vive en `puntuacion` porque
# el score la necesita para orientar las z. Aqui solo marca el optimo en la figura
# y decide como se reduce sobre los ejes que no entran en la pareja.
orientacion = puntuacion.orientacion


# --------------------------------------------------------------------------- #
# Lectura del barrido                                                           #
# --------------------------------------------------------------------------- #

def leer_valores(barrido_dir: Path) -> pd.DataFrame:
    """`barrido_metricas.csv` en formato largo (una fila por combinacion-modelo-metrica).

    Ese CSV es ACUMULATIVO (ver `registro.py`): trae los puntos de todas las
    corridas hechas sobre la carpeta, no solo los de la ultima. Es lo que permite
    que la superficie tenga la rejilla entera aunque se haya barrido en varias
    tandas.
    """
    ruta = Path(barrido_dir) / registro.ARCHIVO_METRICAS
    if not ruta.exists():
        raise SystemExit(
            f"No encuentro {ruta}. Este script dibuja un barrido YA generado: "
            "lanza antes `python -m src.evaluacion.barrido` (o apunta --barrido a "
            "la carpeta que lo contiene)."
        )
    df = pd.read_csv(ruta)
    df["formulacion"] = df["formulacion"].astype(str)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def _combinaciones_recomputadas(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Reconstruye el mapa combinacion -> hiperparametros desde `barrido`.

    Ultimo recurso: solo si la carpeta no tiene ni `combinaciones.json` ni
    `resumen_barrido.md` (barrido interrumpido antes de escribirlos). Vale lo que
    valga la suposicion de que `HIPERPARAMETROS` y los defaults de
    `src/similitud/config.py` no han cambiado desde la corrida, y ademas nombra
    por POSICION, que es justo lo que el registro dejo de hacer; por eso se avisa.
    """
    forms = tuple(f for f in ecfg.FORMULACIONES if f in set(df["formulacion"]))
    ents = tuple(e for e in ecfg.ENTIDADES if e in set(df["entidad"]))
    if not forms or not ents:
        raise SystemExit("El CSV del barrido no declara ninguna formulacion/entidad conocida.")
    seleccion = barrido.combinaciones(forms, ents)
    ejes = barrido._ejes_publicados()
    return {n: barrido._config_efectiva(v, ejes) for n, v in seleccion.items()}


def combinaciones_del_barrido(
    barrido_dir: Path, df: pd.DataFrame
) -> dict[str, dict[str, object]]:
    """Mapa combinacion -> hiperparametros, acotado a las combinaciones del CSV.

    Se lee de la carpeta (`combinaciones.json`, y si no del `resumen_barrido.md`)
    en vez de recalcularse desde `barrido.HIPERPARAMETROS`: esa lista se EDITA
    entre corridas, asi que recalcularla etiquetaria las figuras con los ejes del
    barrido de hoy y no con los que produjeron cada punto. El registro publica los
    valores efectivos con los que se construyo cada combinacion, que es justo lo
    que necesita la superficie.
    """
    reg_dir = registro.ruta(barrido_dir)
    if reg_dir.is_file():
        combos = registro.hiperparametros(registro.cargar(barrido_dir))
    else:
        combos = registro.leer_markdown(barrido_dir)
    if not combos:
        print("  [aviso] la carpeta no tiene combinaciones.json ni "
              "resumen_barrido.md: se recomponen los hiperparametros desde "
              "barrido.HIPERPARAMETROS y los defaults actuales de "
              "src/similitud/config.py. Si esos valores han cambiado desde la "
              "corrida, las etiquetas de los ejes seran incorrectas.")
        combos = _combinaciones_recomputadas(df)

    presentes = [c for c in df["combinacion"].unique()]
    faltan = [c for c in presentes if c not in combos]
    if faltan:
        raise SystemExit(
            f"El CSV trae combinaciones que el indice de la carpeta no describe "
            f"({', '.join(map(str, faltan[:5]))}...): barrido_metricas.csv y "
            "combinaciones.json no son de la misma carpeta."
        )
    return {c: combos[c] for c in sorted(presentes, key=registro.orden_nombre)}


# --------------------------------------------------------------------------- #
# Ejes de la rejilla                                                            #
# --------------------------------------------------------------------------- #

# Los ejes de la superficie son los hiperparametros que REALMENTE se movieron en
# la carpeta. Vive en `registro` porque el resumen del barrido publica los mismos
# y los dos tienen que coincidir.
ejes_variables = registro.ejes_variables


def _fijos(combos: dict[str, dict[str, object]], variables: set[str]) -> dict[str, object]:
    """Ejes constantes en todo el barrido, con su valor (para el pie de figura)."""
    if not combos:
        return {}
    primera = next(iter(combos.values()))
    return {eje: v for eje, v in primera.items() if eje not in variables}


# --------------------------------------------------------------------------- #
# Construccion de la rejilla Z                                                  #
# --------------------------------------------------------------------------- #

def construir_rejilla(
    z_por_combo: dict[str, float],
    combos: dict[str, dict[str, object]],
    eje_x: str, valores_x: list[object],
    eje_y: str, valores_y: list[object],
    reduccion: str,
    mejor: str,
) -> np.ndarray:
    """Matriz Z[len(y), len(x)] con la metrica en cada casilla de la rejilla.

    Cuando el barrido mueve mas de dos ejes, varias combinaciones caen en la
    misma casilla (x, y): se reducen con `reduccion` ("max" = lo mejor alcanzable
    ahi segun `mejor`; "media" = el valor tipico). Las casillas sin ninguna
    combinacion —o con la metrica no calculable— quedan a NaN y la superficie
    simplemente no las dibuja.
    """
    acum: dict[tuple[int, int], list[float]] = {}
    for combo, z in z_por_combo.items():
        if z is None or not np.isfinite(z):
            continue
        conf = combos[combo]
        # Una combinacion que no declare uno de los dos ejes no cae en ninguna
        # casilla de ESTA rejilla (puede pasar con una carpeta acumulada cuyo
        # registro venga de un barrido con otros ejes): se ignora en vez de
        # colocarla donde no esta.
        if conf.get(eje_x) not in valores_x or conf.get(eje_y) not in valores_y:
            continue
        i = valores_y.index(conf[eje_y])
        j = valores_x.index(conf[eje_x])
        acum.setdefault((i, j), []).append(float(z))

    Z = np.full((len(valores_y), len(valores_x)), np.nan)
    for (i, j), zs in acum.items():
        if reduccion == "media":
            Z[i, j] = float(np.mean(zs))
        else:
            Z[i, j] = max(zs) if mejor == "max" else min(zs)
    return Z


def _optimo(Z: np.ndarray, mejor: str) -> tuple[int, int] | None:
    """Indice (i, j) de la mejor casilla, o None si la rejilla esta vacia."""
    if not np.any(np.isfinite(Z)):
        return None
    plano = np.nanargmax(Z) if mejor == "max" else np.nanargmin(Z)
    return np.unravel_index(int(plano), Z.shape)


def en_borde(Z: np.ndarray, idx: tuple[int, int]) -> bool:
    """True si el optimo cae en el borde de la rejilla (barrido posiblemente corto).

    Es la lectura mas util de la figura y la que las tablas del resumen no dan:
    un maximo pegado al borde no es un maximo, es el limite del rango probado.
    Con un solo valor en un eje no hay borde que declarar (todo lo es).
    """
    i, j = idx
    ny, nx = Z.shape
    return (ny > 2 and i in (0, ny - 1)) or (nx > 2 and j in (0, nx - 1))


# --------------------------------------------------------------------------- #
# Figura                                                                        #
# --------------------------------------------------------------------------- #

def _titulo_eje(e: malla.Eje) -> str:
    """Nombre del eje, con la escala entre parentesis si no es la lineal.

    Un eje logaritmico o en posiciones ordinales cambia lo que significa la
    distancia entre dos ticks: tiene que verse en el propio eje, no solo en el pie.
    """
    return e.nombre if e.escala == "lineal" else f"{e.nombre}  ({e.escala})"


def dibujar_superficie(
    plt, Z: np.ndarray, Zf: np.ndarray,
    ex: malla.Eje, ey: malla.Eje,
    metrica: str, modelo: str, mejor: str,
    pie: str, elev: float, azim: float,
):
    """Devuelve la figura de una superficie (sin guardarla).

    `Z` son los valores MEDIDOS en la rejilla del barrido y `Zf` la superficie
    estimada sobre la malla fina de los ejes. Los limites de color y de la z salen
    siempre de `Z`: `Zf` no puede salirse de ese rango (PCHIP no sobrepasa), y
    tomarlos de la estimacion haria que la barra de color anunciara valores que no
    se han medido.
    """
    from matplotlib.colors import Normalize

    Xf, Yf = np.meshgrid(ex.uf, ey.uf)
    X, Y = np.meshgrid(ex.u, ey.u)

    finitos = Z[np.isfinite(Z)]
    lo, hi = float(finitos.min()), float(finitos.max())
    margen = (hi - lo) * 0.35 or max(abs(hi) * 0.1, 1e-3)
    suelo = lo - margen

    # Normalizacion explicita y compartida por superficie, contorno y barra de
    # color. Sin ella `plot_surface` normaliza por el promedio de cada cuadrilatero
    # (no por Z), y la barra de color acabaria anunciando un rango mas estrecho
    # que el de los datos: en una figura cuyo unico contenido son diferencias
    # pequeñas, eso es justo lo que no puede fallar.
    norma = Normalize(vmin=lo, vmax=hi)

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Sin aristas: con la malla fina serian cientos de lineas y taparian la
    # superficie. Lo que hay que poder distinguir es donde estan los puntos
    # medidos, y eso lo hace el `scatter` de mas abajo.
    superficie = ax.plot_surface(
        Xf, Yf, Zf, cmap="viridis", norm=norma, edgecolor="none", linewidth=0,
        rstride=1, cstride=1, antialiased=True, alpha=0.92,
    )

    # Proyeccion de nivel en el suelo: en 3D las diferencias pequeñas se pierden
    # con el escorzo, y el mapa de contorno las recupera sin salir de la figura.
    if np.all(np.isfinite(Zf)) and Zf.size > 1 and hi > lo:
        ax.contourf(Xf, Yf, Zf, zdir="z", offset=suelo, cmap="viridis", norm=norma,
                    alpha=0.55, levels=12)

    # Puntos realmente evaluados: lo demas es la estimacion entre ellos.
    ax.scatter(X[np.isfinite(Z)], Y[np.isfinite(Z)], Z[np.isfinite(Z)],
               color="0.15", s=14, depthshade=False)

    idx = _optimo(Z, mejor)
    if idx is not None:
        i, j = idx
        # La estrella va como TEXTO y no como `scatter`: matplotlib ordena por
        # profundidad la coleccion entera, y con la malla fina la superficie tapa
        # un marcador de un solo punto segun el angulo. El optimo es justo lo que
        # no puede depender de la camara.
        ax.text(ex.u[j], ey.u[i], Z[i, j], "★", color="crimson", fontsize=15,
                ha="center", va="center", zorder=10)
        ax.text(ex.u[j], ey.u[i], Z[i, j] + margen * 0.12,
                f"  {Z[i, j]:.3f}\n  {ex.nombre}={ex.etiquetas[j]}, "
                f"{ey.nombre}={ey.etiquetas[i]}",
                color="crimson", fontsize=8)

    ax.set_xticks(ex.u)
    ax.set_xticklabels(ex.etiquetas, fontsize=8)
    ax.set_yticks(ey.u)
    ax.set_yticklabels(ey.etiquetas, fontsize=8)
    ax.set_xlabel(_titulo_eje(ex), labelpad=10)
    ax.set_ylabel(_titulo_eje(ey), labelpad=10)
    ax.set_zlabel(metrica, labelpad=8)
    ax.set_zlim(suelo, hi + margen * 0.4)

    sentido = "mayor es mejor" if mejor == "max" else "menor es mejor"
    ax.set_title(f"{metrica} — {modelo}\n({sentido})", fontsize=11)
    fig.colorbar(superficie, ax=ax, shrink=0.55, pad=0.12, label=metrica)
    fig.text(0.5, 0.02, pie, ha="center", fontsize=7.5, color="0.35", wrap=True)
    # `tight_layout` no sabe medir las decoraciones de un Axes3D (avisa y no hace
    # nada): los margenes van a mano.
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.90)
    return fig


COLORES_MODELO = ("#3b7dd8", "#e0803a", "#5aa469", "#a45ac0", "#c0455a", "#3ba7a0")


def dibujar_comparativa(
    plt, rejillas: dict[str, tuple[np.ndarray, np.ndarray]],
    ex: malla.Eje, ey: malla.Eje,
    entidad: str, pie: str, elev: float, azim: float,
):
    """Una superficie por modelo en los MISMOS ejes: quien gana y donde.

    Solo tiene sentido con el score compuesto, que es lo unico comparable entre
    modelos (esta z-scoreado dentro de la entidad). Superficies semitransparentes
    con un color por modelo: donde una tapa a las demas por arriba, ese modelo
    domina esa zona de hiperparametros. Cada entrada de `rejillas` es el par
    (medido, estimado) de ese modelo.
    """
    Xf, Yf = np.meshgrid(ex.uf, ey.uf)
    X, Y = np.meshgrid(ex.u, ey.u)
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)

    from matplotlib.patches import Patch
    parches = []
    for i, (modelo, (Z, Zf)) in enumerate(sorted(rejillas.items())):
        color = COLORES_MODELO[i % len(COLORES_MODELO)]
        ax.plot_surface(Xf, Yf, Zf, color=color, edgecolor="none", linewidth=0,
                        rstride=1, cstride=1, alpha=0.55, shade=False)
        ax.scatter(X[np.isfinite(Z)], Y[np.isfinite(Z)], Z[np.isfinite(Z)],
                   color=color, s=16, depthshade=False)
        parches.append(Patch(facecolor=color, alpha=0.7, label=modelo))

    ax.set_xticks(ex.u)
    ax.set_xticklabels(ex.etiquetas, fontsize=8)
    ax.set_yticks(ey.u)
    ax.set_yticklabels(ey.etiquetas, fontsize=8)
    ax.set_xlabel(_titulo_eje(ex), labelpad=10)
    ax.set_ylabel(_titulo_eje(ey), labelpad=10)
    ax.set_zlabel(puntuacion.NOMBRE, labelpad=8)
    ax.set_title(f"{puntuacion.NOMBRE} — {entidad}: que modelo gana y donde\n"
                 "(mayor es mejor)", fontsize=11)
    ax.legend(handles=parches, loc="upper left", fontsize=8, framealpha=0.85)
    fig.text(0.5, 0.02, pie, ha="center", fontsize=7.5, color="0.35", wrap=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.90)
    return fig


# --------------------------------------------------------------------------- #
# Orquestacion                                                                  #
# --------------------------------------------------------------------------- #

PIE_SCORE = (
    f"{puntuacion.NOMBRE}: media ponderada de las metricas llevadas a z-score dentro "
    "de la entidad (heuristica de lectura, no un criterio validado: el protocolo "
    "juzga por convergencia de señales, no por un numero)."
)


def _etiqueta_modelo(form: str, entidad: str, norm: str) -> str:
    return f"F{form}_{entidad}_{norm}"


def _z_json(Z: np.ndarray) -> list[list[float | None]]:
    """Matriz para el HTML: `null` donde no hay valor (JSON no tiene NaN).

    Se redondea a 6 cifras significativas. La malla fina multiplica por ~100 el
    numero de valores embebidos en el HTML y el visor no dibuja mas precision que
    esa; `%g` mantiene las cifras utiles tambien en las metricas muy pequeñas,
    cosa que redondear a un numero fijo de decimales no haria.
    """
    return [[None if not np.isfinite(v) else float(f"{float(v):.6g}") for v in fila]
            for fila in Z]


def _geometria_json(ex: malla.Eje, ey: malla.Eje) -> dict:
    """Ejes de una vista para el visor: etiquetas, coordenadas y malla fina.

    Las coordenadas van normalizadas a [-1, 1] (el cubo en el que dibuja el
    visor), pero conservando el reparto REAL: es lo que hace que el HTML y el PNG
    enseñen la misma superficie. `medidaXf`/`medidaYf` marcan que nodos de la malla
    fina son un valor realmente barrido, para poder dibujar la rejilla medida sobre
    la superficie estimada.
    """
    def de(e: malla.Eje) -> dict:
        return {
            "valores": e.etiquetas,
            "escala": e.escala,
            "coord": [round(float(v), 5) for v in e.normalizar(e.u)],
            "coordF": [round(float(v), 5) for v in e.normalizar(e.uf)],
            "medidaF": [bool(v) for v in e.medidos()],
        }

    x, y = de(ex), de(ey)
    return {
        "ejeX": ex.nombre, "ejeY": ey.nombre,
        "valoresX": x["valores"], "valoresY": y["valores"],
        "escalaX": x["escala"], "escalaY": y["escala"],
        "coordX": x["coord"], "coordY": y["coord"],
        "coordXf": x["coordF"], "coordYf": y["coordF"],
        "medidaXf": x["medidaF"], "medidaYf": y["medidaF"],
    }


def _pie_de_figura(
    ex: malla.Eje, ey: malla.Eje,
    resto: list[str], reduccion: str, mejor: str, fijos: dict[str, object]
) -> str:
    partes = [f"Ejes a escala: {ex.descripcion()}; {ey.descripcion()}."]
    if ex.uf.size > ex.u.size or ey.uf.size > ey.u.size:
        partes.append("La superficie entre los puntos marcados es una ESTIMACION "
                      "(PCHIP monotono): no sobrepasa el rango de los puntos que la "
                      "rodean ni crea optimos nuevos, pero solo esos puntos son datos.")
    if resto:
        como = ("media sobre" if reduccion == "media"
                else f"{'mejor' if mejor == 'max' else 'menor'} valor sobre")
        partes.append(f"Proyeccion: cada casilla es el {como} {', '.join(resto)} "
                      "(no es un corte a valor fijo).")
    if fijos:
        partes.append("Fijos: " + ", ".join(f"{k}={v}" for k, v in fijos.items()) + ".")
    return " ".join(partes)


def generar(
    barrido_dir: Path,
    out_dir: Path,
    metricas_sel: set[str] | None = None,
    modelos_sel: set[str] | None = None,
    reduccion: str = "max",
    elev: float = 26.0,
    azim: float = -128.0,
    escala: str = "auto",
    densidad: int = malla.DENSIDAD,
    mostrar: bool = False,
    con_score: bool = True,
    con_html: bool = True,
) -> int:
    """Dibuja todas las superficies del barrido. Devuelve cuantas figuras escribio."""
    try:
        import matplotlib
        if not mostrar:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Sin figuras (matplotlib no disponible): {e}")

    df = leer_valores(barrido_dir)
    combos = combinaciones_del_barrido(barrido_dir, df)
    if len(combos) < 4:
        print(f"  [aviso] solo hay {len(combos)} combinaciones: la superficie sera "
              "muy pobre (hacen falta al menos 2 valores en cada eje).")

    variables = ejes_variables(combos)
    if len(variables) < 2:
        raise SystemExit(
            "El barrido solo movio "
            f"{len(variables)} hiperparametro(s) ({', '.join(variables) or 'ninguno'}): "
            "una superficie 3D necesita DOS ejes con mas de un valor. Añade otro eje "
            "a HIPERPARAMETROS en src/evaluacion/barrido.py (y comprueba que afecta a "
            "la rejilla que estas construyendo: los F5_* no tocan los modelos F2)."
        )
    fijos = _fijos(combos, set(variables))

    df["modelo"] = [
        _etiqueta_modelo(f, e, n)
        for f, e, n in zip(df["formulacion"], df["entidad"], df["normalizacion"])
    ]

    # El score se calcula ANTES de aplicar los filtros: agrega TODAS las metricas
    # del barrido, no las que el usuario haya pedido dibujar. Filtrar antes daria
    # un "score" distinto en cada corrida segun el flag, que es justo lo que no
    # puede pasar con un numero que se usa para ordenar modelos.
    entidad_de: dict[str, str] = dict(zip(df["modelo"], df["entidad"]))
    if con_score:
        scores = puntuacion.puntuar(df)
        if scores.empty:
            print("  [aviso] no hay metricas puntuables: sin score compuesto.")
            con_score = False
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            scores.to_csv(out_dir / "score.csv", index=False)
            df = pd.concat([df, puntuacion.como_metrica(scores, df)], ignore_index=True)
            print("Score compuesto (media ponderada de las metricas en z, por "
                  "entidad) — mejores:")
            print("\n".join(puntuacion.ranking(scores)))

    if metricas_sel:
        df = df[df["metrica"].isin(metricas_sel)]
    if modelos_sel:
        df = df[df["modelo"].isin(modelos_sel)]
    if df.empty:
        raise SystemExit("El filtro de --metricas/--modelos no deja ninguna serie.")

    out_dir.mkdir(parents=True, exist_ok=True)
    parejas = list(itertools.combinations(variables, 2))
    print(f"Ejes barridos: {', '.join(variables)} -> {len(parejas)} pareja(s).")
    if fijos:
        print("Ejes fijos: " + ", ".join(f"{k}={v}" for k, v in fijos.items()))

    # La geometria de cada eje (escala + malla fina) depende solo de sus valores
    # barridos, asi que se resuelve una vez y la comparten todas las figuras: dos
    # superficies del mismo eje tienen que ser superponibles.
    geo = {nombre: malla.eje(nombre, valores, escala, densidad)
           for nombre, valores in variables.items()}
    print("Escala de los ejes: "
          + "; ".join(f"{e.nombre} -> {e.escala}" for e in geo.values()))

    filas_csv: list[dict] = []
    figuras: list = []
    vistas: list[dict] = []
    n = 0
    vacias: list[str] = []

    for (eje_x, eje_y) in parejas:
        resto = [e for e in variables if e not in (eje_x, eje_y)]
        ex, ey = geo[eje_x], geo[eje_y]
        vx, vy = variables[eje_x], variables[eje_y]
        etq_ejes = f"{eje_x} x {eje_y}"
        # Por modelo, el par (medido, estimado): la comparativa necesita los dos.
        rejillas_score: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for metrica, dfm in df.groupby("metrica", sort=True):
            mejor = orientacion(str(metrica))
            pie = _pie_de_figura(ex, ey, resto, reduccion, mejor, fijos)
            if metrica == puntuacion.NOMBRE:
                pie = PIE_SCORE + " " + pie
            for modelo, dfmm in dfm.groupby("modelo", sort=True):
                z_por_combo = dict(zip(dfmm["combinacion"], dfmm["valor"]))
                Z = construir_rejilla(z_por_combo, combos, eje_x, vx, eje_y, vy,
                                      reduccion, mejor)
                if not np.any(np.isfinite(Z)):
                    # Metrica no calculable para ese modelo (la pureza posicional
                    # del equipo, p. ej.): no hay nada que dibujar.
                    vacias.append(f"{metrica}/{modelo}")
                    continue
                Zf = malla.interpolar(Z, ex, ey)

                # El CSV recoge SOLO los puntos medidos: la superficie estimada es
                # una ayuda de lectura, no un resultado que deba viajar a un
                # analisis externo como si se hubiera evaluado.
                for i, y in enumerate(vy):
                    for j, x in enumerate(vx):
                        filas_csv.append({
                            "metrica": metrica, "modelo": modelo,
                            "eje_x": eje_x, "valor_x": x, "escala_x": ex.escala,
                            "eje_y": eje_y, "valor_y": y, "escala_y": ey.escala,
                            "z": Z[i, j],
                            # Sin ejes sobrantes no se reduce nada: anotar "max"
                            # ahi haria pensar que la celda es un maximo sobre
                            # algo, cuando es el valor de una unica combinacion.
                            "reduccion": reduccion if resto else "",
                            "reducido_sobre": ",".join(resto),
                        })

                fig = dibujar_superficie(plt, Z, Zf, ex, ey,
                                         str(metrica), str(modelo), mejor, pie,
                                         elev, azim)
                nombre = f"{modelo}__{metrica}__{eje_x}_x_{eje_y}.png"
                fig.savefig(out_dir / nombre, dpi=140)
                n += 1

                vistas.append({
                    "metrica": str(metrica), "modelo": str(modelo),
                    "ejes": etq_ejes, "mejor": mejor, "pie": pie,
                    **_geometria_json(ex, ey),
                    "superficies": [{"modelo": str(modelo),
                                     "Z": _z_json(Z), "Zf": _z_json(Zf)}],
                })
                if metrica == puntuacion.NOMBRE:
                    rejillas_score[str(modelo)] = (Z, Zf)

                idx = _optimo(Z, mejor)
                aviso = ""
                if idx is not None and en_borde(Z, idx):
                    aviso = ("   <- optimo en el BORDE de la rejilla: el rango "
                             "barrido puede quedarse corto")
                print(f"  {nombre}{aviso}")

                if mostrar:
                    figuras.append(fig)
                else:
                    plt.close(fig)

        # Comparativa: todos los modelos de una entidad sobre los mismos ejes. Es
        # la unica figura que superpone modelos, y solo vale con el score porque
        # es lo unico comparable entre ellos (z-scoreado dentro de la entidad).
        # Jugador y equipo nunca se mezclan.
        for entidad in sorted({entidad_de.get(m, "?") for m in rejillas_score}):
            de_esa = {m: par for m, par in rejillas_score.items()
                      if entidad_de.get(m) == entidad}
            if len(de_esa) < 2:
                continue
            pie = PIE_SCORE + " " + _pie_de_figura(ex, ey, resto, reduccion, "max", fijos)
            fig = dibujar_comparativa(plt, de_esa, ex, ey, str(entidad), pie,
                                      elev, azim)
            nombre = f"comparativa_{entidad}__{puntuacion.NOMBRE}__{eje_x}_x_{eje_y}.png"
            fig.savefig(out_dir / nombre, dpi=140)
            n += 1
            vistas.append({
                "metrica": puntuacion.NOMBRE, "modelo": f"(comparativa {entidad})",
                "ejes": etq_ejes, "mejor": "max", "pie": pie,
                **_geometria_json(ex, ey),
                "superficies": [{"modelo": m, "Z": _z_json(Z), "Zf": _z_json(Zf)}
                                for m, (Z, Zf) in sorted(de_esa.items())],
            })
            # El ganador se decide sobre los valores MEDIDOS, no sobre la
            # estimacion: nadie ha evaluado los puntos intermedios.
            ganador = max(de_esa.items(),
                          key=lambda kv: np.nanmax(kv[1][0])
                          if np.any(np.isfinite(kv[1][0])) else -np.inf)
            print(f"  {nombre}   <- mejor score de {entidad}: {ganador[0]} "
                  f"({np.nanmax(ganador[1][0]):+.3f})")
            if mostrar:
                figuras.append(fig)
            else:
                plt.close(fig)

    if vacias:
        print(f"  [aviso] {len(vacias)} serie(s) sin ningun valor finito, sin "
              f"figura: {', '.join(sorted(set(vacias)))}")

    if filas_csv:
        pd.DataFrame(filas_csv).to_csv(out_dir / "rejilla_3d.csv", index=False)

    if con_html and vistas:
        ruta = figuras3d_html.escribir(
            vistas, out_dir / "superficies.html",
            f"Superficies del barrido — {Path(barrido_dir).name}",
        )
        print(f"\nVisor interactivo (girar con el raton): {ruta}")

    if mostrar and figuras:
        print(f"\nMostrando {len(figuras)} figura(s) interactivas (cierra las "
              "ventanas para terminar).")
        plt.show()
    return n


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _lista(txt: str | None) -> set[str] | None:
    if not txt:
        return None
    return {v.strip() for v in txt.split(",") if v.strip()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Superficies 3D (hiperparametro x hiperparametro x metrica) "
                    "de un barrido ya generado."
    )
    p.add_argument("--barrido", type=Path, default=ecfg.DEFAULT_OUT_DIR / "barrido",
                   help="Carpeta del barrido (la que contiene barrido_metricas.csv "
                        "y resumen_barrido.md).")
    p.add_argument("--out", type=Path, default=None,
                   help="Destino de las figuras (por defecto <barrido>/figuras3d).")
    p.add_argument("--metricas", type=str, default=None,
                   help="Subconjunto de metricas (coma-separado), p. ej. "
                        "'top1,mrr,knn_accuracy'. Por defecto, todas las del CSV.")
    p.add_argument("--modelos", type=str, default=None,
                   help="Subconjunto de modelos por etiqueta compacta "
                        "(F<formulacion>_<entidad>_<normalizacion>), coma-separado.")
    p.add_argument("--reduccion", choices=("max", "media"), default="max",
                   help="Como se reduce cada casilla cuando el barrido mueve mas "
                        "de dos ejes: 'max' = lo mejor alcanzable ahi (default), "
                        "'media' = el valor tipico. Sin efecto con dos ejes.")
    p.add_argument("--elev", type=float, default=26.0, help="Elevacion de la camara.")
    p.add_argument("--azim", type=float, default=-128.0, help="Azimut de la camara.")
    p.add_argument("--escala", choices=malla.ESCALAS, default="auto",
                   help="Escala de los ejes. 'auto' (default) la elige por eje: "
                        "logaritmica si sus valores son claramente multiplicativos, "
                        "lineal si no. 'ordinal' reparte los valores a intervalos "
                        "iguales (el comportamiento anterior): deforma la superficie, "
                        "pero es lo unico posible con ejes no numericos.")
    p.add_argument("--densidad", type=int, default=malla.DENSIDAD,
                   help="Nodos por eje de la malla con que se estima la superficie "
                        "entre los puntos medidos (PCHIP monotono). Con 0 se dibuja "
                        "la rejilla cruda, sin interpolar. No cambia ningun numero: "
                        "solo la resolucion del dibujo.")
    p.add_argument("--mostrar", action="store_true",
                   help="Abre las figuras en ventanas interactivas (rotables) "
                        "ademas de guardarlas. Con muchas figuras abre muchas "
                        "ventanas: combinalo con --metricas/--modelos.")
    p.add_argument("--sin-score", action="store_true",
                   help="No calcula el score compuesto (ni sus figuras ni "
                        "score.csv): solo las metricas tal cual salen del barrido.")
    p.add_argument("--sin-html", action="store_true",
                   help="No escribe el visor interactivo superficies.html.")
    args = p.parse_args(argv)

    out_dir = args.out or (args.barrido / "figuras3d")
    n = generar(
        barrido_dir=args.barrido, out_dir=out_dir,
        metricas_sel=_lista(args.metricas), modelos_sel=_lista(args.modelos),
        reduccion=args.reduccion, elev=args.elev, azim=args.azim,
        escala=args.escala, densidad=args.densidad,
        mostrar=args.mostrar, con_score=not args.sin_score,
        con_html=not args.sin_html,
    )
    print(f"\nListo: {n} figura(s) en {out_dir} (+ rejilla_3d.csv"
          f"{'' if args.sin_score else ' + score.csv'}"
          f"{'' if args.sin_html else ' + superficies.html'}).")


if __name__ == "__main__":
    main()
