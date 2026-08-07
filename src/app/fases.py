"""Perfil por fases de juego: del vector de features al radar.

La taxonomía de fases es la de `docs/metricas_finales.md`, no la del protocolo de
evaluación de `docs/Como_evaluar.pdf`, que numera fases 0–6 de *validación* y no
tiene nada que ver con esto.

El equipo usa las **7** fases del catálogo. El jugador usa **6**: la séptima
(Perfil/Valor global: VAEP, OBV, Player Vectors, PlayeRank) no es calculable
desde event data abierto, y llenar ese vértice con la posición sería confundir
dos cosas distintas —la posición dice QUÉ es un jugador, no cómo juega—, así que
el radar del jugador es un hexágono y no finge un séptimo eje.

Cómo se calcula cada vértice:

1. Se agrupan las columnas de `feat_display` (z-scores por entidad, media
   ponderada por minutos) según la fase a la que pertenece cada métrica.
2. Las métricas cuyo valor alto es MALO se invierten de signo
   (`FEATURES_INVERTIDAS`), de modo que en todos los ejes «más es mejor» y el
   área del polígono se lee siempre igual.
3. Se promedian los z-scores de la fase. Es una media simple: ponderar sería
   inventar una importancia relativa entre métricas que el proyecto no ha
   medido.
4. Ese promedio se convierte a **percentil** dentro del universo del modelo
   (`percentil_rango`), que es lo que da el rango 0–1 del dibujo. Se prefiere al
   reescalado lineal del z porque este apelotona a casi todas las entidades
   alrededor de 0,5 y aplana la figura.

Aviso que acompaña siempre a la figura: `feat_display` es un resumen a
posteriori (no entra en el ajuste del modelo), así que el radar sirve para
INTERPRETAR una recomendación, no la produce. Y un percentil alto es nivel
relativo dentro del universo cargado, no valor de mercado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Taxonomía ----------------------------------------------------------------


@dataclass(frozen=True)
class Fase:
    """Una fase de juego y las métricas del modelo que la componen."""

    clave: str
    etiqueta: str
    descripcion: str
    features: tuple[str, ...]


# Métricas en las que subir es empeorar. Se les invierte el signo antes de
# promediar para que el eje entero signifique «mejor» hacia fuera; sin esto un
# jugador al que roban muchos balones subiría en Duelos y la figura mentiría.
FEATURES_INVERTIDAS: frozenset[str] = frozenset({
    "dispossessed",   # pérdidas tras ser tackleado
    "dribbled_past",  # veces que le regatean
    "aerial_lost",    # duelos aéreos perdidos
    "ppda",           # pases rivales permitidos por acción defensiva (equipo)
})

FASES_JUGADOR: tuple[Fase, ...] = (
    Fase(
        clave="progresion",
        etiqueta="Progresión",
        descripcion="Hacer avanzar el balón hacia la portería rival.",
        features=(
            "passes", "passes_completed", "progressive_passes",
            "progressive_carries", "passes_into_final_third", "xt",
            "pass_completion_pct",
        ),
    ),
    Fase(
        clave="creacion",
        etiqueta="Creación",
        descripcion="Generar ocasiones para otros.",
        features=("xa", "passes_into_penalty_area", "deep_completions", "sca"),
    ),
    Fase(
        clave="finalizacion",
        etiqueta="Finalización",
        descripcion="Amenaza y acierto en el remate propio.",
        features=(
            "shots", "np_shots", "goals", "np_goals", "shots_on_target",
            "xg", "npxg", "touches_in_att_pen_area", "sot_pct",
            "npxg_per_shot", "np_goals_minus_npxg",
        ),
    ),
    Fase(
        clave="duelos",
        etiqueta="Duelos",
        descripcion="Enfrentamientos 1 contra 1, con y sin balón.",
        features=(
            "take_ons", "take_ons_won", "fouls_drawn", "dispossessed",
            "tackles", "tackles_won", "dribbled_past", "duels_total",
            "duels_won", "take_ons_pct", "tackle_pct", "duels_won_pct",
        ),
    ),
    Fase(
        clave="aereo",
        etiqueta="Juego aéreo",
        descripcion="Dominio en el juego por alto.",
        features=(
            "aerial_won", "aerial_lost", "aerial_won_off", "aerial_won_def",
            "aerial_won_pct",
        ),
    ),
    Fase(
        clave="defensa",
        etiqueta="Defensa",
        descripcion="Trabajo defensivo: presión, recuperación y cobertura.",
        features=(
            "interceptions", "ball_recoveries", "padj_def_actions", "blocks",
            "clearances", "pressures", "pressure_regains", "counterpressures",
        ),
    ),
)

FASES_EQUIPO: tuple[Fase, ...] = (
    Fase(
        clave="posesion",
        etiqueta="Posesión",
        descripcion="Tener y hacer circular el balón.",
        features=(
            "possession_pct", "passes", "passes_completed",
            "pass_completion_pct", "ten_plus_pass_sequences",
        ),
    ),
    Fase(
        clave="progresion",
        etiqueta="Progresión",
        descripcion="Llevar el balón al campo contrario.",
        features=("progressive_passes", "passes_into_final_third", "xt"),
    ),
    Fase(
        clave="creacion",
        etiqueta="Creación",
        descripcion="Generar ocasiones de gol.",
        features=(
            "sca", "gca", "passes_into_penalty_area", "open_play_xg",
            "build_up_attacks",
        ),
    ),
    Fase(
        clave="finalizacion",
        etiqueta="Finalización",
        descripcion="Convertir las ocasiones generadas.",
        features=(
            "shots", "np_shots", "goals", "shots_on_target", "xg", "npxg",
            "xg_per_shot", "sot_pct", "goals_minus_xg",
        ),
    ),
    Fase(
        clave="presion",
        etiqueta="Presión",
        descripcion="Intensidad al recuperar y altura de la recuperación.",
        features=("ppda", "high_turnovers", "pressures", "counterpressures"),
    ),
    Fase(
        clave="territorio",
        etiqueta="Territorio",
        descripcion="Dominio territorial y amplitud del campo usada.",
        features=("field_tilt", "absolute_width", "sequence_start_distance"),
    ),
    Fase(
        clave="estilo",
        etiqueta="Estilo",
        descripcion="Verticalidad: juego directo frente a elaboración.",
        features=("direct_speed", "direct_attacks", "passes_per_sequence",
                  "open_play_sequences"),
    ),
)

FASES: dict[str, tuple[Fase, ...]] = {
    "jugador": FASES_JUGADOR,
    "equipo": FASES_EQUIPO,
}


def fases_de(entidad: str) -> tuple[Fase, ...]:
    """Fases de un tipo de entidad; vacío si la entidad no se conoce."""
    return FASES.get(entidad, ())


def fase_por_feature(entidad: str) -> dict[str, Fase]:
    """Métrica -> fase a la que pertenece, para etiquetar una métrica suelta."""
    return {m: fase for fase in fases_de(entidad) for m in fase.features}


# --- Percentiles --------------------------------------------------------------


def percentil_rango(valores: np.ndarray) -> np.ndarray:
    """Percentil de cada valor dentro del propio vector, en [0, 1].

    Rango medio en los empates (midrank): todas las entidades con el mismo valor
    reciben el mismo percentil, sin que el orden de indexación decida quién va
    delante. Los NaN se propagan como NaN (fase sin métricas medibles) y no
    cuentan para el reparto. Con una sola entidad finita el percentil es 0,5: no
    hay universo contra el que comparar, así que se la sitúa en la mediana en
    vez de darle un 0 o un 1 que sugerirían un extremo.
    """
    valores = np.asarray(valores, dtype=float)
    salida = np.full(valores.shape, np.nan, dtype=float)
    finitos = np.isfinite(valores)
    v = valores[finitos]
    if v.size == 0:
        return salida
    if v.size == 1:
        salida[finitos] = 0.5
        return salida
    _, inverso, cuentas = np.unique(v, return_inverse=True, return_counts=True)
    fin = np.cumsum(cuentas)
    rango_medio = (fin - cuentas + fin - 1) / 2.0
    salida[finitos] = rango_medio[inverso] / (v.size - 1)
    return salida


# --- Matriz de fases ----------------------------------------------------------


@dataclass(frozen=True)
class MatrizFases:
    """Perfil por fases de TODAS las entidades de un modelo.

    Se calcula una vez por artefacto (es O(P·d) y necesita el universo entero
    para los percentiles) y lo cachea `Catalogo`, igual que el propio modelo.

    - `z`: (P, F) promedio de z-scores de la fase, ya con los signos corregidos.
    - `percentil`: (P, F) ese promedio convertido a percentil dentro del modelo.
    - `medibles`: (F,) False en las fases sin ninguna métrica en el artefacto.
    """

    entidad: str
    fases: tuple[Fase, ...]
    z: np.ndarray
    percentil: np.ndarray
    medibles: tuple[bool, ...]


def construir_matriz(
    entidad: str, feat_names: list[str], feat_display: np.ndarray
) -> MatrizFases:
    """Perfil por fases de un modelo a partir de su `feat_display`.

    Una fase cuyas métricas no estén en el artefacto (o la de perfil posicional,
    que por definición no tiene ninguna) queda como NaN y marcada en `medibles`:
    la interfaz la dibuja como hueco, en vez de fingir un 0 que se leería como
    «pésimo en esa fase».
    """
    fases = fases_de(entidad)
    display = np.asarray(feat_display, dtype=float)
    n = display.shape[0] if display.ndim == 2 else 0
    indice = {nombre: i for i, nombre in enumerate(feat_names)}

    z = np.full((n, len(fases)), np.nan, dtype=float)
    medibles: list[bool] = []
    for f, fase in enumerate(fases):
        cols = [indice[m] for m in fase.features if m in indice]
        medibles.append(bool(cols))
        if not cols:
            continue
        bloque = display[:, cols]
        # Signo: en las métricas invertidas, alto = peor.
        signos = np.array(
            [-1.0 if m in FEATURES_INVERTIDAS else 1.0
             for m in fase.features if m in indice]
        )
        z[:, f] = np.nanmean(bloque * signos, axis=1)

    percentil = np.column_stack([percentil_rango(z[:, f]) for f in range(len(fases))]) \
        if len(fases) else np.zeros((n, 0))
    return MatrizFases(
        entidad=entidad,
        fases=fases,
        z=z,
        percentil=percentil,
        medibles=tuple(medibles),
    )


# --- Perfil de una entidad ----------------------------------------------------


@dataclass(frozen=True)
class ValorFase:
    """Un vértice del radar para una entidad concreta."""

    fase: Fase
    percentil: float | None  # None = fase no medible para esta entidad
    z: float | None

    @property
    def etiqueta(self) -> str:
        return self.fase.etiqueta

    @property
    def porcentaje(self) -> str:
        """Percentil listo para pintar («73» o «—»)."""
        if self.percentil is None:
            return "—"
        return f"{self.percentil * 100:.0f}"


def perfil(matriz: MatrizFases, indice: int) -> tuple[ValorFase, ...]:
    """Vértices de la entidad `indice`, en el orden de las fases."""
    salida: list[ValorFase] = []
    for f, fase in enumerate(matriz.fases):
        medible = matriz.medibles[f]
        p = float(matriz.percentil[indice, f]) if medible else float("nan")
        z = float(matriz.z[indice, f]) if medible else float("nan")
        salida.append(ValorFase(
            fase=fase,
            percentil=None if not np.isfinite(p) else p,
            z=None if not np.isfinite(z) else z,
        ))
    return tuple(salida)


# --- Geometría del polígono ---------------------------------------------------
#
# ==========================================================================
#  AQUÍ SE AJUSTA EL TAMAÑO DE LA FIGURA. Son las tres medidas que importan;
#  el cuerpo de letra de los rótulos está en `static/estilo.css`
#  (`.radar-etiquetas text`) y el ancho máximo en pantalla, en `.radar`.
# ==========================================================================
#
# El área de dibujo es un cuadrado de 100x100 con el centro en (50, 50). Todo va
# en esas unidades: la figura escala sola con el `viewBox`, así que subir RADIO
# agranda el polígono Y encoge el texto en pantalla, porque el `viewBox` se
# reparte el mismo ancho de píxeles.
CENTRO = 50.0
RADIO = 33.0            # radio del polígono (el 100 % de un eje)
RADIO_ETIQUETA = 39.0   # a qué distancia del centro se colocan los rótulos

# Hueco alrededor del cuadrado para que quepan los rótulos. Sin él, «Finalización»
# o «Juego aéreo» se salen del `viewBox` y el navegador los RECORTA: el texto
# sigue ahí, pero no se lee. El margen es lateral porque los ejes de arriba y
# abajo van centrados y no sobresalen. Hay un test que comprueba que ningún
# rótulo se sale, para que tocar estas medidas no vuelva a recortarlos.
MARGEN_X = 20.0
MARGEN_Y = 6.0

# `viewBox` resultante. Cuanto más grande, más pequeña se ve la figura entera
# (polígono y texto) dentro del mismo ancho en pantalla.
VIEW_BOX = (
    f"{-MARGEN_X:g} {-MARGEN_Y:g} "
    f"{100.0 + 2 * MARGEN_X:g} {100.0 + 2 * MARGEN_Y:g}"
)

# Un vértice sin dato no se puede dejar en 0 (se leería como «lo peor posible»)
# ni en el borde; se ancla en el centro y la interfaz lo marca aparte.
RADIO_SIN_DATO = 0.0


def _angulo(i: int, n: int) -> float:
    """Ángulo del vértice `i` de `n`, empezando arriba y girando a la derecha."""
    return -math.pi / 2.0 + 2.0 * math.pi * i / n


def vertice(i: int, n: int, radio: float) -> tuple[float, float]:
    """Coordenada del vértice `i` a distancia `radio` del centro."""
    a = _angulo(i, n)
    return (CENTRO + radio * math.cos(a), CENTRO + radio * math.sin(a))


def puntos(valores: list[float | None]) -> str:
    """Atributo `points` de un `<polygon>` a partir de percentiles en [0, 1]."""
    n = len(valores)
    pares = []
    for i, v in enumerate(valores):
        r = RADIO_SIN_DATO if v is None else RADIO * float(v)
        x, y = vertice(i, n, r)
        pares.append(f"{x:.2f},{y:.2f}")
    return " ".join(pares)


def rejilla(n: int, fraccion: float) -> str:
    """Polígono de referencia (todos los vértices al mismo radio)."""
    pares = []
    for i in range(n):
        x, y = vertice(i, n, RADIO * fraccion)
        pares.append(f"{x:.2f},{y:.2f}")
    return " ".join(pares)


@dataclass(frozen=True)
class EtiquetaEje:
    """Rótulo de un vértice, ya colocado para el SVG."""

    texto: str
    x: float
    y: float
    anclaje: str  # text-anchor: start | middle | end


def _etiquetas(valores: tuple[ValorFase, ...]) -> tuple[EtiquetaEje, ...]:
    """Rótulos de los vértices con su posición y alineación.

    El anclaje se deduce del lado del círculo en el que cae el vértice, para que
    el texto no invada la figura: a la derecha empieza en el punto, a la
    izquierda termina en él y arriba/abajo va centrado.
    """
    n = len(valores)
    salida = []
    for i, v in enumerate(valores):
        x, y = vertice(i, n, RADIO_ETIQUETA)
        dx = x - CENTRO
        if abs(dx) < 1.0:
            anclaje = "middle"
        elif dx > 0:
            anclaje = "start"
        else:
            anclaje = "end"
        # Compensa que el texto se alinea por la línea base, no por su centro.
        salida.append(EtiquetaEje(
            texto=v.etiqueta, x=x, y=y + 1.2, anclaje=anclaje
        ))
    return tuple(salida)


# --- Figura lista para la plantilla -------------------------------------------

# Anillos de referencia del fondo (percentiles 25, 50, 75 y 100).
FRACCIONES_REJILLA: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Serie:
    """Un polígono del radar: una entidad dibujada sobre los mismos ejes."""

    nombre: str
    clase: str            # sufijo CSS ('referencia' | 'candidato')
    puntos: str           # atributo `points` del <polygon>
    valores: tuple[ValorFase, ...]


@dataclass(frozen=True)
class Radar:
    """Todo lo que la plantilla necesita para pintar el SVG, ya calculado.

    Se arma en Python y no en Jinja a propósito: la plantilla solo itera, y así
    la geometría (que sí tiene reglas: percentiles, huecos, anclajes) se puede
    probar sin renderizar HTML.
    """

    series: tuple[Serie, ...]
    etiquetas: tuple[EtiquetaEje, ...]
    rejilla: tuple[str, ...]
    radios: tuple[tuple[float, float], ...]   # extremo de cada eje radial
    sin_dato: tuple[str, ...]                 # fases que ninguna serie puede medir
    view_box: str = VIEW_BOX

    @property
    def vacio(self) -> bool:
        return not self.series


def radar(series: list[tuple[str, str, tuple[ValorFase, ...]]]) -> Radar:
    """Construye la figura a partir de `(nombre, clase, perfil)` por entidad.

    Todas las series deben venir de la misma `MatrizFases`, que es lo que
    garantiza que comparten ejes; comparar dos entidades sobre ejes distintos
    daría una figura bonita y sin sentido, así que se rechaza.
    """
    series = [(n, c, v) for n, c, v in series if v]
    if not series:
        return Radar(series=(), etiquetas=(), rejilla=(), radios=(), sin_dato=())
    claves = [tuple(v.fase.clave for v in valores) for _, _, valores in series]
    if len(set(claves)) != 1:
        raise ValueError("Las series del radar no comparten las mismas fases.")

    referencia = series[0][2]
    n = len(referencia)
    construidas = tuple(
        Serie(
            nombre=nombre,
            clase=clase,
            puntos=puntos([v.percentil for v in valores]),
            valores=valores,
        )
        for nombre, clase, valores in series
    )
    sin_dato = tuple(
        referencia[i].etiqueta
        for i in range(n)
        if all(s.valores[i].percentil is None for s in construidas)
    )
    return Radar(
        series=construidas,
        etiquetas=_etiquetas(referencia),
        rejilla=tuple(rejilla(n, f) for f in FRACCIONES_REJILLA),
        radios=tuple(vertice(i, n, RADIO) for i in range(n)),
        sin_dato=sin_dato,
    )
