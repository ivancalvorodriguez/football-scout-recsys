"""Nombres, unidades y definiciones legibles de las métricas del modelo.

Los nombres de feature son los de la BD (`np_goals_minus_npxg`, `padj_def_actions`)
y sirven para el pipeline, no para leerlos en una tabla de scouting. Aquí se
traduce cada uno a español; lo que no esté catalogado cae a una versión legible
del propio nombre en vez de fallar, para que añadir una métrica al pipeline no
tumbe la interfaz.

Tres capas, de menos a más:

- `etiqueta` — el rótulo que acompaña a un número en una tabla.
- `formatear`/`unidad` — cómo se escribe ese número y en qué está expresado.
- `DEFINICIONES` + `glosario_de` — QUÉ mide cada métrica y cómo se calcula, que
  es lo que sirve la página `/glosario`, agrupado por entidad y por fase.

Las definiciones resumen `docs/metricas_finales.md` (catálogo por fase) y
`docs/extraccion_estadisticas.md` (cómo las calcula de verdad este proyecto:
criterio StatsBomb salvo nota, y las aproximaciones declaradas allí). Cuando las
dos difieren manda la segunda, que es lo que el usuario está viendo.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.similitud.consulta import normalizar

from . import fases as cat_fases
from .fases import Fase
from .posiciones import etiqueta_de_slug

PREFIJO_POSICION = "pos_"

ETIQUETAS: dict[str, str] = {
    # --- Progresión -----------------------------------------------------------
    "passes": "Pases intentados",
    "passes_completed": "Pases completados",
    "progressive_passes": "Pases progresivos",
    "progressive_carries": "Conducciones progresivas",
    "passes_into_final_third": "Pases al último tercio",
    "xt": "Amenaza generada (xT)",
    "pass_completion_pct": "% de acierto en el pase",
    # --- Creación -------------------------------------------------------------
    "xa": "Asistencias esperadas (xA)",
    "passes_into_penalty_area": "Pases al área",
    "deep_completions": "Entregas cerca del área",
    "sca": "Acciones que crean tiro (SCA)",
    "gca": "Acciones que crean gol (GCA)",
    "open_play_xg": "xG en juego abierto",
    "build_up_attacks": "Ataques elaborados",
    # --- Finalización ---------------------------------------------------------
    "shots": "Tiros",
    "np_shots": "Tiros sin penaltis",
    "goals": "Goles",
    "np_goals": "Goles sin penaltis",
    "shots_on_target": "Tiros a puerta",
    "xg": "Goles esperados (xG)",
    "npxg": "Goles esperados sin penaltis (npxG)",
    "touches_in_att_pen_area": "Toques en el área rival",
    "sot_pct": "% de tiros a puerta",
    "npxg_per_shot": "Calidad del tiro (npxG por tiro)",
    "xg_per_shot": "Calidad del tiro (xG por tiro)",
    "np_goals_minus_npxg": "Goles menos xG (sin penaltis)",
    "goals_minus_xg": "Goles menos xG",
    # --- Duelos ---------------------------------------------------------------
    "take_ons": "Regates intentados",
    "take_ons_won": "Regates completados",
    "take_ons_pct": "% de regates completados",
    "fouls_drawn": "Faltas recibidas",
    "dispossessed": "Balones perdidos tras entrada rival",
    "tackles": "Entradas",
    "tackles_won": "Entradas ganadas",
    "tackle_pct": "% de éxito en la entrada",
    "dribbled_past": "Veces regateado",
    "duels_total": "Duelos disputados",
    "duels_won": "Duelos ganados",
    "duels_won_pct": "% de duelos ganados",
    # --- Juego aéreo ----------------------------------------------------------
    "aerial_won": "Duelos aéreos ganados",
    "aerial_lost": "Duelos aéreos perdidos",
    "aerial_won_off": "Duelos aéreos ganados en ataque",
    "aerial_won_def": "Duelos aéreos ganados en defensa",
    "aerial_won_pct": "% de duelos aéreos ganados",
    # --- Defensa / presión ----------------------------------------------------
    "interceptions": "Intercepciones",
    "ball_recoveries": "Recuperaciones",
    "padj_def_actions": "Acciones defensivas ajustadas por posesión",
    "blocks": "Bloqueos",
    "clearances": "Despejes",
    "pressures": "Presiones",
    "pressure_regains": "Recuperaciones tras presión",
    "counterpressures": "Contrapresiones",
    "ppda": "Pases rivales por acción defensiva (PPDA)",
    "high_turnovers": "Recuperaciones en campo rival",
    # --- Posesión y estilo (equipo) -------------------------------------------
    "possession_pct": "% de posesión",
    "ten_plus_pass_sequences": "Secuencias de 10 o más pases",
    "open_play_sequences": "Secuencias de juego abierto",
    "passes_per_sequence": "Pases por secuencia",
    "direct_speed": "Verticalidad (metros por segundo de avance)",
    "direct_attacks": "Ataques directos",
    "field_tilt": "Dominio territorial (field tilt)",
    "absolute_width": "Amplitud usada por secuencia",
    "sequence_start_distance": "Altura de inicio de las secuencias",
}


def etiqueta(feature: str) -> str:
    """Nombre de feature -> rótulo en español para la interfaz."""
    if feature.startswith(PREFIJO_POSICION):
        return f"Posición: {etiqueta_de_slug(feature)}"
    conocida = ETIQUETAS.get(feature)
    if conocida is not None:
        return conocida
    return feature.replace("_", " ").capitalize()


# --- Unidades -----------------------------------------------------------------
# Cómo se escribe el valor REAL de cada métrica (el que devuelve `crudos`). Sin
# esto, un 0.7413 de acierto de pase y un 0.7413 de xG se leerían igual, y son
# cosas distintas. Lo que no esté aquí es un conteo y se escribe como tal:
# por 90 minutos en el jugador y por partido en el equipo.
PORCENTAJES: frozenset[str] = frozenset({
    "pass_completion_pct", "take_ons_pct", "aerial_won_pct", "tackle_pct",
    "duels_won_pct", "sot_pct", "possession_pct", "field_tilt",
})
# Valores esperados (xG, xT...) y calidades medias de tiro: dos decimales, que
# es la precisión con la que se publican.
ESPERADOS: frozenset[str] = frozenset({
    "xt", "xa", "xg", "npxg", "open_play_xg", "npxg_per_shot", "xg_per_shot",
})
# Sobre/infrarrendimiento: el signo es la información, así que va explícito.
DIFERENCIAS: frozenset[str] = frozenset({"np_goals_minus_npxg", "goals_minus_xg"})
# Medias por secuencia y ratios sin unidad natural.
RATIOS: frozenset[str] = frozenset({"ppda", "passes_per_sequence"})
METROS: frozenset[str] = frozenset({"absolute_width", "sequence_start_distance"})
VELOCIDADES: frozenset[str] = frozenset({"direct_speed"})


def formatear(feature: str, valor: float | None) -> str:
    """Valor real de una métrica, escrito con su unidad.

    `None` y NaN dan «—»: la métrica no está definida para esa entidad (un
    porcentaje de regate sin ningún regate intentado) o la BD no está disponible.
    No se sustituye por 0, que se leería como un rendimiento nulo real.
    """
    if valor is None or valor != valor:  # NaN != NaN
        return "—"
    if feature.startswith(PREFIJO_POSICION) or feature in PORCENTAJES:
        return f"{valor * 100:.0f} %"
    if feature in ESPERADOS:
        return f"{valor:.2f}"
    if feature in DIFERENCIAS:
        return f"{valor:+.2f}"
    if feature in RATIOS:
        return f"{valor:.2f}"
    if feature in METROS:
        return f"{valor:.1f} m"
    if feature in VELOCIDADES:
        return f"{valor:.2f} m/s"
    return f"{valor:.2f}" if abs(valor) < 10.0 else f"{valor:.1f}"


# Unidad de los conteos, que depende de la entidad: el jugador se normaliza a 90
# minutos y el equipo no (juega el partido completo, y la BD no guarda minutos de
# equipo).
UNIDAD_CONTEO = {
    "jugador": "por 90 minutos",
    "equipo": "por partido",
}
_UNIDAD_CONTEO_CORTA = {"jugador": "por 90 min", "equipo": "por partido"}

# Denominador propio de las métricas que NO son conteos: no se normalizan por
# tiempo sino por otra cosa, y decir «por 90'» de ellas sería falso.
_POR_TIRO: frozenset[str] = frozenset({"npxg_per_shot", "xg_per_shot"})
_POR_SECUENCIA: frozenset[str] = frozenset({
    "passes_per_sequence", "absolute_width", "sequence_start_distance",
})
# Medias del partido que ya vienen calculadas como tales desde la extracción.
_MEDIAS: frozenset[str] = frozenset({"direct_speed", "ppda"})


def unidad(feature: str, entidad: str) -> str:
    """Aclaración de en qué está expresado el valor de una métrica.

    Sin esto, «5,07» y «0,69» se leen igual y no lo son: uno son pases por 90
    minutos y otro xT acumulado por 90 minutos. Devuelve cadena vacía cuando el
    propio número ya lo dice —un porcentaje lleva su `%`— para no repetir.

    Los conteos son el caso por defecto (per-90 o por partido según la entidad),
    de modo que una métrica nueva del pipeline queda etiquetada sin tocar nada;
    lo que se enumera son las excepciones, que se normalizan por otra cosa.
    """
    if feature.startswith(PREFIJO_POSICION):
        return "% de sus minutos"
    if feature in PORCENTAJES:
        return ""            # el propio valor ya lleva el %
    if feature in _POR_TIRO:
        return "por tiro"
    if feature in _POR_SECUENCIA:
        return "media por secuencia"
    if feature in _MEDIAS:
        return "media por partido"
    return _UNIDAD_CONTEO_CORTA.get(entidad, "")


# --- Definiciones -------------------------------------------------------------
# Qué mide cada métrica y cómo se calcula. Una entrada por nombre de feature, no
# por (feature, entidad): las métricas que comparten nombre entre jugador y
# equipo miden lo mismo y solo cambia el denominador, que ya dice `unidad`.
#
# Se escriben con el criterio con el que este proyecto las EXTRAE, que a veces
# es una aproximación declarada de la definición canónica (SCA/GCA, PAdj,
# posesión); cuando es así, se dice, porque un usuario que compare con FBref o
# con StatsBomb tiene que saber por qué no le cuadra el número.

DEFINICIONES: dict[str, str] = {
    # --- Progresión -----------------------------------------------------------
    "passes": "Pases intentados, lleguen o no a un compañero.",
    "passes_completed": "Pases que sí llegan a un compañero.",
    "progressive_passes": (
        "Pases completados que acercan el balón al menos un 25 % de la distancia "
        "que le quedaba al centro de la portería rival. Es el criterio StatsBomb: "
        "FBref y Opta usan otro (10 yardas de avance) y los dos números no son "
        "intercambiables."
    ),
    "progressive_carries": (
        "Conducciones que cumplen el mismo criterio del pase progresivo: reducen "
        "al menos un 25 % de la distancia restante al centro de la portería rival."
    ),
    "passes_into_final_third": (
        "Pases completados que terminan en el tercio ofensivo del campo, sin "
        "contar acciones a balón parado."
    ),
    "xt": (
        "Expected Threat: suma del valor posicional que añaden los pases y las "
        "conducciones, contando solo los incrementos positivos. Usa la "
        "rejilla pública de Karun Singh (12×8), no el xT propietario de StatsBomb."
    ),
    "pass_completion_pct": (
        "Pases completados ÷ pases intentados. Mide precisión, no progresión: "
        "premia el pase seguro hacia atrás, debilidad ya declarada del esquema de "
        "fases de este proyecto."
    ),
    # --- Creación -------------------------------------------------------------
    "xa": (
        "El xG del tiro que sigue al pase, se convierta o no. Mide la calidad de "
        "la ocasión creada, al margen del acierto del rematador."
    ),
    "passes_into_penalty_area": (
        "Pases completados que terminan dentro del área rival, sin contar balón "
        "parado."
    ),
    "deep_completions": (
        "Pases (excluidos los centros) que dejan el balón a 20 metros o menos de "
        "la portería rival."
    ),
    "sca": (
        "Shot-Creating Actions: las dos acciones ofensivas inmediatamente "
        "anteriores a un tiro (pase, conducción, regate o falta recibida). Aquí se "
        "aproximan como las dos últimas acciones del equipo en la posesión que "
        "acaba en tiro."
    ),
    "gca": (
        "Goal-Creating Actions: lo mismo que las SCA, pero contando solo las que "
        "desembocan en gol."
    ),
    "open_play_xg": (
        "xG de los remates de juego abierto (los que StatsBomb marca como Open "
        "Play). Separa la creación estructural de la dependencia del balón parado."
    ),
    "build_up_attacks": (
        "Secuencias de juego abierto de 10 pases o más que acaban en tiro o dejan "
        "al menos un toque en el área rival."
    ),
    # --- Finalización ---------------------------------------------------------
    "shots": "Remates intentados, penaltis incluidos.",
    "np_shots": "Remates intentados sin contar los penaltis.",
    "goals": "Goles marcados, penaltis incluidos.",
    "np_goals": "Goles marcados sin contar los de penalti.",
    "shots_on_target": (
        "Remates que van entre los tres palos, incluidos los que bloquea el último "
        "defensor. No cuenta los penaltis, aunque el nombre no lleve el prefijo "
        "np_ como np_shots o np_goals: es el criterio de StatsBomb y FBref, y es "
        "lo que hace que sot_pct divida dos cuentas comparables."
    ),
    "xg": (
        "Goles esperados: probabilidad de gol de cada remate según el modelo de "
        "StatsBomb (campo shot.statsbomb_xg). Un penalti vale siempre en torno a 0,78, "
        "que es justo lo que distorsiona el perfil de un lanzador."
    ),
    "npxg": (
        "Goles esperados excluyendo los penaltis, que son un remate de valor fijo "
        "y no dicen nada del juego. Es la versión que describe la amenaza que se "
        "genera en el juego."
    ),
    "touches_in_att_pen_area": (
        "Toques dentro del área rival: presencia en la zona de máxima amenaza."
    ),
    "sot_pct": (
        "Remates a puerta ÷ remates intentados, los dos sin contar penaltis. Es "
        "puntería, distinta de la calidad de la ocasión."
    ),
    "npxg_per_shot": (
        "npxG ÷ remates sin penaltis: calidad media del disparo. Distingue a quien "
        "remata desde buenas posiciones de quien acumula xG a base de volumen."
    ),
    "xg_per_shot": (
        "xG ÷ remates: calidad media de la ocasión que el equipo se fabrica."
    ),
    "np_goals_minus_npxg": (
        "Goles sin penaltis menos npxG: cuánto se rinde por encima o por debajo de "
        "lo que decía el modelo. Con pocos remates es más ruido que habilidad "
        "(Davis y Robberechts, 2024), así que se lee con cautela."
    ),
    "goals_minus_xg": (
        "Goles menos xG del equipo: sobre o infrarrendimiento colectivo en el "
        "remate. Mismo aviso que en el jugador: necesita volumen para significar "
        "algo."
    ),
    # --- Duelos ---------------------------------------------------------------
    "take_ons": (
        "Intentos de superar a un rival en un uno contra uno conservando el balón."
    ),
    "take_ons_won": "Regates en los que se supera al rival y se conserva el balón.",
    "take_ons_pct": "Regates completados ÷ regates intentados.",
    "fouls_drawn": (
        "Duelos que el rival solo consigue frenar infringiendo el reglamento."
    ),
    "dispossessed": (
        "Veces que se pierde el balón tras la entrada de un rival. No cuenta los "
        "regates fallados, que son otra cosa. Cuanto más alto, peor."
    ),
    "tackles": "Entradas realizadas sobre un rival que lleva el balón.",
    "tackles_won": "Entradas tras las que el equipo se queda con el balón.",
    "tackle_pct": (
        "Entradas ganadas ÷ (entradas ganadas + veces regateado): éxito en el duelo "
        "defensivo uno contra uno."
    ),
    "dribbled_past": (
        "Veces que un rival supera al jugador en regate. Cuanto más alto, peor."
    ),
    "duels_total": (
        "Duelos disputados de cualquier tipo: entradas, juego aéreo y balones "
        "divididos (el evento 50/50 de StatsBomb). Solapa a propósito con las "
        "métricas específicas de duelo, que lo desglosan."
    ),
    "duels_won": "Duelos ganados de ese total.",
    "duels_won_pct": "Duelos ganados ÷ duelos disputados.",
    # --- Juego aéreo ----------------------------------------------------------
    "aerial_won": (
        "Duelos aéreos ganados. StatsBomb se lo adjudica a quien acaba controlando "
        "el balón, no a quien lo toca primero (criterio de Wyscout)."
    ),
    "aerial_lost": "Duelos aéreos perdidos. Cuanto más alto, peor.",
    "aerial_won_off": (
        "Duelos aéreos ganados en la mitad ofensiva del campo: perfil de rematador."
    ),
    "aerial_won_def": (
        "Duelos aéreos ganados en la mitad defensiva: perfil de defensor por alto."
    ),
    "aerial_won_pct": (
        "Aéreos ganados ÷ aéreos disputados. Con poco volumen engaña, por eso se "
        "lee junto al conteo."
    ),
    # --- Defensa / presión ----------------------------------------------------
    "interceptions": "Balones interceptados en la trayectoria de un pase rival.",
    "ball_recoveries": "Balones sueltos que el jugador recupera para su equipo.",
    "padj_def_actions": (
        "Recuperaciones e intercepciones ajustadas por posesión: quien juega en un "
        "equipo que defiende mucho tiene más ocasiones de intervenir. El ajuste es "
        "simple (acción × 0,5 ÷ posesión rival), no el sigmoide por zonas de "
        "StatsBomb."
    ),
    "blocks": "Remates y pases bloqueados interponiéndose en su trayectoria.",
    "clearances": (
        "Alejar el balón de la zona de peligro sin buscar destinatario."
    ),
    "pressures": (
        "Presiones sobre el rival que lleva el balón. Sale del evento Pressure, "
        "propio de StatsBomb: ningún otro proveedor publica un equivalente, así que "
        "esta métrica no es portable."
    ),
    "pressure_regains": (
        "Presiones tras las que el equipo recupera el balón en 5 segundos o menos."
    ),
    "counterpressures": (
        "Presiones ejercidas en los 5 segundos siguientes a una pérdida en juego "
        "abierto (contrapresión)."
    ),
    "ppda": (
        "Passes Per Defensive Action: pases que el rival completa en su 60 % de "
        "campo por cada acción defensiva propia en esa zona. Cuanto más BAJO, más "
        "arriba y más intensa es la presión."
    ),
    "high_turnovers": (
        "Recuperaciones a 40 metros o menos de la portería rival: robos en zona de "
        "peligro."
    ),
    # --- Posesión y estilo (equipo) -------------------------------------------
    "possession_pct": (
        "Cuota de posesión, aproximada por la cuota de toques del equipo. No es "
        "tiempo de posesión, que el event data no da directamente."
    ),
    "ten_plus_pass_sequences": (
        "Secuencias de juego abierto con 10 pases o más: elaboración larga."
    ),
    "open_play_sequences": (
        "Número de secuencias de juego abierto. Una secuencia es un pasaje de un "
        "solo equipo, y termina con una acción defensiva rival, una interrupción o "
        "un tiro."
    ),
    "passes_per_sequence": (
        "Pases medios por secuencia: la medida más directa de elaborar frente a "
        "jugar directo."
    ),
    "direct_speed": (
        "Metros que la secuencia avanza hacia la portería rival por cada segundo "
        "que dura."
    ),
    "direct_attacks": (
        "Secuencias que arrancan en campo propio, recorren al menos la mitad de "
        "la distancia a la portería rival y acaban en remate o en un toque dentro "
        "del área."
    ),
    "field_tilt": (
        "Cuota de los toques del partido, en el tercio final, que son de este "
        "equipo. Dice dónde se juega, no cuánto se tiene el balón."
    ),
    "absolute_width": (
        "Distancia máxima del balón al eje del campo dentro de cada secuencia, "
        "promediada entre todas."
    ),
    "sequence_start_distance": (
        "Altura del bloque en posesión: distancia media desde la portería propia a "
        "la que el equipo empieza sus secuencias de juego abierto."
    ),
}


# El bloque `pos_*` no es una métrica de juego y no pertenece a ninguna fase (ver
# `servicio._columnas_metricas`), pero sí tiene rótulo, así que se define aparte.
DEFINICION_POSICION = (
    "Fracción de los minutos del jugador disputada en esa posición. Describe qué "
    "es un jugador, no cómo juega: forma parte del vector con el que se ajusta el "
    "modelo, pero no aparece en las listas de métricas ni en el radar de fases."
)


def definicion(feature: str) -> str:
    """Qué mide una métrica; cadena vacía si no está catalogada.

    Vacío y no un texto de relleno: es preferible que la interfaz omita la
    explicación de una métrica nueva a que invente una.
    """
    if feature.startswith(PREFIJO_POSICION):
        return DEFINICION_POSICION
    return DEFINICIONES.get(feature, "")


# --- Glosario por entidad y fase ----------------------------------------------


@dataclass(frozen=True)
class Termino:
    """Una métrica explicada, lista para la página del glosario."""

    feature: str
    etiqueta: str
    definicion: str
    unidad: str
    # True si en esta métrica subir es empeorar (`fases.FEATURES_INVERTIDAS`).
    invertida: bool

    @property
    def orden(self) -> str:
        """Clave alfabética sin acentos ni mayúsculas.

        Con el orden por code point, «Área» acabaría detrás de «Toques».
        """
        return normalizar(self.etiqueta)


@dataclass(frozen=True)
class BloqueFase:
    """Las métricas de una fase de juego, en orden alfabético."""

    fase: Fase
    terminos: tuple[Termino, ...]

    @property
    def clave(self) -> str:
        return self.fase.clave

    @property
    def etiqueta(self) -> str:
        return self.fase.etiqueta

    @property
    def descripcion(self) -> str:
        return self.fase.descripcion


def glosario_de(entidad: str) -> tuple[BloqueFase, ...]:
    """Métricas de una entidad, agrupadas por fase y alfabéticas dentro de cada una.

    La fuente de las fases y de qué métrica va en cada una es `fases.FASES`, la
    misma que alimenta el radar: así el glosario no puede describir un reparto
    distinto del que el usuario ve dibujado. Las fases van en el orden del
    catálogo (que es el de las agujas del reloj en el radar) y las métricas
    dentro de cada una, alfabéticas, que es lo que se busca cuando se consulta un
    glosario: se llega por el nombre, no por el orden de la fase.
    """
    bloques = []
    for fase in cat_fases.fases_de(entidad):
        terminos = sorted(
            (
                Termino(
                    feature=feature,
                    etiqueta=etiqueta(feature),
                    definicion=definicion(feature),
                    unidad=unidad(feature, entidad),
                    invertida=feature in cat_fases.FEATURES_INVERTIDAS,
                )
                for feature in fase.features
            ),
            key=lambda t: t.orden,
        )
        bloques.append(BloqueFase(fase=fase, terminos=tuple(terminos)))
    return tuple(bloques)
