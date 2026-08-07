"""Nombres legibles de las métricas del modelo.

Los nombres de feature son los de la BD (`np_goals_minus_npxg`, `padj_def_actions`)
y sirven para el pipeline, no para leerlos en una tabla de scouting. Aquí se
traduce cada uno a español; lo que no esté catalogado cae a una versión legible
del propio nombre en vez de fallar, para que añadir una métrica al pipeline no
tumbe la interfaz.

Las definiciones exactas de cada métrica están en `docs/metricas_finales.md`
(criterio StatsBomb salvo nota); estas etiquetas son el rótulo, no la definición.
"""

from __future__ import annotations

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
