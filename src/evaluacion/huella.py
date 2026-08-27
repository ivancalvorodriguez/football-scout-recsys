"""Huella de un modelo: que hiperparametros, datos y codigo lo determinan.

El barrido reconstruia la rejilla entera en cada ejecucion, aunque la mayoria de
los modelos fuesen identicos a los de otra combinacion de hiperparametros (un
cambio en `F2_L1` no toca los modelos F5, y la posicion no toca los de equipo).
Este modulo permite saltarse esas reconstrucciones: junto a cada artefacto se
guarda un `<stem>.huella.json` con TODO lo que lo determina, y solo se reutiliza
el artefacto si la huella coincide exactamente con la que se iba a construir
(`coincide`): los hiperparametros del modelo pedido tienen que ser TODOS iguales
a los del guardado, y ademas tienen que cuadrar los datos y el codigo.

La huella tiene tres partes:

1. **Hiperparametros** de `src.similitud.config`, con ALCANCE por celda: en la
   huella de un modelo F2 no entran los `F5_*` ni al reves, y en la de un modelo
   de equipo no entran ni los `PLAYER_*` ni los de posicion. Ese alcance es lo
   que hace que dos combinaciones compartan modelos entre si.
2. **Datos**: tamaño y mtime de la BD. Reextraer estadisticas invalida.
3. **Codigo**: hash del fuente de los modulos que producen el numero, tambien
   con alcance (tocar `slim.py` no invalida los modelos F5). Sin esto la caché
   serviria una S vieja en silencio despues de editar el nucleo numerico, que es
   el fallo mas peligroso que puede tener un cache de resultados.

`config.py` NO entra en el hash de codigo: sus VALORES ya estan en la parte 1, y
si entrase, cambiar un comentario invalidaria la rejilla completa.

El reparto de atributos se valida contra el modulo real: si alguien añade un
hiperparametro a `src.similitud.config` y no lo clasifica aqui, `calcular` falla
en vez de generar huellas incompletas que reutilizarian modelos equivocados.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.similitud import config as scfg
from src.similitud import distancias

SUFIJO = ".huella.json"

# --------------------------------------------------------------------------- #
# Alcance de los hiperparametros: que atributos determinan que celda            #
# --------------------------------------------------------------------------- #
# Derivado de los usos reales de `config.X` en `src/similitud/` (features, data,
# formulacion2, formulacion5, distributional). Si mueves un uso de modulo, mueve
# el atributo aqui.

_COMUNES = ("F_CLIP_Z",)

_POR_ENTIDAD: dict[str, tuple[str, ...]] = {
    "jugador": (
        "PLAYER_COUNT_FEATURES", "PLAYER_RATIO_FEATURES", "PLAYER_DIFF_FEATURES",
        # La posicion solo entra en el vector del jugador; por eso los modelos de
        # equipo son reutilizables entre combinaciones que solo mueven la posicion.
        "USE_POSITION_FEATURES", "POSITION_SCALING", "POSITION_FEATURES",
    ),
    "equipo": (
        "TEAM_COUNT_FEATURES", "TEAM_RATE_FEATURES", "TEAM_RATIO_FEATURES",
        "TEAM_DIFF_FEATURES",
    ),
}

_POR_FORMULACION: dict[str, tuple[str, ...]] = {
    "2": ("F2_N_NEIGHBORS", "F2_BETA", "F2_L1", "F2_MAX_ITER", "F2_TOL"),
    "5": ("F5_EASE_LAMBDA", "F5_RFF_DIM", "F5_RFF_SEED",
          "F5_SINKHORN_REG", "F5_SINKHORN_ITERS"),
}

# El metodo distribucional de la F5 se elige por entidad (formulacion5.py:37).
_METODO_F5 = {"jugador": "F5_METODO_JUGADOR", "equipo": "F5_METODO_EQUIPO"}

# Atributos que NO influyen en el artefacto QUE CONSTRUYE EL BARRIDO: rutas por
# defecto, parametros de consulta, el catalogo del que ya se deriva
# POSITION_FEATURES y las dos tablas que describen lo que SIRVE la app.
#
# `MODELOS_SERVIBLES` no entra en ningun ajuste: selecciona entre artefactos ya
# construidos, cada uno con su propia huella.
#
# `HIPERPARAMETROS_SERVIBLES` si cambia un ajuste, pero solo el de `build` y el
# de `src.incremental.reentrenar`, que son los que lo aplican (ver
# `similitud.config.hiperparametros_servibles`) y no llevan huella. El barrido
# nunca pasa por ahi: construye con `evaluacion.construccion`, donde cada
# combinacion fija sus propios valores y esos SI estan en la huella, uno a uno,
# via `_POR_FORMULACION`. Meterlo aqui como si fuera un hiperparametro mas
# invalidaria la rejilla acumulada cada vez que se promociona un modelo nuevo.
#
# `DISTANCIA_SERVIBLE` esta aqui por lo mismo que `MODELOS_SERVIBLES`: no entra
# en ningun ajuste, solo dice cual de los artefactos ya construidos sirve la app.
# La distancia con la que SE AJUSTA una celda no es un hiperparametro de
# `config`: es parte de la identidad de la celda, y va en `celda` (ver `calcular`).
#
# Y por eso este modulo NO lo usa como valor por defecto en ningun sitio: lo que
# falta cuando una huella o una meta antigua no declara distancia es la geometria
# HISTORICA (`distancias.POR_DEFECTO`, la euclidea, la unica que habia entonces),
# que desde el 25-8-2026 ya no es la que sirve la app. Confundirlos releeria toda
# la rejilla acumulada como si fuera mahalanobis.
_IRRELEVANTES = frozenset({
    "DEFAULT_DB_PATH", "DEFAULT_MODEL_DIR", "DEFAULT_TOP_K", "POSITIONS_25",
    "MODELOS_SERVIBLES", "HIPERPARAMETROS_SERVIBLES", "DISTANCIA_SERVIBLE",
})

# --------------------------------------------------------------------------- #
# Alcance del codigo                                                            #
# --------------------------------------------------------------------------- #
# `config.py` se excluye a proposito (ver docstring del modulo).

# `warm.py` entra aunque un ajuste EN FRIO (el que hace el barrido) no reutilice
# nada: lo llaman las dos formulaciones, y si un cambio ahi hiciera que se diera
# por reutilizable algo que no lo es, la S cambiaria. Preferimos invalidar la
# cache de mas a servir un modelo construido con otro codigo.
#
# `distancias.py` entra en las dos formulaciones: define la geometria (y las
# constantes que la afinan, como el piso del blanqueo o las pasadas de IRLS), asi
# que tocarlo cambia cualquier artefacto que no sea euclideo — y hasta el
# euclideo si se toca la funcion que todos comparten.
_CODIGO_COMUN = ("features.py", "data.py", "modelo.py", "warm.py", "distancias.py")
_CODIGO_POR_FORMULACION: dict[str, tuple[str, ...]] = {
    # formulacion5 usa `slim` para el EASE, asi que slim.py entra en las dos.
    "2": ("formulacion2.py", "slim.py"),
    "5": ("formulacion5.py", "distributional.py", "slim.py"),
}

_DIR_SIMILITUD = Path(__file__).resolve().parent.parent / "similitud"

_NOMBRE_HIPERPARAMETRO = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _hiperparametros_declarados() -> set[str]:
    """Union de todos los atributos clasificados en este modulo."""
    declarados = set(_COMUNES) | _IRRELEVANTES | set(_METODO_F5.values())
    for grupo in (*_POR_ENTIDAD.values(), *_POR_FORMULACION.values()):
        declarados |= set(grupo)
    return declarados


def _comprobar_cobertura() -> None:
    """Falla si `config` tiene un hiperparametro sin clasificar.

    Es la red de seguridad del alcance: un atributo nuevo sin clasificar quedaria
    fuera de la huella y la caché podria devolver un modelo construido con otro
    valor. Preferimos romper el barrido a servir un modelo equivocado.
    """
    reales = {
        n for n in vars(scfg)
        if _NOMBRE_HIPERPARAMETRO.match(n) and not callable(getattr(scfg, n))
    }
    sin_clasificar = sorted(reales - _hiperparametros_declarados())
    if sin_clasificar:
        raise RuntimeError(
            "hiperparametros de src.similitud.config sin clasificar en "
            f"src/evaluacion/huella.py: {sin_clasificar}. Añadelos al alcance que "
            "les corresponda (o a _IRRELEVANTES si no afectan al artefacto) para "
            "que la caché del barrido no reutilice modelos equivocados."
        )


def alcance(formulacion: str, entidad: str) -> list[str]:
    """Atributos de config que determinan la celda (formulacion, entidad).

    Publico porque el barrido lo usa para saber que hiperparametros tiene sentido
    barrer con la rejilla que se le ha pedido.
    """
    attrs = [*_COMUNES, *_POR_ENTIDAD[entidad], *_POR_FORMULACION[formulacion]]
    if formulacion == "5":
        attrs.append(_METODO_F5[entidad])
    return sorted(attrs)


def alcance_datos(entidad: str) -> list[str]:
    """Atributos que determinan la MATRIZ DE FEATURES de esa entidad.

    Es el subconjunto de `alcance` que no depende de la formulacion: lo que fija
    `construccion.crear_contexto` (que columnas se derivan, como se recorta el
    z-score, si entra el bloque de posicion). Un trabajador del barrido lo usa
    como clave para reutilizar un contexto entre tareas: dos celdas con el mismo
    alcance de datos comparten la matriz X exactamente, aunque ajusten
    formulaciones o hiperparametros distintos.
    """
    return sorted([*_COMUNES, *_POR_ENTIDAD[entidad]])


def alcance_ajuste(formulacion: str, entidad: str) -> list[str]:
    """Atributos que determinan el AJUSTE, dada ya la matriz de features.

    El complemento de `alcance_datos` dentro de `alcance`. Sirve para saber si la
    W instancia-instancia cacheada en un contexto reutilizado sigue valiendo: la W
    depende de los `F2_*`, asi que hay que descartarla en cuanto cambian.
    """
    datos = set(alcance_datos(entidad))
    return [attr for attr in alcance(formulacion, entidad) if attr not in datos]


def _hash_codigo(formulacion: str) -> str:
    """sha256 del fuente de los modulos que producen el numero de esa celda."""
    h = hashlib.sha256()
    for nombre in sorted({*_CODIGO_COMUN, *_CODIGO_POR_FORMULACION[formulacion]}):
        h.update(nombre.encode("utf-8"))
        h.update((_DIR_SIMILITUD / nombre).read_bytes())
    return h.hexdigest()


def _huella_datos(db_path: Path) -> dict:
    st = Path(db_path).stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def procedencia(db_path: Path) -> dict:
    """Con que datos y que codigo se esta evaluando AHORA (para el barrido acumulado).

    No identifica una celda concreta: identifica la EJECUCION. El barrido lo anota
    junto a cada combinacion que evalua, para poder avisar de que una carpeta
    acumula metricas producidas con una BD o un nucleo numerico distintos (que no
    son comparables entre si aunque compartan tabla). Se hashean SIEMPRE las dos
    formulaciones, tambien las que esa ejecucion no construya: si dependiera de
    `--formulaciones`, dos ejecuciones de la misma carpeta pareceran discrepar solo
    por haberse acotado distinto.
    """
    return {
        "datos": _huella_datos(db_path),
        "codigo": {f: _hash_codigo(f) for f in sorted(_CODIGO_POR_FORMULACION)},
    }


def calcular(formulacion: str, entidad: str, normalizacion: str, db_path: Path,
             excluir_ligas: tuple[str, ...] = (),
             distancia: str = distancias.POR_DEFECTO) -> dict:
    """Huella de la celda con la configuracion VIGENTE en `src.similitud.config`.

    Se llama dentro del `_config_temporal` de la combinacion, asi que recoge sus
    valores.

    ``excluir_ligas`` forma parte de la huella y no es un adorno: un artefacto
    ajustado sin una liga cubre OTRO universo de entidades que el mismo ajuste con
    ella. Sin esto, dos ejecuciones sobre la misma carpeta —una con hold-out y
    otra sin el— se reutilizarian mutuamente la cache y las metricas describirian
    un modelo que no es el que dicen.

    ``distancia`` va en la CELDA y no en los hiperparametros: no es un valor que
    se afine dentro de un modelo, es que modelo es. Dos celdas que solo difieren
    en ella son artefactos distintos, con nombre distinto en disco y fila propia
    en las tablas — exactamente igual que la normalizacion.
    """
    _comprobar_cobertura()
    h = {
        "version": 1,
        "celda": {
            "formulacion": formulacion,
            "entidad": entidad,
            "normalizacion": normalizacion,
            "distancia": distancia,
        },
        "hiperparametros": {
            attr: getattr(scfg, attr) for attr in alcance(formulacion, entidad)
        },
        "datos": _huella_datos(db_path),
        "excluir_ligas": sorted(excluir_ligas),
        "codigo": _hash_codigo(formulacion),
    }
    # Se devuelve ya en forma canonica JSON: varios hiperparametros son tuplas
    # (PLAYER_RATIO_FEATURES...) y json las relee como listas, asi que sin esto
    # una huella recien calculada nunca seria igual a la misma huella leida de
    # disco y la cache jamas acertaria.
    return json.loads(json.dumps(h, sort_keys=True))


# --------------------------------------------------------------------------- #
# Persistencia junto al artefacto                                               #
# --------------------------------------------------------------------------- #

def ruta(model_dir: Path, stem: str) -> Path:
    return Path(model_dir) / f"{stem}{SUFIJO}"


def escribir(model_dir: Path, stem: str, h: dict) -> Path:
    """Guarda la huella. Se llama DESPUES de construir con exito."""
    p = ruta(model_dir, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(h, ensure_ascii=False, indent=2, sort_keys=True),
                 encoding="utf-8")
    return p


def leer(model_dir: Path, stem: str) -> dict | None:
    """Huella guardada, o None si no existe o esta corrupta."""
    p = ruta(model_dir, stem)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def invalidar(model_dir: Path, stem: str) -> None:
    """Borra la huella ANTES de reconstruir.

    Orden deliberado: si la construccion peta a mitad, el artefacto queda a
    medias pero sin huella, asi que nunca se reutilizara.
    """
    ruta(model_dir, stem).unlink(missing_ok=True)


def artefacto_completo(model_dir: Path, stem: str) -> bool:
    """El par .npz + .json existe (una huella sin artefacto no sirve)."""
    d = Path(model_dir)
    return (d / f"{stem}.npz").is_file() and (d / f"{stem}.json").is_file()


def hiperparametros_iguales(guardada: dict, pedida: dict) -> bool:
    """True si el modelo pedido y el guardado tienen los MISMOS hiperparametros.

    Comparacion clave a clave, y exigiendo ademas el mismo juego de claves: un
    artefacto cuya huella declare menos atributos (escrita con un alcance
    anterior) no vale como "igual", porque de los que le faltan no se sabe con
    que valor se construyo.

    Solo mira la parte 1 de la huella. Que los hiperparametros sean identicos no
    basta para reutilizar el artefacto —tambien tienen que cuadrar la celda, los
    datos y el hash del codigo—; eso lo comprueba `coincide`.
    """
    a = guardada.get("hiperparametros")
    b = pedida.get("hiperparametros")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.keys() != b.keys():
        return False
    return all(a[attr] == b[attr] for attr in b)


def _celda(h: dict | None) -> dict:
    """La celda de una huella, con la distancia rellenada si no la declara.

    Una huella escrita antes de que la distancia fuera parte de la identidad no
    la trae, y era `euclidea` (la unica que habia). Sin esta lectura, cada
    artefacto de las carpetas de barrido ya existentes contaria como «otra
    celda» y se reconstruiria aunque nada hubiera cambiado.
    """
    celda = dict((h or {}).get("celda") or {})
    celda.setdefault("distancia", distancias.POR_DEFECTO)
    return celda


def coincide(model_dir: Path, stem: str, h: dict) -> bool:
    """El artefacto guardado se construyo con exactamente esta huella.

    Se comprueba parte por parte, y todas tienen que dar igual: mismos
    hiperparametros (todos), misma celda, mismos datos y mismo codigo. Basta con
    que una difiera para reconstruir.
    """
    if not artefacto_completo(model_dir, stem):
        return False
    guardada = leer(model_dir, stem)
    if guardada is None:
        return False
    return (
        guardada.get("version") == h["version"]
        and _celda(guardada) == _celda(h)
        and hiperparametros_iguales(guardada, h)
        and guardada.get("datos") == h["datos"]
        and guardada.get("codigo") == h["codigo"]
    )


# --------------------------------------------------------------------------- #
# Adopcion de artefactos anteriores a la caché (opt-in)                         #
# --------------------------------------------------------------------------- #
# Los modelos construidos antes de que existiera este modulo no tienen huella,
# asi que por defecto se reconstruyen. `adoptar` permite recuperarlos SI su
# metadata concuerda con la configuracion vigente: se les escribe la huella y
# quedan cargados como si los hubiera construido esta ejecucion. Es una decision
# del usuario (`--adoptar-existentes`), no un comportamiento por defecto: hay una
# parte de la huella que un artefacto viejo no permite verificar (el hash del
# codigo y la huella de la BD), y adoptarlo equivale a afirmar que ni el nucleo
# numerico ni los datos han cambiado desde que se construyo.

# Campos del `meta` del artefacto contrastables con la config vigente. Los que no
# aparecen aqui (F2_MAX_ITER, F2_TOL, F_CLIP_Z...) no se guardan en el meta y por
# tanto no se pueden verificar.
_META_CONTRASTABLE: dict[str, dict[str, str]] = {
    "2": {"n_neighbors": "F2_N_NEIGHBORS", "beta": "F2_BETA", "l1": "F2_L1"},
    "5": {"ease_lambda": "F5_EASE_LAMBDA", "rff_dim": "F5_RFF_DIM",
          "sinkhorn_reg": "F5_SINKHORN_REG"},
}


def _meta_artefacto(model_dir: Path, stem: str) -> dict | None:
    p = Path(model_dir) / f"{stem}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def adoptar(model_dir: Path, stem: str, entidad: str, h: dict) -> bool:
    """Escribe la huella `h` para un artefacto ya existente si su meta concuerda.

    Devuelve False (y no escribe nada) si falta el artefacto, si algun
    hiperparametro guardado difiere del vigente, o si la posicion esta implicada:
    el meta registra `feat_names`, que delata SI hay columnas `pos_*` pero no con
    que escalado entraron, asi que los tres modos de `POSITION_SCALING` son
    indistinguibles a posteriori y esos modelos nunca se adoptan. Como el modelo
    base lleva posicion, en la practica solo se adoptan los de equipo.
    """
    if not artefacto_completo(model_dir, stem):
        return False
    guardado = _meta_artefacto(model_dir, stem)
    if guardado is None:
        return False

    meta = guardado.get("meta", {})
    celda = _celda(h)
    if meta.get("normalizacion") != celda["normalizacion"]:
        return False
    # Un artefacto sin `distancia` en su meta es euclideo (es lo unico que se
    # construia entonces), asi que solo se adopta como euclideo.
    if (meta.get("distancia") or distancias.POR_DEFECTO) != celda["distancia"]:
        return False
    if guardado.get("formulacion") != celda["formulacion"]:
        return False
    if guardado.get("entidad") != entidad:
        return False

    # La posicion no es verificable a posteriori: solo se adopta si ni el
    # artefacto la tiene ni la config la pide.
    tiene_pos = any(str(c).startswith("pos_") for c in guardado.get("feat_names", []))
    pide_pos = entidad == "jugador" and bool(scfg.USE_POSITION_FEATURES)
    if tiene_pos or pide_pos:
        return False

    for campo, attr in _META_CONTRASTABLE[celda["formulacion"]].items():
        valor = meta.get(campo)
        # None = no aplica a ese metodo (p. ej. sinkhorn_reg con mmd): no informa.
        if valor is not None and valor != getattr(scfg, attr):
            return False

    escribir(model_dir, stem, h)
    return True
