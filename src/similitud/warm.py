"""Estado reutilizable entre ajustes: soporte del warm start.

Un modelo servible (`modelo.py`) guarda lo que hace falta para RESPONDER
consultas: la S entidad-entidad y el indice de entidades. Eso no basta para
reentrenar reaprovechando el trabajo anterior, porque el ajuste vive un nivel mas
abajo (la W instancia-instancia de la Formulacion 2, el ancho del kernel o la
matriz de costes OT de la Formulacion 5). Este modulo guarda ESE nivel.

El estado se serializa aparte del artefacto servible (`<stem>.warm.npz`), para
que la app y los CLI de consulta sigan viendo exactamente los mismos ficheros que
antes y para poder borrarlo sin perder el modelo: sin estado, el reentrenamiento
simplemente arranca en frio.

Piezas:

- `EstadoWarm`: lo aprendido, mas la identidad de cada observacion
  (entidad + partido) y una huella de su vector de features.
- `emparejar`: cruza el estado previo con la matriz de features actual y decide,
  fila a fila y entidad a entidad, que sigue siendo lo mismo. Todo lo que
  reutilizan las formulaciones se apoya en esa decision.
- `inicializador_f2`: traduce la W anterior al espacio de indices de ahora, que
  es lo que `slim.slim_instancia` consume como punto de arranque.

Que la reutilizacion sea EXACTA o solo una buena inicializacion depende de la
pieza: los costes OT y los embeddings de una nube intacta son identitos numero a
numero; el arranque del coordinate descent solo ahorra iteraciones (el optimo es
unico, ver `slim.elasticnet_no_negativo`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, distancias
from .distancias import Espacio
from .features import MatrizFeatures
from .modelo import stem_artefacto

# Sufijo del fichero de estado, junto al artefacto servible.
SUFIJO_ESTADO = ".warm.npz"


def hiper_features() -> dict:
    """Ajustes de la capa de features que cambian el valor de X.

    No estan en `feat_names` (las columnas se llaman igual con cualquiera de
    ellos) pero mueven todos los numeros, asi que forman parte de la identidad
    del ajuste: con otros valores, el estado anterior no es comparable.
    """
    return {
        "clip_z": config.F_CLIP_Z,
        "usa_posicion": config.USE_POSITION_FEATURES,
        "escala_posicion": config.POSITION_SCALING,
    }


def ruta_estado(
    model_dir: Path, formulacion: str, entidad: str, normalizacion: str,
    distancia: str = distancias.POR_DEFECTO,
) -> Path:
    """Ruta del estado warm de un modelo (misma convencion que `modelo.guardar`).

    El nombre lo fija `modelo.stem_artefacto`, que es quien manda: el estado
    warm siempre va al lado de su artefacto y con su mismo nombre.
    """
    stem = stem_artefacto(formulacion, entidad, normalizacion, distancia)
    return Path(model_dir) / f"{stem}{SUFIJO_ESTADO}"


def huellas_filas(X: np.ndarray) -> np.ndarray:
    """Huella de 64 bits de cada fila de X, para detectar si ha cambiado.

    Se hashean los bytes crudos del vector. Con las estadisticas de
    normalizacion congeladas (ver `features.construir`), una observacion cuyos
    conteos no han cambiado produce EXACTAMENTE los mismos bytes, asi que la
    igualdad de huella es fiable; si algo cambia (aunque sea el ultimo bit) la
    huella difiere y la fila se trata como nueva, que es el lado seguro del
    error. Se suma 0.0 para normalizar el -0.0 a 0.0: son el mismo numero pero
    distintos bytes.
    """
    X = np.ascontiguousarray(X, dtype=np.float64) + 0.0
    salida = np.empty(X.shape[0], dtype=np.uint64)
    for i in range(X.shape[0]):
        digest = hashlib.blake2b(X[i].tobytes(), digest_size=8).digest()
        salida[i] = int.from_bytes(digest, "little")
    return salida


@dataclass
class EstadoWarm:
    """Lo que un ajuste deja para el siguiente."""

    formulacion: str
    entidad: str
    normalizacion: str
    feat_names: list[str]
    hiper: dict

    # Identidad de cada observacion (fila de X) y huella de su vector.
    obs_entity: np.ndarray            # (M,) entity_id
    obs_match: np.ndarray             # (M,) match_id
    obs_hash: np.ndarray              # (M,) uint64
    entity_ids: np.ndarray            # (P,) ordenados, como en el modelo

    # Distancia con la que se ajusto y lo que esa distancia aprendio de los datos
    # (hoy solo el blanqueo de mahalanobis). Se guarda por el mismo motivo que
    # `sigma`: proyectar o reentrenar con una geometria REESTIMADA pondria a las
    # entidades nuevas en otro espacio que las del modelo. Un estado escrito
    # antes de que la distancia fuera un parametro se relee como `euclidea`, que
    # es la que tenia.
    distancia: str = distancias.POR_DEFECTO
    espacio_L: np.ndarray | None = None

    # Formulacion 2: W dispersa por columnas, en formato CSC.
    w_ptr: np.ndarray | None = None   # (M+1,) inicio de cada columna en w_idx
    w_idx: np.ndarray | None = None   # (nnz,) fila r con peso
    w_val: np.ndarray | None = None   # (nnz,) valor W_rs

    # Formulacion 5.
    sigma: float | None = None            # ancho del kernel RBF (metodo mmd)
    costos: np.ndarray | None = None      # (P,P) costes OT (metodo sinkhorn)

    @property
    def n_observaciones(self) -> int:
        return int(self.obs_entity.shape[0])

    @property
    def n_entidades(self) -> int:
        return int(self.entity_ids.shape[0])

    def espacio(self) -> Espacio:
        """La geometria del ajuste, lista para volver a usarla tal cual.

        Es lo que consumen el reentrenamiento (para no reestimar el blanqueo con
        los datos ampliados) y `foldin` (para proyectar en el mismo espacio).
        """
        return Espacio(self.distancia, L=self.espacio_L)

    def columna(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, valores) de la columna s de W. Vacias si no hay W guardada."""
        if self.w_ptr is None or self.w_idx is None or self.w_val is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        ini, fin = int(self.w_ptr[s]), int(self.w_ptr[s + 1])
        return self.w_idx[ini:fin], self.w_val[ini:fin]

    # --- Serializacion --------------------------------------------------------
    def guardar(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "obs_entity": self.obs_entity,
            "obs_match": self.obs_match,
            "obs_hash": self.obs_hash,
            "entity_ids": self.entity_ids,
        }
        for nombre in ("w_ptr", "w_idx", "w_val", "costos", "espacio_L"):
            valor = getattr(self, nombre)
            if valor is not None:
                arrays[nombre] = valor
        arrays["__meta__"] = np.frombuffer(
            json.dumps({
                "formulacion": self.formulacion,
                "entidad": self.entidad,
                "normalizacion": self.normalizacion,
                "feat_names": list(self.feat_names),
                "hiper": self.hiper,
                "sigma": self.sigma,
                "distancia": self.distancia,
            }, ensure_ascii=False).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def cargar(cls, path: Path) -> "EstadoWarm":
        datos = np.load(Path(path), allow_pickle=False)
        meta = json.loads(bytes(datos["__meta__"]).decode("utf-8"))
        opcional = lambda k: datos[k] if k in datos.files else None  # noqa: E731
        return cls(
            formulacion=meta["formulacion"],
            entidad=meta["entidad"],
            normalizacion=meta["normalizacion"],
            feat_names=list(meta["feat_names"]),
            hiper=dict(meta["hiper"]),
            obs_entity=datos["obs_entity"],
            obs_match=datos["obs_match"],
            obs_hash=datos["obs_hash"],
            entity_ids=datos["entity_ids"],
            w_ptr=opcional("w_ptr"),
            w_idx=opcional("w_idx"),
            w_val=opcional("w_val"),
            sigma=meta.get("sigma"),
            costos=opcional("costos"),
            # Un estado anterior a que la distancia fuera un parametro no lo
            # declara: era euclidea, que es lo unico que habia.
            distancia=str(meta.get("distancia") or distancias.POR_DEFECTO),
            espacio_L=opcional("espacio_L"),
        )


def identidad_observaciones(mf: MatrizFeatures) -> tuple[np.ndarray, np.ndarray]:
    """(entity_id, match_id) de cada fila. Falla si el df no traia el partido.

    Sin `match_id` no hay forma de decir "esta fila es la misma de antes": una
    entidad tiene varias observaciones y todas comparten `entity_id`.
    """
    if mf.match_id is None:
        raise ValueError(
            "la matriz de features no lleva match_id: no se puede identificar "
            "cada observacion para reutilizar un ajuste anterior"
        )
    return np.asarray(mf.entity_id), np.asarray(mf.match_id)


def estado_base(
    mf: MatrizFeatures, formulacion: str, entidad: str, normalizacion: str,
    entity_ids: np.ndarray, hiper: dict, espacio: Espacio | None = None,
) -> EstadoWarm:
    """Estado con la parte comun (identidad de filas); las formulaciones anaden lo suyo.

    ``espacio`` es la geometria del ajuste (`src.similitud.distancias`): se
    guarda entera —nombre y, si lo tiene, el blanqueo— para que el siguiente
    ajuste y las proyecciones la reutilicen en vez de reestimarla.
    """
    obs_entity, obs_match = identidad_observaciones(mf)
    esp = espacio if espacio is not None else Espacio(distancias.POR_DEFECTO)
    return EstadoWarm(
        formulacion=formulacion,
        entidad=entidad,
        normalizacion=normalizacion,
        feat_names=list(mf.feat_names),
        hiper=dict(hiper),
        obs_entity=obs_entity,
        obs_match=obs_match,
        obs_hash=huellas_filas(mf.X),
        entity_ids=np.asarray(entity_ids),
        distancia=esp.nombre,
        espacio_L=esp.L,
    )


def comprimir_W(
    cols_idx: list[np.ndarray], cols_val: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W dispersa por columnas -> (ptr, idx, val) en formato CSC."""
    tamanos = np.array([c.size for c in cols_idx], dtype=np.int64)
    ptr = np.zeros(len(cols_idx) + 1, dtype=np.int64)
    np.cumsum(tamanos, out=ptr[1:])
    idx = (np.concatenate(cols_idx) if cols_idx else np.empty(0)).astype(np.int64)
    val = (np.concatenate(cols_val) if cols_val else np.empty(0)).astype(float)
    return ptr, idx, val


# --- Emparejamiento entre un ajuste anterior y los datos de ahora ------------
@dataclass
class Emparejamiento:
    """Que se puede reaprovechar del estado previo, fila a fila y entidad a entidad."""

    fila_previa: np.ndarray   # (M,) indice de la fila en el estado previo, -1 si nueva
    fila_igual: np.ndarray    # (M,) existia Y su vector de features no ha cambiado
    ent_previa: np.ndarray    # (P,) indice de la entidad en el estado previo, -1 si nueva
    ent_igual: np.ndarray     # (P,) existia Y su nube de observaciones es identica
    motivo: str = ""          # por que no se reutilizo nada (vacio si si)
    incompatible: bool = False

    @property
    def hay_reutilizacion(self) -> bool:
        return bool(self.fila_previa.size) and bool((self.fila_previa >= 0).any())

    def resumen(self) -> dict:
        return {
            "observaciones": int(self.fila_previa.size),
            "observaciones_reutilizadas": int((self.fila_previa >= 0).sum()),
            "observaciones_intactas": int(self.fila_igual.sum()),
            "observaciones_nuevas": int((self.fila_previa < 0).sum()),
            "entidades": int(self.ent_previa.size),
            "entidades_nuevas": int((self.ent_previa < 0).sum()),
            "entidades_intactas": int(self.ent_igual.sum()),
        }


def _sin_reutilizacion(n_obs: int, n_ent: int, motivo: str) -> Emparejamiento:
    return Emparejamiento(
        fila_previa=np.full(n_obs, -1, dtype=np.int64),
        fila_igual=np.zeros(n_obs, dtype=bool),
        ent_previa=np.full(n_ent, -1, dtype=np.int64),
        ent_igual=np.zeros(n_ent, dtype=bool),
        motivo=motivo,
        incompatible=True,
    )


def compatible(
    estado: EstadoWarm, mf: MatrizFeatures, hiper: dict,
    distancia: str | None = None,
) -> str:
    """Motivo por el que el estado NO sirve, o cadena vacia si sirve.

    Se exige el mismo vector de features, la misma DISTANCIA y los mismos
    hiperparametros: con otros, el optimo es otro y arrancar del anterior seria
    arrancar de la solucion de un problema distinto (valido pero sin sentido, y
    enganoso al informar del ahorro).

    La distancia esta ademas en el nombre del fichero, asi que en la practica no
    se pueden confundir dos estados; se comprueba igual porque nada impide pasar
    un `EstadoWarm` cargado a mano y porque el fallo seria silencioso.
    """
    if list(estado.feat_names) != list(mf.feat_names):
        return "el vector de features ha cambiado (otras columnas o en otro orden)"
    if distancia is not None and estado.distancia != distancias.validar(distancia):
        return (f"el ajuste anterior media con la distancia {estado.distancia!r} "
                f"y ahora se pide {distancia!r}")
    distintos = [
        k for k in set(estado.hiper) | set(hiper)
        if estado.hiper.get(k) != hiper.get(k)
    ]
    if distintos:
        return f"cambiaron hiperparametros: {', '.join(sorted(distintos))}"
    return ""


def emparejar(
    estado: EstadoWarm | None,
    mf: MatrizFeatures,
    entity_ids: np.ndarray,
    hiper: dict,
    distancia: str | None = None,
) -> Emparejamiento:
    """Cruza el estado previo con los datos actuales.

    `fila_igual` marca las observaciones que existian y cuyo vector es identico:
    son las unicas sobre las que se puede reutilizar algo de forma EXACTA.
    `ent_igual` marca las entidades cuya nube entera esta intacta (mismas
    observaciones, todas sin cambios), que es la condicion para reaprovechar su
    coste OT contra otra entidad tambien intacta.
    """
    n_obs, n_ent = mf.X.shape[0], len(entity_ids)
    if estado is None:
        return _sin_reutilizacion(n_obs, n_ent, "no hay estado previo")
    motivo = compatible(estado, mf, hiper, distancia=distancia)
    if motivo:
        return _sin_reutilizacion(n_obs, n_ent, motivo)

    obs_entity, obs_match = identidad_observaciones(mf)
    previo_por_clave = {
        (int(e), int(m)): i
        for i, (e, m) in enumerate(zip(estado.obs_entity, estado.obs_match))
    }
    fila_previa = np.full(n_obs, -1, dtype=np.int64)
    for i in range(n_obs):
        j = previo_por_clave.get((int(obs_entity[i]), int(obs_match[i])))
        if j is not None:
            fila_previa[i] = j

    hash_ahora = huellas_filas(mf.X)
    fila_igual = np.zeros(n_obs, dtype=bool)
    hay = fila_previa >= 0
    fila_igual[hay] = estado.obs_hash[fila_previa[hay]] == hash_ahora[hay]

    # Entidades: indice previo por busqueda binaria (entity_ids va ordenado).
    entity_ids = np.asarray(entity_ids)
    ent_previa = np.full(n_ent, -1, dtype=np.int64)
    if estado.entity_ids.size:
        pos = np.searchsorted(estado.entity_ids, entity_ids)
        pos_valida = np.clip(pos, 0, estado.entity_ids.size - 1)
        acierta = estado.entity_ids[pos_valida] == entity_ids
        ent_previa[acierta] = pos_valida[acierta]

    # Una entidad esta intacta si todas sus filas de ahora estaban y no cambiaron,
    # y ademas tiene el MISMO numero de filas que antes (si tuviera menos, seria
    # otra nube aunque las que quedan sean identicas).
    fila_ent = np.searchsorted(entity_ids, obs_entity)
    n_ahora = np.bincount(fila_ent, minlength=n_ent)
    n_iguales = np.bincount(fila_ent, weights=fila_igual.astype(float), minlength=n_ent)
    n_antes = np.bincount(
        np.searchsorted(estado.entity_ids, estado.obs_entity),
        minlength=estado.n_entidades,
    ) if estado.entity_ids.size else np.zeros(0, dtype=np.int64)
    ent_igual = np.zeros(n_ent, dtype=bool)
    for p in range(n_ent):
        q = int(ent_previa[p])
        if q < 0:
            continue
        ent_igual[p] = (n_iguales[p] == n_ahora[p]) and (n_ahora[p] == n_antes[q])

    return Emparejamiento(
        fila_previa=fila_previa,
        fila_igual=fila_igual,
        ent_previa=ent_previa,
        ent_igual=ent_igual,
    )


def inicializador_f2(estado: EstadoWarm, emp: Emparejamiento):
    """Callable `inicial(s, cand)` para `slim.slim_instancia`.

    Traduce la columna `s` de la W anterior al espacio de indices de ahora: para
    cada vecina candidata se busca que fila era antes y, si aquella columna le
    daba peso, se arranca con ese peso. Las candidatas nuevas (o las que antes no
    tenian peso) arrancan en 0, que es de donde arranca un ajuste en frio.

    Devuelve None cuando la observacion es nueva o cuando su columna anterior
    estaba vacia: en ambos casos el arranque es el vector cero y no hace falta
    construirlo.
    """
    fila_previa = emp.fila_previa

    def inicial(s: int, cand: np.ndarray) -> np.ndarray | None:
        p = int(fila_previa[s])
        if p < 0:
            return None
        idx, val = estado.columna(p)
        if idx.size == 0:
            return None
        pesos = {int(r): float(v) for r, v in zip(idx, val)}
        previas = fila_previa[cand]
        w0 = np.zeros(cand.size, dtype=float)
        for k, anterior in enumerate(previas):
            if anterior >= 0:
                w0[k] = pesos.get(int(anterior), 0.0)
        return w0

    return inicial


def costos_reutilizables(
    estado: EstadoWarm, emp: Emparejamiento, n_ent: int
) -> np.ndarray | None:
    """Matriz (P,P) de costes OT conocidos, NaN donde hay que recalcular.

    Solo se da por bueno el coste de un par cuyas DOS nubes estan intactas: el
    coste depende de las dos, asi que tocar una invalida toda su fila y su
    columna.
    """
    if estado.costos is None:
        return None
    previos = np.full((n_ent, n_ent), np.nan, dtype=float)
    intactas = np.where(emp.ent_igual)[0]
    if intactas.size == 0:
        return previos
    origen = emp.ent_previa[intactas]
    previos[np.ix_(intactas, intactas)] = estado.costos[np.ix_(origen, origen)]
    return previos
