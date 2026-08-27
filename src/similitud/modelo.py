"""Artefacto de modelo entrenado: indexado de entidades, guardado y carga.

Un modelo entrenado (para una formulacion y un tipo de entidad) es:
- `S`: matriz entidad-entidad de puntuacion final para servir top-k (diag = 0).
- indice de entidades: ids y nombres alineados con las filas/columnas de S.
- `feat_display`: media ponderada por minutos de las features z-scored por
  entidad, SOLO para interpretar las recomendaciones (nunca entra en el ajuste;
  se calcula despues de entrenar). Respeta el principio central: el modelo no se
  alimenta de esta agregacion.
- metadatos: formulacion, entidad, nombres de feature, hiperparametros.

Se serializa como `<form>_<entidad>.npz` (arrays) + `.json` (texto/meta).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import distancias
from .features import MatrizFeatures


def stem_artefacto(
    formulacion: str, entidad: str, normalizacion: str,
    distancia: str = distancias.POR_DEFECTO,
) -> str:
    """Nombre base de un artefacto (y, con otro sufijo, de su estado warm).

    `formulacion5_jugador_por_liga`, y con la distancia detras cuando NO es la
    euclidea: `formulacion5_jugador_por_liga_manhattan`. La asimetria es
    deliberada y es la misma que ya tenia la normalizacion (`cargar_modelo` cae
    al nombre sin sufijo): la euclidea es la distancia con la que se construyo
    todo lo que hay en `outputs/`, y darle sufijo obligaria a renombrar miles de
    ficheros —artefactos, estados warm y huellas de ~1.100 combinaciones de
    barrido— para no cambiar ni un numero. Los CSV y las tablas, en cambio, SI
    la nombran siempre (`euclidea`), porque ahi el coste es cero y lo que
    importa es que ningun resultado quede sin declarar con que geometria se midio.
    """
    base = f"formulacion{formulacion}_{entidad}_{normalizacion}"
    distancia = distancias.validar(distancia)
    return base if distancia == distancias.POR_DEFECTO else f"{base}_{distancia}"


@dataclass
class IndiceEntidades:
    """Mapeo entre observaciones (filas) y entidades (0..P-1)."""

    ids: np.ndarray          # (P,) id de cada entidad
    names: list[str]         # (P,) nombre de cada entidad
    row_entity: np.ndarray   # (M,) indice de entidad de cada observacion


def indexar_entidades(mf: MatrizFeatures) -> IndiceEntidades:
    """Asigna a cada entidad un indice 0..P-1, en orden creciente de `entity_id`.

    El orden lo fija `np.unique` (por id, no por aparicion) y es estable entre
    ejecuciones, que es lo que importa: las filas de `S` se alinean con `ids`.
    """
    ids, inverse = np.unique(mf.entity_id, return_inverse=True)
    # Nombre de cada id (el id es la identidad; el nombre es solo para mostrar).
    names_by_id = {int(i): n for i, n in zip(mf.entity_id, mf.entity_name)}
    names = [str(names_by_id[int(i)]) for i in ids]
    return IndiceEntidades(ids=ids, names=names, row_entity=inverse.astype(np.int64))


def features_display(mf: MatrizFeatures, idx: IndiceEntidades) -> np.ndarray:
    """Media ponderada por minutos de las features z-scored por entidad.

    SOLO para interpretar (mostrar que metricas destacan). No se usa para
    entrenar ningun modelo: la agregacion aqui es un resumen a posteriori.
    """
    n_ent = len(idx.ids)
    d = mf.X.shape[1]
    suma = np.zeros((n_ent, d), dtype=float)
    masa = np.zeros(n_ent, dtype=float)
    np.add.at(suma, idx.row_entity, mf.X * mf.weight[:, None])
    np.add.at(masa, idx.row_entity, mf.weight)
    masa[masa == 0.0] = 1.0
    return suma / masa[:, None]


def ligas_por_entidad(mf: MatrizFeatures, idx: IndiceEntidades) -> dict[str, list[str]]:
    """Ligas (competition-season) en las que aparece cada entidad.

    Se serializa en `meta` para poder etiquetar candidatos con su liga sin tener
    que reabrir la BD. Se indexa por `entity_id` (como str, por JSON) y NO por
    nombre: dos jugadores pueden llamarse igual y agrupar por nombre fusionaria
    sus ligas. Para pasar de indice de fila de `S` a esta clave:
    `str(modelo.entity_ids[i])`.
    """
    ligas: dict[str, list[str]] = {str(i): [] for i in idx.ids}
    seen: dict[str, set[str]] = {str(i): set() for i in idx.ids}
    for row, ent in enumerate(idx.row_entity):
        liga = str(mf.league[row])
        clave = str(idx.ids[int(ent)])
        if liga not in seen[clave]:
            seen[clave].add(liga)
            ligas[clave].append(liga)
    return ligas


@dataclass
class ModeloSimilitud:
    """Modelo servible: S + indice + features de display + metadatos."""

    formulacion: str
    entidad: str
    S: np.ndarray
    entity_ids: np.ndarray
    entity_names: list[str]
    feat_names: list[str]
    feat_display: np.ndarray
    meta: dict

    def _stem(self, sufijo: str | None = None) -> str:
        base = f"formulacion{self.formulacion}_{self.entidad}"
        if sufijo:
            return f"{base}_{sufijo}"
        return base

    def guardar(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # El nombre del artefacto refleja siempre la normalizacion del modelo, y
        # la distancia cuando no es la euclidea (ver `stem_artefacto`).
        sufijo = self.meta.get("normalizacion")
        stem = (
            stem_artefacto(self.formulacion, self.entidad, sufijo,
                           self.meta.get("distancia") or distancias.POR_DEFECTO)
            if sufijo else self._stem()
        )
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            S=self.S,
            entity_ids=self.entity_ids,
            feat_display=self.feat_display,
        )
        meta = {
            "formulacion": self.formulacion,
            "entidad": self.entidad,
            "entity_names": self.entity_names,
            "feat_names": self.feat_names,
            "meta": self.meta,
        }
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_dir / f"{stem}.npz"


def cargar_modelo(
    model_dir: Path,
    formulacion: str,
    entidad: str,
    normalizacion: str = "por_liga",
    distancia: str = distancias.POR_DEFECTO,
) -> ModeloSimilitud:
    """Carga un modelo guardado (`<form>_<entidad>[_<norm>][_<distancia>]`).

    `normalizacion` y `distancia` componen el nombre del artefacto con
    `stem_artefacto`. Los dos defaults (`por_liga`, `euclidea`) son los
    historicos, para no romper el flujo previo.

    Hay DOS fallbacks, los dos acotados al default correspondiente y por el mismo
    motivo: caer a un artefacto que no declara una variante solo es seguro cuando
    lo que se pide es la variante que ese artefacto tenia.

    - Sin sufijo de distancia = euclidea, siempre (es lo que se construyo antes
      de que la distancia fuera un parametro). Pedir `manhattan` y servir el
      artefacto sin sufijo daria un ranking de otra geometria, asi que ahi se
      exige el nombre completo y, si no esta, se lanza `FileNotFoundError`.
    - Sin sufijo de normalizacion = `por_liga`, como hasta ahora.
    """
    model_dir = Path(model_dir)
    base = f"formulacion{formulacion}_{entidad}"
    distancia = distancias.validar(distancia)
    candidatos = [
        model_dir / f"{stem_artefacto(formulacion, entidad, normalizacion, distancia)}.npz"
    ]
    # Compatibilidad: el artefacto antiguo sin sufijo solo vale como
    # `por_liga` + `euclidea`, que es con lo que se construyo.
    if normalizacion == "por_liga" and distancia == distancias.POR_DEFECTO:
        candidatos.append(model_dir / f"{base}.npz")
    stem = None
    for npz in candidatos:
        if npz.exists():
            stem = npz.stem
            break
    if stem is None:
        raise FileNotFoundError(
            f"No se encontro modelo para {base} (normalizacion={normalizacion!r}, "
            f"distancia={distancia!r}) en {model_dir}. Probado: "
            + ", ".join(p.name for p in candidatos)
        )
    arr = np.load(model_dir / f"{stem}.npz", allow_pickle=False)
    meta = json.loads((model_dir / f"{stem}.json").read_text(encoding="utf-8"))
    return ModeloSimilitud(
        formulacion=meta["formulacion"],
        entidad=meta["entidad"],
        S=arr["S"],
        entity_ids=arr["entity_ids"],
        entity_names=list(meta["entity_names"]),
        feat_names=list(meta["feat_names"]),
        feat_display=arr["feat_display"],
        meta=meta["meta"],
    )


def top_k(modelo: ModeloSimilitud, i: int, k: int) -> list[tuple[int, float]]:
    """Indices y puntuaciones de las k entidades mas similares a la entidad i.

    Descarta la propia entidad y los candidatos con score exactamente 0: en la
    Formulacion 2 la matriz S es muy dispersa (W solo relaciona vecinas), asi que
    un 0 significa "sin relacion aprendida", no "similar"; incluirlos rellenaria
    el top-k con entidades arbitrarias en orden de indice. Puede devolver menos de
    k resultados si la entidad esta poco conectada (comportamiento honesto).

    El ranking es el del MODELO y no se filtra por quien consulta: si la base de
    datos elegida no tiene a una de las candidatas, sigue saliendo (el artefacto
    trae su nombre, su liga y su vector) y es la app la que la rotula como «no
    esta en estos datos». Esconderla tiraria informacion que el modelo si tiene.
    """
    fila = modelo.S[i].copy()
    fila[i] = -np.inf  # nunca recomendarse a si misma
    orden = np.argsort(fila)[::-1]
    salida: list[tuple[int, float]] = []
    for j in orden:
        score = fila[j]
        if score == 0.0 or not np.isfinite(score):
            continue  # sin relacion aprendida (o la propia entidad)
        salida.append((int(j), float(score)))
        if len(salida) >= k:
            break
    return salida
