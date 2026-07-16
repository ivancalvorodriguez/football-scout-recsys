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

from .features import MatrizFeatures


@dataclass
class IndiceEntidades:
    """Mapeo entre observaciones (filas) y entidades (0..P-1)."""

    ids: np.ndarray          # (P,) id de cada entidad
    names: list[str]         # (P,) nombre de cada entidad
    row_entity: np.ndarray   # (M,) indice de entidad de cada observacion


def indexar_entidades(mf: MatrizFeatures) -> IndiceEntidades:
    """Asigna a cada entidad un indice 0..P-1 (orden de primera aparicion)."""
    ids, first_idx, inverse = np.unique(
        mf.entity_id, return_index=True, return_inverse=True
    )
    # `np.unique` ordena por id; recuperamos el nombre de cada id.
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
    """Ligas (competition-season) en las que aparece cada entidad, ordenadas.

    Se serializa en `meta` para que el comparador pueda etiquetar candidatos con
    su liga sin tener que reabrir la BD.
    """
    ligas: dict[str, list[str]] = {n: [] for n in idx.names}
    seen: dict[str, set[str]] = {n: set() for n in idx.names}
    for row, ent in enumerate(idx.row_entity):
        liga = str(mf.league[row])
        nombre = idx.names[int(ent)]
        if liga not in seen[nombre]:
            seen[nombre].add(liga)
            ligas[nombre].append(liga)
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
        # El nombre del artefacto refleja siempre la normalizacion del modelo.
        sufijo = self.meta.get("normalizacion")
        stem = self._stem(sufijo=sufijo)
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
) -> ModeloSimilitud:
    """Carga un modelo guardado (`<form>_<entidad>[_<normalizacion>]` en `model_dir`).

    `normalizacion` añade el sufijo al nombre del artefacto. Default `por_liga`
    para no romper el flujo previo (los modelos antiguos no tenian sufijo, asi
    que se resuelve a `formulacion{2,5}_{entidad}.{npz,json}` directamente).

    El fallback al artefacto sin sufijo SOLO aplica al default historico
    `por_liga`: un modelo antiguo no lleva normalizacion global, asi que caer a el
    cuando se pide `global` serviria en silencio un modelo equivocado. Para
    cualquier normalizacion distinta de `por_liga` se exige el artefacto con
    sufijo y, si no existe, se lanza `FileNotFoundError`.
    """
    model_dir = Path(model_dir)
    base = f"formulacion{formulacion}_{entidad}"
    candidatos = [model_dir / f"{base}_{normalizacion}.npz"]
    # Compatibilidad: el artefacto antiguo sin sufijo solo vale como `por_liga`.
    if normalizacion == "por_liga":
        candidatos.append(model_dir / f"{base}.npz")
    stem = None
    for npz in candidatos:
        if npz.exists():
            stem = npz.stem
            break
    if stem is None:
        raise FileNotFoundError(
            f"No se encontro modelo para {base} (normalizacion={normalizacion!r}) "
            f"en {model_dir}. Probado: "
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
