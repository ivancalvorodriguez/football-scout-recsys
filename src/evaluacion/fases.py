"""Las fases del protocolo de evaluacion (`docs/Como_evaluar.pdf`).

Cada funcion implementa una fase y devuelve filas/estructuras que el orquestador
(`evaluar.py`) vuelca a CSV y al informe. Fases:

- Fase 0  sanity checks     -> `fase0_sanity`
- Fase 1  auto-similitud    -> `fase1_autosimilitud`  (pilar; mitades pares/impares)
- Fase 2  estabilidad       -> `fase2_estabilidad`    (bootstrap + RBO)
- Fase 3  triangulacion     -> `fase3_triangulacion`  (test de Mantel)
- Fase 4  aporte del denoising (F5) -> se calcula dentro de `fase1_autosimilitud`
- Fase 5  downstream + beyond-accuracy -> `fase5_downstream`
- Fase 6  validacion humana -> no automatizable (se documenta en el informe).
"""

from __future__ import annotations

import numpy as np

from . import config, construccion, datos, metricas
from .construccion import Contexto, SplitVirtual


# --------------------------------------------------------------------------- #
# Utilidades de ranking sobre una matriz S                                     #
# --------------------------------------------------------------------------- #

def top_k_indices(S: np.ndarray, i: int, k: int, excluir_cero: bool = True) -> list[int]:
    """Indices del top-k de la fila i (misma semantica que `modelo.top_k`).

    Descarta la propia entidad y, si ``excluir_cero``, los scores exactamente 0
    (en la F2 un 0 es "sin relacion aprendida", no un candidato valido).

    Escalabilidad: se filtran primero los candidatos invalidos (propia entidad,
    no finitos, ceros) y se selecciona el top-k con ``argpartition`` (O(P)) en vez
    de ordenar toda la fila (O(P log P)). El resultado es identico al de un
    argsort completo: los unicos empates masivos posibles son los ceros, que se
    eliminan antes de particionar, de modo que entre los candidatos retenidos los
    empates son de medida nula. Este es el bucle mas caliente de la evaluacion
    (la Fase 2 lo llama B x P veces), asi que la diferencia importa al crecer P.
    """
    fila = S[i]
    valido = np.isfinite(fila)
    valido[i] = False
    if excluir_cero:
        valido &= fila != 0.0
    cand = np.flatnonzero(valido)
    if cand.size == 0:
        return []
    vals = fila[cand].astype(float)
    if cand.size > k:
        # top-k sin ordenar el resto: argpartition deja los k mayores al final.
        corte = np.argpartition(vals, cand.size - k)[cand.size - k:]
        cand, vals = cand[corte], vals[corte]
    orden = np.argsort(vals, kind="mergesort")[::-1]
    return [int(cand[j]) for j in orden]


# --------------------------------------------------------------------------- #
# Fase 0 - Sanity checks                                                        #
# --------------------------------------------------------------------------- #

def fase0_sanity(
    modelo, roles: dict[int, str], ctx: Contexto, formulacion: str, reportar=None
) -> dict:
    """Asimetria de la S CRUDA y pureza posicional del top-1 (jugadores).

    - Asimetria = ||S_cruda - S_cruda^T|| / ||S_cruda||, medida sobre la S ANTES
      de simetrizar (el artefacto servible ya se simetriza, asi que su asimetria
      es ~0 y no informa de nada). Es el sanity check del PDF: cuanta senal
      direccional corrige el pipeline. Umbral ``ASIMETRIA_ALTA``: por encima, la W
      (F2) / el denoising de EASE (F5) es muy asimetrico y conviene revisar que
      simetrizar sea razonable.
    - Pureza@1 = fraccion de entidades cuyo top-1 comparte rol grueso. Umbral del
      PDF: >80-90%; por debajo, el modelo captura volumen, no estilo (o hay bug).

    ``reportar`` (opcional): callback ``reportar(hecho, total)`` invocado por cada
    entidad del recorrido de pureza (solo jugador). Hoy es instantaneo, pero el
    recorrido es O(P^2) y se instrumenta de cara al escalado del pool.
    """
    S = modelo.S
    S_cruda = construccion.reconstruir_S_cruda(ctx, formulacion)
    norm_c = np.linalg.norm(S_cruda)
    asimetria = float(np.linalg.norm(S_cruda - S_cruda.T) / norm_c) if norm_c else 0.0

    pureza = float("nan")
    n_eval = 0
    if modelo.entidad == "jugador":
        aciertos = 0
        P = S.shape[0]
        for i in range(P):
            if reportar is not None:  # al entrar: cubre tambien las filas que se saltan
                reportar(i + 1, P)
            rol_i = roles.get(int(modelo.entity_ids[i]))
            if rol_i is None or rol_i == config.ROL_DESCONOCIDO:
                continue
            top1 = top_k_indices(S, i, 1)
            if not top1:
                continue
            rol_j = roles.get(int(modelo.entity_ids[top1[0]]))
            n_eval += 1
            if rol_j == rol_i:
                aciertos += 1
        pureza = aciertos / n_eval if n_eval else float("nan")

    return {
        "asimetria": asimetria,
        "pureza_top1": pureza,
        "n_pureza": n_eval,
    }


def face_validity(modelo, nombres_query: list[str], k: int = 5) -> list[str]:
    """Devuelve lineas 'query -> top-k' para revision cualitativa manual."""
    from src.similitud.consulta import resolver, ErrorResolucion

    lineas: list[str] = []
    for nombre in nombres_query:
        try:
            i = resolver(nombre, modelo.entity_names)
        except ErrorResolucion:
            continue
        top = top_k_indices(modelo.S, i, k)
        vecinos = ", ".join(modelo.entity_names[j] for j in top) or "(sin vecinos)"
        lineas.append(f"  {modelo.entity_names[i]} -> {vecinos}")
    return lineas


# --------------------------------------------------------------------------- #
# Fase 1 - Auto-similitud (pilar) + Fase 4 (aporte del denoising F5)            #
# --------------------------------------------------------------------------- #

def _rangos_direccion(
    S: np.ndarray, split: SplitVirtual, origen_half: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rangos del emparejamiento de mitades en una direccion (bipartito).

    ``origen_half`` = 0 significa consultar con la mitad A y recuperar entre las
    mitades B (protocolo de-anonimizacion de Decroos & Davis: query anonima ->
    pool etiquetado). Devuelve (rangos, reales_evaluados, tamano_pool).
    """
    destino_half = 1 - origen_half
    pool = np.where(split.half == destino_half)[0]
    pos_en_pool = {int(v): p for p, v in enumerate(pool)}
    rangos: list[int] = []
    reales: list[int] = []
    for (vA, vB), real in zip(split.pares_AB, split.real_evaluables):
        q, t = (vA, vB) if origen_half == 0 else (vB, vA)
        fila = S[q, pool]
        rangos.append(metricas.rango_del_objetivo(fila, pos_en_pool[int(t)]))
        reales.append(int(real))
    return np.array(rangos, dtype=float), np.array(reales, dtype=int), len(pool)


def _resumen_autosim(rangos: np.ndarray, pool: int) -> dict:
    fila = {"n": int(rangos.size), "pool": int(pool),
            "azar_top1": 1.0 / pool if pool else float("nan"),
            "mrr": metricas.mrr(rangos)}
    for k in config.KS_AUTOSIM:
        fila[f"top{k}"] = metricas.recall_at_k(rangos, k)
    return fila


def fase1_autosimilitud(ctx: Contexto, formulacion: str, reportar=None) -> dict:
    """Construye el modelo virtual (mitades pares/impares) y mide self-retrieval.

    Reporta top-1/5/10 y MRR con la tasa de azar (1/N) y el tamano del pool, en
    ambas direcciones (A->B y B->A) y su media, estratificado por minutos
    (cabeza/torso/cola) para jugadores. Para la F5 mide ademas el aporte del
    denoising de EASE (Fase 4): auto-similitud PRE-EASE vs POST-EASE con test de
    permutacion pareado sobre el reciproco del rango.

    ``reportar`` (opcional): callback ``reportar(hecho, total)`` que se reenvia al
    bucle interno caro (ajuste SLIM de la F2 / OT de la F5) para seguir el avance.
    """
    split = construccion.split_par_impar(ctx)

    if formulacion == "5":
        S_post, S_pre = construccion.reconstruir_S_f5(
            ctx, split.row_entity, split.n_virt, progreso=reportar)
        S = S_post
    else:
        S = construccion.reconstruir_S(
            ctx, formulacion, split.row_entity, split.n_virt, progreso=reportar)
        S_pre = None

    r_ab, reales, pool_ab = _rangos_direccion(S, split, origen_half=0)
    r_ba, _, pool_ba = _rangos_direccion(S, split, origen_half=1)
    r_media = np.concatenate([r_ab, r_ba])

    resultado = {
        "global": _resumen_autosim(r_media, (pool_ab + pool_ba) // 2),
        "A->B": _resumen_autosim(r_ab, pool_ab),
        "B->A": _resumen_autosim(r_ba, pool_ba),
        "estratos": {},
        "denoising": None,
    }

    # Estratificacion por minutos (solo jugador; el equipo juega siempre completo).
    if ctx.entidad == "jugador":
        min_real = np.array([
            ctx.minutos_obs[ctx.idx_real.row_entity == r].sum() for r in reales
        ])
        etiq = datos.terciles(min_real)
        for t in (0, 1, 2):
            m = etiq == t
            if m.sum() == 0:
                continue
            # Pool = todas las mitades del destino (no varia por estrato).
            resultado["estratos"][datos.NOMBRE_TERCIL[t]] = _resumen_autosim(
                np.concatenate([r_ab[m], r_ba[m]]), (pool_ab + pool_ba) // 2
            )

    # Fase 4: aporte del denoising (F5).
    if formulacion == "5":
        r_pre, _, pool_pre = _rangos_direccion(S_pre, split, origen_half=0)
        dif_rr = (1.0 / r_ab) - (1.0 / r_pre)  # reciproco post - pre, por par
        media, p = metricas.permutacion_pareado(dif_rr, config.PERM_PAREADO, config.SEED)
        resultado["denoising"] = {
            "pre": _resumen_autosim(r_pre, pool_pre),
            "post": _resumen_autosim(r_ab, pool_ab),
            "delta_mrr": metricas.mrr(r_ab) - metricas.mrr(r_pre),
            "delta_rr_medio": media,
            "p_pareado": p,
        }

    return resultado


# --------------------------------------------------------------------------- #
# Fase 2 - Estabilidad (bootstrap + RBO)                                        #
# --------------------------------------------------------------------------- #

def fase2_estabilidad(
    ctx: Contexto, formulacion: str, S_base: np.ndarray, B: int,
    reportar=None,
) -> dict:
    """RBO@10 medio del top-k ante remuestreo bootstrap de partidos.

    Reconstruye S con multiplicidades bootstrap por observacion (reutilizando la
    estructura aprendida) y compara el top-k de cada entidad contra el del modelo
    servible (``S_base``). Reporta RBO@10 medio (p=0.9) con IC bootstrap. Umbral
    del PDF: RBO@10 > 0.7 = estable; < 0.5 = preocupante.

    ``reportar`` (opcional): callable ``reportar(hecho, total)`` que se invoca tras
    cada remuestreo, para que el orquestador muestre el progreso del bucle interno
    (es la parte mas cara de toda la evaluacion). Si es None, no imprime nada.
    """
    P = len(ctx.idx_real.ids)
    row = ctx.idx_real.row_entity
    rng = np.random.default_rng(config.SEED)

    # Entidades-consulta cuyo RBO se promedia. Con P grande se submuestrea una
    # vez (fijo por seed) para no rankear O(B x P) veces (ver N_ESTABILIDAD_MAX);
    # None reproduce el comportamiento exacto sobre todas las entidades.
    tope = config.N_ESTABILIDAD_MAX
    if tope is not None and P > tope:
        queries = np.sort(rng.choice(P, size=tope, replace=False))
    else:
        queries = np.arange(P)
    base_lists = {i: top_k_indices(S_base, int(i), config.K_LISTA) for i in queries}
    consultables = [int(i) for i in queries if base_lists[int(i)]]

    rbos_por_b: list[float] = []
    for b in range(B):
        mult = construccion.multiplicidades_bootstrap(ctx, None, rng)
        w = ctx.mf.weight * mult
        # Observaciones con multiplicidad 0 -> masa 0: en la etapa OT (equipo F5)
        # su log(masa) es -inf de forma esperada (quedan excluidas del transporte);
        # se silencia ese aviso, que no indica error numerico.
        with np.errstate(divide="ignore", invalid="ignore"):
            S_b = construccion.reconstruir_S(ctx, formulacion, row, P, weight=w)
        vals = [
            metricas.rbo(base_lists[i], top_k_indices(S_b, i, config.K_LISTA), config.RBO_P)
            for i in consultables
        ]
        rbos_por_b.append(float(np.mean(vals)) if vals else float("nan"))
        if reportar is not None:
            reportar(b + 1, B)

    rbos = np.array(rbos_por_b, dtype=float)
    lo, hi = metricas.ic_bootstrap(rbos)
    return {
        "rbo_medio": float(np.nanmean(rbos)),
        "ic_bajo": lo,
        "ic_alto": hi,
        "n_consultables": len(consultables),
        "B": B,
    }


# --------------------------------------------------------------------------- #
# Fase 3 - Triangulacion (test de Mantel)                                       #
# --------------------------------------------------------------------------- #

def _kendall_medio(A: np.ndarray, Bm: np.ndarray, cap: int, seed: int) -> float:
    """Kendall tau media POR ENTIDAD entre dos matrices S alineadas.

    Para cada entidad-consulta se rankean las demas por A y por B y se mide su
    concordancia (Kendall tau-b); se promedia. Complementa al Mantel (acuerdo
    global) mostrando el acuerdo del *ranking* de vecinos por entidad, que es lo
    que consume el recomendador. O(n^2) por consulta => se acota a ``cap``
    entidades (submuestra fija por seed). Devuelve la tau media en [-1, 1].
    """
    P = A.shape[0]
    rng = np.random.default_rng(seed)
    sel = (np.sort(rng.choice(P, size=cap, replace=False))
           if P > cap else np.arange(P))
    As, Bs = A[np.ix_(sel, sel)], Bm[np.ix_(sel, sel)]
    taus = []
    for i in range(len(sel)):
        otros = np.arange(len(sel)) != i          # excluye la diagonal (propia entidad)
        taus.append(metricas.kendall_tau(As[i, otros], Bs[i, otros]))
    return float(np.mean(taus)) if taus else float("nan")


def fase3_triangulacion(modelos: dict, ctxs: dict, reportar=None) -> list[dict]:
    """Test de Mantel entre matrices S de modelos independientes + Kendall tau.

    Cubre las cuatro comparaciones del PDF (Fase 3):
    - **F2 vs F5** (post-EASE) para cada (entidad, norm): ¿capturan la misma senal?
      Se acompana de la Kendall tau media por entidad (donde divergen los rankings).
    - **por_liga vs global** para cada (formulacion, entidad): efecto de la norma.
    - **F2 agregada vs F5 distribucional cruda** (S_pre, sin EASE): valida que la
      agregacion Sigma-W de la F2 PRESERVA la estructura distribucional directa
      (pregunta critica de la F2 en el PDF, p. 8).
    - **F5 pre-EASE vs post-EASE**: valida que el denoising de EASE NO destruye la
      estructura metrica de la S distribucional (p. 9).

    r alto => misma senal subyacente (umbral orientativo r>0.6). Se prioriza la
    magnitud de r sobre el p-valor (el PDF avisa de inflacion por autocorrelacion
    posicional). Las dos ultimas comparaciones necesitan la S_pre de la F5, que no
    esta en disco: se reconstruye por (entidad, norm) desde ``ctxs``.

    ``reportar`` (opcional): callback ``reportar(hecho, total)`` invocado tras cada
    comparacion de Mantel (cada una es un test de permutacion completo).
    """
    filas: list[dict] = []
    rng = np.random.default_rng(config.SEED)
    _s_pre: dict = {}  # cache de la S distribucional cruda de la F5 por (entidad, norm)

    def _f5_pre(entidad, norm):
        clave = (entidad, norm)
        if clave not in _s_pre:
            ctx = ctxs[clave]
            _, s = construccion.reconstruir_S_f5(
                ctx, ctx.idx_real.row_entity, len(ctx.idx_real.ids))
            _s_pre[clave] = (s, np.asarray(ctx.idx_real.ids))
        return _s_pre[clave]

    def _mantel(A, ids_a, Bm, ids_b, etiqueta, entidad, con_kendall):
        if not np.array_equal(ids_a, ids_b):
            return  # deberian coincidir (misma BD); si no, no es comparable
        P = A.shape[0]
        # Submuestreo: con P grande, permutar el triangulo superior completo 999
        # veces es inviable. Se estima r y p sobre una submuestra fija de entidades.
        if P > config.N_MANTEL_MAX:
            sel = np.sort(rng.choice(P, size=config.N_MANTEL_MAX, replace=False))
            As, Bs = A[np.ix_(sel, sel)], Bm[np.ix_(sel, sel)]
        else:
            As, Bs = A, Bm
        r, p = metricas.mantel(As, Bs, config.MANTEL_PERMUTACIONES, config.SEED)
        tau = _kendall_medio(A, Bm, config.N_KENDALL_MAX, config.SEED) if con_kendall else float("nan")
        filas.append({"comparacion": etiqueta, "entidad": entidad,
                      "n_entidades": As.shape[0], "r": r, "p": p, "kendall": tau})

    # Se enumeran primero todas las comparaciones disponibles para conocer el
    # total y poder reportar avance (cada Mantel es un test de permutacion caro).
    # Cada tupla: (A, ids_a, B, ids_b, etiqueta, entidad, con_kendall).
    comparaciones: list[tuple] = []
    for entidad in config.ENTIDADES:
        for norm in config.NORMALIZACIONES:
            a = modelos.get(("2", entidad, norm))
            b = modelos.get(("5", entidad, norm))
            if a is not None and b is not None:
                comparaciones.append((a.S, a.entity_ids, b.S, b.entity_ids,
                                      f"F2 vs F5 ({norm})", entidad, True))
                # F2 agregada vs F5 distribucional CRUDA (sin EASE): ¿la agregacion
                # de la F2 preserva la estructura distribucional directa?
                s_pre, ids_pre = _f5_pre(entidad, norm)
                comparaciones.append((a.S, a.entity_ids, s_pre, ids_pre,
                                      f"F2 vs F5-distrib. cruda ({norm})", entidad, False))
                # F5 pre-EASE vs post-EASE: ¿el denoising preserva la estructura?
                comparaciones.append((s_pre, ids_pre, b.S, b.entity_ids,
                                      f"F5 pre vs post-EASE ({norm})", entidad, False))
    for entidad in config.ENTIDADES:
        for form in config.FORMULACIONES:
            a = modelos.get((form, entidad, "por_liga"))
            b = modelos.get((form, entidad, "global"))
            if a is not None and b is not None:
                comparaciones.append((a.S, a.entity_ids, b.S, b.entity_ids,
                                      f"por_liga vs global (F{form})", entidad, False))

    total = len(comparaciones)
    for i, (A, ida, Bm, idb, etiqueta, entidad, con_k) in enumerate(comparaciones, start=1):
        _mantel(A, ida, Bm, idb, etiqueta, entidad, con_k)
        if reportar is not None:
            reportar(i, total)
    return filas


# --------------------------------------------------------------------------- #
# Fase 5 - Downstream + beyond-accuracy                                         #
# --------------------------------------------------------------------------- #

def fase5_downstream(
    modelo, roles: dict[int, str], minutos: dict[int, float], reportar=None
) -> dict:
    """Tarea downstream (clasificacion posicional k-NN) + metricas beyond-accuracy.

    - k-NN posicional (jugador): predice el rol por voto de los k vecinos en S;
      accuracy y F1 macro (metrica objetiva de si S captura estilo/rol).
    - Coverage: fraccion de entidades que aparecen en algun top-k (sesgo de
      popularidad si es baja).
    - Diversity intra-lista: distancia media entre los perfiles del top-k.
    - Popularidad: correlacion de Spearman entre el in-degree (veces recomendado)
      y el volumen (minutos), para detectar sesgo hacia jugadores de alto volumen.

    ``reportar`` (opcional): callback ``reportar(hecho, total)`` invocado al
    construir las listas top-k (el recorrido O(P^2) que domina al escalar el pool).
    """
    S = modelo.S
    P = S.shape[0]
    listas = []
    for i in range(P):
        listas.append(top_k_indices(S, i, config.K_LISTA))
        if reportar is not None:
            reportar(i + 1, P)

    # Clasificacion posicional (solo jugador).
    acc = f1 = float("nan")
    n_knn = 0
    if modelo.entidad == "jugador":
        y_true, y_pred = [], []
        for i in range(P):
            rol_i = roles.get(int(modelo.entity_ids[i]))
            if rol_i is None or rol_i == config.ROL_DESCONOCIDO:
                continue
            vecinos = listas[i][: config.KNN_K]
            votos = [roles.get(int(modelo.entity_ids[j])) for j in vecinos]
            votos = [v for v in votos if v and v != config.ROL_DESCONOCIDO]
            if not votos:
                continue
            vals, cuentas = np.unique(votos, return_counts=True)
            y_pred.append(vals[np.argmax(cuentas)])
            y_true.append(rol_i)
        if y_true:
            y_true = np.array(y_true); y_pred = np.array(y_pred)
            acc = float(np.mean(y_true == y_pred))
            f1 = metricas.f1_macro(y_true, y_pred)
            n_knn = len(y_true)

    # Coverage.
    aparecen = set()
    for lst in listas:
        aparecen.update(lst)
    coverage = len(aparecen) / P if P else float("nan")

    # Diversity intra-lista (distancia euclidea media entre perfiles del top-k).
    F = modelo.feat_display
    divs = []
    for lst in listas:
        if len(lst) < 2:
            continue
        sub = F[lst]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        iu = np.triu_indices(len(lst), k=1)
        divs.append(float(np.mean(d[iu])))
    diversity = float(np.mean(divs)) if divs else float("nan")

    # Popularidad: in-degree vs volumen (jugador).
    indeg = np.zeros(P, dtype=float)
    for lst in listas:
        for j in lst:
            indeg[j] += 1
    pop_corr = float("nan")
    if modelo.entidad == "jugador":
        vol = np.array([minutos.get(int(modelo.entity_ids[i]), 0.0) for i in range(P)])
        pop_corr = metricas.spearman(indeg, vol)

    return {
        "knn_accuracy": acc,
        "knn_f1_macro": f1,
        "n_knn": n_knn,
        "coverage": coverage,
        "diversity": diversity,
        "indegree_medio": float(np.mean(indeg)),
        "popularidad_spearman": pop_corr,
    }
