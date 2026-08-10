"""Estado warm: identidad de las filas, emparejamiento y arranque del ajuste.

Todo lo que el reentrenamiento incremental reutiliza cuelga de estas piezas. Si
`emparejar` diese por intacta una fila que ha cambiado, se arrastraria trabajo de
un problema distinto y la S resultante seria silenciosamente incorrecta: por eso
las pruebas insisten mas en lo que NO debe reutilizarse que en lo que si.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import slim, warm
from src.similitud.features import MatrizFeatures


def matriz(X, entity_id, match_id, feat_names=None) -> MatrizFeatures:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return MatrizFeatures(
        X=X,
        feat_names=feat_names or [f"f{i}" for i in range(X.shape[1])],
        entity_id=np.asarray(entity_id),
        entity_name=np.array([f"E{e}" for e in entity_id]),
        weight=np.ones(n),
        league=np.array(["1-1"] * n),
        match_id=np.asarray(match_id),
    )


HIPER = {"beta": 1.0, "l1": 0.5}


def estado_de(mf: MatrizFeatures, entity_ids) -> warm.EstadoWarm:
    return warm.estado_base(mf, "2", "jugador", "por_liga", np.asarray(entity_ids), HIPER)


# --- Huellas ------------------------------------------------------------------
def test_misma_matriz_misma_huella():
    X = np.arange(12, dtype=float).reshape(4, 3)
    assert np.array_equal(warm.huellas_filas(X), warm.huellas_filas(X.copy()))


def test_cambiar_un_valor_cambia_solo_su_huella():
    X = np.arange(12, dtype=float).reshape(4, 3)
    Y = X.copy()
    Y[2, 1] += 1e-12
    a, b = warm.huellas_filas(X), warm.huellas_filas(Y)
    assert a[2] != b[2]
    assert np.array_equal(np.delete(a, 2), np.delete(b, 2))


def test_cero_negativo_cuenta_como_cero():
    """-0.0 y 0.0 son el mismo numero con bytes distintos: no debe contar como cambio."""
    X = np.array([[0.0, 1.0]])
    Y = np.array([[-0.0, 1.0]])
    assert warm.huellas_filas(X)[0] == warm.huellas_filas(Y)[0]


# --- Emparejamiento -----------------------------------------------------------
def test_sin_estado_previo_no_se_reutiliza_nada():
    mf = matriz([[1.0, 2.0], [3.0, 4.0]], [10, 20], [1, 1])
    emp = warm.emparejar(None, mf, np.array([10, 20]), HIPER)
    assert emp.incompatible and emp.motivo
    assert not emp.hay_reutilizacion
    assert (emp.fila_previa == -1).all()
    assert not emp.fila_igual.any()


def test_datos_identicos_se_reutilizan_enteros():
    mf = matriz([[1.0, 2.0], [3.0, 4.0]], [10, 20], [1, 1])
    emp = warm.emparejar(estado_de(mf, [10, 20]), mf, np.array([10, 20]), HIPER)
    assert emp.fila_igual.all()
    assert emp.ent_igual.all()
    assert emp.resumen()["observaciones_nuevas"] == 0


def test_entidad_nueva_no_tiene_predecesora():
    antes = matriz([[1.0, 2.0]], [10], [1])
    estado = estado_de(antes, [10])
    ahora = matriz([[1.0, 2.0], [9.0, 9.0]], [10, 20], [1, 1])
    emp = warm.emparejar(estado, ahora, np.array([10, 20]), HIPER)
    assert emp.fila_igual[0] and not emp.fila_igual[1]
    assert emp.ent_previa[1] == -1
    assert emp.ent_igual[0] and not emp.ent_igual[1]


def test_fila_modificada_se_empareja_pero_no_es_igual():
    antes = matriz([[1.0, 2.0]], [10], [1])
    estado = estado_de(antes, [10])
    ahora = matriz([[1.0, 2.5]], [10], [1])
    emp = warm.emparejar(estado, ahora, np.array([10]), HIPER)
    assert emp.fila_previa[0] == 0        # es la misma observacion...
    assert not emp.fila_igual[0]          # ...pero su vector cambio
    assert not emp.ent_igual[0]


def test_entidad_con_un_partido_mas_no_esta_intacta():
    """Su nube crece: los costes OT contra ella ya no valen aunque lo viejo no cambie."""
    antes = matriz([[1.0, 2.0]], [10], [1])
    estado = estado_de(antes, [10])
    ahora = matriz([[1.0, 2.0], [1.5, 2.5]], [10, 10], [1, 2])
    emp = warm.emparejar(estado, ahora, np.array([10]), HIPER)
    assert emp.fila_igual[0] and not emp.fila_igual[1]
    assert not emp.ent_igual[0]


def test_entidad_que_pierde_un_partido_no_esta_intacta():
    antes = matriz([[1.0, 2.0], [1.5, 2.5]], [10, 10], [1, 2])
    estado = estado_de(antes, [10])
    ahora = matriz([[1.0, 2.0]], [10], [1])
    emp = warm.emparejar(estado, ahora, np.array([10]), HIPER)
    assert emp.fila_igual[0]
    assert not emp.ent_igual[0]


# --- Compatibilidad -----------------------------------------------------------
def test_otro_vector_de_features_invalida_el_estado():
    antes = matriz([[1.0, 2.0]], [10], [1], feat_names=["a", "b"])
    estado = estado_de(antes, [10])
    ahora = matriz([[1.0, 2.0]], [10], [1], feat_names=["a", "c"])
    emp = warm.emparejar(estado, ahora, np.array([10]), HIPER)
    assert emp.incompatible and "features" in emp.motivo


def test_otros_hiperparametros_invalidan_el_estado():
    mf = matriz([[1.0, 2.0]], [10], [1])
    emp = warm.emparejar(estado_de(mf, [10]), mf, np.array([10]), {"beta": 2.0, "l1": 0.5})
    assert emp.incompatible and "beta" in emp.motivo


def test_sin_match_id_no_se_puede_identificar_la_observacion():
    mf = matriz([[1.0, 2.0]], [10], [1])
    mf.match_id = None
    with pytest.raises(ValueError, match="match_id"):
        warm.identidad_observaciones(mf)


# --- Serializacion ------------------------------------------------------------
def test_el_estado_sobrevive_al_disco(tmp_path):
    mf = matriz([[1.0, 2.0], [3.0, 4.0]], [10, 20], [1, 1])
    estado = estado_de(mf, [10, 20])
    estado.w_ptr, estado.w_idx, estado.w_val = warm.comprimir_W(
        [np.array([1]), np.array([0])], [np.array([0.5]), np.array([0.25])])
    estado.sigma = 1.75
    estado.costos = np.array([[0.0, 2.0], [2.0, 0.0]])

    leido = warm.EstadoWarm.cargar(estado.guardar(tmp_path / "e.warm.npz"))

    assert leido.formulacion == "2" and leido.entidad == "jugador"
    assert leido.feat_names == estado.feat_names
    assert leido.hiper == HIPER
    assert leido.sigma == pytest.approx(1.75)
    assert np.array_equal(leido.obs_hash, estado.obs_hash)
    assert np.allclose(leido.costos, estado.costos)
    idx, val = leido.columna(0)
    assert idx.tolist() == [1] and val == pytest.approx([0.5])


def test_estado_sin_partes_opcionales_se_guarda_igual(tmp_path):
    """Un estado de la Formulacion 5 no lleva W; pedir una columna da vacio, no falla."""
    mf = matriz([[1.0, 2.0]], [10], [1])
    leido = warm.EstadoWarm.cargar(estado_de(mf, [10]).guardar(tmp_path / "e.warm.npz"))
    assert leido.w_ptr is None and leido.costos is None and leido.sigma is None
    idx, val = leido.columna(0)
    assert idx.size == 0 and val.size == 0


def test_comprimir_W_conserva_cada_columna():
    cols_idx = [np.array([2, 5]), np.array([], dtype=np.int64), np.array([1])]
    cols_val = [np.array([0.1, 0.2]), np.array([]), np.array([0.3])]
    ptr, idx, val = warm.comprimir_W(cols_idx, cols_val)
    assert ptr.tolist() == [0, 2, 2, 3]
    assert idx.tolist() == [2, 5, 1]
    assert val == pytest.approx([0.1, 0.2, 0.3])


def test_ruta_estado_sigue_el_nombre_del_artefacto(tmp_path):
    ruta = warm.ruta_estado(tmp_path, "5", "equipo", "global")
    assert ruta.name == "formulacion5_equipo_global.warm.npz"


# --- Inicializador de la Formulacion 2 ---------------------------------------
def test_el_inicializador_traduce_los_pesos_al_nuevo_orden():
    """La observacion 0 de antes es la 2 de ahora: su peso debe seguirla."""
    antes = matriz([[1.0, 0.0], [0.0, 1.0]], [10, 20], [1, 1])
    estado = estado_de(antes, [10, 20])
    # Columna 1 de antes: la fila 0 contribuye con 0.7.
    estado.w_ptr, estado.w_idx, estado.w_val = warm.comprimir_W(
        [np.array([], dtype=np.int64), np.array([0])], [np.array([]), np.array([0.7])])

    # Ahora llega una entidad nueva delante y todo se desplaza una posicion.
    ahora = matriz([[5.0, 5.0], [1.0, 0.0], [0.0, 1.0]], [5, 10, 20], [1, 1, 1])
    emp = warm.emparejar(estado, ahora, np.array([5, 10, 20]), HIPER)
    inicial = warm.inicializador_f2(estado, emp)

    # La fila 2 de ahora era la 1 de antes; su vecina 1 de ahora era la 0.
    w0 = inicial(2, np.array([0, 1]))
    assert w0 == pytest.approx([0.0, 0.7])


def test_el_inicializador_arranca_en_frio_lo_que_no_existia():
    antes = matriz([[1.0, 0.0]], [10], [1])
    estado = estado_de(antes, [10])
    estado.w_ptr, estado.w_idx, estado.w_val = warm.comprimir_W(
        [np.array([], dtype=np.int64)], [np.array([])])
    ahora = matriz([[1.0, 0.0], [9.0, 9.0]], [10, 20], [1, 1])
    emp = warm.emparejar(estado, ahora, np.array([10, 20]), HIPER)
    inicial = warm.inicializador_f2(estado, emp)
    assert inicial(1, np.array([0])) is None    # observacion nueva
    assert inicial(0, np.array([1])) is None    # existia, pero sin pesos previos


# --- Costes OT reutilizables --------------------------------------------------
def test_solo_se_reutiliza_el_coste_de_pares_intactos():
    antes = matriz([[1.0, 0.0], [0.0, 1.0]], [10, 20], [1, 1])
    estado = estado_de(antes, [10, 20])
    estado.costos = np.array([[0.0, 4.0], [4.0, 0.0]])

    # 20 cambia de vector; entra 30 nueva.
    ahora = matriz([[1.0, 0.0], [0.0, 2.0], [7.0, 7.0]], [10, 20, 30], [1, 1, 1])
    emp = warm.emparejar(estado, ahora, np.array([10, 20, 30]), HIPER)
    previos = warm.costos_reutilizables(estado, emp, 3)

    assert np.isnan(previos[0, 1])   # 20 cambio -> su coste no vale
    assert np.isnan(previos[0, 2])   # 30 es nueva
    # Con una sola entidad intacta no queda ningun PAR reutilizable (la diagonal
    # no cuenta: `costos_sinkhorn` solo recorre b > a y la deja a 0).
    assert np.isnan(previos[np.triu_indices(3, k=1)]).all()


def test_se_reutiliza_el_bloque_de_las_entidades_intactas():
    antes = matriz([[1.0, 0.0], [0.0, 1.0]], [10, 20], [1, 1])
    estado = estado_de(antes, [10, 20])
    estado.costos = np.array([[0.0, 4.0], [4.0, 0.0]])
    ahora = matriz([[1.0, 0.0], [0.0, 1.0], [7.0, 7.0]], [10, 20, 30], [1, 1, 1])
    emp = warm.emparejar(estado, ahora, np.array([10, 20, 30]), HIPER)
    previos = warm.costos_reutilizables(estado, emp, 3)

    assert previos[0, 1] == pytest.approx(4.0)   # par viejo intacto
    assert np.isnan(previos[0, 2]) and np.isnan(previos[1, 2])


def test_sin_costes_guardados_no_hay_nada_que_reutilizar():
    mf = matriz([[1.0, 0.0]], [10], [1])
    estado = estado_de(mf, [10])
    emp = warm.emparejar(estado, mf, np.array([10]), HIPER)
    assert warm.costos_reutilizables(estado, emp, 1) is None


# --- Warm start del coordinate descent ---------------------------------------
def test_el_warm_start_llega_al_mismo_optimo():
    """beta > 0 hace el objetivo estrictamente convexo: el optimo es unico."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 8))
    y = rng.normal(size=30)
    frio = slim.elasticnet_no_negativo(A, y, beta=1.0, l1=0.1, max_iter=500, tol=1e-10)
    caliente = slim.elasticnet_no_negativo(
        A, y, beta=1.0, l1=0.1, max_iter=500, tol=1e-10, w_init=frio + 0.3)
    assert caliente == pytest.approx(frio, abs=1e-6)


def test_arrancar_en_el_optimo_no_lo_mueve():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    optimo = slim.elasticnet_no_negativo(A, y, beta=1.0, l1=0.1, max_iter=500, tol=1e-12)
    otra = slim.elasticnet_no_negativo(
        A, y, beta=1.0, l1=0.1, max_iter=1, tol=1e-12, w_init=optimo)
    assert otra == pytest.approx(optimo, abs=1e-9)


def test_el_arranque_negativo_se_recorta_a_la_region_factible():
    A = np.eye(3)
    y = np.array([1.0, 1.0, 1.0])
    w = slim.elasticnet_no_negativo(A, y, beta=1.0, l1=0.0, w_init=np.array([-5.0, 0.0, 1.0]))
    assert (w >= 0).all()


def test_arranque_con_forma_equivocada_falla():
    A = np.eye(3)
    y = np.ones(3)
    with pytest.raises(ValueError, match="w_init"):
        slim.elasticnet_no_negativo(A, y, beta=1.0, l1=0.0, w_init=np.zeros(2))


def test_slim_instancia_con_arranque_da_el_mismo_resultado():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 6))
    args = dict(n_neighbors=6, beta=1.0, l1=0.05, max_iter=500, tol=1e-10)
    idx_frio, val_frio = slim.slim_instancia(X, **args)
    idx_cal, val_cal = slim.slim_instancia(
        X, **args, inicial=lambda s, cand: np.full(cand.size, 0.05))
    for a, b in zip(val_frio, val_cal):
        assert np.sort(b) == pytest.approx(np.sort(a), abs=1e-6)
    for a, b in zip(idx_frio, idx_cal):
        assert set(a.tolist()) == set(b.tolist())
