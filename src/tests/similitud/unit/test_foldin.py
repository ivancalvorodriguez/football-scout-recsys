"""Proyeccion (fold-in) de una entidad que no esta en el modelo.

Lo que se vigila:

- Que la entidad nueva entre en el MISMO espacio que las del modelo: mismo mapa
  RFF y mismo ancho de kernel. Si `proyectar` estimase el sigma de los datos de
  la consulta, el coseno resultante compararia dos espacios distintos y ninguna
  prueba de forma lo detectaria — por eso hay una que fija el resultado contra
  `distributional.similitud_mmd`, que es la definicion.
- Que cada proyeccion declare su FIDELIDAD: la F5 reproduce la etapa 1 del modelo
  y la F2 solo una direccion de su agregacion. Las dos se sirven, pero no
  significan lo mismo y el consumidor tiene que poder distinguirlas.
- Que se niegue a proyectar donde de verdad no esta definido (F5 con Sinkhorn) en
  vez de devolver un numero.
- Que una entidad del modelo sin observaciones en estos datos no se cuele en el
  top-k con una puntuacion de relleno.
- Que preparar una vez y proyectar muchas de la MISMA respuesta que proyectar de
  una en una: es una optimizacion, no otro calculo.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import config, distributional, foldin
from src.similitud.features import MatrizFeatures
from src.similitud.modelo import ModeloSimilitud

SIGMA = 1.5


def _mf(ids: list[int], X: np.ndarray, weight: np.ndarray | None = None
        ) -> MatrizFeatures:
    ids = np.asarray(ids, dtype=np.int64)
    return MatrizFeatures(
        X=np.asarray(X, dtype=float),
        feat_names=[f"f{i}" for i in range(np.shape(X)[1])],
        entity_id=ids,
        entity_name=np.array([f"e{i}" for i in ids]),
        weight=np.ones(len(ids)) if weight is None else np.asarray(weight, float),
        league=np.array(["L1"] * len(ids)),
        match_id=np.arange(len(ids), dtype=np.int64),
    )


def _modelo(ids: list[int], formulacion: str = "5", metodo: str = "mmd",
            rff_dim: int = 64) -> ModeloSimilitud:
    n = len(ids)
    return ModeloSimilitud(
        formulacion=formulacion,
        entidad="jugador",
        S=np.zeros((n, n)),
        entity_ids=np.asarray(ids, dtype=np.int64),
        entity_names=[f"e{i}" for i in ids],
        feat_names=["f0", "f1"],
        feat_display=np.zeros((n, 2)),
        meta={"metodo_distribucional": metodo, "rff_dim": rff_dim},
    )


class TestQueSePuedeProyectar:
    def test_la_f5_con_mmd_es_fiel(self) -> None:
        """Su etapa 1 esta definida fuera de muestra: es la misma magnitud."""
        assert foldin.fidelidad(_modelo([1, 2])) == foldin.FIEL

    def test_la_f2_es_aproximada(self) -> None:
        """Su similitud agrega bloques de W por los DOS lados y de una entidad
        nueva solo se resuelve uno: se sirve, pero marcada."""
        assert foldin.fidelidad(_modelo([1, 2], formulacion="2")) == foldin.APROXIMADA
        assert foldin.puede_proyectar(_modelo([1, 2], formulacion="2")) is True

    def test_la_f5_con_sinkhorn_no_se_puede(self) -> None:
        """Proyectar exigiria el coste OT contra las P nubes del modelo."""
        assert foldin.fidelidad(_modelo([1, 2], metodo="sinkhorn")) is None
        assert foldin.puede_proyectar(_modelo([1, 2], metodo="sinkhorn")) is False

    def test_proyectar_lo_rechaza_en_vez_de_inventarselo(self) -> None:
        mf = _mf([1, 1, 2, 2, 9, 9], np.random.default_rng(0).normal(size=(6, 2)))
        with pytest.raises(foldin.EntidadNoProyectable):
            foldin.proyectar(_modelo([1, 2], metodo="sinkhorn"), mf, 9, sigma=SIGMA)

    def test_la_f5_sin_sigma_no_se_lo_inventa(self) -> None:
        """El ancho de kernel sale del ajuste; estimarlo de la consulta pondria a
        la entidad nueva en otro espacio."""
        mf = _mf([1, 1, 2, 2, 9, 9], np.random.default_rng(0).normal(size=(6, 2)))
        with pytest.raises(foldin.EntidadNoProyectable, match="sigma"):
            foldin.proyectar(_modelo([1, 2]), mf, 9)


class TestProyeccion:
    def _caso(self, rng=None):
        rng = rng or np.random.default_rng(7)
        ids = [1, 1, 1, 2, 2, 3, 3, 9, 9, 9]
        X = rng.normal(size=(len(ids), 2))
        return _modelo([1, 2, 3]), _mf(ids, X), X, ids

    def test_devuelve_una_puntuacion_por_entidad_del_modelo(self) -> None:
        modelo, mf, _, _ = self._caso()
        p = foldin.proyectar(modelo, mf, 9, sigma=SIGMA)
        assert p.puntuacion.shape == (3,)
        assert list(p.ids) == [1, 2, 3]
        assert p.entidad_id == 9 and p.n_observaciones == 3

    def test_coincide_con_la_similitud_mmd_del_pipeline(self) -> None:
        """La definicion: proyectar tiene que dar lo MISMO que ajustar con la
        entidad dentro y leer su fila de la S distribucional. Es lo que garantiza
        que la entidad nueva cae en el espacio del modelo y no en otro."""
        modelo, mf, _, ids = self._caso()
        p = foldin.proyectar(modelo, mf, 9, sigma=SIGMA, rff_dim=64)

        # Misma matriz de features, ajustando las cuatro entidades a la vez.
        orden = {1: 0, 2: 1, 3: 2, 9: 3}
        idx = np.array([orden[i] for i in ids])
        antiguo = config.F5_RFF_DIM
        try:
            config.F5_RFF_DIM = 64
            S = distributional.similitud_mmd(mf, idx, 4, sigma=SIGMA)
        finally:
            config.F5_RFF_DIM = antiguo
        assert p.puntuacion == pytest.approx(S[3, :3])

    def test_el_top_va_de_mayor_a_menor(self) -> None:
        modelo, mf, _, _ = self._caso()
        p = foldin.proyectar(modelo, mf, 9, sigma=SIGMA)
        puntos = [s for _, s in p.top(3)]
        assert puntos == sorted(puntos, reverse=True)

    def test_el_top_devuelve_ids_del_modelo(self) -> None:
        modelo, mf, _, _ = self._caso()
        assert {i for i, _ in foldin.proyectar(modelo, mf, 9, sigma=SIGMA).top(2)} <= {1, 2, 3}

    def test_un_top_de_cero_o_negativo_no_devuelve_nada(self) -> None:
        modelo, mf, _, _ = self._caso()
        p = foldin.proyectar(modelo, mf, 9, sigma=SIGMA)
        assert p.top(0) == [] and p.top(-3) == []

    def test_se_parece_mas_a_quien_tiene_las_mismas_observaciones(self) -> None:
        """Control de sentido: una entidad clonada de la 2 debe puntuar mas alto
        con la 2 que con nadie."""
        X = np.array([[0.0, 0.0], [0.1, 0.0],       # entidad 1
                      [5.0, 5.0], [5.1, 5.0],       # entidad 2
                      [-4.0, 9.0], [-4.1, 9.0],     # entidad 3
                      [5.0, 5.0], [5.1, 5.0]])      # entidad 9 = clon de la 2
        mf = _mf([1, 1, 2, 2, 3, 3, 9, 9], X)
        p = foldin.proyectar(_modelo([1, 2, 3]), mf, 9, sigma=SIGMA)
        assert p.top(1)[0][0] == 2

    def test_los_minutos_pesan_en_el_embedding(self) -> None:
        """El embedding es la media PONDERADA por minutos: cambiar los pesos de
        las observaciones de la entidad nueva cambia su proyeccion."""
        X = np.array([[0.0, 0.0], [4.0, 4.0], [0.0, 0.0], [4.0, 4.0]])
        mf_a = _mf([1, 2, 9, 9], X, weight=[1.0, 1.0, 9.0, 1.0])
        mf_b = _mf([1, 2, 9, 9], X, weight=[1.0, 1.0, 1.0, 9.0])
        modelo = _modelo([1, 2])
        a = foldin.proyectar(modelo, mf_a, 9, sigma=SIGMA).puntuacion
        b = foldin.proyectar(modelo, mf_b, 9, sigma=SIGMA).puntuacion
        assert not np.allclose(a, b)
        assert a[0] > a[1] and b[1] > b[0]     # cada una tira hacia su lado


class TestLoQueNoSePuede:
    def test_una_entidad_que_ya_esta_en_el_modelo(self) -> None:
        """Tiene su fila en S: proyectarla seria darle una respuesta peor."""
        mf = _mf([1, 1, 2, 2], np.random.default_rng(0).normal(size=(4, 2)))
        with pytest.raises(foldin.EntidadNoProyectable, match="ya esta en el modelo"):
            foldin.proyectar(_modelo([1, 2]), mf, 1, sigma=SIGMA)

    def test_una_entidad_sin_observaciones_en_estos_datos(self) -> None:
        mf = _mf([1, 1, 2, 2], np.random.default_rng(0).normal(size=(4, 2)))
        with pytest.raises(foldin.EntidadNoProyectable, match="no tiene observaciones"):
            foldin.proyectar(_modelo([1, 2]), mf, 77, sigma=SIGMA)

    def test_una_entidad_del_modelo_ausente_de_la_bd_no_entra_en_el_top(self) -> None:
        """Sin observaciones su embedding seria nulo y puntuaria 0 de casualidad,
        colandose por delante de similitudes bajas pero reales."""
        mf = _mf([1, 1, 9, 9], np.random.default_rng(0).normal(size=(4, 2)))
        p = foldin.proyectar(_modelo([1, 2, 3]), mf, 9, sigma=SIGMA)
        assert np.isneginf(p.puntuacion[1]) and np.isneginf(p.puntuacion[2])
        assert [i for i, _ in p.top(3)] == [1]


class TestProyeccionF2:
    """La proyeccion aproximada: columnas de W resueltas contra el modelo.

    No se contrasta contra `slim_instancia` como la F5 contra `similitud_mmd`,
    porque NO pretende reproducirla (es una sola direccion de la agregacion). Lo
    que tiene que cumplir es la escala y el sentido: mismo denominador que
    `agregar_W_a_entidades` y mas puntuacion para quien de verdad se le parece.
    """

    def _caso(self):
        X = np.array([[0.0, 0.0], [0.2, 0.1],       # entidad 1
                      [5.0, 5.0], [5.2, 4.9],       # entidad 2
                      [5.1, 5.1], [5.0, 4.8]])      # entidad 9 = parecida a la 2
        return _modelo([1, 2], formulacion="2"), _mf([1, 1, 2, 2, 9, 9], X)

    def test_puntua_a_todas_las_entidades_del_modelo(self) -> None:
        modelo, mf = self._caso()
        p = foldin.proyectar(modelo, mf, 9)
        assert p.puntuacion.shape == (2,) and p.fidelidad == foldin.APROXIMADA

    def test_se_parece_mas_a_quien_reconstruye_sus_observaciones(self) -> None:
        modelo, mf = self._caso()
        assert foldin.proyectar(modelo, mf, 9).top(1)[0][0] == 2

    def test_no_necesita_sigma(self) -> None:
        """La F2 no tiene kernel: exigirlo cerraria la puerta a proyectar equipos."""
        modelo, mf = self._caso()
        assert np.isfinite(foldin.proyectar(modelo, mf, 9).puntuacion).any()

    def test_usa_los_hiperparametros_del_ARTEFACTO(self) -> None:
        """Resolver la columna con otra regularizacion la sacaria del ajuste al
        que se la quiere pegar: los valores salen del `meta`, no de `config`."""
        modelo, mf = self._caso()
        suave = foldin.proyectar(modelo, mf, 9).puntuacion
        modelo.meta["l1"] = 1e6      # con esta penalizacion no sobrevive ningun peso
        duro = foldin.proyectar(modelo, mf, 9).puntuacion
        assert not np.allclose(suave, duro)
        assert np.allclose(duro[np.isfinite(duro)], 0.0)

    def test_divide_por_la_masa_como_la_agregacion_del_ajuste(self) -> None:
        """Sin el denominador N_p*N_q, la entidad con mas partidos gana siempre
        (es el error que hacia inservible el primer intento de proyectar la F2)."""
        X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],  # 1: 4 obs
                      [0.0, 0.0],                                       # 2: 1 obs
                      [0.0, 0.0]])                                      # 9
        p = foldin.proyectar(_modelo([1, 2], formulacion="2"), _mf([1, 1, 1, 1, 2, 9], X), 9)
        # Identicas: la unica diferencia entre 1 y 2 es cuantas observaciones tienen.
        assert p.puntuacion[0] == pytest.approx(p.puntuacion[1])


class TestSimilitudEntreNuevas:
    """La S entre entidades que NINGUNA esta en el modelo (F2).

    Aqui no se pierde media magnitud —las columnas de las dos entidades se
    resuelven al medir—, asi que lo exigible es lo maximo: que sea EXACTAMENTE la
    agregacion del ajuste. Por eso la primera prueba la contrasta contra
    `slim_instancia` + `agregar_W_a_entidades`, que es la definicion, igual que la
    proyeccion F5 se contrasta contra `similitud_mmd`.
    """

    def _caso(self, ids: list[int], X: np.ndarray, modelo_ids: list[int],
              n_neighbors: int):
        mf = _mf(ids, X)
        modelo = _modelo(modelo_ids, formulacion="2")
        modelo.meta["n_neighbors"] = n_neighbors
        grupos = {e: g for g, e in enumerate(dict.fromkeys(ids))}
        grupo = np.array([grupos[i] for i in ids], dtype=np.int64)
        return foldin.preparar(modelo, mf), mf, grupo, len(grupos)

    def test_es_la_agregacion_del_AJUSTE_cuando_no_hay_anclas(self) -> None:
        """La propiedad de la que depende que `gen_top1` signifique lo mismo que
        el `top1` de la Fase 1: sin observaciones del modelo de por medio, esto
        tiene que ser el pipeline real, no algo parecido."""
        from src.similitud import slim

        rng = np.random.default_rng(5)
        ids = [10, 10, 10, 11, 11, 12, 12, 12, 13, 13]
        X = rng.normal(size=(len(ids), 3))
        # El modelo no comparte ninguna entidad con `mf`: no hay anclas.
        ref, mf, grupo, n = self._caso(ids, X, [999], n_neighbors=9)

        obtenida = foldin.similitud_entre_nuevas(ref, mf, grupo, n)
        cols_idx, cols_val = slim.slim_instancia(
            X, n_neighbors=9, beta=config.F2_BETA, l1=config.F2_L1,
            max_iter=config.F2_MAX_ITER, tol=config.F2_TOL)
        esperada = slim.agregar_W_a_entidades(
            cols_idx, cols_val, grupo, np.asarray(mf.weight, dtype=float), n)
        assert obtenida == pytest.approx(esperada)

    def test_sale_simetrica_y_con_diagonal_cero(self) -> None:
        """Las dos direcciones se agregan y se promedian como en el ajuste; es lo
        que distingue esto de la proyeccion contra el modelo."""
        rng = np.random.default_rng(6)
        ids = [10, 10, 11, 11, 12, 12]
        ref, mf, grupo, n = self._caso(
            ids, rng.normal(size=(len(ids), 3)), [999], n_neighbors=4)
        S = foldin.similitud_entre_nuevas(ref, mf, grupo, n)
        assert S.shape == (n, n)
        assert S == pytest.approx(S.T)
        assert np.diag(S) == pytest.approx(np.zeros(n))

    def test_puntua_mas_alto_a_quien_de_verdad_se_le_parece(self) -> None:
        """Control de sentido: dos grupos con observaciones casi iguales tienen
        que salir mas parecidos entre si que con el tercero, que esta lejos.

        Las features van a escala de z-score (norma ~1) porque el umbral L1 es
        absoluto: con vectores diminutos ningun peso sobrevive al
        soft-thresholding y la S saldria toda a cero — cierto, pero no probaria
        nada."""
        X = np.array([[1.0, 1.0], [1.1, 0.9],       # grupo 0
                      [0.9, 1.05], [1.05, 1.0],     # grupo 1 = pegado al 0
                      [-2.0, 3.0], [-2.1, 3.1]])    # grupo 2 = lejos
        ref, mf, grupo, n = self._caso(
            [10, 10, 11, 11, 12, 12], X, [999], n_neighbors=4)
        S = foldin.similitud_entre_nuevas(ref, mf, grupo, n)
        assert S[0, 1] > S[0, 2] and S[0, 1] > S[1, 2]

    def test_ninguna_observacion_se_reconstruye_consigo_misma(self) -> None:
        """El `w_ss = 0` del ajuste. Sin excluirla, cada columna se explicaria con
        su propia fila, la W seria la identidad y la S entre grupos, cero."""
        X = np.array([[1.0, 1.0, 0.9], [1.2, 0.8, 1.1],
                      [0.9, 1.1, 1.0], [1.1, 0.95, 0.95]])
        ref, mf, grupo, n = self._caso(
            [10, 10, 11, 11], X, [999], n_neighbors=99)
        S = foldin.similitud_entre_nuevas(ref, mf, grupo, n)
        assert S[0, 1] > 0.0

    def test_las_anclas_compiten_pero_no_reciben(self) -> None:
        """Las observaciones del modelo entran en el diccionario —es lo que hace
        selectiva la reconstruccion— y cambian el resultado, pero la S que sale es
        solo la de los grupos: no puede tener mas filas que grupos hay.

        Con el ancla pegada a los dos grupos, parte del peso que se repartian
        entre ellos se lo lleva ella y la similitud baja. Ese es exactamente su
        papel: sin candidatas del modelo, dos entidades nuevas se explicarian solo
        la una a la otra por no haber nada mejor."""
        X = np.array([[1.0, 1.0], [1.1, 0.9],       # grupo (entidad 10)
                      [0.9, 1.05], [1.05, 1.0],     # grupo (entidad 11)
                      [1.0, 1.0], [1.02, 0.99]])    # entidad 1: ancla del modelo
        mf = _mf([10, 10, 11, 11, 1, 1], X)
        grupo = np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)

        con = _modelo([1], formulacion="2")
        sin = _modelo([999], formulacion="2")
        con.meta["n_neighbors"] = sin.meta["n_neighbors"] = 4
        S_con = foldin.similitud_entre_nuevas(
            foldin.preparar(con, mf), mf, grupo, 2)
        S_sin = foldin.similitud_entre_nuevas(
            foldin.preparar(sin, mf), mf, grupo, 2)
        assert S_con.shape == S_sin.shape == (2, 2)
        assert 0.0 < S_con[0, 1] < S_sin[0, 1]

    def test_una_entidad_del_modelo_desdoblada_no_cuenta_dos_veces(self) -> None:
        """El ambito «dentro» de la Fase 7: sus filas son grupo Y son del modelo.
        Si siguieran de anclas, cada mitad se reconstruiria con las filas que ya
        es y la comparacion con el ambito «fuera» no mediria lo mismo."""
        rng = np.random.default_rng(9)
        ids = [1, 1, 2, 2]                      # las dos estan en el modelo
        mf = _mf(ids, rng.normal(size=(4, 3)))
        ref = foldin.preparar(_modelo([1, 2], formulacion="2"), mf)
        # Grupos = las dos mitades de la entidad 1; la 2 se queda de ancla.
        grupo = np.array([0, 1, -1, -1], dtype=np.int64)
        S = foldin.similitud_entre_nuevas(ref, mf, grupo, 2)
        assert S.shape == (2, 2) and np.isfinite(S).all()

    def test_sin_grupos_devuelve_una_matriz_vacia_y_no_revienta(self) -> None:
        rng = np.random.default_rng(10)
        mf = _mf([1, 1, 2, 2], rng.normal(size=(4, 2)))
        ref = foldin.preparar(_modelo([1, 2], formulacion="2"), mf)
        S = foldin.similitud_entre_nuevas(
            ref, mf, np.full(4, -1, dtype=np.int64), 0)
        assert S.shape == (0, 0)

    def test_la_f5_se_rechaza_porque_ya_tiene_su_via(self) -> None:
        """Dos nubes nuevas se comparan por el coseno de sus embeddings, que no
        cuesta ningun ajuste: pasar por aqui seria pagarlo para nada."""
        rng = np.random.default_rng(12)
        mf = _mf([1, 1, 9, 9], rng.normal(size=(4, 2)))
        ref = foldin.preparar(_modelo([1]), mf, sigma=SIGMA)
        with pytest.raises(foldin.EntidadNoProyectable, match="embeddings"):
            foldin.similitud_entre_nuevas(
                ref, mf, np.array([-1, -1, 0, 1]), 2)

    def test_informa_del_avance_columna_a_columna(self) -> None:
        """Es el bucle caro de la fase 7 en la F2 (una regresion por observacion):
        sin reportero, la barra se queda parada minutos."""
        rng = np.random.default_rng(13)
        ids = [10, 10, 11, 11]
        ref, mf, grupo, n = self._caso(
            ids, rng.normal(size=(4, 3)), [999], n_neighbors=3)
        vistos: list[tuple[int, int]] = []
        foldin.similitud_entre_nuevas(
            ref, mf, grupo, n, progreso=lambda h, t: vistos.append((h, t)))
        assert vistos == [(1, 4), (2, 4), (3, 4), (4, 4)]


class TestReferenciaReutilizada:
    def test_preparar_una_vez_da_lo_mismo_que_proyectar_una_a_una(self) -> None:
        """Es una optimizacion para proyectar en lote (una liga entera, o una
        sesion de busqueda), no otro calculo."""
        rng = np.random.default_rng(3)
        ids = [1, 1, 2, 2, 3, 3, 8, 8, 9, 9]
        mf = _mf(ids, rng.normal(size=(len(ids), 2)))
        modelo = _modelo([1, 2, 3])
        ref = foldin.preparar(modelo, mf, sigma=SIGMA)
        for eid in (8, 9):
            suelta = foldin.proyectar(modelo, mf, eid, sigma=SIGMA)
            lote = foldin.proyectar_desde(ref, mf, eid)
            assert lote.puntuacion == pytest.approx(suelta.puntuacion)
            assert lote.fidelidad == suelta.fidelidad

    def test_tambien_en_la_f2(self) -> None:
        rng = np.random.default_rng(4)
        ids = [1, 1, 2, 2, 9, 9]
        mf = _mf(ids, rng.normal(size=(len(ids), 2)))
        modelo = _modelo([1, 2], formulacion="2")
        ref = foldin.preparar(modelo, mf)
        assert (foldin.proyectar_desde(ref, mf, 9).puntuacion
                == pytest.approx(foldin.proyectar(modelo, mf, 9).puntuacion))


class TestEmbeddings:
    def test_salen_normalizados(self) -> None:
        rng = np.random.default_rng(1)
        mf = _mf([1, 1, 2, 2], rng.normal(size=(4, 3)))
        mu = foldin.embeddings(mf, np.array([0, 0, 1, 1]), 2, SIGMA, 32)
        assert np.linalg.norm(mu, axis=1) == pytest.approx([1.0, 1.0])

    def test_el_sigma_cambia_el_espacio(self) -> None:
        """Por eso `proyectar` lo exige y no lo estima: con otro ancho, la
        entidad nueva no seria comparable con las del modelo."""
        rng = np.random.default_rng(2)
        mf = _mf([1, 1, 2, 2], rng.normal(size=(4, 3)))
        idx = np.array([0, 0, 1, 1])
        a = foldin.embeddings(mf, idx, 2, 0.5, 32)
        b = foldin.embeddings(mf, idx, 2, 5.0, 32)
        assert not np.allclose(a, b)


class TestPiezas:
    """La proyeccion descompuesta por observacion, para poder repesarla gratis."""

    def _caso(self, formulacion: str = "5"):
        rng = np.random.default_rng(11)
        ids = [1, 1, 2, 2, 3, 3, 9, 9, 9]
        mf = _mf(ids, rng.normal(size=(len(ids), 2)))
        modelo = _modelo([1, 2, 3], formulacion=formulacion)
        sigma = SIGMA if formulacion == "5" else None
        return modelo, mf, foldin.preparar(modelo, mf, sigma=sigma)

    @pytest.mark.parametrize("formulacion", ["5", "2"])
    def test_sin_repesar_da_la_misma_proyeccion(self, formulacion) -> None:
        """Es la condicion para poder usarlas: si el atajo no reprodujera el
        camino normal, el remuestreo mediria otra cosa."""
        modelo, mf, ref = self._caso(formulacion)
        normal = foldin.proyectar_desde(ref, mf, 9)
        atajo = foldin.piezas(ref, mf, 9).puntuar()
        assert atajo.puntuacion == pytest.approx(normal.puntuacion)
        assert atajo.fidelidad == normal.fidelidad
        assert atajo.n_observaciones == normal.n_observaciones

    @pytest.mark.parametrize("formulacion", ["5", "2"])
    def test_repesar_equivale_a_proyectar_con_esos_pesos(self, formulacion) -> None:
        """Lo que hace legitimo el bootstrap: cambiar las masas por la via barata
        tiene que dar lo mismo que cambiarlas en los datos y volver a proyectar."""
        modelo, mf, ref = self._caso(formulacion)
        pesos = np.array([3.0, 0.0, 1.0])          # multiplicidades tipo bootstrap
        repesada = foldin.piezas(ref, mf, 9).puntuar(pesos)

        suya = np.asarray(mf.entity_id) == 9
        w = np.asarray(mf.weight, dtype=float).copy()
        w[suya] = pesos
        otra = _mf([int(i) for i in mf.entity_id], mf.X, weight=w)
        directa = foldin.proyectar_desde(
            foldin.preparar(modelo, otra, sigma=ref.sigma), otra, 9)
        assert repesada.puntuacion == pytest.approx(directa.puntuacion)

    def test_una_entidad_del_modelo_se_rechaza_igual(self) -> None:
        _modelo_, mf, ref = self._caso()
        with pytest.raises(foldin.EntidadNoProyectable, match="ya esta en el modelo"):
            foldin.piezas(ref, mf, 1)

    def test_sin_masa_no_divide_por_cero(self) -> None:
        """Un remuestreo puede dejar a la entidad sin ninguna observacion."""
        _modelo_, mf, ref = self._caso()
        p = foldin.piezas(ref, mf, 9).puntuar(np.zeros(3))
        assert np.isfinite(p.puntuacion[np.isfinite(p.puntuacion)]).all()
