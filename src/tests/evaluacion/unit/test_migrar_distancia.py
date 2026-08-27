"""Migracion de lo construido antes de que la distancia fuera un eje.

Lo que tiene que cumplir un script que reescribe miles de ficheros de resultados
ya producidos: **no cambiar ni un numero**, ser idempotente (relanzarlo no puede
encadenar sufijos) y no simular escrituras que luego no ocurren.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluacion import migrar_distancia as M, registro


def _csv(ruta: Path, **extra) -> Path:
    filas = [{"combinacion": "v01", "modelo": "F5_jugador_por_liga",
              "formulacion": 5, "entidad": "jugador", "normalizacion": "por_liga",
              "fase": 1, "metrica": "top1", "valor": 0.42, **extra}]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def _json(ruta: Path, datos: dict) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


class TestEtiqueta:
    def test_anade_la_distancia(self) -> None:
        assert M.etiquetar("F5_jugador_por_liga") == "F5_jugador_por_liga_euclidea"

    def test_es_idempotente(self) -> None:
        """Relanzar la migracion no puede dar `..._euclidea_euclidea`."""
        una = M.etiquetar("F5_jugador_por_liga")
        assert M.etiquetar(una) == una

    def test_no_toca_una_etiqueta_de_otra_geometria(self) -> None:
        assert M.etiquetar("F2_equipo_global_manhattan") == "F2_equipo_global_manhattan"


class TestCsv:
    def test_anade_la_columna_y_el_sufijo(self, tmp_path: Path) -> None:
        ruta = _csv(tmp_path / "barrido_metricas.csv")
        assert M.migrar_csv(ruta, aplicar=True)
        df = pd.read_csv(ruta)
        assert df["distancia"].tolist() == ["euclidea"]
        assert df["modelo"].tolist() == ["F5_jugador_por_liga_euclidea"]

    def test_la_columna_va_junto_a_los_otros_ejes(self, tmp_path: Path) -> None:
        """Los cuatro ejes del modelo se leen juntos y en el mismo orden que en
        las tablas del resumen."""
        ruta = _csv(tmp_path / "m.csv")
        M.migrar_csv(ruta, aplicar=True)
        columnas = list(pd.read_csv(ruta).columns)
        assert columnas.index("distancia") == columnas.index("normalizacion") + 1

    def test_no_cambia_ningun_valor(self, tmp_path: Path) -> None:
        """Es una migracion de etiquetas: no recalcula nada."""
        ruta = _csv(tmp_path / "m.csv")
        antes = pd.read_csv(ruta)
        M.migrar_csv(ruta, aplicar=True)
        despues = pd.read_csv(ruta)
        assert despues["valor"].tolist() == antes["valor"].tolist()
        assert len(despues) == len(antes)

    def test_es_idempotente(self, tmp_path: Path) -> None:
        ruta = _csv(tmp_path / "m.csv")
        M.migrar_csv(ruta, aplicar=True)
        assert not M.migrar_csv(ruta, aplicar=True)
        assert pd.read_csv(ruta)["modelo"].tolist() == ["F5_jugador_por_liga_euclidea"]

    def test_un_csv_ya_migrado_no_se_toca(self, tmp_path: Path) -> None:
        ruta = _csv(tmp_path / "m.csv", distancia="manhattan")
        assert not M.migrar_csv(ruta, aplicar=True)
        assert pd.read_csv(ruta)["distancia"].tolist() == ["manhattan"]

    def test_un_csv_sin_columna_modelo_se_ignora(self, tmp_path: Path) -> None:
        """En outputs/ hay CSV que no son resultados de evaluacion."""
        ruta = tmp_path / "otro.csv"
        pd.DataFrame([{"competicion": "La Liga", "partidos": 380}]).to_csv(
            ruta, index=False)
        assert not M.migrar_csv(ruta, aplicar=True)

    def test_un_csv_vacio_no_rompe(self, tmp_path: Path) -> None:
        ruta = tmp_path / "vacio.csv"
        ruta.write_text("", encoding="utf-8")
        assert not M.migrar_csv(ruta, aplicar=True)


class TestArtefactos:
    def test_la_meta_declara_la_distancia(self, tmp_path: Path) -> None:
        ruta = _json(tmp_path / "formulacion5_jugador_por_liga.json", {
            "formulacion": "5", "entidad": "jugador", "entity_names": ["a"],
            "feat_names": ["xg"], "meta": {"normalizacion": "por_liga"},
        })
        assert M.migrar_meta(ruta, aplicar=True)
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        assert datos["meta"]["distancia"] == "euclidea"
        assert datos["meta"]["normalizacion"] == "por_liga"   # no se pierde nada

    def test_la_huella_declara_la_distancia_en_su_celda(self, tmp_path: Path) -> None:
        ruta = _json(tmp_path / "x.huella.json", {
            "version": 1,
            "celda": {"formulacion": "5", "entidad": "jugador",
                      "normalizacion": "por_liga"},
            "hiperparametros": {"F_CLIP_Z": 6.0}, "codigo": "abc",
        })
        assert M.migrar_huella(ruta, aplicar=True)
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        assert datos["celda"]["distancia"] == "euclidea"
        assert datos["hiperparametros"] == {"F_CLIP_Z": 6.0}

    def test_un_json_que_no_es_ni_meta_ni_huella_se_ignora(self, tmp_path: Path) -> None:
        ruta = _json(tmp_path / "conjunto.json", {"nombre": "base", "usuario": "ivan"})
        assert not M.migrar_meta(ruta, aplicar=True)
        assert json.loads(ruta.read_text(encoding="utf-8")) == {
            "nombre": "base", "usuario": "ivan"}

    def test_un_json_ilegible_no_tumba_la_migracion(self, tmp_path: Path) -> None:
        ruta = tmp_path / "roto.json"
        ruta.write_text("{no es json", encoding="utf-8")
        assert not M.migrar_meta(ruta, aplicar=True)

    def test_el_linaje_de_produccion_anota_cada_artefacto(self, tmp_path: Path) -> None:
        ruta = _json(tmp_path / "produccion.json", {
            "promovidos": [{"entidad": "jugador", "formulacion": "5",
                            "combinacion": "v83"}],
        })
        assert M.migrar_produccion(ruta, aplicar=True)
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        assert datos["promovidos"][0]["distancia"] == "euclidea"
        assert datos["distancia"] == "euclidea"
        assert datos["promovidos"][0]["combinacion"] == "v83"


class TestRecorrido:
    @pytest.fixture
    def arbol(self, tmp_path: Path) -> Path:
        _csv(tmp_path / "barrido_metricas.csv")
        _csv(tmp_path / "v01" / "fase0_sanity.csv")
        _json(tmp_path / "v01" / "modelo" / "m.huella.json",
              {"celda": {"formulacion": "5", "entidad": "jugador",
                         "normalizacion": "por_liga"}})
        _json(tmp_path / "v01" / "modelo" / "m.json",
              {"formulacion": "5", "entidad": "jugador", "meta": {}})
        return tmp_path

    def test_recorre_toda_la_carpeta(self, arbol: Path) -> None:
        r = M.migrar(arbol, aplicar=True)
        assert r.total == 4
        assert len(r.csv) == 2 and len(r.huellas) == 1 and len(r.metas) == 1

    def test_simular_no_escribe_nada(self, arbol: Path) -> None:
        """Es el modo por defecto: primero se ve el alcance, luego se aplica."""
        antes = {p: p.read_bytes() for p in arbol.rglob("*") if p.is_file()}
        r = M.migrar(arbol, aplicar=False)
        assert r.total == 4                      # los cuenta...
        assert all(p.read_bytes() == b for p, b in antes.items())   # ...y no toca

    def test_relanzarla_no_encuentra_nada(self, arbol: Path) -> None:
        M.migrar(arbol, aplicar=True)
        assert M.migrar(arbol, aplicar=True).total == 0

    def test_los_resumenes_no_se_regeneran_si_no_se_piden(self, arbol: Path) -> None:
        """La migracion toca datos; rehacer una salida entera es otra decision."""
        (arbol / "resumen_barrido.md").write_text("viejo", encoding="utf-8")
        r = M.migrar(arbol, aplicar=True)
        assert r.resumenes == []
        assert (arbol / "resumen_barrido.md").read_text(encoding="utf-8") == "viejo"


class TestResumen:
    """`--regenerar-resumen`: reescribir la VISTA con el dato ya migrado.

    Un `resumen_barrido.md` escrito antes de que la geometria fuera un eje no
    dice con que se midio cada fila, y el markdown no se parchea: se vuelve a
    generar desde `barrido_metricas.csv`, que si lo dice.
    """

    @pytest.fixture
    def barrido_dir(self, tmp_path: Path) -> Path:
        _csv(tmp_path / "barrido_metricas.csv", distancia="euclidea",
             modelo="F5_jugador_por_liga_euclidea")
        _json(tmp_path / "combinaciones.json",
              {"version": registro.VERSION,
               "combinaciones": {"v01": {"hiperparametros": {"F5_RFF_DIM": 1152}}}})
        (tmp_path / "resumen_barrido.md").write_text(
            "| formulacion | entidad | normalizacion | v01 |\n", encoding="utf-8")
        return tmp_path

    def test_la_tabla_pasa_a_declarar_la_distancia(self, barrido_dir: Path) -> None:
        assert M.regenerar_resumen(barrido_dir, aplicar=True)
        texto = (barrido_dir / "resumen_barrido.md").read_text(encoding="utf-8")
        assert "| formulacion | entidad | normalizacion | distancia | v01 |" in texto
        assert "| F5 | jugador | por_liga | euclidea | 0.420 |" in texto

    def test_la_cabecera_no_finge_una_evaluacion(self, barrido_dir: Path) -> None:
        """Regenerar no mide nada: anunciar fases y bootstrap seria falso."""
        M.regenerar_resumen(barrido_dir, aplicar=True)
        texto = (barrido_dir / "resumen_barrido.md").read_text(encoding="utf-8")
        assert "Regenerado el" in texto
        assert "bootstrap" not in texto.splitlines()[2]

    def test_no_cambia_ningun_valor(self, barrido_dir: Path) -> None:
        """El CSV es la fuente: la regeneracion lo lee y no lo reescribe."""
        antes = (barrido_dir / "barrido_metricas.csv").read_bytes()
        M.regenerar_resumen(barrido_dir, aplicar=True)
        assert (barrido_dir / "barrido_metricas.csv").read_bytes() == antes

    def test_simular_no_escribe(self, barrido_dir: Path) -> None:
        antes = (barrido_dir / "resumen_barrido.md").read_bytes()
        assert M.regenerar_resumen(barrido_dir, aplicar=False)
        assert (barrido_dir / "resumen_barrido.md").read_bytes() == antes

    def test_una_carpeta_sin_metricas_no_es_un_barrido(self, tmp_path: Path) -> None:
        assert not M.regenerar_resumen(tmp_path, aplicar=True)
        assert not (tmp_path / "resumen_barrido.md").exists()

    def test_el_recorrido_encuentra_cada_carpeta_de_barrido(
        self, barrido_dir: Path
    ) -> None:
        """El CSV acumulado es lo que marca una carpeta como salida de barrido:
        los CSV por fase de las subcarpetas no cuentan."""
        _csv(barrido_dir / "v01" / "fase0_sanity.csv")
        assert M.carpetas_de_barrido(barrido_dir) == [barrido_dir]

    def test_va_despues_de_migrar_el_csv(self, tmp_path: Path) -> None:
        """En una carpeta sin migrar, la vista se genera del dato YA migrado: el
        resumen sale con la distancia aunque el CSV llegase sin la columna."""
        _csv(tmp_path / "barrido_metricas.csv")            # sin `distancia`
        _json(tmp_path / "combinaciones.json",
              {"version": registro.VERSION,
               "combinaciones": {"v01": {"hiperparametros": {"F5_RFF_DIM": 1152}}}})
        r = M.migrar(tmp_path, aplicar=True, resumenes=True)
        assert len(r.csv) == 1 and len(r.resumenes) == 1
        texto = (tmp_path / "resumen_barrido.md").read_text(encoding="utf-8")
        assert "| F5 | jugador | por_liga | euclidea | 0.420 |" in texto

    def test_avisa_de_lo_que_el_registro_no_conoce(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Sin `combinaciones.json` (ni resumen del que reconstruirlo) las tablas
        se quedan sin columnas: se dice, en vez de entregar un resumen mudo."""
        _csv(tmp_path / "barrido_metricas.csv", distancia="euclidea")
        assert M.regenerar_resumen(tmp_path, aplicar=True)
        assert "v01" in capsys.readouterr().out
