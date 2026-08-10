"""Ejecutor de tareas largas: lanzamiento, seguimiento y exclusión mutua.

Se ejecutan procesos de verdad (`python -c ...`, milisegundos) en vez de simular
el subproceso: lo que hay que comprobar aquí es justamente el pegamento con el
sistema operativo — que la salida llega, que el `\\r` de la barra de progreso no
se traga las líneas y que el código de salida se traduce bien.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from src.app import tareas
from src.app.tareas import GestorTareas, Tarea, TareaEnCurso, _leer_progreso


@pytest.fixture
def gestor_py(tmp_path: Path) -> GestorTareas:
    return GestorTareas(cwd=tmp_path)


def esperar(gestor: GestorTareas, limite: float = 20.0) -> Tarea:
    """Bloquea hasta que no quede ninguna tarea en curso."""
    fin = time.time() + limite
    while time.time() < fin and gestor.en_curso() is not None:
        time.sleep(0.02)
    ultima = gestor.ultima()
    assert ultima is not None
    return ultima


def lanzar_python(gestor: GestorTareas, codigo: str, titulo: str = "prueba") -> Tarea:
    """Ejecuta `python -c <codigo>` con la maquinaria del gestor, en el hilo actual.

    `lanzar` compone siempre `python -m <modulo registrado>` — eso es
    precisamente lo que se quiere blindar y se prueba aparte. Para ejercitar la
    captura de salida hace falta un programa a medida, así que se llama a
    `_ejecutar` con el comando ya puesto; síncrono, además, para no tener que
    sincronizar con el hilo en cada aserción.
    """
    tarea = Tarea(id="t", tipo="prueba", titulo=titulo,
                  comando=[sys.executable, "-c", codigo])
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    gestor._ejecutar(tarea)
    return tarea


# --- Lectura del progreso -----------------------------------------------------
def test_lee_el_porcentaje_del_reentrenamiento():
    progreso, paso = _leer_progreso("    ajuste:  54%  (27313/50579)")
    assert progreso == pytest.approx(27313 / 50579)
    assert paso == ""


def test_lee_el_avance_de_la_ingesta():
    progreso, paso = _leer_progreso("  [12/306] 3890561 Hoffenheim vs Schalke 04")
    assert progreso == pytest.approx(12 / 306)
    assert "12" in paso and "306" in paso


def test_lee_la_etiqueta_del_modelo_en_curso():
    progreso, paso = _leer_progreso("[formulacion 2 | jugador | por_liga]")
    assert progreso is None
    assert paso == "formulacion 2 | jugador | por_liga"


def test_una_linea_cualquiera_no_aporta_progreso():
    assert _leer_progreso("Copia de seguridad: outputs/_backup/x.db") == (None, "")


def test_total_cero_no_divide_entre_cero():
    progreso, _ = _leer_progreso("  [0/0] nada")
    assert progreso is None


# --- Ejecución ----------------------------------------------------------------
def test_captura_la_salida_y_el_exito(gestor_py: GestorTareas):
    tarea = lanzar_python(gestor_py, "print('hola'); print('adios')")
    assert tarea.estado == "terminada" and tarea.codigo == 0
    assert tarea.lineas == ["hola", "adios"]
    assert tarea.progreso == 1.0


def test_un_fallo_deja_la_tarea_como_fallida(gestor_py: GestorTareas):
    tarea = lanzar_python(gestor_py, "import sys; print('mal'); sys.exit(3)")
    assert tarea.estado == "fallida" and tarea.codigo == 3
    assert "mal" in tarea.lineas


def test_tambien_recoge_lo_que_va_a_stderr(gestor_py: GestorTareas):
    """`stderr` se mezcla con `stdout`: un traceback tiene que verse en la página."""
    tarea = lanzar_python(gestor_py, "import sys; print('roto', file=sys.stderr)")
    assert "roto" in tarea.lineas


def test_la_barra_de_progreso_llega_como_lineas(gestor_py: GestorTareas):
    """El hijo reescribe la misma línea con `\\r`; en modo texto llega separada."""
    codigo = (
        "import sys\n"
        "for p in (10, 50, 100):\n"
        "    print('    ajuste: %3d%%  (%d/100)' % (p, p), end='\\r')\n"
        "print()\n"
    )
    tarea = lanzar_python(gestor_py, codigo)
    assert tarea.progreso == 1.0
    # Las de progreso se colapsan en una sola: solo queda la ultima.
    de_progreso = [x for x in tarea.lineas if "ajuste:" in x]
    assert len(de_progreso) == 1 and "100%" in de_progreso[0]


def test_el_log_no_crece_sin_limite(gestor_py: GestorTareas, monkeypatch):
    monkeypatch.setattr(tareas, "MAX_LINEAS", 5)
    tarea = lanzar_python(gestor_py, "for i in range(50): print('linea', i)")
    assert len(tarea.lineas) == 5
    assert tarea.lineas[-1] == "linea 49"     # se conservan las ULTIMAS


def test_un_comando_inexistente_no_revienta_el_hilo(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path)
    tarea = Tarea(id="t", tipo="prueba", titulo="x",
                  comando=[str(tmp_path / "no-existe.exe")])
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    gestor._ejecutar(tarea)
    assert tarea.estado == "fallida"
    assert any("No se pudo lanzar" in x for x in tarea.lineas)


# --- Exclusión mutua e historial ---------------------------------------------
def test_no_se_admite_una_segunda_tarea_a_la_vez(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path)
    gestor._actual = Tarea(id="ocupado", tipo="ingerir", titulo="Ocupado",
                           comando=["x"])
    with pytest.raises(TareaEnCurso, match="en curso"):
        gestor.lanzar("ingerir", "Otra", ["--paquete", "x"])


def test_un_tipo_desconocido_se_rechaza(tmp_path: Path):
    with pytest.raises(ValueError, match="desconocido"):
        GestorTareas(cwd=tmp_path).lanzar("borrar_todo", "Nope", [])


def test_el_comando_se_compone_con_el_modulo_registrado(tmp_path: Path, monkeypatch):
    """El ejecutable y el módulo son constantes: el usuario no elige qué se corre."""
    lanzadas = {}

    def falso(self, tarea):
        lanzadas["comando"] = tarea.comando
        self._cerrar(tarea, codigo=0)

    monkeypatch.setattr(GestorTareas, "_ejecutar", falso)
    gestor = GestorTareas(cwd=tmp_path)
    gestor.lanzar("ingerir", "x", ["--paquete", "sitio raro; rm -rf /"])
    esperar(gestor)

    assert lanzadas["comando"][:3] == [sys.executable, "-m", tareas.MODULOS["ingerir"]]
    # El argumento viaja como UN elemento de la lista: no hay shell que lo parta.
    assert lanzadas["comando"][-1] == "sitio raro; rm -rf /"


def test_las_tareas_guardadas_no_crecen_sin_limite(tmp_path: Path, monkeypatch):
    """Se conservan unas pocas; la página solo enseña la última."""
    monkeypatch.setattr(GestorTareas, "_ejecutar",
                        lambda self, tarea: self._cerrar(tarea, codigo=0))
    gestor = GestorTareas(cwd=tmp_path, max_historial=3)
    for i in range(6):
        gestor.lanzar("ingerir", f"tarea {i}", [])
    assert len(gestor._tareas) == 3
    assert gestor.ultima().titulo == "tarea 5"


def test_sin_tareas_no_hay_ultima(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path)
    assert gestor.ultima() is None and gestor.en_curso() is None


def test_el_dict_es_serializable_y_completo(gestor_py: GestorTareas):
    tarea = lanzar_python(gestor_py, "print('ok')")
    datos = tarea.como_dict()
    assert datos["estado"] == "terminada"
    assert datos["lineas"] == ["ok"]
    assert isinstance(datos["comando"], str)
    assert datos["segundos"] >= 0
