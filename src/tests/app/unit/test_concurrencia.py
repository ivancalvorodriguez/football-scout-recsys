"""Lo que se rompe al salir de «un proceso, un usuario, mi portatil».

Tres cosas, y todas se han elegido a favor de fallar pronto y ruidosamente:

- **La clave de firma** no se autogenera en produccion. Con varios procesos cada
  uno inventaria la suya y las sesiones se rechazarian entre si; el usuario
  entraria y a la siguiente peticion estaria fuera, sin ningun error visible.
- **Una tarea colgada** ya no bloquea el servidor para siempre: hay plazo maximo
  y hay boton de cancelar, y las dos cosas liberan el gestor.
- **El subproceso sobrevive al reinicio de la app** (el hilo es daemon, el
  `Popen` no), asi que queda marca en disco para poder avisar.

Los procesos se lanzan de verdad (`python -c`, milisegundos): lo que se prueba es
justamente el pegamento con el sistema operativo.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.factoria import crear_app
from src.app.tareas import (
    GestorTareas,
    Tarea,
    TareaAjena,
    huerfano,
    leer_marca,
    pid_vivo,
)


def esperar_a_que_acabe(gestor: GestorTareas, limite: float = 20.0) -> None:
    fin = time.time() + limite
    while time.time() < fin and gestor.en_curso() is not None:
        time.sleep(0.02)
    assert gestor.en_curso() is None, "la tarea no se cerro dentro del limite"


def lanzar_dormilon(gestor: GestorTareas, segundos: float = 30.0,
                    usuario: str | None = "ana") -> Tarea:
    """Arranca un proceso que no hace nada durante mucho rato.

    Es el subproceso colgado del enunciado: no escribe en su salida (asi que el
    bucle que la lee esta BLOQUEADO) y no termina por su cuenta.
    """
    tarea = Tarea(id="dormilon", tipo="prueba", titulo="Durmiendo",
                  comando=[sys.executable, "-c", f"import time; time.sleep({segundos})"],
                  usuario=usuario)
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    import threading

    threading.Thread(target=gestor._ejecutar, args=(tarea,), daemon=True).start()
    # Se espera a que el `Popen` exista: sin el no hay nada que matar.
    fin = time.time() + 10.0
    while time.time() < fin and tarea.proceso is None:
        time.sleep(0.01)
    assert tarea.proceso is not None, "el subproceso no llego a arrancar"
    return tarea


# --- Clave de firma -----------------------------------------------------------
def test_en_produccion_sin_clave_no_arranca(dir_modelos: Path, monkeypatch):
    """Fallar al arrancar es lo correcto: la alternativa es un servidor en pie
    con sesiones que se invalidan solas."""
    monkeypatch.delenv(config_app.ENV_CLAVE, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        crear_app(dir_modelos, produccion=True, testing=True)
    mensaje = str(excinfo.value)
    assert config_app.ENV_CLAVE in mensaje
    # Y el mensaje tiene que decir COMO arreglarlo, no solo que falta.
    assert "token_urlsafe" in mensaje


def test_en_produccion_con_clave_arranca(dir_modelos: Path, monkeypatch):
    monkeypatch.setenv(config_app.ENV_CLAVE, "una-clave-de-produccion")
    app = crear_app(dir_modelos, produccion=True, testing=True)
    assert app.config["SECRET_KEY"] == "una-clave-de-produccion"


def test_en_desarrollo_se_genera_una_y_se_avisa(dir_modelos: Path, monkeypatch, caplog):
    import logging

    monkeypatch.delenv(config_app.ENV_CLAVE, raising=False)
    with caplog.at_level(logging.WARNING):
        app = crear_app(dir_modelos, testing=True)
    assert app.config["SECRET_KEY"]
    assert config_app.ENV_CLAVE in caplog.text


def test_dos_apps_de_desarrollo_no_comparten_clave(dir_modelos: Path, monkeypatch):
    """Es justo por esto por lo que en produccion no se autogenera: dos procesos
    acabarian con claves distintas."""
    monkeypatch.delenv(config_app.ENV_CLAVE, raising=False)
    primera = crear_app(dir_modelos, testing=True)
    segunda = crear_app(dir_modelos, testing=True)
    assert primera.config["SECRET_KEY"] != segunda.config["SECRET_KEY"]


def test_produccion_y_debug_no_se_pueden_combinar(monkeypatch, capsys):
    """`--debug` enciende el depurador de Werkzeug, que ejecuta codigo arbitrario
    desde el navegador."""
    from src.app.__main__ import main

    monkeypatch.setenv(config_app.ENV_CLAVE, "clave")
    with pytest.raises(SystemExit) as excinfo:
        main(["--produccion", "--debug"])
    assert excinfo.value.code == 2
    assert "incompatibles" in capsys.readouterr().err


def test_sin_clave_el_cli_de_produccion_no_arranca(monkeypatch, capsys):
    from src.app.__main__ import main

    monkeypatch.delenv(config_app.ENV_CLAVE, raising=False)
    assert main(["--produccion"]) == 1
    assert config_app.ENV_CLAVE in capsys.readouterr().err


# --- Plazo maximo -------------------------------------------------------------
@pytest.mark.lento
def test_una_tarea_que_se_pasa_del_plazo_se_mata(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path, duracion_max=0.5)
    tarea = lanzar_dormilon(gestor)
    esperar_a_que_acabe(gestor)

    assert tarea.estado == "fallida"
    assert tarea.motivo == "timeout"
    assert any("excedió el máximo" in linea for linea in tarea.lineas)


@pytest.mark.lento
def test_el_plazo_libera_el_gestor(tmp_path: Path):
    """Era el fallo de fondo: `_actual` no se soltaba nunca y `/datos` quedaba
    inservible para TODAS las cuentas hasta reiniciar el servidor."""
    gestor = GestorTareas(cwd=tmp_path, duracion_max=0.5)
    lanzar_dormilon(gestor)
    esperar_a_que_acabe(gestor)

    assert gestor.en_curso() is None
    # Y se puede volver a lanzar, que es la comprobacion que importa.
    otra = gestor.lanzar("validar", "Otra", ["--ayuda-inexistente"], usuario="ana")
    assert otra is not None


@pytest.mark.lento
def test_una_tarea_normal_no_la_mata_el_plazo(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path, duracion_max=30.0)
    tarea = Tarea(id="t", tipo="prueba", titulo="Rapida",
                  comando=[sys.executable, "-c", "print('hecho')"])
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    gestor._ejecutar(tarea)

    assert tarea.estado == "terminada"
    assert tarea.motivo == ""


def test_sin_plazo_configurado_no_se_vigila(tmp_path: Path):
    """`duracion_max=None` tiene que seguir funcionando (es lo que usan las
    pruebas antiguas), sin reventar al componer el mensaje de cierre."""
    gestor = GestorTareas(cwd=tmp_path, duracion_max=None)
    tarea = Tarea(id="t", tipo="prueba", titulo="Rapida",
                  comando=[sys.executable, "-c", "print('hecho')"])
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    gestor._ejecutar(tarea)
    assert tarea.estado == "terminada"


# --- Cancelacion --------------------------------------------------------------
@pytest.mark.lento
def test_cancelar_mata_la_tarea_y_libera_el_gestor(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path)
    tarea = lanzar_dormilon(gestor, usuario="ana")

    gestor.cancelar(tarea.id, "ana")
    esperar_a_que_acabe(gestor)

    assert tarea.estado == "fallida"
    assert tarea.motivo == "cancelada"
    assert gestor.en_curso() is None


@pytest.mark.lento
def test_un_usuario_no_puede_cancelar_la_tarea_de_otro(tmp_path: Path):
    gestor = GestorTareas(cwd=tmp_path)
    tarea = lanzar_dormilon(gestor, usuario="ana")
    try:
        with pytest.raises(TareaAjena):
            gestor.cancelar(tarea.id, "bruno")
        assert gestor.en_curso() is tarea      # sigue viva
    finally:
        gestor.cancelar(tarea.id, "ana")
        esperar_a_que_acabe(gestor)


def test_cancelar_una_tarea_que_no_existe(tmp_path: Path):
    with pytest.raises(TareaAjena):
        GestorTareas(cwd=tmp_path).cancelar("no-existe", "ana")


@pytest.mark.lento
def test_no_se_puede_cancelar_por_id_ajeno_aunque_se_acierte(tmp_path: Path):
    """Conocer el id no basta: tambien tiene que ser suya."""
    gestor = GestorTareas(cwd=tmp_path)
    tarea = lanzar_dormilon(gestor, usuario="ana")
    try:
        with pytest.raises(TareaAjena):
            gestor.cancelar("dormilon", None)
    finally:
        gestor.cancelar(tarea.id, "ana")
        esperar_a_que_acabe(gestor)


# --- Marca en disco -----------------------------------------------------------
@pytest.mark.lento
def test_mientras_corre_hay_marca_en_disco(tmp_path: Path):
    marca = tmp_path / "tarea_en_curso.json"
    gestor = GestorTareas(cwd=tmp_path, ruta_marca=marca)
    tarea = lanzar_dormilon(gestor, usuario="ana")
    try:
        # La marca la escribe el hilo justo despues del `Popen`.
        fin = time.time() + 10.0
        while time.time() < fin and not marca.exists():
            time.sleep(0.01)
        datos = json.loads(marca.read_text(encoding="utf-8"))
        assert datos["tipo"] == "prueba"
        assert datos["usuario"] == "ana"
        assert datos["pid"] == tarea.proceso.pid   # el del HIJO, no el de la app
        assert datos["inicio"] > 0
    finally:
        gestor.cancelar(tarea.id, "ana")
        esperar_a_que_acabe(gestor)


@pytest.mark.lento
def test_al_terminar_la_marca_se_borra(tmp_path: Path):
    marca = tmp_path / "tarea_en_curso.json"
    gestor = GestorTareas(cwd=tmp_path, ruta_marca=marca)
    tarea = Tarea(id="t", tipo="prueba", titulo="Rapida",
                  comando=[sys.executable, "-c", "print('hecho')"])
    gestor._actual = tarea
    gestor._tareas.append(tarea)
    gestor._ejecutar(tarea)
    assert not marca.exists()


def test_pid_vivo_reconoce_al_proceso_actual():
    """En Windows NO se puede usar `os.kill(pid, 0)`: alli mataria el proceso."""
    assert pid_vivo(os.getpid())


def test_pid_vivo_dice_que_no_de_un_pid_imposible():
    assert not pid_vivo(0)
    assert not pid_vivo(-1)


def test_sin_marca_no_hay_huerfano(tmp_path: Path):
    assert leer_marca(tmp_path / "no_existe.json") is None
    assert huerfano(tmp_path / "no_existe.json") is None
    assert huerfano(None) is None


def test_una_marca_ilegible_no_rompe_el_arranque(tmp_path: Path):
    marca = tmp_path / "marca.json"
    marca.write_text("{roto", encoding="utf-8")
    assert leer_marca(marca) is None
    assert huerfano(marca) is None


def test_una_marca_con_un_pid_vivo_es_un_huerfano(tmp_path: Path):
    """Reiniciar la app NO mata el subproceso: el hilo era daemon, el Popen no."""
    marca = tmp_path / "marca.json"
    marca.write_text(json.dumps({"pid": os.getpid(), "tipo": "reentrenar",
                                 "usuario": "ana", "inicio": 0}), encoding="utf-8")
    detectado = huerfano(marca)
    assert detectado is not None
    assert detectado["tipo"] == "reentrenar"


def test_una_marca_de_un_proceso_muerto_no_es_huerfano(tmp_path: Path):
    marca = tmp_path / "marca.json"
    marca.write_text(json.dumps({"pid": 0, "tipo": "reentrenar"}), encoding="utf-8")
    assert huerfano(marca) is None


def test_el_arranque_avisa_del_huerfano(dir_modelos: Path, tmp_path: Path,
                                        monkeypatch, caplog):
    import logging

    marca = tmp_path / "marca.json"
    marca.write_text(json.dumps({"pid": os.getpid(), "tipo": "ingerir",
                                 "usuario": "ana", "inicio": 0}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        # `testing=False`: los avisos de arranque no se dan en las pruebas para
        # no ensuciar cada montaje, asi que aqui se pide una app «de verdad».
        crear_app(dir_modelos, ruta_marca=marca, testing=False)
    assert "TODAVÍA EN MARCHA" in caplog.text
    assert "corrupción" in caplog.text


def test_el_arranque_avisa_de_un_servidor_multiproceso(dir_modelos: Path,
                                                       monkeypatch, caplog):
    """gunicorn -w N rompe el «una tarea a la vez» y el sondeo de /datos/tarea."""
    import logging

    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    with caplog.at_level(logging.WARNING):
        crear_app(dir_modelos, testing=False)
    assert "multiproceso" in caplog.text
    assert "waitress" in caplog.text
