"""CLI de cuentas: alta, cambio de contrasena, baja y listado.

No hay registro desde el navegador a proposito, asi que este comando ES la unica
forma de crear una cuenta. Lo que se comprueba: que la contrasena nunca llega por
argumento (queda en el historial del shell y en la lista de procesos), que se
pide dos veces, y que nada de lo que se imprime la contiene.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app.auth import Usuarios, verificar
from src.app.usuarios import MIN_LARGO, main

CONTRASENA = "contrasena-valida"


class Teclado:
    """Sustituto de `getpass`: devuelve respuestas preparadas, sin tty."""

    def __init__(self, *respuestas: str) -> None:
        self.respuestas = list(respuestas)
        self.preguntas: list[str] = []

    def __call__(self, aviso: str = "") -> str:
        self.preguntas.append(aviso)
        return self.respuestas.pop(0)


@pytest.fixture
def fichero(tmp_path: Path) -> Path:
    return tmp_path / "usuarios.json"


def _correr(fichero: Path, *args: str, teclado: Teclado | None = None) -> int:
    return main([*args, "--fichero", str(fichero)],
                pedir=teclado or Teclado(CONTRASENA, CONTRASENA))


# --- Alta ---------------------------------------------------------------------
def test_alta_crea_la_cuenta(fichero: Path):
    assert _correr(fichero, "--alta", "ivan", "--nombre", "Iván Calvo") == 0
    cuenta = Usuarios(fichero).obtener("ivan")
    assert cuenta.nombre == "Iván Calvo"
    assert verificar(cuenta, CONTRASENA)


def test_la_contrasena_se_pide_dos_veces(fichero: Path):
    teclado = Teclado(CONTRASENA, CONTRASENA)
    _correr(fichero, "--alta", "ivan", teclado=teclado)
    assert len(teclado.preguntas) == 2


def test_si_no_coinciden_no_se_crea_nada(fichero: Path):
    teclado = Teclado(CONTRASENA, "otra-cosa-distinta")
    assert _correr(fichero, "--alta", "ivan", teclado=teclado) == 1
    assert not fichero.exists()


def test_una_contrasena_corta_se_rechaza(fichero: Path):
    corta = "a" * (MIN_LARGO - 1)
    assert _correr(fichero, "--alta", "ivan",
                   teclado=Teclado(corta, corta)) == 1
    assert not fichero.exists()


def test_no_se_puede_dar_de_alta_dos_veces(fichero: Path, capsys):
    _correr(fichero, "--alta", "ivan")
    assert _correr(fichero, "--alta", "IVAN") == 1
    assert "--contrasena" in capsys.readouterr().err


def test_el_nombre_visible_es_opcional(fichero: Path):
    _correr(fichero, "--alta", "ivan")
    assert Usuarios(fichero).obtener("ivan").nombre == "ivan"


def test_un_usuario_vacio_se_rechaza(fichero: Path):
    assert _correr(fichero, "--alta", "   ") == 1


def test_la_contrasena_no_se_puede_pasar_por_argumento(fichero: Path):
    """Lo que se escribe en la linea de comandos queda en el historial del shell
    y es visible en la lista de procesos de la maquina.

    `--contrasena` existe, pero es el NOMBRE DE LA CUENTA a la que cambiarsela:
    el valor lo pide `getpass`. Si algun dia alguien le anadiera un segundo
    argumento con la contrasena, esta prueba lo cazaria.
    """
    with pytest.raises(SystemExit):
        main(["--fichero", str(fichero), "--alta", "ivan", "--password", "secreta"],
             pedir=Teclado(CONTRASENA, CONTRASENA))

    teclado = Teclado(CONTRASENA, CONTRASENA)
    _correr(fichero, "--alta", "ivan", teclado=teclado)
    # La unica via por la que ha entrado la contrasena es el teclado sin eco.
    assert len(teclado.respuestas) == 0


def test_nada_de_lo_impreso_lleva_la_contrasena(fichero: Path, capsys):
    _correr(fichero, "--alta", "ivan", "--nombre", "Iván")
    salida = capsys.readouterr()
    assert CONTRASENA not in salida.out
    assert CONTRASENA not in salida.err


# --- Cambio de contrasena -----------------------------------------------------
def test_cambiar_la_contrasena(fichero: Path):
    _correr(fichero, "--alta", "ivan", "--nombre", "Iván")
    nueva = "otra-contrasena-larga"
    assert _correr(fichero, "--contrasena", "ivan",
                   teclado=Teclado(nueva, nueva)) == 0

    cuenta = Usuarios(fichero).obtener("ivan")
    assert verificar(cuenta, nueva)
    assert not verificar(cuenta, CONTRASENA)
    assert cuenta.nombre == "Iván"      # el nombre visible se conserva


def test_cambiar_la_de_una_cuenta_que_no_existe(fichero: Path, capsys):
    assert _correr(fichero, "--contrasena", "fantasma") == 1
    assert "No existe" in capsys.readouterr().err


def test_cambiar_deja_una_sal_nueva(fichero: Path):
    """Reusar la sal facilitaria correlacionar la contrasena vieja con la nueva."""
    _correr(fichero, "--alta", "ivan")
    antes = Usuarios(fichero).obtener("ivan").sal
    _correr(fichero, "--contrasena", "ivan",
            teclado=Teclado("otra-contrasena-larga", "otra-contrasena-larga"))
    assert Usuarios(fichero).obtener("ivan").sal != antes


# --- Baja y listado -----------------------------------------------------------
def test_baja_elimina_la_cuenta(fichero: Path):
    _correr(fichero, "--alta", "ivan")
    assert _correr(fichero, "--baja", "ivan") == 0
    assert Usuarios(fichero).obtener("ivan") is None


def test_baja_de_una_cuenta_que_no_existe(fichero: Path):
    assert _correr(fichero, "--baja", "fantasma") == 1


def test_la_baja_avisa_de_que_lo_suyo_sigue_ahi(fichero: Path, capsys):
    """Borrar una cuenta no destruye horas de entrenamiento."""
    _correr(fichero, "--alta", "ivan")
    _correr(fichero, "--baja", "ivan")
    assert "siguen en disco" in capsys.readouterr().out


def test_listar_las_cuentas(fichero: Path, capsys):
    _correr(fichero, "--alta", "ana", "--nombre", "Ana Ruiz")
    _correr(fichero, "--alta", "bruno", "--nombre", "Bruno Díaz")
    _correr(fichero, "--listar")
    salida = capsys.readouterr().out
    assert "ana" in salida and "Ana Ruiz" in salida
    assert "bruno" in salida and "Bruno Díaz" in salida


def test_listar_sin_cuentas_lo_dice(fichero: Path, capsys):
    _correr(fichero, "--listar")
    assert "No hay ninguna cuenta" in capsys.readouterr().out


def test_sin_argumentos_se_muestra_la_ayuda(fichero: Path, capsys):
    assert _correr(fichero) == 2
    assert "--alta" in capsys.readouterr().out
