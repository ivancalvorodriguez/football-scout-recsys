"""Lanzador de pruebas de mutacion (mutmut) desde Windows a traves de WSL.

Por que WSL
-----------
`mutmut` "prueba a las pruebas": muta el codigo fuente (p. ej. `>`->`>=`,
`+`->`-`, `and`->`or`) y comprueba que ALGUNA prueba falle. Un mutante que
SOBREVIVE (ninguna prueba lo detecta) señala una linea que la suite ejecuta pero
no verifica de verdad.

mutmut NO corre en Windows nativo (usa `os.fork` y el modulo `resource`; su 3.x
bloquea Windows explicitamente). Por eso vive en un venv de Linux bajo WSL. Este
modulo es un puente: se ejecuta con el Python de Windows pero delega en `mutmut`
dentro de WSL, sobre el mismo arbol de codigo (`/mnt/c/...`). El desarrollo y la
suite normal (`pytest`) siguen en Windows; solo la mutacion pasa por WSL.

La configuracion de mutmut esta en `setup.cfg` (`[mutmut]`); el venv y sus
parches los prepara `tools/setup_mutmut_wsl.sh`, que este lanzador invoca solo
la primera vez. Diseño y detalles en `docs/pruebas.md`.

Uso (desde la raiz del repo, en Windows)
----------------------------------------
    python -m src.tests.mutacion                     # muta TODO src (mutmut run)
    python -m src.tests.mutacion --modulo similitud.slim   # solo ese modulo
    python -m src.tests.mutacion results             # resumen (supervivientes)
    python -m src.tests.mutacion show src.similitud.slim.x_ease__mutmut_3
    python -m src.tests.mutacion browse              # TUI interactiva

Cualquier subcomando de mutmut (run/results/show/browse/apply/tests-for-mutant/
print-time-estimates/export-cicd-stats) se reenvia tal cual.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[2]
VENV_MUTMUT = "$HOME/.venvs/tfg-mut"

# Subcomandos que mutmut entiende: si el primer argumento es uno de ellos, se
# reenvia todo sin tocar.
SUBCOMANDOS = {
    "run", "results", "show", "browse", "apply",
    "tests-for-mutant", "print-time-estimates", "export-cicd-stats",
}


def _construir_comando_mutmut(argv: list[str]) -> list[str]:
    """Traduce los argumentos del lanzador a un comando de mutmut."""
    if argv and argv[0] in SUBCOMANDOS:
        return argv  # paso directo

    # Modo por defecto: `run`, con filtro opcional por modulo.
    modulo = None
    resto: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--modulo":
            if i + 1 >= len(argv):
                raise SystemExit("--modulo necesita un valor, p. ej. 'similitud.slim'.")
            modulo = argv[i + 1]
            i += 2
        else:
            resto.append(argv[i])
            i += 1

    cmd = ["run"]
    if modulo is not None:
        # mutmut filtra los mutantes A EJECUTAR por nombre; nuestros modulos se
        # importan bajo el paquete `src`, de ahi el prefijo.
        cmd.append(f"src.{modulo}.*")
    return cmd + resto


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    mutmut_cmd = _construir_comando_mutmut(argv)
    mutmut_cmd_q = " ".join(shlex.quote(a) for a in mutmut_cmd)

    # Ruta Windows del repo -> ruta WSL (via wslpath, en el propio shell remoto).
    win_repo = str(RAIZ_REPO)
    # Script bash: sitúa el cwd en el repo, asegura el venv y lanza mutmut.
    remoto = (
        "set -e; "
        f'cd "$(wslpath -a {shlex.quote(win_repo)})"; '
        f'if [ ! -x "{VENV_MUTMUT}/bin/mutmut" ]; then '
        "echo '>> Preparando entorno de mutacion (primera vez)...'; "
        "bash tools/setup_mutmut_wsl.sh; fi; "
        f'"{VENV_MUTMUT}/bin/mutmut" {mutmut_cmd_q}'
    )

    print(f"[mutacion] WSL: mutmut {mutmut_cmd_q}\n", flush=True)
    try:
        completado = subprocess.run(["wsl.exe", "-e", "bash", "-lc", remoto])
    except FileNotFoundError:
        raise SystemExit(
            "No se encontro wsl.exe. Las pruebas de mutacion requieren WSL "
            "(ver docs/pruebas.md)."
        )
    return completado.returncode


if __name__ == "__main__":
    raise SystemExit(main())
