#!/usr/bin/env bash
# Prepara el entorno de pruebas de mutacion (mutmut) dentro de WSL/Linux.
#
# mutmut no funciona en Windows nativo (usa os.fork y el modulo `resource`, y su
# version 3.x bloquea Windows explicitamente). Por eso las mutaciones se corren
# en un venv de Linux bajo WSL. Este script:
#   1. Crea el venv en ~/.venvs/tfg-mut (si no existe).
#   2. Instala las dependencias de requirements-mutmut.txt.
#   3. Parchea dos incompatibilidades de mutmut 3.6.0 con el layout de este
#      proyecto, donde `src` es un PAQUETE real (los modulos se importan como
#      `src.similitud.slim`, no `similitud.slim`):
#        a) format_utils.get_mutant_name elimina el prefijo "src." al derivar el
#           nombre de modulo desde la ruta -> la clave esperada no casa con la
#           que registran las pruebas (`src....`). Se desactiva ese recorte.
#        b) record_trampoline_hit asegura que el nombre NO empiece por "src.".
#           Se relaja ese assert.
#      Los dos parches son idempotentes (solo tocan la linea original).
#
# Uso (desde WSL, en la raiz del repo):   bash tools/setup_mutmut_wsl.sh
# Normalmente se invoca solo, a traves de `python -m src.tests.mutacion`.
set -euo pipefail

VENV="${TFG_MUT_VENV:-$HOME/.venvs/tfg-mut}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$VENV/bin/mutmut" ]]; then
    echo ">> Creando venv de mutacion en $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$REPO_DIR/requirements-mutmut.txt"
fi

SP="$("$VENV/bin/python" -c 'import mutmut, os; print(os.path.dirname(mutmut.__file__))')"

# Parche (a): no recortar el prefijo "src." del nombre de modulo.
FMT="$SP/utils/format_utils.py"
if grep -q '^    module_name = strip_prefix(module_name, prefix="src.")' "$FMT"; then
    echo ">> Parcheando $FMT (recorte de 'src.')"
    sed -i 's/^    module_name = strip_prefix(module_name, prefix="src.")/    # (parche tfg) \`src\` es paquete real: no se recorta el prefijo/' "$FMT"
fi

# Parche (b): relajar el assert que rechaza nombres que empiezan por "src.".
MAIN="$SP/__main__.py"
if grep -q '^    assert not name.startswith("src.")' "$MAIN"; then
    echo ">> Parcheando $MAIN (assert 'src.')"
    sed -i 's/^    assert not name.startswith("src.").*/    pass  # (parche tfg) \`src\` es paquete real del proyecto/' "$MAIN"
fi

echo ">> Entorno de mutacion listo: $VENV"
