"""Resolucion de entidades por nombre para los CLI.

La regla de negocio importante: ante ambiguedad se falla en vez de elegir. En
scouting, devolver el jugador equivocado pasa desapercibido.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.similitud import consulta


class TestNormalizar:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("Messi", "messi"),
            ("MESSI", "messi"),
            ("  Messi  ", "messi"),
            ("Lionel Andrés Messi", "lionel andres messi"),
            ("Çağlar Söyüncü", "caglar soyuncu"),
            ("Müller", "muller"),
            ("Nicolò Barella", "nicolo barella"),
        ],
    )
    def test_quita_acentos_mayusculas_y_espacios(self, entrada: str, esperado: str) -> None:
        assert consulta.normalizar(entrada) == esperado

    def test_es_idempotente(self) -> None:
        una = consulta.normalizar("Lionel Andrés Messi")
        assert consulta.normalizar(una) == una


class TestResolver:
    NOMBRES = ["Lionel Andres Messi Cuccittini", "Memphis Depay", "Pedro Gonzalez Lopez"]

    def test_coincidencia_exacta(self) -> None:
        assert consulta.resolver("Memphis Depay", self.NOMBRES) == 1

    def test_coincidencia_exacta_sin_distinguir_mayusculas(self) -> None:
        assert consulta.resolver("memphis depay", self.NOMBRES) == 1

    def test_coincidencia_exacta_ignorando_acentos(self) -> None:
        assert consulta.resolver("Lionel Andrés Messi Cuccittini", self.NOMBRES) == 0

    def test_coincidencia_parcial_unica(self) -> None:
        """Escribir el nombre completo de StatsBomb a mano no es realista."""
        assert consulta.resolver("Messi", self.NOMBRES) == 0

    def test_la_coincidencia_exacta_gana_a_la_parcial(self) -> None:
        """Si "Messi" existe como entidad, "Messi" debe resolver a ella y no ser
        ambiguo con "Messi Cuccittini".
        """
        nombres = ["Messi Cuccittini", "Messi", "Otro Messi"]
        assert consulta.resolver("Messi", nombres) == 1

    def test_varias_coincidencias_parciales_son_ambiguas(self) -> None:
        nombres = ["Lionel Messi", "Messi Cuccittini"]
        with pytest.raises(consulta.EntidadAmbigua):
            consulta.resolver("Messi", nombres)

    def test_el_error_de_ambiguedad_lista_los_candidatos(self) -> None:
        nombres = ["Lionel Messi", "Messi Cuccittini"]
        with pytest.raises(consulta.EntidadAmbigua, match="Lionel Messi"):
            consulta.resolver("Messi", nombres)

    def test_la_lista_de_candidatos_se_trunca(self) -> None:
        nombres = [f"Jugador Comun {i}" for i in range(20)]
        with pytest.raises(consulta.EntidadAmbigua) as exc:
            consulta.resolver("Comun", nombres)
        assert "..." in str(exc.value)

    def test_sin_coincidencias(self) -> None:
        with pytest.raises(consulta.EntidadNoEncontrada, match="Nadie"):
            consulta.resolver("Nadie", self.NOMBRES)

    def test_universo_vacio(self) -> None:
        with pytest.raises(consulta.EntidadNoEncontrada):
            consulta.resolver("Messi", [])

    def test_los_errores_son_de_dominio_no_systemexit(self) -> None:
        """Cada script decide si abortar (`probar`) o seguir con la siguiente
        referencia (`comparar`): por eso son LookupError y no SystemExit.
        """
        assert issubclass(consulta.EntidadNoEncontrada, consulta.ErrorResolucion)
        assert issubclass(consulta.EntidadAmbigua, consulta.ErrorResolucion)
        assert issubclass(consulta.ErrorResolucion, LookupError)


class TestConfigurarConsola:
    def test_fuerza_utf8_en_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La consola de Windows (cp1252) aborta con nombres como 'Söyüncü'."""
        llamadas = {}
        falso = SimpleNamespace(
            reconfigure=lambda **kw: llamadas.update(kw)
        )
        monkeypatch.setattr(sys, "stdout", falso)
        consulta.configurar_consola()
        assert llamadas == {"encoding": "utf-8", "errors": "replace"}

    def test_un_stdout_sin_reconfigure_no_rompe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdout", SimpleNamespace())
        consulta.configurar_consola()   # no debe lanzar
