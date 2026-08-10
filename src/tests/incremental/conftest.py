"""Fixtures del flujo incremental: paquetes de partidos sinteticos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.tests import factories


def escribir_paquete(
    raiz: Path,
    *,
    competition_id: int = 1,
    season_id: int = 100,
    competition_name: str = "Liga Sintetica",
    season_name: str = "2019/2020",
    match_id: int = 7001,
) -> dict[str, Any]:
    """Paquete valido de un solo partido, con el formato que exige `paquete.py`."""
    match, eventos, alineaciones = factories.partido_minimo(match_id)
    factories.escribir_open_data(
        raiz,
        competiciones=[factories.fila_competicion(
            competition_id, season_id,
            competition_name=competition_name, season_name=season_name)],
        partidos={(competition_id, season_id): [match]},
        eventos={match_id: eventos},
        alineaciones={match_id: alineaciones},
    )
    return {"raiz": raiz, "match": match, "eventos": eventos,
            "alineaciones": alineaciones, "match_id": match_id}


@pytest.fixture
def paquete_valido(tmp_path: Path) -> dict[str, Any]:
    return escribir_paquete(tmp_path / "paquete")


def reescribir(path: Path, contenido: Any) -> None:
    """Sobreescribe un JSON del paquete (para fabricar paquetes rotos)."""
    path.write_text(json.dumps(contenido, ensure_ascii=False), encoding="utf-8")
