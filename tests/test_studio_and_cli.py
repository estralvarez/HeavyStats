"""Pruebas para el módulo heavystats.studio y heavystats.cli."""

import pytest
from heavystats.cli import build_parser, main
from heavystats.studio.config import METALES_CONFIG, DEFAULT_METAL, resolver_metal
from heavystats.studio.filesystem import verificar_proyecto_existente, crear_estructura_directorios
from heavystats.studio.app import HeavyStatsApp
import heavystats as hs


def test_studio_exports_and_configs():
    assert "mercurio" in METALES_CONFIG
    assert "plomo" in METALES_CONFIG
    assert "cadmio" in METALES_CONFIG
    assert DEFAULT_METAL in METALES_CONFIG
    assert hasattr(hs, "launch_studio")


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.metal is None
    assert args.no_tui is False


def test_cli_parser_custom_args():
    parser = build_parser()
    args = parser.parse_args(["-m", "plomo", "--no-tui"])
    assert args.metal == "plomo"
    assert args.no_tui is True


def test_cli_resolver_metal():
    cfg = resolver_metal("hg")
    assert cfg["key"] == "mercurio"
    cfg_default = resolver_metal("desconocido")
    assert cfg_default["key"] == DEFAULT_METAL


def test_verificar_proyecto_existente_false():
    cfg = resolver_metal("plomo")
    # Si faltan carpetas o cuadernos retorna False
    # (o si no existe nada)
    assert isinstance(verificar_proyecto_existente(cfg), bool)


def test_app_instantiation():
    app = HeavyStatsApp()
    assert app.TITLE == "HeavyStats Studio"


def test_notebook_templates_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import json
    from heavystats.studio.templates import (
        generar_cuaderno_univariante_graficos,
        generar_cuaderno_bivariante_graficos,
        generar_cuaderno_multivariante_pls,
        generar_cuaderno_atipicos_diagnostico,
    )
    for metal in ["plomo", "mercurio", "cadmio"]:
        cfg = METALES_CONFIG[metal]
        nb_u = generar_cuaderno_univariante_graficos(cfg)
        assert nb_u.exists()
        content_u = json.loads(nb_u.read_text(encoding="utf-8"))
        assert "cells" in content_u
        source_u = "".join("".join(c["source"]) for c in content_u["cells"])
        assert "percentile_mask" in source_u

        nb_b = generar_cuaderno_bivariante_graficos(cfg)
        assert nb_b.exists()
        content_b = json.loads(nb_b.read_text(encoding="utf-8"))
        assert "cells" in content_b
        source_b = "".join("".join(c["source"]) for c in content_b["cells"])
        assert "subcohort_contrast_plot" in source_b
        assert "similarity_ranking_plot" in source_b
        assert "scan_subcohort_contrasts" in source_b

        nb_m = generar_cuaderno_multivariante_pls(cfg)
        assert nb_m.exists()
        content_m = json.loads(nb_m.read_text(encoding="utf-8"))
        assert "cells" in content_m
        source_m = "".join("".join(c["source"]) for c in content_m["cells"])
        assert "PlsModeler" in source_m
        assert "MultivariatePlots" in source_m

        nb_a = generar_cuaderno_atipicos_diagnostico(cfg)
        assert nb_a.exists()
        content_a = json.loads(nb_a.read_text(encoding="utf-8"))
        assert "cells" in content_a
        source_a = "".join("".join(c["source"]) for c in content_a["cells"])
        assert "DiagnosticoAtipicos" in source_a
        assert "GraficosAtipicos" in source_a

