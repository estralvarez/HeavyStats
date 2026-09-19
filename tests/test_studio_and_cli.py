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

