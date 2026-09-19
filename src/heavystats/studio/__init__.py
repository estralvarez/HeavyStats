"""
HeavyStats Studio: Módulo de generación interactiva y visual (TUI/CLI) de proyectos bioestadísticos.
"""

from heavystats.studio.app import HeavyStatsApp, iniciar_tui
from heavystats.studio.config import METALES_CONFIG, DEFAULT_METAL, resolver_metal
from heavystats.studio.pipeline import ejecutar_pipeline
from heavystats.studio.filesystem import verificar_proyecto_existente, auditar_dataset

__all__ = [
    "HeavyStatsApp",
    "iniciar_tui",
    "METALES_CONFIG",
    "DEFAULT_METAL",
    "resolver_metal",
    "ejecutar_pipeline",
    "verificar_proyecto_existente",
    "auditar_dataset",
]
