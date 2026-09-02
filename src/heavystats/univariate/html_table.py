"""
Módulo de compatibilidad para renderizado HTML de tablas univariantes.
Centraliza la lógica en heavystats.html_utils para cumplir con el principio DRY.
"""

from heavystats.html_utils import (
    get_publication_css,
    format_html_str,
    wrap_html_container,
    render_html_table,
    BaseReport,
)

# Alias para compatibilidad interna
_format_html_str = format_html_str

__all__ = [
    "get_publication_css",
    "format_html_str",
    "_format_html_str",
    "wrap_html_container",
    "render_html_table",
    "BaseReport",
]
