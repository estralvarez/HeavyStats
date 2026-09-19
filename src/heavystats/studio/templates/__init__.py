"""
Plantillas estructuradas de cuadernos Jupyter para HeavyStats Studio.
"""

from heavystats.studio.templates.validacion import generar_cuaderno_validacion
from heavystats.studio.templates.filtros import generar_cuaderno_filtros
from heavystats.studio.templates.univariante import (
    generar_cuaderno_univariante_tablas,
    generar_cuaderno_univariante_graficos,
)
from heavystats.studio.templates.bivariante import (
    generar_cuaderno_bivariante_tablas,
    generar_cuaderno_bivariante_graficos,
)

__all__ = [
    "generar_cuaderno_validacion",
    "generar_cuaderno_filtros",
    "generar_cuaderno_univariante_tablas",
    "generar_cuaderno_univariante_graficos",
    "generar_cuaderno_bivariante_tablas",
    "generar_cuaderno_bivariante_graficos",
]
