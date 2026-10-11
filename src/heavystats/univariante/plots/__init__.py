"""
Subpaquete de gráficos univariantes para HeavyStats.
Organizado de forma modular en distribuciones cuantitativas, variables categóricas
y figuras compuestas/mosaicos de determinantes de exposición toxicológica.
"""

from heavystats.univariante.plots.base import BasePlots
from heavystats.univariante.plots.distribucion import (
    DistribucionPlots,
    obtener_mascara_percentiles,
    get_percentile_mask,
    mascara_percentiles,
)
from heavystats.univariante.plots.categorico import CategoricoPlots, desglosar_multirrespuesta
from heavystats.univariante.plots.exposicion import ExposicionPlots


class UnivariatePlots(BasePlots, DistribucionPlots, CategoricoPlots, ExposicionPlots):
    """
    Clase principal de visualizaciones univariantes con tipografía y jerarquía en negrita (#0f172a).
    Orquesta los pipelines para gráficos de distribución, evaluación de normalidad Q-Q,
    diagramas de barras categóricos y mosaicos de factores de exposición.
    """

    # Canvas y Mosaico
    canvas = BasePlots.canvas
    crear_lienzo = BasePlots.canvas
    create_canvas = BasePlots.canvas
    mosaic = BasePlots.crear_mosaico
    mosaico = BasePlots.crear_mosaico
    crear_mosaico = BasePlots.crear_mosaico

    # Pipelines y funciones simplificadas en español
    plot_distribucion = DistribucionPlots.plot_distribucion
    distribucion = DistribucionPlots.plot_distribucion
    obtener_mascara_percentiles = DistribucionPlots.obtener_mascara_percentiles
    get_percentile_mask = DistribucionPlots.obtener_mascara_percentiles
    mascara_percentiles = DistribucionPlots.obtener_mascara_percentiles

    plot_histograma = DistribucionPlots.plot_histograma
    histograma = DistribucionPlots.plot_histograma
    plot_histograma_estratificado = DistribucionPlots.plot_histograma_estratificado
    histograma_estratificado = DistribucionPlots.plot_histograma_estratificado
    plot_histograma_apilado = DistribucionPlots.plot_histograma_estratificado
    histograma_apilado = DistribucionPlots.plot_histograma_estratificado

    plot_boxplot = DistribucionPlots.plot_boxplot
    boxplot = DistribucionPlots.plot_boxplot

    plot_qq = DistribucionPlots.plot_qq
    qq = DistribucionPlots.plot_qq

    plot_categorico = CategoricoPlots.plot_categorico
    categorico = CategoricoPlots.plot_categorico

    plot_multirrespuesta = CategoricoPlots.plot_multirrespuesta
    multirrespuesta = CategoricoPlots.plot_multirrespuesta

    plot_factores = ExposicionPlots.plot_factores
    plot_factores_exposicion = ExposicionPlots.plot_factores
    factores = ExposicionPlots.plot_factores


__all__ = [
    "BasePlots",
    "DistribucionPlots",
    "CategoricoPlots",
    "ExposicionPlots",
    "UnivariatePlots",
    "desglosar_multirrespuesta",
    "obtener_mascara_percentiles",
    "get_percentile_mask",
    "mascara_percentiles",
]

