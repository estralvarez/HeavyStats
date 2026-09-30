"""
Subpaquete de gráficos univariantes para HeavyStats.
Organizado de forma modular en distribuciones cuantitativas, variables categóricas
y figuras compuestas/mosaicos de determinantes de exposición toxicológica.
"""

from heavystats.univariate.plots.base import BasePlots
from heavystats.univariate.plots.distribucion import DistribucionPlotsMixin
from heavystats.univariate.plots.categorico import CategoricoPlotsMixin, desglosar_multirrespuesta
from heavystats.univariate.plots.exposicion import ExposicionPlotsMixin


class UnivariatePlots(BasePlots, DistribucionPlotsMixin, CategoricoPlotsMixin, ExposicionPlotsMixin):
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
    plot_distribucion = DistribucionPlotsMixin.plot_distribucion
    distribucion = DistribucionPlotsMixin.plot_distribucion

    plot_histograma = DistribucionPlotsMixin.plot_histograma
    histograma = DistribucionPlotsMixin.plot_histograma
    plot_histograma_estratificado = DistribucionPlotsMixin.plot_histograma_estratificado
    histograma_estratificado = DistribucionPlotsMixin.plot_histograma_estratificado
    plot_histograma_apilado = DistribucionPlotsMixin.plot_histograma_estratificado
    histograma_apilado = DistribucionPlotsMixin.plot_histograma_estratificado

    plot_boxplot = DistribucionPlotsMixin.plot_boxplot
    boxplot = DistribucionPlotsMixin.plot_boxplot

    plot_qq = DistribucionPlotsMixin.plot_qq
    qq = DistribucionPlotsMixin.plot_qq

    plot_categorico = CategoricoPlotsMixin.plot_categorico
    categorico = CategoricoPlotsMixin.plot_categorico

    plot_multirrespuesta = CategoricoPlotsMixin.plot_multirrespuesta
    multirrespuesta = CategoricoPlotsMixin.plot_multirrespuesta

    plot_factores = ExposicionPlotsMixin.plot_factores
    plot_factores_exposicion = ExposicionPlotsMixin.plot_factores
    factores = ExposicionPlotsMixin.plot_factores


__all__ = [
    "BasePlots",
    "UnivariatePlots",
    "desglosar_multirrespuesta",
]

