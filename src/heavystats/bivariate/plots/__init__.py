"""
Subpaquete de gráficos bivariantes para HeavyStats.
Estructurado de forma modular para análisis bivariante epidemiológico y toxicológico:
  1. Comparación de Carga Corporal (Cuantitativa vs Cualitativa, Mann-Whitney U, Kruskal-Wallis H).
  2. Gradientes Continuos y Co-Exposición (Correlación Spearman/Bootstrap, Pearson/OLS, matrices).
  3. Patrones Dietarios y Mosaicos Editoriales (Radar polar, boxplots dietarios integrados, mosaicos grid).
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import pandas as pd
import matplotlib.pyplot as plt

from heavystats.bivariate.plots.base import (
    BivariateBasePlots,
    DEFAULT_BIOMEDICAL_METAL_LABELS,
    DEFAULT_REFERENCE_LABELS,
)
from heavystats.bivariate.plots.comparacion import ComparacionPlotsMixin
from heavystats.bivariate.plots.correlacion import CorrelacionPlotsMixin
from heavystats.bivariate.plots.dietario import DietarioPlotsMixin
from heavystats.bivariate.plots.mosaico import MosaicoPlotsMixin


class BivariatePlots(
    BivariateBasePlots,
    ComparacionPlotsMixin,
    CorrelacionPlotsMixin,
    DietarioPlotsMixin,
    MosaicoPlotsMixin,
):
    """
    Clase principal para construir gráficos bivariantes con estética editorial científica.
    Anotaciones estadísticas directas en corchetes (brackets), tamaños de muestra (n=...),
    límites toxicológicos contextualizados y matrices de co-exposición bivariantes.
    """
    pass


# =============================================================================
# Funciones Modulares de Conveniencia a Nivel de Módulo
# =============================================================================

def compare_groups_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).compare_groups()."""
    return BivariatePlots(df).compare_groups(*args, **kwargs)


def correlation_analysis_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).correlation_analysis()."""
    return BivariatePlots(df).correlation_analysis(*args, **kwargs)


def coexposure_matrix_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, Any]:
    """Función de conveniencia para BivariatePlots(df).coexposure_matrix()."""
    return BivariatePlots(df).coexposure_matrix(*args, **kwargs)


def diet_radar_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Función de conveniencia para BivariatePlots(df).plot_diet_radar()."""
    return BivariatePlots(df).plot_diet_radar(*args, **kwargs)


def diet_boxplots_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, Any]:
    """Función de conveniencia para BivariatePlots(df).plot_diet_boxplots()."""
    return BivariatePlots(df).plot_diet_boxplots(*args, **kwargs)


__all__ = [
    "BivariateBasePlots",
    "BivariatePlots",
    "compare_groups_plot",
    "correlation_analysis_plot",
    "coexposure_matrix_plot",
    "diet_radar_plot",
    "diet_boxplots_plot",
    "DEFAULT_BIOMEDICAL_METAL_LABELS",
    "DEFAULT_REFERENCE_LABELS",
]
