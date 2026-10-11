"""
Subpaquete de gráficos bivariantes para HeavyStats.
Estructurado de forma modular para análisis bivariante epidemiológico y toxicológico:
  1. Comparación de Carga Corporal (Cuantitativa vs Cualitativa, Mann-Whitney U, Kruskal-Wallis H).
  2. Gradientes Continuos y Co-Exposición (Correlación Spearman/Bootstrap, Pearson/OLS, matrices).
  3. Patrones Dietarios y Mosaicos Editoriales (Radar polar, boxplots dietarios integrados).
  4. Análisis de Subcohortes y Factores Compartidos (Rankings correlacionales, contrastes con IDs y consensos).
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import pandas as pd
import matplotlib.pyplot as plt

from heavystats.bivariante.plots.base import (
    BivariateBasePlots,
    DEFAULT_BIOMEDICAL_METAL_LABELS,
    DEFAULT_REFERENCE_LABELS,
)
from heavystats.bivariante.plots.comparacion import ComparacionPlots
from heavystats.bivariante.plots.correlacion import CorrelacionPlots
from heavystats.bivariante.plots.dietario import DietarioPlots
from heavystats.bivariante.plots.subcohorte import SubcohortPlots


class BivariatePlots(
    BivariateBasePlots,
    ComparacionPlots,
    CorrelacionPlots,
    DietarioPlots,
    SubcohortPlots,
):
    """
    Clase principal para construir gráficos bivariantes con estética editorial científica.
    Anotaciones estadísticas directas en corchetes (brackets), tamaños de muestra (n=...),
    límites toxicológicos contextualizados, matrices de co-exposición, análisis de subcohortes
    y radiografía epidemiológica de cuestionarios en mapas de calor normalizados.
    Hereda del motor BasePlots para soportar canvas, mosaicos y estilizado editorial unificado.
    """

    # Canvas y Mosaico (compatibilidad total con BasePlots y UnivariatePlots)
    canvas = BivariateBasePlots.canvas
    crear_lienzo = BivariateBasePlots.canvas
    create_canvas = BivariateBasePlots.canvas
    mosaic = BivariateBasePlots.crear_mosaico
    mosaico = BivariateBasePlots.crear_mosaico
    crear_mosaico = BivariateBasePlots.crear_mosaico
    close_all = BivariateBasePlots.close_all

    # 1. Comparación de Carga Corporal
    compare_groups = ComparacionPlots.compare_groups
    plot_compare = ComparacionPlots.compare_groups
    plot_compare_groups = ComparacionPlots.compare_groups
    plot_binary = ComparacionPlots.plot_binary
    plot_categorical = ComparacionPlots.plot_categorical
    plot_categorico = ComparacionPlots.plot_categorical
    plot_dummies = ComparacionPlots.plot_dummies
    comparar_grupos = ComparacionPlots.compare_groups
    plot_comparacion = ComparacionPlots.compare_groups
    comparacion = ComparacionPlots.compare_groups

    # 2. Correlación y Co-Exposición
    correlation_analysis = CorrelacionPlots.correlation_analysis
    plot_correlation = CorrelacionPlots.correlation_analysis
    plot_correlation_analysis = CorrelacionPlots.correlation_analysis
    plot_correlacion = CorrelacionPlots.correlation_analysis
    correlacion = CorrelacionPlots.correlation_analysis
    coexposure_matrix = CorrelacionPlots.coexposure_matrix
    plot_coexposure_matrix = CorrelacionPlots.coexposure_matrix
    matriz_coexposicion = CorrelacionPlots.coexposure_matrix
    plot_matriz_coexposicion = CorrelacionPlots.coexposure_matrix

    # 3. Hábitos Dietarios
    ordinal_trend_plot = DietarioPlots.ordinal_trend_plot
    plot_dietary = DietarioPlots.ordinal_trend_plot
    plot_ordinal_trend = DietarioPlots.ordinal_trend_plot
    tendencia_ordinal = DietarioPlots.ordinal_trend_plot
    plot_tendencia_ordinal = DietarioPlots.ordinal_trend_plot
    plot_diet_boxplots = DietarioPlots.plot_diet_boxplots
    plot_diet_grid = DietarioPlots.plot_diet_boxplots
    mosaico_dietario = DietarioPlots.plot_diet_boxplots
    plot_mosaico_dietario = DietarioPlots.plot_diet_boxplots
    plot_diet_radar = DietarioPlots.plot_diet_radar
    plot_radar = DietarioPlots.plot_diet_radar
    radar_dietario = DietarioPlots.plot_diet_radar
    plot_radar_dietario = DietarioPlots.plot_diet_radar

    # 4. Subcohortes y Contrastes
    similarity_ranking_plot = SubcohortPlots.similarity_ranking_plot
    plot_similarity_ranking = SubcohortPlots.similarity_ranking_plot
    ranking_similitud = SubcohortPlots.similarity_ranking_plot
    plot_ranking_similitud = SubcohortPlots.similarity_ranking_plot
    subcohort_contrast_plot = SubcohortPlots.subcohort_contrast_plot
    plot_subcohort_contrast = SubcohortPlots.subcohort_contrast_plot
    contraste_subcohorte = SubcohortPlots.subcohort_contrast_plot
    plot_contraste_subcohorte = SubcohortPlots.subcohort_contrast_plot
    shared_factors_plot = SubcohortPlots.shared_factors_plot
    plot_shared_factors = SubcohortPlots.shared_factors_plot
    factores_compartidos = SubcohortPlots.shared_factors_plot
    plot_factores_compartidos = SubcohortPlots.shared_factors_plot
    subcohort_mosaic_plot = SubcohortPlots.subcohort_mosaic_plot
    plot_subcohort_mosaic = SubcohortPlots.subcohort_mosaic_plot
    mosaico_subcohorte = SubcohortPlots.subcohort_mosaic_plot
    plot_mosaico_subcohorte = SubcohortPlots.subcohort_mosaic_plot
    scan_subcohort_contrasts = SubcohortPlots.scan_subcohort_contrasts
    escanear_contrastes = SubcohortPlots.scan_subcohort_contrasts
    escanear_subcohortes = SubcohortPlots.scan_subcohort_contrasts


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


def similarity_ranking_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Función de conveniencia para BivariatePlots(df).similarity_ranking_plot()."""
    return BivariatePlots(df).similarity_ranking_plot(*args, **kwargs)


def subcohort_contrast_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """Función de conveniencia para BivariatePlots(df).subcohort_contrast_plot()."""
    return BivariatePlots(df).subcohort_contrast_plot(*args, **kwargs)


def shared_factors_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Función de conveniencia para BivariatePlots(df).shared_factors_plot()."""
    return BivariatePlots(df).shared_factors_plot(*args, **kwargs)


def subcohort_mosaic_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, Any]]:
    """Función de conveniencia para BivariatePlots(df).subcohort_mosaic_plot()."""
    return BivariatePlots(df).subcohort_mosaic_plot(*args, **kwargs)


def scan_subcohort_contrasts(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Función de conveniencia para BivariatePlots(df).scan_subcohort_contrasts()."""
    return BivariatePlots(df).scan_subcohort_contrasts(*args, **kwargs)


# Aliases en español a nivel de módulo
comparar_grupos_plot = compare_groups_plot
correlacion_analisis_plot = correlation_analysis_plot
matriz_coexposicion_plot = coexposure_matrix_plot
radar_dietario_plot = diet_radar_plot
mosaico_dietario_plot = diet_boxplots_plot
ranking_similitud_plot = similarity_ranking_plot
contraste_subcohorte_plot = subcohort_contrast_plot
factores_compartidos_plot = shared_factors_plot
mosaico_subcohorte_plot = subcohort_mosaic_plot


__all__ = [
    "BivariateBasePlots",
    "BivariatePlots",
    "compare_groups_plot",
    "correlation_analysis_plot",
    "coexposure_matrix_plot",
    "diet_radar_plot",
    "diet_boxplots_plot",
    "similarity_ranking_plot",
    "subcohort_contrast_plot",
    "shared_factors_plot",
    "subcohort_mosaic_plot",
    "scan_subcohort_contrasts",
    "comparar_grupos_plot",
    "correlacion_analisis_plot",
    "matriz_coexposicion_plot",
    "radar_dietario_plot",
    "mosaico_dietario_plot",
    "ranking_similitud_plot",
    "contraste_subcohorte_plot",
    "factores_compartidos_plot",
    "mosaico_subcohorte_plot",
    "DEFAULT_BIOMEDICAL_METAL_LABELS",
    "DEFAULT_REFERENCE_LABELS",
]
