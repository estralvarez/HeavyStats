"""
Subpaquete de gráficos bivariantes para HeavyStats.
Estructurado de forma modular para análisis bivariante epidemiológico y toxicológico:
  1. Comparación de Carga Corporal (Cuantitativa vs Cualitativa, Mann-Whitney U, Kruskal-Wallis H).
  2. Gradientes Continuos y Co-Exposición (Correlación Spearman/Bootstrap, Pearson/OLS, matrices).
  3. Patrones Dietarios y Mosaicos Editoriales (Radar polar, boxplots dietarios integrados, mosaicos grid).
  4. Análisis de Subcohortes y Factores Compartidos (Rankings correlacionales, contrastes con IDs y consensos).
  5. Radiografía Epidemiológica de la Encuesta (Mapas de calor temáticos y síntesis global normalizada).
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
from heavystats.bivariate.plots.subcohorte import SubcohortPlotsMixin
from heavystats.bivariate.plots.radiografia import RadiografiaPlotsMixin


class BivariatePlots(
    BivariateBasePlots,
    ComparacionPlotsMixin,
    CorrelacionPlotsMixin,
    DietarioPlotsMixin,
    MosaicoPlotsMixin,
    SubcohortPlotsMixin,
    RadiografiaPlotsMixin,
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
    compare_groups = ComparacionPlotsMixin.compare_groups
    plot_compare = ComparacionPlotsMixin.compare_groups
    plot_compare_groups = ComparacionPlotsMixin.compare_groups
    plot_binary = ComparacionPlotsMixin.plot_binary
    plot_categorical = ComparacionPlotsMixin.plot_categorical
    plot_categorico = ComparacionPlotsMixin.plot_categorical
    plot_dummies = ComparacionPlotsMixin.plot_dummies
    comparar_grupos = ComparacionPlotsMixin.compare_groups
    plot_comparacion = ComparacionPlotsMixin.compare_groups
    comparacion = ComparacionPlotsMixin.compare_groups

    # 2. Correlación y Co-Exposición
    correlation_analysis = CorrelacionPlotsMixin.correlation_analysis
    plot_correlation = CorrelacionPlotsMixin.correlation_analysis
    plot_correlation_analysis = CorrelacionPlotsMixin.correlation_analysis
    plot_correlacion = CorrelacionPlotsMixin.correlation_analysis
    correlacion = CorrelacionPlotsMixin.correlation_analysis
    coexposure_matrix = CorrelacionPlotsMixin.coexposure_matrix
    plot_coexposure_matrix = CorrelacionPlotsMixin.coexposure_matrix
    matriz_coexposicion = CorrelacionPlotsMixin.coexposure_matrix
    plot_matriz_coexposicion = CorrelacionPlotsMixin.coexposure_matrix

    # 3. Hábitos Dietarios
    ordinal_trend_plot = DietarioPlotsMixin.ordinal_trend_plot
    plot_dietary = DietarioPlotsMixin.ordinal_trend_plot
    plot_ordinal_trend = DietarioPlotsMixin.ordinal_trend_plot
    tendencia_ordinal = DietarioPlotsMixin.ordinal_trend_plot
    plot_tendencia_ordinal = DietarioPlotsMixin.ordinal_trend_plot
    plot_diet_boxplots = DietarioPlotsMixin.plot_diet_boxplots
    plot_diet_grid = DietarioPlotsMixin.plot_diet_boxplots
    mosaico_dietario = DietarioPlotsMixin.plot_diet_boxplots
    plot_mosaico_dietario = DietarioPlotsMixin.plot_diet_boxplots
    plot_diet_radar = DietarioPlotsMixin.plot_diet_radar
    plot_radar = DietarioPlotsMixin.plot_diet_radar
    radar_dietario = DietarioPlotsMixin.plot_diet_radar
    plot_radar_dietario = DietarioPlotsMixin.plot_diet_radar
    risk_algorithm_plots = DietarioPlotsMixin.risk_algorithm_plots
    plot_risk_algorithm = DietarioPlotsMixin.risk_algorithm_plots
    algoritmo_riesgo = DietarioPlotsMixin.risk_algorithm_plots
    plot_algoritmo_riesgo = DietarioPlotsMixin.risk_algorithm_plots

    # 4. Mosaico Bivariante Integrado
    plot_grid = MosaicoPlotsMixin.plot_grid
    grid = MosaicoPlotsMixin.plot_grid
    mosaico_bivariante = MosaicoPlotsMixin.plot_grid
    plot_mosaico = MosaicoPlotsMixin.plot_grid

    # 5. Subcohortes y Contrastes
    similarity_ranking_plot = SubcohortPlotsMixin.similarity_ranking_plot
    plot_similarity_ranking = SubcohortPlotsMixin.similarity_ranking_plot
    ranking_similitud = SubcohortPlotsMixin.similarity_ranking_plot
    plot_ranking_similitud = SubcohortPlotsMixin.similarity_ranking_plot
    subcohort_contrast_plot = SubcohortPlotsMixin.subcohort_contrast_plot
    plot_subcohort_contrast = SubcohortPlotsMixin.subcohort_contrast_plot
    contraste_subcohorte = SubcohortPlotsMixin.subcohort_contrast_plot
    plot_contraste_subcohorte = SubcohortPlotsMixin.subcohort_contrast_plot
    shared_factors_plot = SubcohortPlotsMixin.shared_factors_plot
    plot_shared_factors = SubcohortPlotsMixin.shared_factors_plot
    factores_compartidos = SubcohortPlotsMixin.shared_factors_plot
    plot_factores_compartidos = SubcohortPlotsMixin.shared_factors_plot
    subcohort_mosaic_plot = SubcohortPlotsMixin.subcohort_mosaic_plot
    plot_subcohort_mosaic = SubcohortPlotsMixin.subcohort_mosaic_plot
    mosaico_subcohorte = SubcohortPlotsMixin.subcohort_mosaic_plot
    plot_mosaico_subcohorte = SubcohortPlotsMixin.subcohort_mosaic_plot
    scan_subcohort_contrasts = SubcohortPlotsMixin.scan_subcohort_contrasts
    escanear_contrastes = SubcohortPlotsMixin.scan_subcohort_contrasts
    escanear_subcohortes = SubcohortPlotsMixin.scan_subcohort_contrasts

    # 6. Radiografía Epidemiológica
    survey_block_heatmap = RadiografiaPlotsMixin.survey_block_heatmap
    mapa_calor_bloque = RadiografiaPlotsMixin.survey_block_heatmap
    plot_survey_block_heatmap = RadiografiaPlotsMixin.survey_block_heatmap
    plot_survey_radiography = RadiografiaPlotsMixin.plot_survey_radiography
    radiografia_encuesta = RadiografiaPlotsMixin.plot_survey_radiography
    plot_radiografia_encuesta = RadiografiaPlotsMixin.plot_survey_radiography
    plot_survey_synthesis = RadiografiaPlotsMixin.plot_survey_synthesis
    sintesis_encuesta = RadiografiaPlotsMixin.plot_survey_synthesis
    plot_sintesis_encuesta = RadiografiaPlotsMixin.plot_survey_synthesis
    plot_survey_radiography_suite = RadiografiaPlotsMixin.plot_survey_radiography
    suite_radiografia = RadiografiaPlotsMixin.plot_survey_radiography
    plot_suite_radiografia = RadiografiaPlotsMixin.plot_survey_radiography
    plot_epidemiological_radiography = RadiografiaPlotsMixin.plot_survey_radiography
    plot_radiography_heatmap = RadiografiaPlotsMixin.survey_block_heatmap


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


def survey_block_heatmap_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).survey_block_heatmap()."""
    return BivariatePlots(df).survey_block_heatmap(*args, **kwargs)


def survey_radiography_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """Función de conveniencia para BivariatePlots(df).plot_survey_radiography()."""
    return BivariatePlots(df).plot_survey_radiography(*args, **kwargs)


def survey_synthesis_plot(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).plot_survey_synthesis()."""
    return BivariatePlots(df).plot_survey_synthesis(*args, **kwargs)


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
radiografia_encuesta_plot = survey_radiography_plot
sintesis_encuesta_plot = survey_synthesis_plot


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
    "survey_block_heatmap_plot",
    "survey_radiography_plot",
    "survey_synthesis_plot",
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
    "radiografia_encuesta_plot",
    "sintesis_encuesta_plot",
    "DEFAULT_BIOMEDICAL_METAL_LABELS",
    "DEFAULT_REFERENCE_LABELS",
]
