"""
Módulo de Análisis y Visualización Bivariante de HeavyStats (`heavystats.bivariante`).
Implementa pruebas de hipótesis (Mann-Whitney, Kruskal-Wallis, t-Student, ANOVA),
correlaciones no paramétricas (Spearman, Kendall), motor gráfico editorial a 300 DPI y compilación en PDF.
"""

from functools import partial
from heavystats.bivariante.tests import (
    mann_whitney_test,
    independent_t_test,
    kruskal_wallis_test,
    anova_oneway_test,
    dunn_posthoc_test,
    jonckheere_terpstra_test,
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    spearman_matrix,
    adjust_pvalues,
    hodges_lehmann_2sample,
    bootstrap_ci_diff_medians,
)
from heavystats.bivariante.plots import (
    BivariatePlots,
    compare_groups_plot,
    correlation_analysis_plot,
    coexposure_matrix_plot,
    diet_radar_plot,
    diet_boxplots_plot,
    similarity_ranking_plot,
    subcohort_contrast_plot,
    shared_factors_plot,
    subcohort_mosaic_plot,
    scan_subcohort_contrasts,
    comparar_grupos_plot,
    correlacion_analisis_plot,
    matriz_coexposicion_plot,
    radar_dietario_plot,
    mosaico_dietario_plot,
    ranking_similitud_plot,
    contraste_subcohorte_plot,
    factores_compartidos_plot,
    mosaico_subcohorte_plot,
)
from heavystats.bivariante.constants import (
    DEFAULT_PRIMARY_METALS,
    DEFAULT_METAL_LIMITS,
    DEFAULT_METAL_LODS,
    DEFAULT_METAL_CUTOFFS,
    DEFAULT_METAL_PAIRS,
    DEFAULT_BIVARIATE_GROUPS,
    CDC_LEAD_REFERENCE_VALUE,
    EPA_MERCURY_REFERENCE_VALUE,
    OMS_CADMIUM_REFERENCE_VALUE,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)

# Nombres canónicos en español
GraficosBivariantes = BivariatePlots

# Pruebas estadísticas en español
prueba_mann_whitney = mann_whitney_test
prueba_t_independiente = independent_t_test
prueba_kruskal_wallis = kruskal_wallis_test
prueba_anova_un_factor = anova_oneway_test
prueba_posthoc_dunn = dunn_posthoc_test
prueba_jonckheere_terpstra = jonckheere_terpstra_test
correlacion_spearman = spearman_correlation
correlacion_pearson = pearson_correlation
correlacion_kendall = kendall_correlation
matriz_spearman = spearman_matrix
ajustar_pvalores = adjust_pvalues
estimador_hodges_lehmann = hodges_lehmann_2sample
ic_bootstrap_diferencia_medianas = bootstrap_ci_diff_medians

# Constantes en español
METALES_PRIMARIOS_POR_DEFECTO = DEFAULT_PRIMARY_METALS
LIMITES_METALES_POR_DEFECTO = DEFAULT_METAL_LIMITS
LODS_METALES_POR_DEFECTO = DEFAULT_METAL_LODS
PUNTOS_CORTE_METALES_POR_DEFECTO = DEFAULT_METAL_CUTOFFS
PARES_METALES_POR_DEFECTO = DEFAULT_METAL_PAIRS
GRUPOS_BIVARIANTES_POR_DEFECTO = DEFAULT_BIVARIATE_GROUPS

__all__ = [
    # Motor gráfico
    "GraficosBivariantes",
    "BivariatePlots",
    "comparar_grupos_plot",
    "correlacion_analisis_plot",
    "matriz_coexposicion_plot",
    "radar_dietario_plot",
    "mosaico_dietario_plot",
    "ranking_similitud_plot",
    "contraste_subcohorte_plot",
    "factores_compartidos_plot",
    "mosaico_subcohorte_plot",
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

    # Pruebas estadísticas
    "prueba_mann_whitney",
    "prueba_t_independiente",
    "prueba_kruskal_wallis",
    "prueba_anova_un_factor",
    "prueba_posthoc_dunn",
    "prueba_jonckheere_terpstra",
    "correlacion_spearman",
    "correlacion_pearson",
    "correlacion_kendall",
    "matriz_spearman",
    "ajustar_pvalores",
    "estimador_hodges_lehmann",
    "ic_bootstrap_diferencia_medianas",
    "mann_whitney_test",
    "independent_t_test",
    "kruskal_wallis_test",
    "anova_oneway_test",
    "dunn_posthoc_test",
    "jonckheere_terpstra_test",
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "spearman_matrix",
    "adjust_pvalues",
    "hodges_lehmann_2sample",
    "bootstrap_ci_diff_medians",

    # Constantes
    "METALES_PRIMARIOS_POR_DEFECTO",
    "LIMITES_METALES_POR_DEFECTO",
    "LODS_METALES_POR_DEFECTO",
    "PUNTOS_CORTE_METALES_POR_DEFECTO",
    "PARES_METALES_POR_DEFECTO",
    "GRUPOS_BIVARIANTES_POR_DEFECTO",
    "DEFAULT_PRIMARY_METALS",
    "DEFAULT_METAL_LIMITS",
    "DEFAULT_METAL_LODS",
    "DEFAULT_METAL_CUTOFFS",
    "DEFAULT_METAL_PAIRS",
    "DEFAULT_BIVARIATE_GROUPS",
    "CDC_LEAD_REFERENCE_VALUE",
    "EPA_MERCURY_REFERENCE_VALUE",
    "OMS_CADMIUM_REFERENCE_VALUE",
    "DIET_ORDINAL_MAP",
    "DIET_ORDINAL_LABELS",
]
