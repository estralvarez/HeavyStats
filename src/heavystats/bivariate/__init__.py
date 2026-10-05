"""
Módulo de Visualización Bivariante de HeavyStats (`heavystats.bivariate`).
Implementa el motor gráfico editorial y los contrastes estadísticos bivariantes.
"""

from heavystats.bivariate.tests import (
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
from heavystats.bivariate.plots import (
    BivariatePlots,
    compare_groups_plot,
    correlation_analysis_plot,
    coexposure_matrix_plot,
    diet_radar_plot,
    diet_boxplots_plot,
)
from heavystats.bivariate.constants import (
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

__all__ = [
    "BivariatePlots",
    "compare_groups_plot",
    "correlation_analysis_plot",
    "coexposure_matrix_plot",
    "diet_radar_plot",
    "diet_boxplots_plot",

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
