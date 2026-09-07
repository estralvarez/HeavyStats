"""
Módulo de Análisis Bivariante de HeavyStats (`heavystats.bivariate`).
Implementa el protocolo de análisis estadístico bivariante bioestadístico y toxicológico.
"""

from heavystats.bivariate.tests import (
    mann_whitney_test,
    kruskal_wallis_test,
    dunn_posthoc_test,
    jonckheere_terpstra_test,
    spearman_correlation,
    spearman_matrix,
    fisher_chi2_test,
    adjust_pvalues,
    hodges_lehmann_2sample,
    bootstrap_ci_diff_medians,
    rank_bivariate_associations,
    collinearity_matrix,
)
from heavystats.bivariate.tables import (
    BivariateTables,
    BivariateTableReport,
)
from heavystats.bivariate.plots import (
    BivariatePlots,
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
    "BivariateTables",
    "BivariateTableReport",
    "BivariatePlots",
    "mann_whitney_test",
    "kruskal_wallis_test",
    "dunn_posthoc_test",
    "jonckheere_terpstra_test",
    "spearman_correlation",
    "spearman_matrix",
    "fisher_chi2_test",
    "adjust_pvalues",
    "hodges_lehmann_2sample",
    "bootstrap_ci_diff_medians",
    "rank_bivariate_associations",
    "collinearity_matrix",
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
