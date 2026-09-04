"""
Módulo de Análisis Bivariantes de HeavyStats.
"""

from heavystats.bivariate.tests import (
    mann_whitney_test,
    welch_ttest_log,
    kruskal_wallis_test,
    dunn_posthoc_test,
    jonckheere_terpstra_test,
    spearman_correlation,
    spearman_matrix,
    fisher_chi2_test,
    adjust_pvalues,
    hodges_lehmann_2sample,
    bootstrap_ci_diff_medians,
)
from heavystats.bivariate.tables import (
    BivariateTables,
    BivariateTableReport,
)
from heavystats.bivariate.plots import (
    BivariatePlots,
)
from heavystats.bivariate.constants import (
    DEFAULT_PRIMARY_BINARY_VARS,
    DEFAULT_PRIMARY_METALS,
    DEFAULT_EXPLORATORY_GROUPS,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)

__all__ = [
    "BivariateTables",
    "BivariateTableReport",
    "BivariatePlots",
    "mann_whitney_test",
    "welch_ttest_log",
    "kruskal_wallis_test",
    "dunn_posthoc_test",
    "jonckheere_terpstra_test",
    "spearman_correlation",
    "spearman_matrix",
    "fisher_chi2_test",
    "adjust_pvalues",
    "hodges_lehmann_2sample",
    "bootstrap_ci_diff_medians",
    "DEFAULT_PRIMARY_BINARY_VARS",
    "DEFAULT_PRIMARY_METALS",
    "DEFAULT_EXPLORATORY_GROUPS",
    "DIET_ORDINAL_MAP",
    "DIET_ORDINAL_LABELS",
]
