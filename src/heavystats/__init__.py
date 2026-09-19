__version__ = "0.4.0"

from heavystats.validation import ValidationReport, validate_data
from heavystats.comparation import (
    ComparationReport,
    compare_groups,
)
from heavystats.cleaning import (
    VariableTypeReport,
    VariablesTableReport,
    columns_type,
    categorical_columns,
    numerical_columns,
    variables_table,
    get_analytical_sample,
    select_metal,
    load_data,
    standardize_boolean_columns,
    desaggregate_multiple_responses,
    encode_dietary_frequencies,
    create_composite_indicators,
)
from heavystats.univariate import (
    UnivariateTables,
    UnivariatePlots,
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    CDC_BMI_REFERENCE,
    get_label,
)
from heavystats.bivariate import (
    BivariateTables,
    BivariateTableReport,
    BivariatePlots,
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
    DEFAULT_METAL_LIMITS,
    DEFAULT_METAL_LODS,
    DEFAULT_METAL_CUTOFFS,
    DEFAULT_METAL_PAIRS,
)
from heavystats.version_checker import check_for_updates

__all__ = [
    "__version__",
    "ValidationReport",
    "validate_data",
    "ComparationReport",
    "compare_groups",
    "VariableTypeReport",
    "VariablesTableReport",
    "columns_type",
    "categorical_columns",
    "numerical_columns",
    "variables_table",
    "get_analytical_sample",
    "select_metal",
    "load_data",
    "standardize_boolean_columns",
    "desaggregate_multiple_responses",
    "encode_dietary_frequencies",
    "create_composite_indicators",
    "UnivariateTables",
    "UnivariatePlots",
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
    "DEFAULT_LABELS_MAP",
    "DEFAULT_CUSTOM_PARAMS",
    "DEFAULT_PERMISSIBLE_LIMITS",
    "DEFAULT_METAL_LIMITS",
    "DEFAULT_METAL_LODS",
    "DEFAULT_METAL_CUTOFFS",
    "DEFAULT_METAL_PAIRS",
    "CDC_BMI_REFERENCE",
    "get_label",
    "check_for_updates",
    "launch_studio",
]


def launch_studio():
    """Lanza la interfaz de usuario de terminal interactiva de HeavyStats Studio."""
    from heavystats.studio.app import iniciar_tui
    return iniciar_tui()

# Verificación de versiones no bloqueante en segundo plano al importar
try:
    check_for_updates(__version__, async_check=True)
except Exception:
    pass


def __getattr__(name: str):
    if name == "df":
        from heavystats.cleaning import df
        return df
    raise AttributeError(f"module {__name__} has no attribute {name}")
