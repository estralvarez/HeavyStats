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
)

__all__ = [
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
]

def __getattr__(name: str):
    if name == "df":
        from heavystats.cleaning import df
        return df
    raise AttributeError(f"module {__name__} has no attribute {name}")
