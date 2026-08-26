from heavystats.validation import ValidationReport, validate_data
from heavystats.cleaning import (
    VariableTypeReport,
    VariablesTableReport,
    columns_type,
    categorical_columns,
    numerical_columns,
    variables_table,
    get_analytical_sample,
    select_metal,
    load_default_data,
)

__all__ = [
    "ValidationReport",
    "validate_data",
    "VariableTypeReport",
    "VariablesTableReport",
    "columns_type",
    "categorical_columns",
    "numerical_columns",
    "variables_table",
    "get_analytical_sample",
    "select_metal",
    "load_default_data",
]

def __getattr__(name: str):
    if name == "df":
        from heavystats.cleaning import df
        return df
    raise AttributeError(f"module {__name__} has no attribute {name}")
