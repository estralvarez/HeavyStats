import pandas as pd
import heavystats as hs


def test_load_data():
    df = hs.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_columns_type():
    df = hs.load_data()
    report = hs.columns_type(df)
    assert isinstance(report, hs.VariableTypeReport)
    df_report = report.to_dataframe()
    assert isinstance(df_report, pd.DataFrame)
    assert "Variable" in df_report.columns
    assert "Tipo de Dato" in df_report.columns


def test_categorical_and_numerical_columns():
    df = hs.load_data()
    cat_cols = hs.categorical_columns(df)
    num_cols = hs.numerical_columns(df)
    assert isinstance(cat_cols, pd.DataFrame)
    assert isinstance(num_cols, pd.DataFrame)


def test_variables_table():
    df = hs.load_data()
    var_table = hs.variables_table(df)
    assert isinstance(var_table, hs.VariablesTableReport)
    assert isinstance(var_table.to_dataframe(), pd.DataFrame)
