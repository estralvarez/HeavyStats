import pandas as pd
import heavystats as hs
from heavystats.univariate import UnivariateTables


def test_univariate_tables_init():
    df = hs.load_data()
    tables = UnivariateTables(df)
    assert tables is not None


def test_univariate_metal_summary():
    df = hs.load_data()
    tables = UnivariateTables(df)
    # Check if there are metal columns in df
    metal_cols = [c for c in ["Plomo_Sangre", "Mercurio_Sangre", "Cadmio_Sangre", "Arsenico_Orina"] if c in df.columns]
    if metal_cols:
        summary = tables.metal_summary(metals=metal_cols)
        assert summary is not None
        assert isinstance(summary.to_dataframe(), pd.DataFrame)
        assert len(summary.to_dataframe()) > 0


def test_univariate_categorical_summary():
    df = hs.load_data()
    tables = UnivariateTables(df)
    cat_cols = tables.default_categorical_cols
    if cat_cols:
        summary = tables.categorical_summary(columns=cat_cols[:3])
        assert summary is not None
        assert isinstance(summary.to_dataframe(), pd.DataFrame)
