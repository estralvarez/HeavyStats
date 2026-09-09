import numpy as np
import pandas as pd
import heavystats as hs
from heavystats.bivariate import (
    adjust_pvalues,
    mann_whitney_test,
    spearman_matrix,
    spearman_correlation,
)


def test_adjust_pvalues():
    raw_p = [0.001, 0.01, 0.04, 0.05, 0.20]
    bh_p = adjust_pvalues(raw_p, method="fdr_bh")
    assert len(bh_p) == len(raw_p)
    assert np.all(bh_p >= np.asarray(raw_p))
    assert np.all(bh_p <= 1.0)

    bonf_p = adjust_pvalues(raw_p, method="bonferroni")
    assert np.isclose(bonf_p[0], 0.005)


def test_spearman_correlation():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    res = spearman_correlation(x, y)
    assert isinstance(res, dict)
    assert np.isclose(res.get("rho", 0), 1.0)
    assert res.get("p_val", 1.0) < 0.01


def test_bivariate_spearman_matrix():
    df = hs.load_data()
    metal_cols = [c for c in ["Plomo_Sangre", "Mercurio_Sangre", "Cadmio_Sangre"] if c in df.columns]
    if len(metal_cols) >= 2:
        matrix_report = spearman_matrix(df, metal_cols=metal_cols)
        assert matrix_report is not None
        df_mat = matrix_report.to_dataframe()
        assert isinstance(df_mat, pd.DataFrame)
