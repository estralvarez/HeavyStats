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


def test_qualitative_association_and_summary():
    df = pd.DataFrame({
        "Consumo_Pescado": ["Alto", "Alto", "Bajo", "Bajo", "Alto", "Bajo", "Alto", "Bajo", "Alto", "Alto"],
        "Amalgama": ["Si", "No", "Si", "No", "No", "Si", "Si", "No", "Si", "No"],
        "Hg_Alto": ["Si", "Si", "No", "No", "Si", "No", "Si", "No", "No", "Si"]
    })
    
    # Test single 2x2 test
    res = hs.bivariate.qualitative_association_test(df["Consumo_Pescado"], df["Hg_Alto"])
    assert "p_val_fisher" in res
    assert "odds_ratio" in res
    assert "relative_risk" in res
    assert "cramer_v" in res

    # Test BivariateTables.qualitative_association
    assoc_rep = hs.bivariate.BivariateTables.qualitative_association(df, "Consumo_Pescado", "Hg_Alto")
    assert assoc_rep is not None
    assert isinstance(assoc_rep.to_dataframe(), pd.DataFrame)

    # Test BivariateTables.qualitative_summary
    screen_rep = hs.bivariate.BivariateTables.qualitative_summary(
        df,
        target_col="Hg_Alto",
        feature_cols=["Consumo_Pescado", "Amalgama"],
        target_positive="Si"
    )
    assert screen_rep is not None
    df_screen = screen_rep.to_dataframe()
    assert len(df_screen) == 2
    assert "Odds Ratio [IC 95%]" in df_screen.columns or "Fisher (p)" in df_screen.columns


def test_compare_groups_parametric_and_nonparametric():
    np.random.seed(42)
    df = pd.DataFrame({
        "Grupo2": ["Control"] * 10 + ["Expuesto"] * 10,
        "Grupo3": ["Bajo"] * 7 + ["Medio"] * 7 + ["Alto"] * 6,
        "Mercurio": np.concatenate([np.random.normal(2.0, 0.5, 10), np.random.normal(5.5, 1.2, 10)]),
        "Plomo": np.random.normal(10.0, 2.0, 20)
    })

    # Test 2 groups - Non-parametric
    rep_np2 = hs.bivariate.BivariateTables.compare_groups(
        df, group_col="Grupo2", continuous_cols=["Mercurio", "Plomo"], method="nonparametric"
    )
    assert rep_np2 is not None
    df_np2 = rep_np2.to_dataframe()
    assert len(df_np2) == 2
    assert "Estadístico" in df_np2.columns or "p-valor" in df_np2.columns

    # Test 2 groups - Parametric
    rep_p2 = hs.bivariate.BivariateTables.compare_groups(
        df, group_col="Grupo2", continuous_cols=["Mercurio"], method="parametric"
    )
    assert rep_p2 is not None
    df_p2 = rep_p2.to_dataframe()
    assert len(df_p2) == 2
    assert "t de Welch" in df_p2.columns or "Valor p" in df_p2.columns


    # Test >2 groups - Non-parametric (Kruskal-Wallis)
    rep_np3 = hs.bivariate.BivariateTables.compare_groups(
        df, group_col="Grupo3", continuous_cols=["Mercurio"], method="nonparametric"
    )
    assert rep_np3 is not None
    df_np3 = rep_np3.to_dataframe()
    assert "Kruskal-Wallis H" in df_np3.columns

    # Test >2 groups - Parametric (ANOVA)
    rep_p3 = hs.bivariate.BivariateTables.compare_groups(
        df, group_col="Grupo3", continuous_cols=["Mercurio"], method="parametric"
    )
    assert rep_p3 is not None
    df_p3 = rep_p3.to_dataframe()
    assert "ANOVA F" in df_p3.columns



def test_correlation_analysis_parametric_and_nonparametric():
    np.random.seed(42)
    x = np.linspace(1, 20, 20)
    y = 2.5 * x + np.random.normal(0, 2, 20)
    z = np.random.normal(0, 1, 20)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    # Non-parametric (Spearman & Kendall)
    rep_np = hs.bivariate.BivariateTables.correlation_analysis(
        df, target_col="Y", continuous_cols=["X", "Z"], method="nonparametric"
    )
    assert rep_np is not None
    df_np = rep_np.to_dataframe()
    assert len(df_np) == 2
    assert "Spearman" in df_np.iloc[0]["Metodo"]

    # Parametric (Pearson & Linear Regression)
    rep_p = hs.bivariate.BivariateTables.correlation_analysis(
        df, target_col="Y", continuous_cols=["X", "Z"], method="parametric"
    )
    assert rep_p is not None
    df_p = rep_p.to_dataframe()
    assert len(df_p) == 2
    assert "Pearson" in df_p.iloc[0]["Metodo"]
    assert "Pendiente (Beta)" in df_p.columns


def test_multivariate_screening_battery():
    np.random.seed(42)
    n = 20
    grupo = ["A"] * 10 + ["B"] * 10
    habito = ["Si"] * 12 + ["No"] * 8
    edad = np.random.uniform(20, 60, n)
    pescado = np.random.uniform(0, 10, n)
    hg = 0.5 * pescado + np.random.normal(2, 0.5, n)

    df = pd.DataFrame({
        "Grupo": grupo,
        "Habito": habito,
        "Edad": edad,
        "Pescado": pescado,
        "Mercurio": hg
    })

    screening_rep = hs.bivariate.BivariateTables.multivariate_screening(
        df,
        target_col="Mercurio",
        candidate_features=["Grupo", "Habito", "Edad", "Pescado"],
        method="nonparametric",
        fdr_alpha=0.10
    )
    assert screening_rep is not None
    df_screen = screening_rep.to_dataframe()
    assert len(df_screen) == 4
    assert "FDR p-valor (BH)" in df_screen.columns
    assert "Prioridad Multivariable" in df_screen.columns
    assert "Rank" in df_screen.columns
    assert "Colinealidad (rho max)" in df_screen.columns

