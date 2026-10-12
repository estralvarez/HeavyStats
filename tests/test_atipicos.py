"""
Pruebas unitarias para el módulo de diagnóstico de valores atípicos (heavystats.atipicos).
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from heavystats.atipicos import (
    OutlierDiagnostics,
    OutlierPlots,
    UnivariateOutlierReport,
    OutlierDiagnosticReport,
)


@pytest.fixture
def sample_cohort():
    np.random.seed(42)
    n = 20
    # 19 casos normales + 1 caso extremo (punto palanca y concentración alta)
    y = np.random.normal(2.0, 0.4, size=n)
    y[-1] = 3.8 # caso alto

    x1 = np.random.uniform(0, 1, size=n)
    x1[-1] = 0.95
    x2 = np.random.uniform(0, 1, size=n)
    x2[-1] = 0.90
    x3 = np.random.uniform(5, 12, size=n)

    df = pd.DataFrame({
        "Muestra_Codificada": [f"M-{i+1:02d}" for i in range(n)],
        "Mercurio_ug_L": y,
        "Dieta_Pescado": x1,
        "Exposicion_Taller": x2,
        "Edad": x3,
    })
    return df


def test_outlier_diagnostics_univariate(sample_cohort):
    diag = OutlierDiagnostics(
        df=sample_cohort,
        target_col="Mercurio_ug_L",
        id_col="Muestra_Codificada",
        n_components=2
    )
    kpis = diag.get_kpis()

    assert kpis["n_obs"] == 20
    assert kpis["mean"] > 0
    assert kpis["std"] > 0
    assert kpis["median"] > 0
    assert kpis["tukey_fences"][1] > kpis["median"]
    assert "grubbs" in kpis
    assert kpis["grubbs"]["g_crit"] > 0


def test_outlier_diagnostics_multivariate(sample_cohort):
    diag = OutlierDiagnostics(
        df=sample_cohort,
        target_col="Mercurio_ug_L",
        id_col="Muestra_Codificada",
        n_components=2
    )
    df_sum = diag.get_summary_dataframe()

    assert len(df_sum) == 20
    assert "Hotelling_T2" in df_sum.columns
    assert "Leverage" in df_sum.columns
    assert "Cooks_D" in df_sum.columns
    assert "Diagnostico" in df_sum.columns

    # Caso M-20 debe tener el mayor T2 o Leverage
    m20_row = df_sum[df_sum["Muestra"] == "M-20"].iloc[0]
    assert m20_row["Leverage"] > 0.15


def test_outlier_plots(sample_cohort):
    diag = OutlierDiagnostics(
        df=sample_cohort,
        target_col="Mercurio_ug_L",
        id_col="Muestra_Codificada",
        n_components=2
    )
    op = OutlierPlots(diag)

    fig1, _ = op.plot_univariate_fences()
    assert fig1 is not None

    fig2, _ = op.plot_latent_scores()
    assert fig2 is not None

    fig3, _ = op.plot_influence_space()
    assert fig3 is not None

    fig4, _ = op.plot_leverage_cook()
    assert fig4 is not None

    fig5, _ = op.plot_dashboard_2x2()
    assert fig5 is not None


def test_outlier_reports(sample_cohort):
    diag = OutlierDiagnostics(
        df=sample_cohort,
        target_col="Mercurio_ug_L",
        id_col="Muestra_Codificada",
        n_components=2
    )

    rep_uni = UnivariateOutlierReport(diag)
    df_uni = rep_uni.to_dataframe()
    assert not df_uni.empty
    csv_uni = rep_uni.to_csv()
    assert len(csv_uni) > 0
    json_uni = rep_uni.to_json()
    assert "Estadísticos" in json_uni
    html_uni = rep_uni.to_html()
    assert "<table" in html_uni

    rep_diag = OutlierDiagnosticReport(diag)
    df_diag = rep_diag.to_dataframe()
    assert not df_diag.empty
    csv_diag = rep_diag.to_csv()
    assert len(csv_diag) > 0
    json_diag = rep_diag.to_json()
    assert "Muestra" in json_diag
    html_diag = rep_diag.to_html()
    assert "<table" in html_diag

    # Verificar accesibilidad en español
    from heavystats.atipicos import DiagnosticoAtipicos
    assert DiagnosticoAtipicos is OutlierDiagnostics

