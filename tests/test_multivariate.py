"""
Pruebas unitarias para el módulo de análisis multivariante de HeavyStats (heavystats.multivariante).
"""

import pytest
import numpy as np
import pandas as pd

from heavystats.multivariante import (
    MultivariateScreening,
    LassoModeler,
    PlsModeler,
    MultivariatePipeline,
    MultivariateTableReport,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 25
    x_diet = np.random.choice([0, 1, 2, 3, 4], size=n)
    x_binary = np.random.choice(["SI", "NO"], size=n)
    x_sector = np.random.choice(["Norte", "Sur", "Centro"], size=n)
    x_continuous = np.random.uniform(5, 12, size=n)

    # Variable respuesta con señal fuerte en x_diet y x_binary
    noise = np.random.normal(0, 0.2, size=n)
    binary_num = (x_binary == "SI").astype(float)
    y = np.exp(0.5 + 0.3 * x_diet + 0.4 * binary_num + 0.05 * x_continuous + noise)

    df = pd.DataFrame({
        "Alim_Pescados": x_diet,
        "Es_Expuesto": x_binary,
        "Sector": x_sector,
        "Edad": x_continuous,
        "Mercurio_ug_L": y,
    })
    return df


def test_multivariate_screening(synthetic_data):
    screening = MultivariateScreening(
        data=synthetic_data,
        target_col="Mercurio_ug_L",
        n_boot=100,
        random_state=42
    )
    res_df = screening.run()

    assert not res_df.empty
    assert "Variable" in res_df.columns
    assert "p_valor" in res_df.columns
    assert "q_fdr" in res_df.columns
    assert "Prioridad" in res_df.columns
    assert "Direccion" in res_df.columns

    # Alim_Pescados debe ser de prioridad Alta o Intermedia
    pescados_row = res_df[res_df["Variable"] == "Alim_Pescados"].iloc[0]
    assert pescados_row["p_valor"] < 0.10


def test_lasso_modeler(synthetic_data):
    X = synthetic_data[["Alim_Pescados", "Edad"]].copy()
    y = np.log(synthetic_data["Mercurio_ug_L"].values)

    lasso = LassoModeler(cv_folds=3, random_state=42)
    lasso.fit(X, y)

    assert len(lasso.coef_) == 2
    summary = lasso.summary_df
    assert not summary.empty
    assert "Beta_Estandarizado" in summary.columns
    assert "Efecto_Lasso" in summary.columns


def test_pls_modeler(synthetic_data):
    X = synthetic_data[["Alim_Pescados", "Edad"]].copy()
    y = np.log(synthetic_data["Mercurio_ug_L"].values)

    pls = PlsModeler(n_components=2)
    pls.fit(X, y)

    assert len(pls.vip_) == 2
    assert np.all(pls.vip_ >= 0)
    summary = pls.summary_df
    assert not summary.empty
    assert "VIP" in summary.columns
    assert "Coef_PLS" in summary.columns


def test_multivariate_pipeline_and_export(synthetic_data, tmp_path):
    pipeline = MultivariatePipeline(
        data=synthetic_data,
        target_col="Mercurio_ug_L",
        log_transform_target=True,
        random_state=42
    )

    scr_df = pipeline.run_screening()
    assert not scr_df.empty

    fit_res = pipeline.fit_models(min_priority="Todas", lasso_cv_folds=3, pls_components=2)
    assert "consolidated" in fit_res
    assert "report" in fit_res

    report = pipeline.get_report()
    assert isinstance(report, MultivariateTableReport)

    # Exportar LaTeX
    latex_path = str(tmp_path / "tabla_test.tex")
    latex_code = pipeline.export_latex(latex_path)
    assert "\\begin{table}" in latex_code
    assert "\\toprule" in latex_code
    assert "\\bottomrule" in latex_code

    # Exportar HTML
    html_path = str(tmp_path / "reporte_test.html")
    pipeline.export_html(html_path)
    assert (tmp_path / "reporte_test.html").exists()
