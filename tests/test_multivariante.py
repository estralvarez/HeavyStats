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

    # Exportar JSON y CSV
    json_path = str(tmp_path / "tabla_test.json")
    json_code = pipeline.export_json(json_path)
    assert (tmp_path / "tabla_test.json").exists()
    assert len(json_code) > 0

    csv_path = str(tmp_path / "tabla_test.csv")
    csv_code = pipeline.export_csv(csv_path)
    assert (tmp_path / "tabla_test.csv").exists()
    assert len(csv_code) > 0

    # Exportar HTML
    html_path = str(tmp_path / "reporte_test.html")
    pipeline.export_html(html_path)
    assert (tmp_path / "reporte_test.html").exists()



def test_prepare_pls_dataframe():
    from heavystats.multivariante import prepare_pls_dataframe

    # Crear dataset con variables representativas de la encuesta
    np.random.seed(42)
    n = 20
    df_raw = pd.DataFrame({
        "Muestra_Codificada": range(1, n + 1),
        "Edad": np.random.randint(6, 12, n),
        "Sexo": ["femenino", "masculino"] * 10,
        "Es_Expuesto": ["SI", "NO"] * 10,
        "Score_Riesgo": np.random.randint(1, 8, n),
        "Mercurio_ug_L": np.random.uniform(1.2, 4.0, n),
        "Alim_Cereales": np.random.randint(0, 5, n),
        "Alim_Leguminosas": np.random.randint(0, 5, n),
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Alim_Pescados": np.random.randint(0, 5, n),
        "Salud_Fuma": ["SI", "NO"] * 10,
        "Salud_Actividad": ["NO", "SI"] * 10,
        "Salud_Transporte_caminar": [1, 0] * 10,
        "Salud_Transporte_vehiculo": [0, 1] * 10,
        "Salud_Agua_pozo_profundo": [0, 0] * 10,
        "Salud_Agua_filtrada": [1, 1] * 10,
        "Exposicion_Talleres_mecanico": [1, 0] * 10,
        "Exposicion_Cualquier_Taller": [1, 0] * 10,
    })

    pls_df = prepare_pls_dataframe(
        df_raw,
        target_metal="Mercurio_ug_L",
        id_col="Muestra_Codificada",
        include_domains_totals=True,
        normalize_diet=True,
    )

    assert isinstance(pls_df, pd.DataFrame)
    assert len(pls_df) == n
    assert "Muestra_Codificada" in pls_df.columns
    assert "Mercurio_ug_L" in pls_df.columns
    assert "Edad" in pls_df.columns
    assert "Sexo" in pls_df.columns
    assert "Es_Expuesto" in pls_df.columns
    assert "Tot_Dieta_Norm" in pls_df.columns
    assert "Tot_Habitos_Norm" in pls_df.columns
    assert "Tot_Movilidad_Norm" in pls_df.columns
    assert "Tot_Agua_Norm" in pls_df.columns
    assert "Tot_Exp_Antropica_Norm" in pls_df.columns
    assert "Exposicion_Total_Norm" in pls_df.columns

    # Verificar que los valores dietarios normalizados están en [0.0 - 1.0]
    assert pls_df["Alim_Pescados"].max() <= 1.0
    assert pls_df["Alim_Pescados"].min() >= 0.0

    # Verificar que todas las columnas excepto el ID son numéricas y sin NaNs
    feature_cols = [c for c in pls_df.columns if c != "Muestra_Codificada"]
    for c in feature_cols:
        assert np.issubdtype(pls_df[c].dtype, np.number)
        assert not pls_df[c].isna().any()

    # Verificar que se puede ajustar directamente PlsModeler con esta matriz
    X_mat = pls_df.drop(columns=["Muestra_Codificada", "Mercurio_ug_L"])
    y_vec = pls_df["Mercurio_ug_L"]

    from heavystats.multivariante import PlsModeler
    pls = PlsModeler(n_components=2)
    pls.fit(X_mat, y_vec)
    assert len(pls.vip_) == X_mat.shape[1]
    assert not pls.summary_df.empty

