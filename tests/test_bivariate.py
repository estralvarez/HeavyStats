import numpy as np
import pandas as pd
import heavystats as hs
from heavystats.bivariate import (
    adjust_pvalues,
    mann_whitney_test,
    kruskal_wallis_test,
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


def test_bivariate_plots_three_pillars():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "Grupo2": ["Control"] * 10 + ["Expuesto"] * 10,
        "Grupo3": ["Bajo"] * 7 + ["Medio"] * 7 + ["Alto"] * 6,
        "Consumo_Pescado": ["Alto", "Alto", "Bajo", "Bajo", "Alto", "Bajo", "Alto", "Bajo", "Alto", "Alto"] * 2,
        "Amalgama": ["Si", "No", "Si", "No", "No", "Si", "Si", "No", "Si", "No"] * 2,
        "Hg_Alto": ["Si", "Si", "No", "No", "Si", "No", "Si", "No", "No", "Si"] * 2,
        "Edad": np.random.uniform(20, 60, n),
        "Pescado_Frec": np.random.uniform(0, 10, n),
        "Mercurio": np.concatenate([np.random.normal(2.0, 0.4, 10), np.random.normal(5.8, 1.1, 10)]),
        "Plomo": np.random.normal(8.0, 1.5, n),
        "Score_Riesgo": np.random.uniform(1, 9, n),
    })

    bp = hs.bivariate.BivariatePlots(df)

    # PILAR 1: Cuantitativa vs Cualitativa (Dual: No Paramétrico y Paramétrico)
    fig1_np, ax1_np = bp.compare_groups(quantitative="Mercurio", group="Grupo2", method="nonparametric")
    assert fig1_np is not None
    plt.close(fig1_np)

    fig1_p, ax1_p = bp.compare_groups(quantitative="Mercurio", group="Grupo2", method="parametric")
    assert fig1_p is not None
    plt.close(fig1_p)

    fig1_k, ax1_k = bp.compare_groups(quantitative="Mercurio", group="Grupo3", method="nonparametric")
    assert fig1_k is not None
    plt.close(fig1_k)

    # Compatibilidad Pilar 1
    fig1_leg, ax1_leg = bp.metal_by_group(group_col="Grupo2", metal="Mercurio")
    assert fig1_leg is not None
    plt.close(fig1_leg)

    # PILAR 2: Cuantitativa vs Cuantitativa (Dual: Spearman/Bootstrap y Pearson/OLS)
    fig2_np, ax2_np = bp.correlation_analysis(x="Edad", y="Mercurio", method="nonparametric")
    assert fig2_np is not None
    plt.close(fig2_np)

    fig2_p, ax2_p = bp.correlation_analysis(x="Edad", y="Mercurio", method="parametric")
    assert fig2_p is not None
    plt.close(fig2_p)

    fig2_mat, ax2_mat = bp.coexposure_matrix(variables=["Mercurio", "Plomo", "Edad"], kind="heatmap")
    assert fig2_mat is not None
    plt.close(fig2_mat)

    fig2_scat_mat, axes2_scat_mat = bp.coexposure_matrix(variables=["Mercurio", "Plomo", "Edad"], kind="scatter")
    assert fig2_scat_mat is not None
    plt.close(fig2_scat_mat)

    fig2_ord, ax2_ord = bp.ordinal_trend_plot(ordinal_col="Grupo3", metal="Mercurio")
    assert fig2_ord is not None
    plt.close(fig2_ord)

    # Compatibilidad Pilar 2
    fig2_scat, ax2_scat = bp.scatter_continuous(continuous_col="Edad", metal="Mercurio")
    assert fig2_scat is not None
    plt.close(fig2_scat)

    # Funciones de conveniencia a nivel de paquete
    fig_f1, _ = hs.bivariate.compare_groups_plot(df, quantitative="Mercurio", group="Grupo2")
    assert fig_f1 is not None
    plt.close(fig_f1)

    fig_f2, _ = hs.bivariate.correlation_analysis_plot(df, x="Edad", y="Mercurio")
    assert fig_f2 is not None
    plt.close(fig_f2)

    fig_f3, _ = hs.bivariate.coexposure_matrix_plot(df, variables=["Mercurio", "Plomo"])
    assert fig_f3 is not None
    plt.close(fig_f3)

    plt.close("all")


def test_bivariate_grouping_dummies_and_multiple():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_test = pd.DataFrame({
        "Sexo": ["F", "F", "F", "M", "M", "M"],
        "Sector": ["Norte", "Norte", "Sur", "Sur", "Sur", "Norte"],
        "Exposicion_Lugares_canale": [1, 0, 0, 0, 0, 0],
        "Exposicion_Lugares_canales": [0, 1, 0, 0, 0, 0],
        "Exposicion_Lugares_rios": [0, 1, 1, 0, 1, 0],
        "Exposicion_Talleres_mecanico": [1, 1, 0, 0, 1, 0],
        "Exposicion_Industrias_fabrica_metales": [0, 0, 1, 1, 0, 0],
        "Plomo_ug_dL": [2.5, 3.2, 4.1, 1.8, 5.0, 2.9],
    })

    bp = hs.bivariate.BivariatePlots(df_test)

    # 1. Verificar consolidación de Exposicion_Lugares_canale
    assert "Exposicion_Lugares_canale" not in bp.df.columns
    assert "Exposicion_Lugares_canales" in bp.df.columns
    assert bp.df["Exposicion_Lugares_canales"].sum() == 2

    # 2. plot_categorical con lista de dummies (incluyendo canale como alias)
    fig_d, ax_d = bp.plot_categorical(
        metal="Plomo_ug_dL",
        group_col=["Exposicion_Lugares_canales", "Exposicion_Lugares_rios", "Exposicion_Lugares_canale"]
    )
    assert fig_d is not None
    xticks_d = [t.get_text() for t in ax_d.get_xticklabels()]
    assert any("Canales" in t for t in xticks_d)
    assert any("Ríos" in t or "Rios" in t for t in xticks_d)
    plt.close(fig_d)

    # 3. plot_categorical con dimensión por nombre
    fig_lug, ax_lug = bp.plot_categorical(metal="Plomo_ug_dL", group_col="lugares")
    assert fig_lug is not None
    plt.close(fig_lug)

    fig_all, ax_all = bp.plot_categorical(metal="Plomo_ug_dL", group_col="all_dummies")
    assert fig_all is not None
    plt.close(fig_all)

    # 4. plot_categorical con interacción de columnas categóricas
    fig_inter, ax_inter = bp.plot_categorical(metal="Plomo_ug_dL", group_col=["Sector", "Sexo"])
    assert fig_inter is not None
    plt.close(fig_inter)

    # 5. Método explícito plot_dummies
    fig_dum, ax_dum = bp.plot_dummies(metal="Plomo_ug_dL", dimension="talleres")
    assert fig_dum is not None
    plt.close(fig_dum)

    plt.close("all")


def test_plot_diet_radar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 20
    df_radar = pd.DataFrame({
        "Plomo_ug_dL": np.random.uniform(1.0, 5.0, n),
        "Alim_Cereales": np.random.randint(0, 5, n),
        "Alim_Leguminosas": np.random.randint(0, 5, n),
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Alim_Pescados": np.random.randint(0, 5, n),
        "Alim_Frutas": np.random.randint(0, 5, n),
    })

    bp = hs.bivariate.BivariatePlots(df_radar)

    # 1. Estratificación por mediana
    fig_med, ax_med = bp.plot_diet_radar(metal="Plomo_ug_dL", stratify_by="median")
    assert fig_med is not None
    assert ax_med is not None
    thetaticks = [t.get_text() for t in ax_med.get_xticklabels()]
    assert "Cereales" in thetaticks
    assert "Pescados" in thetaticks
    plt.close(fig_med)

    # 2. Estratificación por terciles
    fig_terc, ax_terc = bp.plot_diet_radar(metal="Plomo_ug_dL", stratify_by="terciles")
    assert fig_terc is not None
    plt.close(fig_terc)

    # 3. Paneles divididos contiguos (split_panels=True)
    fig_split, axes_split = bp.plot_diet_radar(metal="Plomo_ug_dL", stratify_by="median", split_panels=True)
    assert fig_split is not None
    assert len(axes_split) == 2
    plt.close(fig_split)

    # 4. Función de conveniencia diet_radar_plot
    fig_conv, ax_conv = hs.bivariate.diet_radar_plot(df_radar, metal="Plomo_ug_dL", split_panels=True)
    assert fig_conv is not None
    assert len(ax_conv) == 2
    plt.close(fig_conv)

    # 4. Comprobación de error si no existen columnas Alim_
    df_no_alim = pd.DataFrame({"Plomo_ug_dL": [1.0, 2.0], "Otro": [1, 2]})
    bp_no_alim = hs.bivariate.BivariatePlots(df_no_alim)
    try:
        bp_no_alim.plot_diet_radar(metal="Plomo_ug_dL")
        assert False, "Debería haber levantado ValueError"
    except ValueError:
        pass

    plt.close("all")


def test_plot_diet_boxplots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 30
    df_box = pd.DataFrame({
        "Plomo_ug_dL": np.random.uniform(1.0, 5.0, n),
        "Alim_Cereales": np.random.randint(0, 5, n),
        "Alim_Leguminosas": np.random.randint(0, 5, n),
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Alim_Pescados": np.random.randint(0, 5, n),
    })

    bp = hs.bivariate.BivariatePlots(df_box)

    # 1. Modo grid no paramétrico (Jonckheere-Terpstra)
    fig_grid, axes_grid = bp.plot_diet_boxplots(metal="Plomo_ug_dL", layout="grid", ncols=2, method="nonparametric")
    assert fig_grid is not None
    assert axes_grid.size == 4
    plt.close(fig_grid)

    # 2. Modo grid paramétrico (Pearson)
    fig_param, axes_param = bp.plot_diet_boxplots(metal="Plomo_ug_dL", layout="grid", ncols=2, method="parametric")
    assert fig_param is not None
    assert axes_param.size == 4
    plt.close(fig_param)

    # 3. Verificación de que layout='consolidated' levanta ValueError tras su eliminación
    try:
        bp.plot_diet_boxplots(metal="Plomo_ug_dL", layout="consolidated")
        assert False, "Debería haber levantado ValueError para layout='consolidated'"
    except ValueError as e:
        assert "eliminado" in str(e)

    # 4. Prueba de plot_dietary con métodos paramétrico y no paramétrico
    fig_d_np, ax_d_np = bp.plot_dietary(ordinal_col="Alim_Carnes", metal="Plomo_ug_dL", method="nonparametric")
    assert fig_d_np is not None and ax_d_np is not None
    plt.close(fig_d_np)

    fig_d_p, ax_d_p = bp.plot_dietary(ordinal_col="Alim_Carnes", metal="Plomo_ug_dL", method="parametric")
    assert fig_d_p is not None and ax_d_p is not None
    plt.close(fig_d_p)

    # 5. Función de conveniencia diet_boxplots_plot
    fig_conv, axes_conv = hs.bivariate.diet_boxplots_plot(df_box, metal="Plomo_ug_dL", layout="grid", ncols=2)
    assert fig_conv is not None
    assert axes_conv.size == 4
    plt.close(fig_conv)

    # 6. Error si no hay columnas de dieta
    df_no_alim = pd.DataFrame({"Plomo_ug_dL": [1.0, 2.0], "Otro": [1, 2]})
    bp_no = hs.bivariate.BivariatePlots(df_no_alim)
    try:
        bp_no.plot_diet_boxplots(metal="Plomo_ug_dL")
        assert False, "Debería haber levantado ValueError"
    except ValueError:
        pass

    plt.close("all")


def test_subcohort_and_radiography_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 20
    df_test = pd.DataFrame({
        "Muestra_Codificada": range(1, n + 1),
        "Mercurio_ug_L": np.sort(np.random.uniform(0.5, 4.5, n)),
        "Alim_Cereales": np.random.randint(0, 5, n),
        "Alim_Leguminosas": [2] * 7 + [0, 1, 3, 4] * 3 + [1],
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Alim_Pescados": np.random.randint(0, 5, n),
        "Salud_Fuma": ["NO", "SI"] * 10,
        "Salud_Actividad": ["SI", "NO"] * 10,
        "Salud_Bombillos": ["NO"] * 20,
        "Salud_Techo": ["NO"] * 20,
        "Salud_Joyeria": ["SI"] * 10 + ["NO"] * 10,
        "Salud_Transporte_caminar": [1, 0] * 10,
        "Salud_Transporte_publico": [0, 1] * 10,
        "Salud_Transporte_vehiculo": [1, 1] * 10,
        "Salud_Agua_pozo_profundo": [0, 1] * 10,
        "Salud_Agua_filtrada": [1, 0] * 10,
        "Salud_Agua_mineral_embotellada": [0, 0] * 10,
        "Exposicion_Talleres_carpinteria": [0] * 20,
        "Exposicion_Talleres_latoneria": [0, 1] * 10,
        "Exposicion_Talleres_mecanico": [1, 0] * 10,
        "Exposicion_Lugares_canales": [0] * 20,
        "Exposicion_Lugares_estacion_gasolina": [1, 0] * 10,
        "Exposicion_Lugares_llenadora_gas_natural": [0] * 20,
        "Exposicion_Lugares_rectificadora_motores": [0] * 20,
        "Exposicion_Lugares_rios": [0, 1] * 10,
        "Exposicion_Industrias_fabrica_metales": [0] * 20,
        "Exposicion_Industrias_fabrica_productos_quimicos": [0] * 20,
        "Exposicion_Cualquier_Taller": [1, 0] * 10,
        "Exposicion_Cualquier_Industria": [0] * 20,
        "Exposicion_Cualquier_Lugar_Riesgo": [1, 0] * 10,
        "Salud_Cualquier_Agua_Riesgo": [0, 1] * 10,
    })

    bp = hs.bivariate.BivariatePlots(df_test)

    # 1. Similarity Ranking Plot
    fig_sim, ax_sim, df_sim = bp.similarity_ranking_plot(
        target_metal="Mercurio_ug_L",
        highlight_item="Alim_Leguminosas",
        method="nonparametric",
    )
    assert fig_sim is not None
    assert ax_sim is not None
    assert isinstance(df_sim, pd.DataFrame)
    assert not df_sim.empty
    plt.close(fig_sim)

    # 1b. Similarity Ranking Plot (Parametric Pearson)
    fig_sim_p, ax_sim_p, df_sim_p = bp.similarity_ranking_plot(
        target_metal="Mercurio_ug_L",
        highlight_item="Alim_Leguminosas",
        method="parametric",
    )
    assert fig_sim_p is not None
    assert "Pearson_r" in df_sim_p.columns
    plt.close(fig_sim_p)

    # 2. Subcohort Contrast Plot (Nonparametric, auto-detected mode with greatest count)
    fig_sub, ax_sub, stats_sub = bp.subcohort_contrast_plot(
        target_metal="Mercurio_ug_L",
        criterion_col="Alim_Leguminosas",
        criterion_val=None, # Auto-detecta moda (opción con mayor cantidad de muestras: 2, n=7)
        method="nonparametric",
        annotate_ids=False,
    )
    assert fig_sub is not None
    assert ax_sub is not None
    assert stats_sub["n_subcohort"] == 7
    assert stats_sub["n_rest"] == 13
    assert stats_sub["method"] == "nonparametric"
    assert "p_value" in stats_sub
    plt.close(fig_sub)

    # 2b. Subcohort Contrast Plot (Parametric Welch t-test)
    fig_sub_p, ax_sub_p, stats_sub_p = bp.subcohort_contrast_plot(
        target_metal="Mercurio_ug_L",
        criterion_col="Alim_Leguminosas",
        method="parametric",
    )
    assert fig_sub_p is not None
    assert stats_sub_p["method"] == "parametric"
    assert "t_stat" in stats_sub_p
    assert "hedges_g" in stats_sub_p
    plt.close(fig_sub_p)

    # 3. Subcohort Mosaic Plot (Panel A + B + C)
    fig_mos, axes_mos, res_mos = bp.subcohort_mosaic_plot(
        target_metal="Mercurio_ug_L",
        criterion_col="Alim_Leguminosas",
        criterion_val=2,
        include_shared_factors=True,
    )
    assert fig_mos is not None
    assert len(axes_mos) == 3
    assert "similarity_df" in res_mos
    assert "contrast_stats" in res_mos
    plt.close(fig_mos)

    # 4. Survey Radiography Plot (Suite de 7 paneles)
    res_rad = bp.plot_survey_radiography(
        target_metal="Mercurio_ug_L",
        id_col="Muestra_Codificada",
    )
    assert isinstance(res_rad, dict)
    assert len(res_rad) == 7
    assert "bloque1_dieta" in res_rad
    assert "bloque7_exposicion_total" in res_rad

    # 5. Scan Subcohort Contrasts (Escaneo automatizado)
    df_scan = bp.scan_subcohort_contrasts(
        target_metal="Mercurio_ug_L",
        candidate_cols=["Alim_Leguminosas", "Alim_Carnes", "Alim_Pescados"],
        method="nonparametric",
        plot_significant=False,
    )
    assert isinstance(df_scan, pd.DataFrame)
    assert not df_scan.empty
    assert "p_valor" in df_scan.columns
    assert "Significativo" in df_scan.columns

    plt.close("all")
    for k, (fig_r, ax_r) in res_rad.items():
        assert fig_r is not None
        assert ax_r is not None
        plt.close(fig_r)

    # 5. Funciones funcionales directas
    fig_conv_sim, ax_conv_sim, _ = hs.bivariate.similarity_ranking_plot(
        df_test, target_metal="Mercurio_ug_L"
    )
    assert fig_conv_sim is not None
    plt.close(fig_conv_sim)

    fig_conv_sub, ax_conv_sub, _ = hs.bivariate.subcohort_contrast_plot(
        df_test, target_metal="Mercurio_ug_L", criterion_col="Alim_Leguminosas", criterion_val=2
    )
    assert fig_conv_sub is not None
    plt.close(fig_conv_sub)

    plt.close("all")





