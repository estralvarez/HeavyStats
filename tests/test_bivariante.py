import numpy as np
import pandas as pd
import heavystats as hs
from heavystats.bivariante import (
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

    # 4. Scan Subcohort Contrasts (Escaneo automatizado)
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


def test_trimetal_subcohort_generality():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(123)
    n = 24
    df_tri = pd.DataFrame({
        "Muestra_Codificada": range(1, n + 1),
        "Plomo_ug_dL": np.random.uniform(1.0, 6.0, n),
        "Mercurio_ug_L": np.random.uniform(0.5, 8.0, n),
        "Cadmio_ug_L": np.random.uniform(0.1, 2.5, n),
        "Alim_Leguminosas": [2] * 10 + [0, 1, 3, 4] * 3 + [1, 2],
        "Alim_Pescados": [0] * 8 + [1, 2, 3, 4] * 4,
        "Alim_Vegetales": [2] * 9 + [1, 3, 4] * 5,
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Salud_Fuma": ["NO", "SI"] * 12,
        "Exposicion_Lugares_rios": [0, 1] * 12,
    })

    bp = hs.bivariate.BivariatePlots(df_tri)

    metals_config = [
        ("Plomo_ug_dL", "Alim_Leguminosas", 3.5),
        ("Mercurio_ug_L", "Alim_Pescados", 5.0),
        ("Cadmio_ug_L", "Alim_Vegetales", 1.0),
    ]

    for m_col, food_col, ref_val in metals_config:
        # 1. Similarity ranking with string item_cols and all items
        fig_sim, ax_sim, df_sim = bp.similarity_ranking_plot(target_metal=m_col)
        assert fig_sim is not None
        assert "Spearman_rho" in df_sim.columns
        assert df_sim["Metal"].iloc[0] == m_col
        plt.close(fig_sim)

        # 2. Subcohort contrast with toxicological reference line
        fig_sub, ax_sub, stats_sub = bp.subcohort_contrast_plot(
            target_metal=m_col,
            criterion_col=food_col,
            show_limit=True,
        )
        assert fig_sub is not None
        assert stats_sub["metal"] == m_col
        assert stats_sub["permissible_limit"] == ref_val
        plt.close(fig_sub)

        # 3. Subcohort mosaic plot with automatic canonical criterion
        fig_mos, axes_mos, res_mos = bp.subcohort_mosaic_plot(
            target_metal=m_col,
            include_shared_factors=True,
        )
        assert fig_mos is not None
        assert len(axes_mos) == 3
        assert res_mos["metal"] == m_col
        plt.close(fig_mos)

    plt.close("all")


def test_optional_titles_subcohort():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(99)
    n = 20
    df = pd.DataFrame({
        "Muestra_Codificada": range(1, n + 1),
        "Mercurio_ug_L": np.random.uniform(0.5, 4.5, n),
        "Alim_Pescados": np.random.randint(0, 5, n),
        "Alim_Carnes": np.random.randint(0, 5, n),
        "Salud_Fuma": ["NO", "SI"] * 10,
    })

    bp = hs.bivariate.BivariatePlots(df)

    def get_any_title(a):
        return a.get_title(loc="left") or a.get_title(loc="center")

    # 1. similarity_ranking_plot
    # 1a. Default -> Tiene título
    fig1, ax1, _ = bp.similarity_ranking_plot(target_metal="Mercurio_ug_L")
    assert get_any_title(ax1) != ""
    plt.close(fig1)

    # 1b. show_title=False -> Título vacío
    fig1_no, ax1_no, _ = bp.similarity_ranking_plot(target_metal="Mercurio_ug_L", show_title=False)
    assert get_any_title(ax1_no) == ""
    plt.close(fig1_no)

    # 1c. title="" -> Título vacío
    fig1_empty, ax1_empty, _ = bp.similarity_ranking_plot(target_metal="Mercurio_ug_L", title="")
    assert get_any_title(ax1_empty) == ""
    plt.close(fig1_empty)

    # 1d. title=False -> Título vacío
    fig1_f, ax1_f, _ = bp.similarity_ranking_plot(target_metal="Mercurio_ug_L", title=False)
    assert get_any_title(ax1_f) == ""
    plt.close(fig1_f)

    # 1e. Custom title
    fig1_c, ax1_c, _ = bp.similarity_ranking_plot(target_metal="Mercurio_ug_L", title="Ranking Personalizado")
    assert get_any_title(ax1_c) == "Ranking Personalizado"
    plt.close(fig1_c)

    # 2. subcohort_contrast_plot
    # 2a. Default -> Tiene título
    fig2, ax2, _ = bp.subcohort_contrast_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados")
    assert get_any_title(ax2) != ""
    plt.close(fig2)

    # 2b. show_title=False -> Título vacío
    fig2_no, ax2_no, _ = bp.subcohort_contrast_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", show_title=False)
    assert get_any_title(ax2_no) == ""
    plt.close(fig2_no)

    # 2c. title="" -> Título vacío
    fig2_empty, ax2_empty, _ = bp.subcohort_contrast_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", title="")
    assert get_any_title(ax2_empty) == ""
    plt.close(fig2_empty)

    # 2d. title=False -> Título vacío
    fig2_f, ax2_f, _ = bp.subcohort_contrast_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", title=False)
    assert get_any_title(ax2_f) == ""
    plt.close(fig2_f)

    # 2e. Custom title
    fig2_c, ax2_c, _ = bp.subcohort_contrast_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", title="Contraste Personalizado")
    assert get_any_title(ax2_c) == "Contraste Personalizado"
    plt.close(fig2_c)

    # 3. shared_factors_plot
    mask = df["Alim_Pescados"] >= 2
    # 3a. Default -> Tiene título
    fig3, ax3, _ = bp.shared_factors_plot(subcohort_mask=mask)
    assert get_any_title(ax3) != ""
    plt.close(fig3)

    # 3b. show_title=False -> Título vacío
    fig3_no, ax3_no, _ = bp.shared_factors_plot(subcohort_mask=mask, show_title=False)
    assert get_any_title(ax3_no) == ""
    plt.close(fig3_no)

    # 3c. title="" -> Título vacío
    fig3_empty, ax3_empty, _ = bp.shared_factors_plot(subcohort_mask=mask, title="")
    assert get_any_title(ax3_empty) == ""
    plt.close(fig3_empty)

    # 3d. Custom title
    fig3_c, ax3_c, _ = bp.shared_factors_plot(subcohort_mask=mask, title="Factores de Riesgo")
    assert get_any_title(ax3_c) == "Factores de Riesgo"
    plt.close(fig3_c)

    # 4. subcohort_mosaic_plot
    # 4a. Default -> Todos los paneles tienen título
    fig4, axes4, _ = bp.subcohort_mosaic_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados")
    for ax in axes4.values():
        assert get_any_title(ax) != ""
    plt.close(fig4)

    # 4b. show_titles=False -> Paneles sin título
    fig4_no, axes4_no, _ = bp.subcohort_mosaic_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", show_titles=False)
    for ax in axes4_no.values():
        assert get_any_title(ax) == ""
    plt.close(fig4_no)

    # 4c. title="" -> Paneles sin título
    fig4_empty, axes4_empty, _ = bp.subcohort_mosaic_plot(target_metal="Mercurio_ug_L", criterion_col="Alim_Pescados", title="")
    for ax in axes4_empty.values():
        assert get_any_title(ax) == ""
    plt.close(fig4_empty)

    # 4d. Suptitle y personalización individual de paneles
    fig4_sup, axes4_sup, _ = bp.subcohort_mosaic_plot(
        target_metal="Mercurio_ug_L",
        criterion_col="Alim_Pescados",
        title="Mosaico de Mercurio",
        title_contrast="Panel B Personalizado",
    )
    assert fig4_sup._suptitle is not None and fig4_sup._suptitle.get_text() == "Mosaico de Mercurio"
    assert get_any_title(axes4_sup["ax_contrast"]) == "Panel B Personalizado"
    plt.close(fig4_sup)

    plt.close("all")






