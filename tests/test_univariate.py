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
    summary = tables.metal_summary()
    assert summary is not None
    res_df = summary.to_dataframe()
    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) > 0
    assert "Mediana [RIQ]" in res_df.columns
    assert "Media (DE)" in res_df.columns
    assert "Media Geom. (GSD)" not in res_df.columns


def test_univariate_categorical_summary():
    df = hs.load_data()
    tables = UnivariateTables(df)
    cat_cols = tables.default_categorical_cols
    if cat_cols:
        summary = tables.categorical_summary(columns=cat_cols[:3])
        assert summary is not None
        assert isinstance(summary.to_dataframe(), pd.DataFrame)


def test_desglosar_multirrespuesta():
    s = pd.Series([
        "Agua filtrada; Agua mineral embotellada",
        "Agua filtrada",
        "Agua mineral embotellada",
        "Ninguno",
        None
    ])
    counts = hs.desglosar_multirrespuesta(s, sep=";", exclude_patterns=["ningun"])
    assert "Agua filtrada" in counts.index
    assert counts["Agua filtrada"] == 2
    assert counts["Agua mineral embotellada"] == 2
    assert "Ninguno" not in counts.index


def test_univariate_plots_new_functions(tmp_path):
    import matplotlib.pyplot as plt
    df_test = pd.DataFrame({
        "Mercurio_ug_L": [1.2, 2.5, 3.8, 5.5, 4.1],
        "Salud_Agua": [
            "Agua filtrada",
            "Agua mineral embotellada",
            "Agua filtrada; Agua mineral embotellada",
            "Agua de pozo profundo",
            "Agua filtrada"
        ],
        "Exposicion_Lugares": ["Canales", "Rios; Estacion de Gasolina", "Ninguno", "Canales", "Rios"],
        "Exposicion_Talleres": ["Taller Mecanico", "Carpinteria", "Ninguno", "Taller Mecanico", "Ninguno"],
        "Exposicion_Industrias": ["Fabrica Quimicos", "Ninguno", "Fabrica Metales", "Ninguno", "Fabrica Quimicos"],
        "Alim_Pescados": [0, 1, 2, 3, 2],
        "Salud_Bombillos": ["SI", "NO", "SI", "SI", "NO"],
        "Salud_Fuma": ["NO", "NO", "SI", "NO", "SI"],
        "Sexo": ["femenino", "masculino", "femenino", "masculino", "femenino"],
    })
    uplots = hs.UnivariatePlots(df_test, palette="crest")

    # 1. Test plot_multirrespuesta
    fig1 = uplots.plot_multirrespuesta(
        "Salud_Agua", 
        replace_map={"Agua filtrada; Agua mineral embotellada": "Mixta"},
        save_dir=str(tmp_path),
        save_format=["png", "pdf"]
    )
    assert fig1 is not None
    assert (tmp_path / "Salud_Agua_frecuencia.png").exists()
    assert (tmp_path / "Salud_Agua_frecuencia.pdf").exists()

    # 2. Test plot_factores
    fig2 = uplots.plot_factores(
        dimensions={
            "Exposicion_Lugares": "Lugares de Riesgo",
            "Exposicion_Talleres": "Talleres y Servicios",
            "Exposicion_Industrias": "Industrias Químicas/Metales"
        },
        save_dir=str(tmp_path)
    )
    assert fig2 is not None
    assert (tmp_path / "Factores_Exposicion_agrupado_elegante.png").exists()

    # 3. Test canvas y mosaico con plot_categorico usando índices de color
    fig3, axd = uplots.crear_mosaico(
        [["pescado", "pescado"], ["bombillos", "fumadores"]],
        figsize=(10.5, 7.2),
        bold=True
    )
    uplots.plot_categorico("Alim_Pescados", ax=axd["pescado"], title="Pescado", color=0)
    uplots.plot_categorico("Salud_Bombillos", ax=axd["bombillos"], title="Bombillos", color=1)
    uplots.plot_categorico("Salud_Fuma", ax=axd["fumadores"], title="Fumadores", color=2)
    assert fig3 is not None
    assert len(axd["pescado"].patches) > 0

    # 4. Test plot_multirrespuesta
    fig4 = uplots.plot_multirrespuesta(
        "Alim_Pescados",
        mapping={0: "Nunca", 1: "Rara vez", 2: "A veces", 3: "Frecuentemente"},
        order=["Nunca", "Rara vez", "A veces", "Frecuentemente"],
        title="Frecuencia de Consumo de Pescados y Mariscos",
        save_dir=str(tmp_path)
    )
    assert fig4 is not None

    # 5. Test plot_categorico with bold_annotations, label capitalization and multi-column color rotation
    figs = uplots.plot_categorico(["Salud_Bombillos", "Salud_Fuma"], bold_annotations=True)
    assert len(figs) == 2
    # Check that categories are capitalized (e.g. "Sí", "No")
    tick_labels = [t.get_text() for t in figs[0].axes[0].get_yticklabels()]
    assert all(label[0].isupper() for label in tick_labels if len(label) > 0)
    # Check that both plots have distinct colors
    color1 = figs[0].axes[0].patches[0].get_facecolor()
    color2 = figs[1].axes[0].patches[0].get_facecolor()
    assert color1 != color2

    # 6. Test plot_histograma with bold_annotations and log_scale
    figs_hist = uplots.plot_histograma("Mercurio_ug_L", log_scale=True, bold_annotations=True)
    assert len(figs_hist) == 1
    assert figs_hist[0].axes[0].get_xlabel().startswith("ln [")

    # 6b. Test plot_histograma with hue (stratified / stacked) and plot_histograma_estratificado
    figs_strat1 = uplots.plot_histograma(
        "Mercurio_ug_L",
        hue="Sexo",
        multiple="stack",
        shrink=0.8,
        kde=True,
        palette="Set1",
        save_dir=str(tmp_path),
    )
    assert len(figs_strat1) == 1
    assert (tmp_path / "Mercurio_ug_L_histograma_por_sexo.png").exists()
    assert figs_strat1[0].axes[0].get_legend() is not None

    figs_strat2 = uplots.plot_histograma_estratificado("Mercurio_ug_L", hue="Sexo")
    assert len(figs_strat2) == 1
    assert figs_strat2[0].axes[0].get_legend() is not None

    # 7. Test plot_qq with bold_annotations
    figs_qq = uplots.plot_qq("Mercurio_ug_L", bold_annotations=True)
    assert len(figs_qq) == 1

    # 8. Test plot_boxplot with bold_annotations and log_scale
    figs_box = uplots.plot_boxplot("Mercurio_ug_L", log_scale=True, bold_annotations=True)
    assert len(figs_box) == 1
    assert figs_box[0].axes[0].get_ylabel().startswith("ln [")

    # 9. Test plot_distribucion with bold_annotations and log_scale
    figs_comb = uplots.plot_distribucion("Mercurio_ug_L", log_scale=True, bold_annotations=True)
    assert len(figs_comb) == 1
    # Check that x-axis of bottom panel contains "ln ["
    assert any("ln [" in ax.get_xlabel() for ax in figs_comb[0].axes)

    # 10. Test subplots integration with axes (Cell 5 pattern)
    fig_sub, axes = plt.subplots(1, 2, figsize=(10, 4))
    uplots.plot_histograma("Mercurio_ug_L", ax=axes[0])
    uplots.plot_boxplot("Mercurio_ug_L", ax=axes[1])
    assert axes[0].get_xlabel() != ""
    assert axes[1].get_ylabel() != ""

    plt.close("all")


def test_univariate_plots_simplified_spanish_pipeline():
    df_test = pd.DataFrame({
        "Mercurio_ug_L": [1.5, 2.8, 3.2, 5.8, 4.4],
        "Sexo": ["M", "F", "M", "M", "F"],
        "Salud_Agua": ["Agua filtrada", "Agua mineral embotellada", "Agua de pozo", "Agua filtrada", "Agua mineral embotellada"],
        "Alim_Pescados": [0, 1, 2, 3, 1],
        "Salud_Bombillos": ["SI", "NO", "SI", "SI", "NO"],
        "Salud_Fuma": ["NO", "NO", "SI", "NO", "SI"],
        "Exposicion_Lugares": ["Canales", "Rios", "Ninguno", "Canales", "Rios"],
    })
    uplots = hs.UnivariatePlots(df_test)

    # 1. Pipeline Distribución (Boxplot + Histograma)
    figs_dist = uplots.plot_distribucion("Mercurio_ug_L", log_scale=True)
    assert len(figs_dist) == 1
    assert any("ln [" in ax.get_xlabel() for ax in figs_dist[0].axes)

    # 2. Pipeline Categórico (Barras)
    figs_cat = uplots.plot_categorico("Sexo")
    assert len(figs_cat) == 1

    # 3. Pipeline Multirrespuesta
    fig_multi = uplots.plot_multirrespuesta("Salud_Agua")
    assert fig_multi is not None

    # 4. Pipeline Factores de Exposición agrupados
    fig_fact = uplots.plot_factores_exposicion(dimensions={"Exposicion_Lugares": "Lugares de Riesgo"})
    assert fig_fact is not None

    # 5. Canvas y Mosaico
    fig_c, ax_c = uplots.canvas(nrows=1, ncols=2)
    assert fig_c is not None
    assert len(ax_c) == 2

    hs.UnivariatePlots.close_all()


def test_base_plots_canvas_engine_without_data():
    """Verifica que BasePlots funciona como motor de Canvas sin requerir datos obligatorios."""
    base = hs.BasePlots()
    assert base.df is None

    # 1. Crear lienzo atómico con canvas()
    fig, ax = base.canvas(figsize=(6, 4), bold=True)
    assert fig is not None
    assert ax is not None
    assert ax.spines["bottom"].get_edgecolor() == (15/255, 23/255, 42/255, 1.0) or ax.spines["bottom"].get_color() == "#0f172a"

    # 2. Crear mosaico con mosaic()
    fig_m, axd = base.mosaic([["p1", "p2"]], figsize=(8, 3))
    assert "p1" in axd and "p2" in axd
    base.close_all()


def test_plot_distribucion_percentile_mask():
    """Verifica la máscara de percentiles (<=P25, >P25 y <P75, >=P75) en plot_distribucion y obtener_mascara_percentiles."""
    df = hs.load_data()
    df_hg = hs.select_metal(df, concentration_col="hg")
    uplots = hs.UnivariatePlots(df=df_hg)

    # 1. Test visual pipeline con percentile_mask=True
    figs = uplots.plot_distribucion(
        columns=["Mercurio_ug_L"],
        percentile_mask=True,
        kde=False,
        overlay_points=True
    )
    assert len(figs) == 1
    ax_box, ax_hist = figs[0].axes[0], figs[0].axes[1]
    # Verificar que existen las líneas divisorias en ax_hist
    assert len(ax_hist.lines) >= 2

    # 2. Test del método estadístico obtener_mascara_percentiles
    res = uplots.obtener_mascara_percentiles("Mercurio_ug_L")
    assert "p_low" in res
    assert "p_high" in res
    assert res["n_total"] == 20
    assert res["estratos"]["<=P25"]["n"] == 5
    assert res["estratos"]["<=P25"]["pct"] == 25.0
    assert res["estratos"][">P25 y <P75"]["n"] == 10
    assert res["estratos"][">P25 y <P75"]["pct"] == 50.0
    assert res["estratos"][">=P75"]["n"] == 5
    assert res["estratos"][">=P75"]["pct"] == 25.0
    assert len(res["resumen"]) == 3

    hs.UnivariatePlots.close_all()



