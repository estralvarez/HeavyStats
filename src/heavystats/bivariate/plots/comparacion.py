"""
Módulo de comparación de variables cuantitativas vs cualitativas.
Implementa análisis comparativo de concentraciones séricas frente a factores
sociodemográficos, ambientales y variables dummy de exposición.
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from heavystats.univariate.constants import DEFAULT_LABELS_MAP
from heavystats.bivariate.tests import (
    mann_whitney_test,
    kruskal_wallis_test,
    independent_t_test,
    anova_oneway_test,
)
from heavystats.bivariate.plots.base import (
    BivariateBasePlots,
    DEFAULT_BIOMEDICAL_METAL_LABELS,
    DEFAULT_REFERENCE_LABELS,
)


class ComparacionPlotsMixin:
    """Mixin para contrastes entre variables cuantitativas y cualitativas."""

    def _get_dummy_category_label(self: BivariateBasePlots, col_name: str) -> str:
        """Obtiene la etiqueta limpia para una categoría derivada de una dummy."""
        if col_name in self.labels_map:
            return self.labels_map[col_name]
        if col_name in DEFAULT_LABELS_MAP:
            return DEFAULT_LABELS_MAP[col_name]
        for prefix in ("Exposicion_Lugares_", "Exposicion_Talleres_", "Exposicion_Industrias_", "Salud_"):
            if col_name.startswith(prefix):
                return col_name[len(prefix):].replace("_", " ").title()
        return col_name.replace("_", " ").title()

    def _build_dummy_grouping(
        self: BivariateBasePlots,
        dummy_cols: List[str],
        q_col: str,
        log_scale: bool = False,
        include_unexposed: bool = False,
        custom_label: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str, List[str], List[int], List[str], List[np.ndarray], str]:
        cat_data: Dict[str, np.ndarray] = {}

        for c in dummy_cols:
            if c not in self.df.columns:
                continue
            mask = (
                pd.to_numeric(self.df[c], errors="coerce").fillna(0).isin([1, 1.0])
                | self.df[c].astype(str).str.lower().isin(["1", "si", "sí", "true"])
            )
            vals = pd.to_numeric(self.df.loc[mask, q_col], errors="coerce").dropna()
            if log_scale:
                vals = vals[vals > 0]
            if len(vals) > 0:
                lbl = self._get_dummy_category_label(c)
                cat_data[lbl] = vals.values

        if include_unexposed and dummy_cols:
            present_cols = [c for c in dummy_cols if c in self.df.columns]
            if present_cols:
                unexp_mask = ~(self.df[present_cols].apply(pd.to_numeric, errors="coerce").fillna(0) == 1).any(axis=1)
                u_vals = pd.to_numeric(self.df.loc[unexp_mask, q_col], errors="coerce").dropna()
                if log_scale:
                    u_vals = u_vals[u_vals > 0]
                if len(u_vals) > 0:
                    cat_data["Sin Exposición"] = u_vals.values

        if not cat_data:
            raise ValueError(f"No hay observaciones positivas para las variables dummy especificadas en '{q_col}'.")

        records = []
        for cat_name, arr in cat_data.items():
            for v in arr:
                records.append({"__Grupo__": cat_name, q_col: float(v)})
        plot_df = pd.DataFrame(records)
        cats = list(cat_data.keys())
        counts = [len(cat_data[c]) for c in cats]
        xtick_labels = [f"{self._wrap_text(c, width=13)}\n(n={n})" for c, n in zip(cats, counts)]
        groups_vals = [cat_data[c] for c in cats]
        var_lbl = custom_label or "Categoría de Exposición"
        return plot_df, "__Grupo__", cats, counts, xtick_labels, groups_vals, var_lbl

    def _build_interaction_grouping(
        self: BivariateBasePlots,
        cols: List[str],
        q_col: str,
        log_scale: bool = False,
    ) -> Tuple[pd.DataFrame, str, List[str], List[int], List[str], List[np.ndarray], str]:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Las columnas '{missing}' no se encuentran en el DataFrame.")
        sub_df = self.df[cols + [q_col]].dropna().copy()
        sub_df[q_col] = pd.to_numeric(sub_df[q_col], errors="coerce")
        sub_df = sub_df.dropna()
        if log_scale:
            sub_df = sub_df[sub_df[q_col] > 0]
        if sub_df.empty:
            raise ValueError(f"No hay observaciones válidas para '{q_col}' agrupado por {cols}.")

        combined_name = " - ".join(cols)
        sub_df[combined_name] = sub_df[cols[0]].astype(str)
        for c in cols[1:]:
            sub_df[combined_name] = sub_df[combined_name] + " - " + sub_df[c].astype(str)

        cats = sorted(sub_df[combined_name].unique(), key=lambda x: str(x))
        counts = [int((sub_df[combined_name] == c).sum()) for c in cats]
        xtick_labels = [f"{self._wrap_text(str(c), width=13)}\n(n={n})" for c, n in zip(cats, counts)]
        groups_vals = [sub_df[sub_df[combined_name] == c][q_col].values for c in cats]
        var_lbl = " - ".join([self.get_label(c) for c in cols])
        return sub_df, combined_name, cats, counts, xtick_labels, groups_vals, var_lbl

    def _build_single_grouping(
        self: BivariateBasePlots,
        g_col: str,
        q_col: str,
        log_scale: bool = False,
    ) -> Tuple[pd.DataFrame, str, List[str], List[int], List[str], List[np.ndarray], str]:
        if g_col not in self.df.columns or q_col not in self.df.columns:
            raise ValueError(f"Las columnas '{g_col}' o '{q_col}' no se encuentran en el DataFrame.")
        sub_df = self.df[[g_col, q_col]].dropna().copy()
        sub_df[q_col] = pd.to_numeric(sub_df[q_col], errors="coerce")
        sub_df = sub_df.dropna()
        if log_scale:
            sub_df = sub_df[sub_df[q_col] > 0]
        if sub_df.empty:
            raise ValueError(f"No hay observaciones numéricas válidas para '{q_col}' agrupado por '{g_col}'.")

        cats = sorted(sub_df[g_col].unique(), key=lambda x: str(x))
        counts = [int((sub_df[g_col] == c).sum()) for c in cats]
        xtick_labels = [self._format_category_label(g_col, c, n_c) for c, n_c in zip(cats, counts)]
        groups_vals = [sub_df[sub_df[g_col] == c][q_col].values for c in cats]
        var_lbl = self.get_label(g_col)
        return sub_df, g_col, cats, counts, xtick_labels, groups_vals, var_lbl

    def _prepare_grouping(
        self: BivariateBasePlots,
        g_name: Union[str, Sequence[str]],
        q_col: str,
        log_scale: bool = False,
        include_unexposed: bool = False,
    ) -> Tuple[pd.DataFrame, str, List[str], List[int], List[str], List[np.ndarray], str]:
        DIMENSION_MAP = {
            "talleres": "Exposicion_Talleres",
            "taller": "Exposicion_Talleres",
            "exposicion_talleres": "Exposicion_Talleres",
            "lugares": "Exposicion_Lugares",
            "lugar": "Exposicion_Lugares",
            "exposicion_lugares": "Exposicion_Lugares",
            "industrias": "Exposicion_Industrias",
            "industria": "Exposicion_Industrias",
            "exposicion_industrias": "Exposicion_Industrias",
            "todas_las_dummies": "ALL_DUMMIES",
            "all_dummies": "ALL_DUMMIES",
            "all": "ALL_DUMMIES",
            "todos": "ALL_DUMMIES",
            "todas": "ALL_DUMMIES",
            "todo": "ALL_DUMMIES",
            "factores_ambientales": "ALL_DUMMIES",
            "exposiciones": "ALL_DUMMIES",
            "ambientales": "ALL_DUMMIES",
            "cualquier_exposicion": "COMPOSITE_DUMMIES",
            "exposicion_cualquiera": "COMPOSITE_DUMMIES",
        }

        # Caso de secuencia o lista de variables
        if isinstance(g_name, (list, tuple)):
            expanded_cols = []
            for item in g_name:
                item_key = str(item).lower().strip()
                if item_key in DIMENSION_MAP:
                    dim = DIMENSION_MAP[item_key]
                    if dim == "Exposicion_Talleres":
                        expanded_cols.extend([c for c in self.df.columns if c.startswith("Exposicion_Talleres_")])
                    elif dim == "Exposicion_Lugares":
                        expanded_cols.extend([c for c in self.df.columns if c.startswith("Exposicion_Lugares_")])
                    elif dim == "Exposicion_Industrias":
                        expanded_cols.extend([c for c in self.df.columns if c.startswith("Exposicion_Industrias_")])
                    elif dim == "ALL_DUMMIES":
                        expanded_cols.extend([c for c in self.df.columns if c.startswith(("Exposicion_Talleres_", "Exposicion_Lugares_", "Exposicion_Industrias_"))])
                    elif dim == "COMPOSITE_DUMMIES":
                        expanded_cols.extend([c for c in ["Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Lugar_Riesgo", "Exposicion_Cualquier_Industria"] if c in self.df.columns])
                else:
                    resolved = self._resolve_column(str(item))
                    expanded_cols.append(resolved)

            cols = list(dict.fromkeys(expanded_cols))

            def _is_binary_col(c: str) -> bool:
                if c not in self.df.columns:
                    return False
                u_vals = set(self.df[c].dropna().unique())
                return u_vals.issubset({0, 1, 0.0, 1.0, "0", "1", True, False, "SI", "NO", "Si", "No", "si", "no"})

            all_dummy = len(cols) > 0 and all(_is_binary_col(c) for c in cols)
            if all_dummy:
                return self._build_dummy_grouping(cols, q_col, log_scale=log_scale, include_unexposed=include_unexposed)
            else:
                return self._build_interaction_grouping(cols, q_col, log_scale=log_scale)

        # Caso de un solo string
        g_str = str(g_name).strip()
        g_lower = g_str.lower()

        if g_lower in DIMENSION_MAP:
            dim = DIMENSION_MAP[g_lower]
            if dim == "Exposicion_Talleres":
                cols = [c for c in self.df.columns if c.startswith("Exposicion_Talleres_")]
                var_lbl = "Talleres y Servicios"
            elif dim == "Exposicion_Lugares":
                cols = [c for c in self.df.columns if c.startswith("Exposicion_Lugares_")]
                var_lbl = "Lugares de Riesgo"
            elif dim == "Exposicion_Industrias":
                cols = [c for c in self.df.columns if c.startswith("Exposicion_Industrias_")]
                var_lbl = "Industrias Químicas/Metales"
            elif dim == "ALL_DUMMIES":
                cols = [c for c in self.df.columns if c.startswith(("Exposicion_Talleres_", "Exposicion_Lugares_", "Exposicion_Industrias_"))]
                var_lbl = "Factores de Exposición Ambiental"
            elif dim == "COMPOSITE_DUMMIES":
                cols = [c for c in ["Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Lugar_Riesgo", "Exposicion_Cualquier_Industria"] if c in self.df.columns]
                var_lbl = "Indicadores Compuestos de Exposición"
            return self._build_dummy_grouping(cols, q_col, log_scale=log_scale, include_unexposed=include_unexposed, custom_label=var_lbl)

        resolved_col = self._resolve_column(g_str)
        dummy_matches = [c for c in self.df.columns if c.startswith(f"{resolved_col}_")]
        if dummy_matches:
            return self._build_dummy_grouping(dummy_matches, q_col, log_scale=log_scale, include_unexposed=include_unexposed)

        return self._build_single_grouping(resolved_col, q_col, log_scale=log_scale)

    def compare_groups(
        self: BivariateBasePlots,
        quantitative: Optional[Union[str, Sequence[str]]] = None,
        group: Optional[Union[str, Sequence[str]]] = None,
        method: str = "nonparametric",
        continuous_col: Optional[str] = None,
        group_col: Optional[Union[str, Sequence[str]]] = None,
        metal: Optional[str] = None,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None,
        include_unexposed: bool = False,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 1: Comparación de concentraciones o variables continuas entre categorías de un factor cualitativo,
        múltiples columnas (interacción) o dimensiones de variables dummy de exposición.
        Soporta modo no paramétrico (Mann-Whitney U / Kruskal-Wallis con medianas [RIQ] y stripplot)
        y modo paramétrico (t de Welch / ANOVA con medias ± DE, Hedges' g / η² y marcadores de media).
        """
        q_name = quantitative or continuous_col or metal or "Mercurio_ug_L"
        g_name = group or group_col or "Es_Expuesto"

        q_col = self._resolve_column(q_name)
        plot_df, x_col, cats, counts, xtick_labels, groups_vals, var_lbl = self._prepare_grouping(
            g_name=g_name, q_col=q_col, log_scale=log_scale, include_unexposed=include_unexposed
        )

        calc_figsize = figsize
        if calc_figsize is None:
            fig_w = max(6.5, len(cats) * 1.55) if len(cats) >= 5 else 6.5
            calc_figsize = (fig_w, 5.0)

        fig, ax = plt.subplots(figsize=calc_figsize)

        is_param = method.lower().startswith("param")
        active_palette = palette or (["#0284c7", "#0f766e"] if len(cats) == 2 else self.palette)

        if is_param:
            sns.boxplot(
                data=plot_df,
                x=x_col,
                y=q_col,
                hue=x_col,
                legend=False,
                order=cats,
                palette=active_palette,
                ax=ax,
                width=0.42,
                showmeans=True,
                meanprops=dict(marker="D", markeredgecolor="#0f172a", markerfacecolor="#e11d48", markersize=6.5),
                boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.2),
                medianprops=dict(color="#0f172a", linewidth=1.8),
                whiskerprops=dict(color="#475569", linewidth=1.2),
                capprops=dict(color="#475569", linewidth=1.2),
                showfliers=False,
            )
        else:
            sns.boxplot(
                data=plot_df,
                x=x_col,
                y=q_col,
                hue=x_col,
                legend=False,
                order=cats,
                palette=active_palette,
                ax=ax,
                width=0.40,
                boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.2),
                medianprops=dict(color="#0f172a", linewidth=2.2),
                whiskerprops=dict(color="#475569", linewidth=1.2),
                capprops=dict(color="#475569", linewidth=1.2),
                showfliers=False,
            )

        if show_points:
            sns.stripplot(
                data=plot_df,
                x=x_col,
                y=q_col,
                order=cats,
                color="#b91c1c",
                alpha=0.90,
                size=6.5,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.6,
                ax=ax,
            )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        q_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(q_col, self.get_label(q_col))
        if log_scale:
            q_axis_lbl += " (Escala Log)"

        if title is not None:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel(var_lbl, fontsize=10.5, fontweight="bold", labelpad=8)
        ax.set_ylabel(q_axis_lbl, fontsize=10.5, fontweight="bold", labelpad=8)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(xtick_labels, fontsize=9.0)
        self._clean_spines_and_ticks(ax)

        if show_stats and len(cats) >= 2:
            y_max_data = float(plot_df[q_col].max()) if len(plot_df) > 0 else 1.0
            y_min_data = float(plot_df[q_col].min()) if len(plot_df) > 0 else 0.0
            y_span = (y_max_data - y_min_data) if y_max_data > y_min_data else 1.0

            if log_scale and y_max_data > 0:
                y_bracket = y_max_data * 1.15
                h_bracket = y_bracket * 0.04
                y_top_limit = y_bracket * 1.35
            else:
                y_bracket = y_max_data + 0.08 * y_span
                h_bracket = 0.03 * y_span
                y_top_limit = y_bracket + 0.12 * y_span

            if is_param:
                if len(cats) == 2:
                    g1 = groups_vals[0]
                    g2 = groups_vals[1]
                    tt = independent_t_test(g1, g2, equal_var=False)
                    p_val = tt.get("p_val", np.nan)
                    g_hedges = tt.get("hedges_g", np.nan)
                    diff_m = tt.get("diff_means", np.nan)

                    p_str = self._format_p_val(p_val)
                    g_str = rf"$g = {g_hedges:+.2f}$" if pd.notna(g_hedges) else ""
                    diff_str = rf"$\Delta\mathrm{{Media}} = {diff_m:+.2f}$" if pd.notna(diff_m) else ""

                    stat_parts = [f"Welch t: {p_str}"]
                    if g_str:
                        stat_parts.append(g_str)
                    if diff_str:
                        stat_parts.append(diff_str)
                    stat_text = "  |  ".join(stat_parts)

                    ax.plot([0, 0, 1, 1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                    ax.text(
                        0.5,
                        y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                        stat_text,
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#0f172a",
                    )
                else:
                    an = anova_oneway_test(groups_vals, group_names=[str(c) for c in cats], equal_var=False)
                    p_val = an.get("p_val", np.nan)
                    f_stat = an.get("f_stat", np.nan)
                    eta_sq = an.get("eta_squared", np.nan)

                    p_str = self._format_p_val(p_val)
                    f_str = rf"$F = {f_stat:.2f}$" if pd.notna(f_stat) else "F = N/D"
                    if pd.notna(eta_sq):
                        eta_str = r"$\eta^2 < 0.01$" if eta_sq < 0.01 else rf"$\eta^2 = {eta_sq:.2f}$"
                    else:
                        eta_str = ""

                    stat_parts = [f"ANOVA: {p_str}", f_str]
                    if eta_str:
                        stat_parts.append(eta_str)
                    stat_text = "  |  ".join(stat_parts)

                    ax.plot([0, 0, len(cats)-1, len(cats)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                    ax.text(
                        (len(cats) - 1) * 0.5,
                        y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                        stat_text,
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#0f172a",
                    )
            else:
                if len(cats) == 2:
                    g1 = groups_vals[0]
                    g2 = groups_vals[1]
                    mw = mann_whitney_test(g1, g2)
                    p_val = mw.get("p_val", mw.get("p_value", np.nan))
                    r_rb = mw.get("r_rb", mw.get("rank_biserial", np.nan))
                    diff_med = mw.get("diff_medians", np.nan)

                    p_str = self._format_p_val(p_val)
                    r_str = rf"$r_{{\mathrm{{rb}}}} = {r_rb:+.2f}$" if pd.notna(r_rb) else ""
                    diff_str = rf"$\Delta\mathrm{{Med}} = {diff_med:+.2f}$" if pd.notna(diff_med) else ""

                    stat_parts = [f"Mann–Whitney U: {p_str}"]
                    if r_str:
                        stat_parts.append(r_str)
                    if diff_str:
                        stat_parts.append(diff_str)
                    stat_text = "  |  ".join(stat_parts)

                    ax.plot([0, 0, 1, 1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                    ax.text(
                        0.5,
                        y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                        stat_text,
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#0f172a",
                    )
                else:
                    kw = kruskal_wallis_test(groups_vals)
                    p_val = kw.get("p_val", kw.get("p_value", np.nan))
                    h_stat = kw.get("h_stat", kw.get("h_statistic", np.nan))
                    eps_sq = kw.get("epsilon_sq", kw.get("epsilon_squared", np.nan))

                    p_str = self._format_p_val(p_val)
                    h_str = rf"$H = {h_stat:.2f}$" if pd.notna(h_stat) else "H = N/D"
                    if pd.notna(eps_sq):
                        eps_str = r"$\epsilon^2 < 0.01$" if eps_sq < 0.01 else rf"$\epsilon^2 = {eps_sq:.2f}$"
                    else:
                        eps_str = ""

                    stat_parts = [f"Kruskal–Wallis: {p_str}", h_str]
                    if eps_str:
                        stat_parts.append(eps_str)
                    stat_text = "  |  ".join(stat_parts)

                    ax.plot([0, 0, len(cats)-1, len(cats)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                    ax.text(
                        (len(cats) - 1) * 0.5,
                        y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                        stat_text,
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#0f172a",
                    )

            ax.set_ylim(top=y_top_limit)

        self._save_figure(fig, filepath=filepath)
        return fig, ax

    # Aliases de compatibilidad metodológica
    plot_compare_groups = compare_groups
    plot_categorical = compare_groups
    plot_categorico = compare_groups

    def plot_binary(
        self: BivariateBasePlots,
        quantitative: Optional[Union[str, Sequence[str]]] = None,
        group: Optional[Union[str, Sequence[str]]] = None,
        method: str = "nonparametric",
        continuous_col: Optional[str] = None,
        group_col: Optional[Union[str, Sequence[str]]] = None,
        metal: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Gráfico de cajas comparativo para variables dicotómicas/binarias (2 grupos independientes).
        Diseñado estrictamente para contraste de grupos sin capacidad de mostrar líneas de límite toxicológico.
        """
        q = quantitative or continuous_col or metal
        g = group or group_col
        return self.compare_groups(
            quantitative=q,
            group=g,
            method=method,
            **kwargs,
        )

    def plot_dummies(
        self: BivariateBasePlots,
        metal: str,
        dimension: str = "all",
        dummy_cols: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Diagramas de cajas comparativos a partir de variables dummy de exposición ambiental."""
        group = dummy_cols if dummy_cols is not None else dimension
        return self.compare_groups(quantitative=metal, group=group, method=method, **kwargs)

    def metal_by_group(
        self: BivariateBasePlots,
        group_col: Union[str, Sequence[str]],
        metal: str,
        method: str = "nonparametric",
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper de compatibilidad para compare_groups()."""
        return self.compare_groups(quantitative=metal, group=group_col, method=method, **kwargs)
