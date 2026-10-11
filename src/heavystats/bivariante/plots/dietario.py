"""
Módulo de patrones dietarios y tendencias ordinales.
Implementa tendencias ordinales (Jonckheere-Terpstra), radar charts multivariantes
en coordenadas polares y mosaicos de hábitos alimenticios.
"""

import math
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from heavystats.bivariante.constants import (
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)
from heavystats.bivariante.tests import (
    jonckheere_terpstra_test,
)
from heavystats.bivariante.plots.base import (
    BivariateBasePlots,
    DEFAULT_BIOMEDICAL_METAL_LABELS,
    DEFAULT_REFERENCE_LABELS,
)


class DietarioPlots:
    """Clase base para análisis de patrones dietarios y tendencias ordinales."""

    def ordinal_trend_plot(
        self: BivariateBasePlots,
        ordinal_col: str,
        metal: Optional[str] = None,
        target_metal: Optional[str] = None,
        method: str = "nonparametric",
        ordinal_map: Optional[Dict[str, int]] = None,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = False,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 4.8),
        filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Gráfico de tendencia ordinal (boxplots ordenados de menor a mayor frecuencia)
        para evaluar gradientes de exposición dietaria frente a biomarcadores.

        Parameters
        ----------
        ordinal_col : str
            Variable de consumo o escala ordinal (ej. 'Alim_Carnes').
        metal : str, optional
            Biomarcador de concentración ('Plomo_ug_dL', 'Mercurio_ug_L', 'Cadmio_ug_L', etc.).
        target_metal : str, optional
            Alias de metal.
        method : str, default="nonparametric"
            Método inferencial de tendencia:
            - 'nonparametric' / 'jonckheere': Prueba de Jonckheere–Terpstra (z, p-valor).
            - 'parametric' / 'pearson': Correlación lineal de Pearson (r, p-valor).
            - 'anova': Análisis de varianza (F, p-valor).
        """
        ord_resolved = self._resolve_column(ordinal_col)
        target_m = (
            target_metal
            or metal
            or kwargs.get("target")
            or kwargs.get("target_col")
            or kwargs.get("col_metal")
            or getattr(self, "default_metal", "Mercurio_ug_L")
        )
        metal_resolved = self._resolve_column(target_m)

        if ord_resolved not in self.df.columns or metal_resolved not in self.df.columns:
            raise ValueError(f"Las columnas '{ord_resolved}' o '{metal_resolved}' no se encuentran en el DataFrame.")

        sub_df = self.df[[ord_resolved, metal_resolved]].dropna().copy()
        sub_df[metal_resolved] = pd.to_numeric(sub_df[metal_resolved], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[sub_df[metal_resolved] > 0]

        if ordinal_map is None and (ord_resolved.startswith("Alim_") or ord_resolved.startswith("Consumo_")):
            ordinal_map = DIET_ORDINAL_MAP

        if ordinal_map is not None:
            sub_df["_ord_val"] = sub_df[ord_resolved].apply(
                lambda v: ordinal_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
            sub_df["_ord_val"] = pd.to_numeric(sub_df["_ord_val"], errors="coerce")
            sub_df = sub_df.dropna(subset=["_ord_val"])
            ordered_levels = sorted(sub_df["_ord_val"].unique())
            level_labels = [
                f"{DIET_ORDINAL_LABELS.get(int(lvl), str(lvl))} (n={(sub_df['_ord_val'] == lvl).sum()})"
                for lvl in ordered_levels
            ]
        else:
            ordered_levels = sorted(sub_df[ord_resolved].unique(), key=lambda x: str(x))
            level_labels = [
                f"{str(lvl).title()} (n={(sub_df[ord_resolved] == lvl).sum()})"
                for lvl in ordered_levels
            ]

        fig, ax = plt.subplots(figsize=figsize)
        plot_x_col = "_ord_val" if ordinal_map is not None else ord_resolved
        active_palette = palette or self.palette

        sns.boxplot(
            data=sub_df,
            x=plot_x_col,
            y=metal_resolved,
            hue=plot_x_col,
            legend=False,
            order=ordered_levels,
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
                data=sub_df,
                x=plot_x_col,
                y=metal_resolved,
                order=ordered_levels,
                color="#b91c1c",
                alpha=0.90,
                size=6.0,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.6,
                ax=ax,
            )

        limit_val = self._resolve_limit(metal_resolved, permissible_limit)
        if show_limit and limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
            ref_text = limit_label or DEFAULT_REFERENCE_LABELS.get(metal_resolved, f"Ref: {limit_val:g}")
            ax.text(
                len(ordered_levels) - 0.52,
                limit_val,
                f"{ref_text} ",
                color="#b91c1c",
                va="bottom",
                ha="right",
                fontsize=8.0,
                fontweight="bold",
            )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        var_lbl = self.get_label(ord_resolved)
        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_resolved, self.get_label(metal_resolved))
        if log_scale:
            metal_axis_lbl += " (Escala Log)"

        is_parametric = self._is_parametric(method, **kwargs)
        effective_m = str(kwargs.get("metodo") or kwargs.get("tipo_analisis") or method).lower()
        is_anova = "anova" in effective_m or effective_m == "f"

        if title is not None:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
        else:
            test_tag = "ANOVA" if is_anova else ("Pearson" if is_parametric else "Jonckheere–Terpstra")
            ax.set_title(f"Gradiente de Ingesta: {var_lbl} vs {metal_axis_lbl} ({test_tag})", fontsize=11, fontweight="bold", pad=14)

        ax.set_xlabel(f"Gradiente Ordinal de {var_lbl}", fontsize=10.5, fontweight="bold", labelpad=8)
        ax.set_ylabel(metal_axis_lbl, fontsize=10.5, fontweight="bold", labelpad=8)

        ax.set_xticks(range(len(ordered_levels)))
        ax.set_xticklabels(level_labels, fontsize=9.0)
        self._clean_spines_and_ticks(ax)

        if show_stats and len(ordered_levels) >= 2:
            y_max_data = float(sub_df[metal_resolved].max()) if len(sub_df) > 0 else 1.0
            y_min_data = float(sub_df[metal_resolved].min()) if len(sub_df) > 0 else 0.0
            y_span = (y_max_data - y_min_data) if y_max_data > y_min_data else 1.0

            y_bracket = y_max_data + 0.08 * y_span
            h_bracket = 0.03 * y_span
            y_top_limit = y_bracket + 0.12 * y_span

            groups_vals = [sub_df[sub_df[plot_x_col] == lvl][metal_resolved].values for lvl in ordered_levels]

            if is_parametric:
                if is_anova:
                    f_stat, p_val = stats.f_oneway(*groups_vals)
                    f_val = float(f_stat) if pd.notna(f_stat) else 0.0
                    p_val = float(p_val) if pd.notna(p_val) else 1.0
                    p_str = self._format_p_val(p_val, threshold=0.0001, precision=4)
                    f_str = f"$F = {f_val:.2f}$"
                    stat_text = f"Gradiente Paramétrico (ANOVA): {f_str}  |  {p_str}"
                else:
                    x_vals = pd.to_numeric(sub_df[plot_x_col], errors="coerce")
                    y_vals = pd.to_numeric(sub_df[metal_resolved], errors="coerce")
                    valid_xy = x_vals.notna() & y_vals.notna()
                    if valid_xy.sum() >= 3:
                        r_val, p_val = stats.pearsonr(x_vals[valid_xy], y_vals[valid_xy])
                    else:
                        r_val, p_val = 0.0, 1.0
                    p_str = self._format_p_val(p_val, threshold=0.0001, precision=4)
                    r_str = f"$r = {r_val:+.2f}$"
                    stat_text = f"Gradiente Lineal (Pearson): {r_str}  |  {p_str}"
            else:
                jt = jonckheere_terpstra_test(groups_vals)
                p_val = jt.get("p_val", jt.get("p_value", np.nan))
                z_score = jt.get("z_stat", jt.get("z_score", np.nan))
                p_str = self._format_p_val(p_val, threshold=0.0001, precision=4)
                z_str = rf"$z = {z_score:+.2f}$" if pd.notna(z_score) else ""
                stat_text = f"Tendencia Monótona (Jonckheere–Terpstra): {z_str}  |  {p_str}"

            ax.plot([0, 0, len(ordered_levels)-1, len(ordered_levels)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
            ax.text(
                (len(ordered_levels) - 1) * 0.5,
                y_bracket + 0.01 * y_span,
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

    plot_dietary = ordinal_trend_plot

    def plot_diet_radar(
        self: BivariateBasePlots,
        metal: Optional[str] = None,
        target_metal: Optional[str] = None,
        dietary_cols: Optional[Sequence[str]] = None,
        stratify_by: str = "median",
        split_panels: bool = False,
        palette: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
        """
        Genera un perfil dietario multivariante en coordenadas polares (radar / spider chart)
        utilizando ax.set_thetagrids() para proyectar los grupos de alimentos y contrastar
        la ingesta media según estratos de concentración del metal pesado.
        """
        # 1. Validación y resolución de columna de metal
        target_m = (
            target_metal
            or metal
            or kwargs.get("target")
            or kwargs.get("col_metal")
            or getattr(self, "default_metal", "Mercurio_ug_L")
        )
        metal_resolved = self._resolve_column(target_m)
        if metal_resolved not in self.df.columns:
            matched = [c for c in self.df.columns if str(target_m).strip().lower() in c.lower()]
            if matched:
                metal_resolved = matched[0]
            else:
                raise KeyError(f"La columna de metal '{target_m}' no se encuentra en el DataFrame.")
        metal = metal_resolved

        # 2. Selección de columnas dietarias
        if dietary_cols is None:
            dietary_cols = [c for c in self.df.columns if c.startswith("Alim_")]
        if not dietary_cols:
            raise ValueError("No se encontraron columnas dietarias (con prefijo 'Alim_') en el DataFrame.")

        labels = [c.replace("Alim_", "").replace("_", " ").title() for c in dietary_cols]
        num_vars = len(labels)

        # 3. Ángulos para cada eje polar
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        angles_deg = np.rad2deg(angles)

        # 4. Estratificación según la carga corporal del metal
        clean_df = self.df.dropna(subset=[metal] + list(dietary_cols))
        if clean_df.empty:
            raise ValueError("No existen observaciones válidas después de descartar valores nulos.")

        metal_vals = clean_df[metal]
        metal_base = metal.split("_")[0]
        unit = "µg/dL" if "dl" in metal.lower() else "µg/L"
        metal_unit = unit

        groups = []
        if stratify_by == "median":
            med_val = float(metal_vals.median())
            low_mask = metal_vals < med_val
            high_mask = metal_vals >= med_val
            if palette and len(palette) >= 2:
                col_low, col_high = palette[0], palette[1]
            elif self.palette and isinstance(self.palette, str):
                pal_cols = sns.color_palette(self.palette, 2)
                col_low, col_high = pal_cols[0], pal_cols[1]
            elif self.palette and isinstance(self.palette, (list, tuple)) and len(self.palette) >= 2:
                col_low, col_high = self.palette[0], self.palette[1]
            else:
                col_low, col_high = "#0f766e", "#b91c1c"
            groups.append({
                "label": rf"< Mediana ({med_val:.2f} {unit}, n={low_mask.sum()})",
                "means": clean_df.loc[low_mask, dietary_cols].mean().values,
                "color": col_low,
                "linestyle": "--" if not split_panels else "-",
                "marker": "s",
            })
            groups.append({
                "label": rf"$\geq$ Mediana ({med_val:.2f} {unit}, n={high_mask.sum()})",
                "means": clean_df.loc[high_mask, dietary_cols].mean().values,
                "color": col_high,
                "linestyle": "-",
                "marker": "o",
            })
        elif stratify_by == "terciles":
            t1, t2 = float(metal_vals.quantile(1/3)), float(metal_vals.quantile(2/3))
            low_mask = metal_vals <= t1
            mid_mask = (metal_vals > t1) & (metal_vals <= t2)
            high_mask = metal_vals > t2
            if palette and len(palette) >= 3:
                cols = palette
            elif self.palette and isinstance(self.palette, str):
                cols = sns.color_palette(self.palette, 3)
            elif self.palette and isinstance(self.palette, (list, tuple)) and len(self.palette) >= 3:
                cols = self.palette
            else:
                cols = ["#0f766e", "#d97706", "#b91c1c"]
            groups.append({
                "label": rf"T1 Bajo ($\leq$ {t1:.2f} {unit}, n={low_mask.sum()})",
                "means": clean_df.loc[low_mask, dietary_cols].mean().values,
                "color": cols[0],
                "linestyle": ":" if not split_panels else "-",
                "marker": "^",
            })
            groups.append({
                "label": rf"T2 Medio ({t1:.2f}–{t2:.2f} {unit}, n={mid_mask.sum()})",
                "means": clean_df.loc[mid_mask, dietary_cols].mean().values,
                "color": cols[1],
                "linestyle": "--" if not split_panels else "-",
                "marker": "s",
            })
            groups.append({
                "label": rf"T3 Alto (> {t2:.2f} {unit}, n={high_mask.sum()})",
                "means": clean_df.loc[high_mask, dietary_cols].mean().values,
                "color": cols[2],
                "linestyle": "-",
                "marker": "o",
            })
        else:
            raise ValueError(f"Estratificación '{stratify_by}' no soportada. Use 'median' o 'terciles'.")

        # 5. Creación y Renderizado de Figura Polar (Split vs Single)
        if split_panels:
            calc_figsize = figsize if figsize is not None else (7.5 * len(groups), 7.5)
            fig, axes_raw = plt.subplots(
                1, len(groups),
                figsize=calc_figsize,
                subplot_kw=dict(polar=True),
                dpi=kwargs.get("dpi", 200),
            )
            axes_list = [axes_raw] if len(groups) == 1 else list(axes_raw)
            fig.subplots_adjust(wspace=0.35)

            for i, (ax, grp) in enumerate(zip(axes_list, groups)):
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)

                ax.set_thetagrids(angles_deg, labels, fontsize=9.0, fontweight="bold", color="#0f172a")
                ax.tick_params(axis="x", pad=14)

                ax.set_ylim(0, 4.35)
                ax.set_yticks([1, 2, 3, 4])
                ax.set_yticklabels(["Rara vez (1)", "A veces (2)", "Frecuente (3)", "Diario (4)"], fontsize=7.5, color="#475569")
                ax.set_rlabel_position(345)
                for tick_lbl in ax.get_yticklabels():
                    tick_lbl.set_bbox(dict(facecolor="#ffffff", edgecolor="none", alpha=0.8, pad=1.2))

                ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.8, alpha=0.8)
                ax.spines["polar"].set_color("#94a3b8")
                ax.spines["polar"].set_linewidth(1.0)

                vals = grp["means"]
                vals_closed = np.concatenate([vals, [vals[0]]])
                ax.plot(
                    angles_closed,
                    vals_closed,
                    color=grp["color"],
                    linewidth=2.4,
                    linestyle=grp["linestyle"],
                    marker=grp["marker"],
                    markersize=6.0,
                    zorder=3,
                )
                ax.fill(angles_closed, vals_closed, color=grp["color"], alpha=0.22, zorder=2)

                panel_letter = chr(ord("A") + i)
                panel_title = f"{panel_letter}. {metal_base} {grp['label']}"
                ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=22, color="#0f172a")

            if title is not None:
                fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03)

            return_ax = axes_list
        else:
            calc_figsize = figsize if figsize is not None else (7.5, 7.5)
            fig, ax = plt.subplots(figsize=calc_figsize, subplot_kw=dict(polar=True), dpi=kwargs.get("dpi", 200))

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            ax.set_thetagrids(angles_deg, labels, fontsize=9.5, fontweight="bold", color="#0f172a")
            ax.tick_params(axis="x", pad=16)

            ax.set_ylim(0, 4.35)
            ax.set_yticks([1, 2, 3, 4])
            ax.set_yticklabels(["Rara vez (1)", "A veces (2)", "Frecuente (3)", "Diario (4)"], fontsize=8, color="#475569")

            ax.set_rlabel_position(345)
            for tick_lbl in ax.get_yticklabels():
                tick_lbl.set_bbox(dict(facecolor="#ffffff", edgecolor="none", alpha=0.75, pad=1.5))

            ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.8, alpha=0.8)
            ax.spines["polar"].set_color("#94a3b8")
            ax.spines["polar"].set_linewidth(1.0)

            for grp in groups:
                vals = grp["means"]
                vals_closed = np.concatenate([vals, [vals[0]]])
                ax.plot(
                    angles_closed,
                    vals_closed,
                    color=grp["color"],
                    linewidth=2.2,
                    linestyle=grp["linestyle"],
                    marker=grp["marker"],
                    markersize=5.5,
                    label=grp["label"],
                    zorder=3,
                )
                ax.fill(angles_closed, vals_closed, color=grp["color"], alpha=0.18, zorder=2)

            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=len(groups),
                frameon=True,
                facecolor="#f8fafc",
                edgecolor="#cbd5e1",
                framealpha=0.95,
                fontsize=9,
                title=f"Concentracion de {metal_base} ({metal_unit})",
                title_fontsize=9.5,
            )

            if title is not None:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=24)

            return_ax = ax

        self._save_figure(fig, filepath=filepath)
        return fig, return_ax

    def plot_diet_boxplots(
        self: BivariateBasePlots,
        metal: Optional[str] = None,
        target_metal: Optional[str] = None,
        dietary_cols: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        layout: str = "grid",
        ncols: int = 4,
        palette: Optional[Union[str, Sequence[str]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, Any]:
        """
        Mosaico integrado editorial de boxplots para todos los hábitos alimenticios
        vs la concentración del metal pesado, con prueba inferencial seleccionable.

        Parameters
        ----------
        metal : str, optional
            Biomarcador o metal a evaluar.
        target_metal : str, optional
            Alias de metal.
        dietary_cols : Sequence[str], optional
            Columnas de dieta (por defecto las que inician con 'Alim_').
        method : str, default="nonparametric"
            Método inferencial de contraste:
            - 'nonparametric' / 'jonckheere': Tendencia ordinal de Jonckheere–Terpstra (z, p-valor).
            - 'parametric' / 'pearson': Correlación lineal de Pearson (r, p-valor).
            - 'anova': Análisis de varianza (F, p-valor).
        layout : str, default="grid"
            Distribución visual (modo facetado 'grid'). El modo 'consolidated' ha sido eliminado.
        ncols : int, default=4
            Número de columnas en la cuadrícula.
        palette : str or list, optional
            Paleta de colores.
        figsize : tuple, optional
            Dimensiones de la figura.
        filepath : str, optional
            Ruta para guardar la imagen.
        """
        if str(layout).lower() == "consolidated":
            raise ValueError(
                "El layout 'consolidated' ha sido eliminado de la librería bivariante. "
                "Utilice el layout 'grid' para el mosaico multivariable facetado."
            )
        if str(layout).lower() != "grid":
            raise ValueError(f"Layout '{layout}' no reconocido. Opciones disponibles: 'grid'.")

        target_m = (
            target_metal
            or metal
            or kwargs.get("target")
            or kwargs.get("target_col")
            or kwargs.get("col_metal")
            or getattr(self, "default_metal", "Mercurio_ug_L")
        )
        metal_resolved = self._resolve_column(target_m)
        if metal_resolved not in self.df.columns:
            raise KeyError(f"La columna de metal '{target_m}' no se encuentra en el DataFrame.")

        if dietary_cols is None:
            dietary_cols = [c for c in self.df.columns if c.startswith("Alim_")]
        if not dietary_cols:
            raise ValueError("No se encontraron columnas dietarias (con prefijo 'Alim_') en el DataFrame.")

        clean_df = self.df.dropna(subset=[metal_resolved]).copy()
        clean_df[metal_resolved] = pd.to_numeric(clean_df[metal_resolved], errors="coerce")
        clean_df = clean_df.dropna(subset=[metal_resolved])

        compact_labels = {0: "Nunca", 1: "Rara", 2: "A veces", 3: "Frec.", 4: "Diario"}
        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_resolved, self.get_label(metal_resolved))
        active_palette = palette or self.palette

        n_plots = len(dietary_cols)
        nrows = math.ceil(n_plots / ncols)
        calc_figsize = figsize if figsize is not None else (16.0, 2.9 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=calc_figsize, sharey=True, dpi=kwargs.get("dpi", 200))
        flat_axes = np.array(axes).flatten()

        is_parametric = self._is_parametric(method, **kwargs)
        effective_m = str(kwargs.get("metodo") or kwargs.get("tipo_analisis") or method).lower()
        is_anova = "anova" in effective_m or effective_m == "f"

        for idx, col in enumerate(dietary_cols):
            ax = flat_axes[idx]
            sub = clean_df[[col, metal_resolved]].dropna().copy()
            clean_name = col.replace("Alim_", "").replace("_", " ").title()

            ordered_levels = sorted(sub[col].unique())
            groups_vals = [sub[sub[col] == lvl][metal_resolved].values for lvl in ordered_levels]

            if is_parametric:
                if is_anova:
                    f_stat, p_val = stats.f_oneway(*groups_vals)
                    f_val = float(f_stat) if pd.notna(f_stat) else 0.0
                    p_val = float(p_val) if pd.notna(p_val) else 1.0
                    p_str = self._format_p_val(p_val, threshold=0.001)
                    stat_ann = f" ($F={f_val:.2f}$, {p_str})"
                else:
                    x_vals = pd.to_numeric(sub[col], errors="coerce")
                    y_vals = pd.to_numeric(sub[metal_resolved], errors="coerce")
                    valid_xy = x_vals.notna() & y_vals.notna()
                    if valid_xy.sum() >= 3:
                        r_val, p_val = stats.pearsonr(x_vals[valid_xy], y_vals[valid_xy])
                    else:
                        r_val, p_val = 0.0, 1.0
                    p_str = self._format_p_val(p_val, threshold=0.001)
                    r_str = f"r={r_val:+.2f}" if pd.notna(r_val) else ""
                    stat_ann = f" (${r_str}$, {p_str})" if r_str else f" ({p_str})"
            else:
                jt = jonckheere_terpstra_test(groups_vals)
                p_val = jt.get("p_val", jt.get("p_value", np.nan))
                z_score = jt.get("z_stat", jt.get("z_score", np.nan))
                p_str = self._format_p_val(p_val, threshold=0.001)
                z_str = f"z={z_score:+.2f}" if pd.notna(z_score) else ""
                stat_ann = f" (${z_str}$, {p_str})" if z_str else f" ({p_str})"

            sns.boxplot(
                data=sub,
                x=col,
                y=metal_resolved,
                hue=col,
                legend=False,
                palette=active_palette,
                ax=ax,
                width=0.45,
                boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.1),
                medianprops=dict(color="#0f172a", linewidth=2.0),
                whiskerprops=dict(color="#475569", linewidth=1.1),
                capprops=dict(color="#475569", linewidth=1.1),
                showfliers=False,
            )

            sns.stripplot(
                data=sub,
                x=col,
                y=metal_resolved,
                color="#b91c1c",
                alpha=0.90,
                size=5.2,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.5,
                ax=ax,
            )

            letter = chr(ord('A') + idx) if idx < 26 else str(idx + 1)
            ax.set_title(f"{letter}. {clean_name}{stat_ann}", fontsize=10.0, fontweight="bold", pad=8, color="#0f172a")

            ax.set_xlabel("")
            if idx % ncols == 0:
                ax.set_ylabel(metal_axis_lbl, fontsize=9.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            x_ticks = range(len(ordered_levels))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([compact_labels.get(int(lvl), str(lvl)) for lvl in ordered_levels], fontsize=8.5, fontweight="bold")
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        for j in range(n_plots, len(flat_axes)):
            flat_axes[j].set_visible(False)

        self._save_figure(fig, filepath=filepath)
        return fig, axes

