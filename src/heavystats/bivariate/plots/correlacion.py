"""
Módulo de correlación continua y matrices de co-exposición.
Implementa análisis bivariante entre variables continuas y matrices multimodales
(heatmap triangular y matriz de dispersión compacta).
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.bivariate.constants import DEFAULT_PRIMARY_METALS
from heavystats.bivariate.tests import (
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
)
from heavystats.bivariate.plots.base import BivariateBasePlots, DEFAULT_BIOMEDICAL_METAL_LABELS


class CorrelacionPlotsMixin:
    """Mixin para análisis de gradientes continuos y matrices de co-exposición."""

    def correlation_analysis(
        self: BivariateBasePlots,
        x: Optional[str] = None,
        y: Optional[str] = None,
        method: str = "nonparametric",
        continuous_col: Optional[str] = None,
        metal: Optional[str] = None,
        target: Optional[str] = None,
        target_col: Optional[str] = None,
        log_scale: bool = False,
        show_stats: bool = True,
        n_boot: int = 2000,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.2, 5.0),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Dispersión y correlación entre dos variables cuantitativas continuas.
        Modo no paramétrico: Spearman rho con IC 95% Bootstrap y Kendall tau-b.
        Modo paramétrico: Pearson r con IC 95% Fisher z, recta de regresión OLS y R².
        """
        col_x_raw = x if x is not None else continuous_col
        col_y_raw = y or target or target_col or metal or "Mercurio_ug_L"

        if col_x_raw is None:
            raise ValueError("Debe especificar la variable continua independiente x.")

        col_x = self._resolve_column(col_x_raw)
        col_y = self._resolve_column(col_y_raw)

        if col_x not in self.df.columns or col_y not in self.df.columns:
            raise ValueError(f"Las columnas '{col_x}' o '{col_y}' no se encuentran en el DataFrame.")

        sub_df = self.df[[col_x, col_y]].dropna().copy()
        sub_df[col_x] = pd.to_numeric(sub_df[col_x], errors="coerce")
        sub_df[col_y] = pd.to_numeric(sub_df[col_y], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[(sub_df[col_x] > 0) & (sub_df[col_y] > 0)]

        if sub_df.empty:
            raise ValueError(f"No hay observaciones numéricas válidas para '{col_x}' y '{col_y}'.")

        fig, ax = plt.subplots(figsize=figsize)
        plot_x = np.log(sub_df[col_x]) if log_scale else sub_df[col_x]
        plot_y = np.log(sub_df[col_y]) if log_scale else sub_df[col_y]

        is_param = method.lower().startswith("param")

        sns.regplot(
            x=plot_x,
            y=plot_y,
            ax=ax,
            color="#0284c7",
            scatter_kws={"s": 48, "alpha": 0.75, "edgecolor": "#0f172a", "linewidths": 0.7},
            line_kws={"linewidth": 1.7, "color": "#0369a1"},
        )

        lbl_x = DEFAULT_BIOMEDICAL_METAL_LABELS.get(col_x, self.get_label(col_x)) + (" [ln]" if log_scale else "")
        lbl_y = DEFAULT_BIOMEDICAL_METAL_LABELS.get(col_y, self.get_label(col_y)) + (" [ln]" if log_scale else "")

        ax.set_xlabel(lbl_x, fontsize=10.5, fontweight="bold", labelpad=8)
        ax.set_ylabel(lbl_y, fontsize=10.5, fontweight="bold", labelpad=8)
        self._clean_spines_and_ticks(ax)

        stat_subtitle = ""
        if show_stats and len(sub_df) >= 3:
            if is_param:
                pr = pearson_correlation(sub_df[col_x], sub_df[col_y])
                r_val = pr.get("r", np.nan)
                p_val = pr.get("p_val", np.nan)
                ci_l = pr.get("ci_low", np.nan)
                ci_h = pr.get("ci_high", np.nan)
                r2 = pr.get("r_squared", np.nan)
                slp = pr.get("slope", np.nan)
                icpt = pr.get("intercept", np.nan)

                p_str = self._format_p_val(p_val, threshold=0.0001, precision=4)
                ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""
                eq_str = f"y = {slp:.2f}x + {icpt:.2f}" if (pd.notna(slp) and pd.notna(icpt)) else ""

                parts = [rf"Pearson $r = {r_val:+.3f}$"]
                if ci_str:
                    parts.append(f"IC 95%: {ci_str}")
                if pd.notna(r2):
                    parts.append(rf"$R^2 = {r2:.3f}$")
                if eq_str:
                    parts.append(eq_str)
                parts.append(p_str)
                stat_subtitle = "  |  ".join(parts)
            else:
                sp = spearman_correlation(sub_df[col_x], sub_df[col_y], n_boot=n_boot)
                kd = kendall_correlation(sub_df[col_x], sub_df[col_y])
                rho_val = sp.get("rho", np.nan)
                p_val = sp.get("p_val", sp.get("p_value", np.nan))
                ci_l = sp.get("ci_low", np.nan)
                ci_h = sp.get("ci_high", np.nan)
                tau_val = kd.get("tau", np.nan)
                n_obs = sp.get("n_valid", len(sub_df))

                p_str = self._format_p_val(p_val, threshold=0.0001, precision=4)
                ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""

                parts = [rf"Spearman $\rho_s = {rho_val:+.3f}$"]
                if ci_str:
                    parts.append(f"IC 95%: {ci_str}")
                if pd.notna(tau_val):
                    parts.append(rf"Kendall $\tau_b = {tau_val:+.3f}$")
                parts.extend([p_str, f"n = {n_obs}"])
                stat_subtitle = "  |  ".join(parts)

        if title is not None:
            ax.set_title(title, fontsize=11.0, fontweight="bold", pad=12)

        if show_stats and stat_subtitle:
            ax.text(
                0.96, 0.95,
                stat_subtitle.replace("  |  ", "\n"),
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
            )

        self._save_figure(fig, filepath=filepath)
        return fig, ax

    def coexposure_matrix(
        self: BivariateBasePlots,
        metals: Optional[Sequence[str]] = None,
        variables: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        kind: str = "heatmap",
        n_boot: int = 2000,
        cmap: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Matriz de Co-Exposición y Correlación Multivariable entre Variables Cuantitativas.
        Soporta modo 'heatmap' (triangular editorial con IC 95% Bootstrap) y 'scatter'/'matrix' (compacto con densidades).
        """
        target_cols = variables or metals or DEFAULT_PRIMARY_METALS
        valid_cols = [self._resolve_column(m) for m in target_cols if self._resolve_column(m) in self.df.columns]

        if len(valid_cols) < 2:
            cand_extra = ["Score_Riesgo", "Edad", "IMC", "Peso_kg", "Altura_cm"]
            for c in cand_extra:
                res_c = self._resolve_column(c)
                if res_c in self.df.columns and res_c not in valid_cols:
                    valid_cols.append(res_c)
                if len(valid_cols) >= 3:
                    break

        if len(valid_cols) < 2:
            fig, ax = plt.subplots(figsize=figsize or (6.5, 5.5))
            ax.text(
                0.5, 0.5,
                "Se requieren al menos 2 variables cuantitativas en el DataFrame\npara la matriz de co-exposición.",
                ha="center", va="center", fontsize=9.5, color="#64748b",
            )
            ax.axis("off")
            return fig, ax

        sub_df = self.df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
        k = len(valid_cols)
        is_param = method.lower().startswith("param")
        labels_x = [self._wrap_text(self.get_label(m), width=11) for m in valid_cols]
        labels_y = [self._wrap_text(self.get_label(m), width=14) for m in valid_cols]

        # MODALIDAD 1: MATRIZ DE DISPERSIÓN COMPACTA
        if str(kind).lower() in ("scatter", "matrix", "pairplot"):
            size_calc = (max(8.5, k * 2.2), max(7.5, k * 2.0))
            fig_size = figsize or size_calc
            fig, axes = plt.subplots(k, k, figsize=fig_size, sharex=False, sharey=False)

            tint_cmap = sns.diverging_palette(220, 20, as_cmap=True)

            for i in range(k):
                for j in range(k):
                    ax = axes[i, j] if k > 1 else axes
                    col_y = valid_cols[i]
                    col_x = valid_cols[j]
                    label_x = labels_x[j]
                    label_y = labels_y[i]

                    # Diagonal: Distribución Univariante (KDE + Histograma + Mediana)
                    if i == j:
                        sns.histplot(
                            sub_df[col_x],
                            kde=True,
                            ax=ax,
                            color="#0f766e",
                            edgecolor="#0f172a",
                            linewidth=0.8,
                            stat="density",
                        )
                        med_val = sub_df[col_x].median()
                        ax.axvline(med_val, color="#b91c1c", linestyle="--", linewidth=1.5, alpha=0.85)
                        ax.set_title(f"{label_x}\nMed: {med_val:.2f}", fontsize=9.0, fontweight="bold", pad=4)
                        ax.set_xlabel("")
                        ax.set_ylabel("")

                    # Triángulo Inferior: Dispersión Bivariante con Recta y Banda 95%
                    elif i > j:
                        sns.regplot(
                            x=col_x,
                            y=col_y,
                            data=sub_df,
                            ax=ax,
                            color="#0284c7",
                            scatter_kws={"s": 32, "color": "#b91c1c", "edgecolor": "white", "alpha": 0.9},
                            line_kws={"color": "#0f172a", "linewidth": 1.5},
                            ci=95,
                        )
                        ax.grid(True, linestyle=":", alpha=0.5)

                        if j == 0:
                            ax.set_ylabel(label_y, fontweight="bold", fontsize=9.0)
                        else:
                            ax.set_ylabel("")

                        if i == k - 1:
                            ax.set_xlabel(label_x, fontweight="bold", fontsize=9.0)
                        else:
                            ax.set_xlabel("")

                    # Triángulo Superior: Bloque Estadístico
                    else:
                        if is_param:
                            pr = pearson_correlation(sub_df[col_x], sub_df[col_y])
                            coef_val = pr.get("r", 0.0)
                            p_val = pr.get("p_val", 1.0)
                            ci_l = pr.get("ci_low", np.nan)
                            ci_h = pr.get("ci_high", np.nan)
                            metric_name = "Pearson $r$"
                        else:
                            sp = spearman_correlation(sub_df[col_x], sub_df[col_y], n_boot=n_boot)
                            coef_val = sp.get("rho", 0.0)
                            p_val = sp.get("p_val", 1.0)
                            ci_l = sp.get("ci_low", np.nan)
                            ci_h = sp.get("ci_high", np.nan)
                            metric_name = r"Spearman $\rho_s$"

                        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                        p_str = self._format_p_val(p_val)

                        norm_val = np.clip((coef_val + 1.0) / 2.0, 0.0, 1.0)
                        bg_rgb = tint_cmap(norm_val)
                        ax.set_facecolor((*bg_rgb[:3], 0.16))

                        for spine in ax.spines.values():
                            spine.set_color("#cbd5e1")
                            spine.set_linewidth(1.0)

                        ci_str = f"IC 95%: [{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else ""
                        text_box = f"{metric_name}\n{coef_val:+.2f}{stars}\n\n{p_str}\n{ci_str}".strip()
                        ax.text(
                            0.5, 0.5, text_box,
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=9.0, fontweight="bold",
                            color="#0f172a",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Formato de ticks visibles
                    for tick in ax.get_xticklabels():
                        tick.set_fontweight("bold")
                        tick.set_fontsize(8.5)
                    for tick in ax.get_yticklabels():
                        tick.set_fontweight("bold")
                        tick.set_fontsize(8.5)

            if title is not None:
                fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

            self._save_figure(fig, filepath=filepath)
            return fig, axes

        # MODALIDAD 2: HEATMAP TRIANGULAR EDITORIAL REFINADO
        corr_matrix = np.zeros((k, k))
        annot_matrix = np.empty((k, k), dtype=object)

        for i in range(k):
            for j in range(k):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    annot_matrix[i, j] = "1.00"
                elif i > j:
                    col_i = valid_cols[i]
                    col_j = valid_cols[j]
                    if is_param:
                        pr = pearson_correlation(sub_df[col_i], sub_df[col_j])
                        coef_val = pr.get("r", 0.0)
                        p_val = pr.get("p_val", 1.0)
                        ci_l = pr.get("ci_low", np.nan)
                        ci_h = pr.get("ci_high", np.nan)
                    else:
                        sp = spearman_correlation(sub_df[col_i], sub_df[col_j], n_boot=n_boot)
                        coef_val = sp.get("rho", 0.0)
                        p_val = sp.get("p_val", 1.0)
                        ci_l = sp.get("ci_low", np.nan)
                        ci_h = sp.get("ci_high", np.nan)

                    stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                    corr_matrix[i, j] = coef_val
                    corr_matrix[j, i] = coef_val

                    if pd.notna(ci_l) and pd.notna(ci_h):
                        annot_matrix[i, j] = f"{coef_val:+.2f}{stars}\n({ci_l:.2f}, {ci_h:.2f})"
                    else:
                        annot_matrix[i, j] = f"{coef_val:+.2f}{stars}"
                else:
                    annot_matrix[i, j] = ""

        size_calc = (max(8.5, k * 1.3), max(7.2, k * 1.1))
        fig_size = figsize or size_calc
        fig, ax = plt.subplots(figsize=fig_size)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        heatmap_cmap = cmap or "vlag"
        cbar_lbl = r"Coeficiente de Correlación de Pearson ($r$)" if is_param else r"Coeficiente de Correlación de Spearman ($\rho_s$)"

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot_matrix,
            fmt="",
            cmap=heatmap_cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            linewidths=1.5,
            linecolor="white",
            xticklabels=labels_x,
            yticklabels=labels_y,
            cbar_kws={"shrink": 0.80, "label": cbar_lbl},
            ax=ax,
            annot_kws={"size": max(7.5, 11 - k * 0.5), "weight": "bold"},
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontweight="bold", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontweight="bold", fontsize=8.5)

        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.label.set_fontweight("bold")
            cbar.ax.yaxis.label.set_fontsize(9.5)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontweight("bold")

        if title is not None:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

        self._save_figure(fig, filepath=filepath)
        return fig, ax

    # Aliases de compatibilidad metodológica
    plot_coexposure_matrix = coexposure_matrix
    plot_correlation_analysis = correlation_analysis
    scatter_continuous = correlation_analysis
    scatter_metals = correlation_analysis

    def correlation_matrix(self: BivariateBasePlots, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para coexposure_matrix()."""
        return self.coexposure_matrix(*args, **kwargs)

    plot_metal_matrix = correlation_matrix
