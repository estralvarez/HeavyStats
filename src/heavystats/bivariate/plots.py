"""
Gráficos bivariantes con estética científica y calidad de publicación editorial.
Implementa comparaciones de metales por grupos (boxplots con puntos y anotación estadística en brackets),
matrices de correlación anotadas con intervalos Bootstrap, pairplots de co-exposición inter-metales,
gráficos de dispersión continuos con regresión y 95% CI, tendencias ordinales (Jonckheere-Terpstra),
cuadrículas exploratorias integradas, volcano plots de screening FDR y forest plots de tamaños de efecto.
"""

import os
import math
import re
import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)
from heavystats.bivariate.constants import (
    DEFAULT_PRIMARY_METALS,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)
from heavystats.bivariate.tests import (
    mann_whitney_test,
    kruskal_wallis_test,
    spearman_correlation,
    jonckheere_terpstra_test,
)

# Etiquetas toxicológicas y de unidades estándar para publicaciones biomédicas
DEFAULT_BIOMEDICAL_METAL_LABELS: Dict[str, str] = {
    "Plomo_ug_dL": "Plomo en sangre (µg/dL)",
    "Mercurio_ug_L": "Mercurio en sangre (µg/L)",
    "Cadmio_ug_L": "Cadmio en sangre (µg/L)",
    "Plomo": "Plomo en sangre (µg/dL)",
    "Mercurio": "Mercurio en sangre (µg/L)",
    "Cadmio": "Cadmio en sangre (µg/L)",
    "pb": "Plomo en sangre (µg/dL)",
    "hg": "Mercurio en sangre (µg/L)",
    "cd": "Cadmio en sangre (µg/L)",
}

# Fuentes y especificación técnica rigurosa de límites de referencia toxicológicos
# - Plomo: CDC (2021) Blood Lead Reference Value (BLRV) infantil = 3.5 µg/dL (actualización del valor 5.0 de 2012)
# - Mercurio: EPA / OMS valor de referencia en sangre total = 5.0 µg/L
# - Cadmio: OMS / ATSDR nivel de referencia en sangre = 1.0 µg/L
DEFAULT_REFERENCE_LABELS: Dict[str, str] = {
    "Plomo_ug_dL": "Ref. CDC BLRV (3.5 µg/dL)",
    "Mercurio_ug_L": "Ref. EPA / OMS (5.0 µg/L)",
    "Cadmio_ug_L": "Ref. OMS (1.0 µg/L)",
    "Plomo": "Ref. CDC BLRV (3.5 µg/dL)",
    "Mercurio": "Ref. EPA / OMS (5.0 µg/L)",
    "Cadmio": "Ref. OMS (1.0 µg/L)",
    "pb": "Ref. CDC BLRV (3.5 µg/dL)",
    "hg": "Ref. EPA / OMS (5.0 µg/L)",
    "cd": "Ref. OMS (1.0 µg/L)",
}


class BivariatePlots:
    """
    Clase para construir gráficos bivariantes utilizando Seaborn y Matplotlib.
    Diseñada con estética editorial científica, anotaciones estadísticas directas
    en corchetes (brackets), tamaños de muestra (n=...) en ejes y límites toxicológicos
    debidamente contextualizados sin recuadros que obstruyan los datos.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        palette: Union[str, Sequence[str]] = "crest",
        style: str = "ticks",
        context: str = "notebook",
        labels_map: Optional[Dict[str, str]] = None,
        rc: Optional[Dict[str, Any]] = None
    ):
        self.df = df.copy()
        self.palette = palette
        self.style = style
        self.context = context
        self.labels_map = DEFAULT_LABELS_MAP.copy()
        if labels_map is not None:
            self.labels_map.update(labels_map)

        self.rc = DEFAULT_CUSTOM_PARAMS.copy()
        if rc is not None:
            self.rc.update(rc)

        sns.set_theme(
            style=self.style,
            palette=self.palette if isinstance(self.palette, str) else None,
            rc=self.rc,
            context=self.context
        )
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

    def get_label(self, col: str) -> str:
        """Obtiene la etiqueta limpia de una variable."""
        return get_label(col, self.labels_map)

    def _resolve_limit(self, col: str, permissible_limit: Optional[float] = None) -> Optional[float]:
        """Resuelve el límite permisible de referencia toxicológica."""
        if permissible_limit is not None:
            return permissible_limit
        return DEFAULT_PERMISSIBLE_LIMITS.get(col)

    def _format_category_label(self, group_col: str, cat_val: Any, count: int) -> str:
        """Formatea el nombre de la categoría incluyendo el tamaño muestral (n=...)."""
        val_str = str(cat_val).strip()
        val_lower = val_str.lower()
        
        # Mapeos semánticos contextuales limpios
        if group_col in ("Es_Expuesto", "es_expuesto") or group_col.startswith("Exposicion_") or group_col.startswith("Salud_"):
            if val_lower in ("0", "0.0", "false", "no", "no expuesto", "no_expuesto"):
                clean_name = "No expuesto" if "expuesto" in group_col.lower() else "No"
            elif val_lower in ("1", "1.0", "true", "si", "sí", "expuesto"):
                clean_name = "Expuesto" if "expuesto" in group_col.lower() else "Sí"
            else:
                clean_name = val_str.replace("_", " ").title()
        elif group_col.lower() == "sexo":
            if val_lower in ("f", "fem", "femenino"):
                clean_name = "Femenino"
            elif val_lower in ("m", "masc", "masculino"):
                clean_name = "Masculino"
            else:
                clean_name = val_str.title()
        else:
            clean_name = val_str.replace("_", " ").title()

        return f"{clean_name} (n={count})"

    # =========================================================================
    # 1. Comparación de Metales por Grupos (Boxplots con Stripplot y Brackets)
    # =========================================================================

    def metal_by_group(
        self,
        group_col: str,
        metal: str,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.5, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de distribución comparativa (boxplot con puntos individuales superpuestos)
        para una concentración de metal entre las categorías de una variable cualitativa.
        
        Diseño optimizado para publicación científica:
        - Eje Y con nombre toxicológico y unidades en sangre (ej. Plomo en sangre (µg/dL)).
        - Eje X con nombres limpios y tamaño muestral explícito (n=...).
        - Mediana destacada y puntos individuales con jitter controlado.
        - Línea de referencia toxicológica con etiqueta directa en el extremo (sin leyenda obstructiva).
        - Bracket superior con contraste no paramétrico (Mann-Whitney U o Kruskal-Wallis).
        """
        if group_col not in self.df.columns or metal not in self.df.columns:
            raise ValueError(f"Las columnas '{group_col}' o '{metal}' no se encuentran en el DataFrame.")

        sub_df = self.df[[group_col, metal]].dropna().copy()
        sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[sub_df[metal] > 0]

        fig, ax = plt.subplots(figsize=figsize)
        active_palette = palette or (["#0284c7", "#0f766e"] if sub_df[group_col].nunique() == 2 else self.palette)

        # Ordenar niveles de categoría
        cats = sorted(sub_df[group_col].unique(), key=lambda x: str(x))
        counts = [int((sub_df[group_col] == c).sum()) for c in cats]
        xtick_labels = [self._format_category_label(group_col, c, n_c) for c, n_c in zip(cats, counts)]

        # Boxplot base (bordes sobrios y mediana sólida)
        sns.boxplot(
            data=sub_df,
            x=group_col,
            y=metal,
            hue=group_col,
            legend=False,
            order=cats,
            palette=active_palette,
            ax=ax,
            width=0.40,
            boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.2),
            medianprops=dict(color="#0f172a", linewidth=2.2),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
            showfliers=False
        )

        # Puntos individuales (stripplot con borde sutil para visualización clara)
        if show_points:
            sns.stripplot(
                data=sub_df,
                x=group_col,
                y=metal,
                order=cats,
                color="#0f172a",
                alpha=0.65,
                size=6.5,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.5,
                ax=ax
            )

        # Línea de referencia toxicológica con etiqueta directa sobre la línea (sin leyenda flotante)
        limit_val = self._resolve_limit(metal, permissible_limit)
        if show_limit and limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85
            )
            # Etiqueta directa al margen derecho
            ref_text = limit_label or DEFAULT_REFERENCE_LABELS.get(metal, f"Ref: {limit_val:g}")
            ax.text(
                len(cats) - 0.52,
                limit_val,
                f"{ref_text} ",
                color="#b91c1c",
                va="bottom",
                ha="right",
                fontsize=8.0,
                fontweight="semibold"
            )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        var_lbl = self.get_label(group_col)
        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
        if log_scale:
            metal_axis_lbl += " (Escala Log)"

        # Títulos y etiquetas limpias
        plot_title = title if title is not None else f"{metal_axis_lbl.split('(')[0].strip()} según {var_lbl}"
        ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel("", fontsize=9.5)
        ax.set_ylabel(metal_axis_lbl, fontsize=10, fontweight="medium", labelpad=8)
        
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(xtick_labels, fontsize=9.5)
        ax.tick_params(axis="both", labelsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Bracket superior con contraste no paramétrico (Mann-Whitney / Kruskal-Wallis)
        if show_stats and len(cats) >= 2:
            y_max_data = float(sub_df[metal].max()) if len(sub_df) > 0 else 1.0
            y_min_data = float(sub_df[metal].min()) if len(sub_df) > 0 else 0.0
            y_span = (y_max_data - y_min_data) if y_max_data > y_min_data else 1.0

            if log_scale and y_max_data > 0:
                y_bracket = y_max_data * 1.15
                h_bracket = y_bracket * 0.04
                y_top_limit = y_bracket * 1.35
            else:
                y_bracket = y_max_data + 0.08 * y_span
                h_bracket = 0.03 * y_span
                y_top_limit = y_bracket + 0.12 * y_span

            if len(cats) == 2:
                g1 = sub_df[sub_df[group_col] == cats[0]][metal].values
                g2 = sub_df[sub_df[group_col] == cats[1]][metal].values
                mw = mann_whitney_test(g1, g2)
                p_val = mw.get("p_val", mw.get("p_value", np.nan))
                r_rb = mw.get("r_rb", mw.get("rank_biserial", np.nan))
                diff_med = mw.get("diff_medians", np.nan)

                if pd.notna(p_val):
                    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
                else:
                    p_str = "p = N/D"

                r_str = f"r_rb = {r_rb:+.2f}" if pd.notna(r_rb) else "r_rb = N/D"
                diff_str = f"ΔMed = {diff_med:+.2f}" if pd.notna(diff_med) else ""

                stat_text = f"Mann–Whitney U: {p_str}  |  {r_str}"
                if diff_str:
                    stat_text += f"  |  {diff_str}"

                # Trazar bracket entre x=0 y x=1
                ax.plot([0, 0, 1, 1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                ax.text(
                    0.5,
                    y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                    stat_text,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="medium",
                    color="#0f172a"
                )
            else:
                groups_vals = [sub_df[sub_df[group_col] == c][metal].values for c in cats]
                kw = kruskal_wallis_test(groups_vals)
                p_val = kw.get("p_val", kw.get("p_value", np.nan))
                h_stat = kw.get("h_stat", kw.get("h_statistic", np.nan))
                eps_sq = kw.get("epsilon_sq", kw.get("epsilon_squared", np.nan))

                if pd.notna(p_val):
                    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
                else:
                    p_str = "p = N/D"

                h_str = f"H = {h_stat:.2f}" if pd.notna(h_stat) else "H = N/D"
                eps_str = f"ε² = {eps_sq:.2f}" if pd.notna(eps_sq) else "ε² = N/D"

                stat_text = f"Kruskal–Wallis: {p_str}  |  {h_str}  |  {eps_str}"
                ax.plot([0, 0, len(cats)-1, len(cats)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
                ax.text(
                    (len(cats) - 1) * 0.5,
                    y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                    stat_text,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="medium",
                    color="#0f172a"
                )

            ax.set_ylim(top=y_top_limit)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_binary(
        self,
        metal: str,
        group_col: str,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.5, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico e intuitivo para variables dicotómicas/binarias (ej. Es_Expuesto, Sexo)."""
        return self.metal_by_group(
            group_col=group_col,
            metal=metal,
            log_scale=log_scale,
            show_points=show_points,
            show_stats=show_stats,
            show_limit=show_limit,
            permissible_limit=permissible_limit,
            limit_label=limit_label,
            palette=palette,
            title=title,
            figsize=figsize,
            filepath=filepath,
        )

    def plot_categorical(
        self,
        metal: str,
        group_col: str,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico e intuitivo para variables categóricas/politómicas (>2 grupos, ej. Sector, Institución)."""
        return self.metal_by_group(
            group_col=group_col,
            metal=metal,
            log_scale=log_scale,
            show_points=show_points,
            show_stats=show_stats,
            show_limit=show_limit,
            permissible_limit=permissible_limit,
            limit_label=limit_label,
            palette=palette,
            title=title,
            figsize=figsize,
            filepath=filepath,
        )

    # =========================================================================
    # 2. Correlaciones Inter-Metales (Heatmap y Pairplot)
    # =========================================================================

    def correlation_matrix(
        self,
        metals: Optional[List[str]] = None,
        n_boot: int = 2000,
        cmap: str = "Blues",
        figsize: Tuple[float, float] = (6.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un heatmap de correlación de Spearman entre metales (Pb, Hg, Cd)
        anotando en cada celda el coeficiente rho, el valor p y el intervalo de confianza Bootstrap al 95%.
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        k = len(metals)
        rho_mat = np.ones((k, k))
        p_mat = np.zeros((k, k))
        annot_mat = np.empty((k, k), dtype=object)
        metal_labels = [DEFAULT_BIOMEDICAL_METAL_LABELS.get(m, self.get_label(m)) for m in metals]

        for i in range(k):
            for j in range(k):
                if i == j:
                    annot_mat[i, j] = "1.00\n[Diag.]"
                else:
                    sp = spearman_correlation(self.df[metals[i]], self.df[metals[j]], n_boot=n_boot)
                    rho_val = sp.get("rho", np.nan)
                    p_val = sp.get("p_val", sp.get("p_value", np.nan))
                    ci_l = sp.get("ci_low", np.nan)
                    ci_h = sp.get("ci_high", np.nan)

                    rho_mat[i, j] = rho_val
                    p_mat[i, j] = p_val
                    p_str = f"p={p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else "p<0.001"
                    ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else "[—]"
                    annot_mat[i, j] = f"ρ = {rho_val:+.2f}\n{p_str}\n{ci_str}"

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            rho_mat,
            annot=annot_mat,
            fmt="",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=1.2,
            linecolor="#ffffff",
            xticklabels=metal_labels,
            yticklabels=metal_labels,
            cbar_kws={"label": "Coeficiente de Spearman (ρ)", "shrink": 0.8},
            ax=ax,
            annot_kws={"fontsize": 8.5, "weight": "medium"}
        )

        ax.set_title("Correlación de Spearman entre Biomarcadores de Metales Pesados\n(con Intervalos de Confianza Bootstrap al 95%)", fontsize=10.5, fontweight="bold", pad=12)
        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_metal_matrix(
        self,
        metals: Optional[List[str]] = None,
        n_boot: int = 2000,
        cmap: str = "Blues",
        figsize: Tuple[float, float] = (6.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias directo para correlation_matrix()."""
        return self.correlation_matrix(
            metals=metals,
            n_boot=n_boot,
            cmap=cmap,
            figsize=figsize,
            filepath=filepath,
        )

    def pairplot_metals(
        self,
        metals: Optional[List[str]] = None,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (7.5, 6.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, Any]:
        """
        Matriz pareada de dispersión inter-metales con densidades univariantes KDE en la diagonal,
        gráficos de dispersión y recta de regresión en el triángulo inferior, y coeficientes de Spearman en el superior.
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        sub_df = self.df[metals].dropna().copy()
        for m in metals:
            sub_df[m] = pd.to_numeric(sub_df[m], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            for m in metals:
                sub_df = sub_df[sub_df[m] > 0]

        k = len(metals)
        fig, axes = plt.subplots(k, k, figsize=figsize)
        metal_labels = [DEFAULT_BIOMEDICAL_METAL_LABELS.get(m, self.get_label(m)) for m in metals]

        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                m_i, m_j = metals[i], metals[j]

                if i == j:
                    # Diagonal: Densidad KDE
                    plot_data = np.log(sub_df[m_i]) if log_scale else sub_df[m_i]
                    sns.kdeplot(plot_data, fill=True, color="#0284c7", alpha=0.35, ax=ax, linewidth=1.5)
                    ax.set_ylabel("")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                elif i > j:
                    # Triángulo inferior: Scatter + Regresión
                    px = np.log(sub_df[m_j]) if log_scale else sub_df[m_j]
                    py = np.log(sub_df[m_i]) if log_scale else sub_df[m_i]
                    sns.regplot(
                        x=px,
                        y=py,
                        ax=ax,
                        color="#0284c7",
                        scatter_kws={"s": 32, "alpha": 0.70, "edgecolor": "#0f172a", "linewidths": 0.5},
                        line_kws={"linewidth": 1.4, "color": "#0369a1"}
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                else:
                    # Triángulo superior: Anotación estadística
                    sp = spearman_correlation(sub_df[m_j], sub_df[m_i], n_boot=1000)
                    rho_val = sp.get("rho", np.nan)
                    p_val = sp.get("p_val", sp.get("p_value", np.nan))
                    ci_l = sp.get("ci_low", np.nan)
                    ci_h = sp.get("ci_high", np.nan)

                    p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else "p < 0.001"
                    ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""
                    
                    is_sig = pd.notna(p_val) and p_val < 0.05
                    txt_color = "#0f172a" if is_sig else "#64748b"
                    font_w = "bold" if is_sig else "normal"

                    ax.text(
                        0.5, 0.5,
                        f"ρ = {rho_val:+.2f}\n{p_str}\n{ci_str}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9.0, fontweight=font_w, color=txt_color
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)

                if i == k - 1:
                    ax.set_xlabel(metal_labels[j] + (" [ln]" if log_scale else ""), fontsize=8.5, fontweight="medium")
                else:
                    ax.set_xlabel("")

                if j == 0 and i != 0:
                    ax.set_ylabel(metal_labels[i] + (" [ln]" if log_scale else ""), fontsize=8.5, fontweight="medium")
                elif i != j:
                    ax.set_ylabel("")

                ax.tick_params(axis="both", labelsize=8.0)

        fig.suptitle("Matriz Pareada de Co-Exposición Toxicológica (Pb, Hg, Cd)", fontsize=11.5, fontweight="bold", y=0.99)
        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_metal_pairplot(
        self,
        metals: Optional[List[str]] = None,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (7.5, 6.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, Any]:
        """Alias directo para pairplot_metals()."""
        return self.pairplot_metals(
            metals=metals,
            log_scale=log_scale,
            figsize=figsize,
            filepath=filepath,
        )

    # =========================================================================
    # 3. Dispersión Continua (Scatter con Regresión e Intervalos Bootstrap)
    # =========================================================================

    def scatter_metals(
        self,
        metal_x: str,
        metal_y: str,
        log_scale: bool = False,
        show_stats: bool = True,
        n_boot: int = 2000,
        figsize: Tuple[float, float] = (6.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de dispersión bivariante entre dos variables continuas (o dos metales)
        con recta de regresión y banda de confianza al 95%, incluyendo anotación estadística compacta.
        """
        if metal_x not in self.df.columns or metal_y not in self.df.columns:
            raise ValueError(f"Las columnas '{metal_x}' o '{metal_y}' no se encuentran en el DataFrame.")

        sub_df = self.df[[metal_x, metal_y]].dropna().copy()
        sub_df[metal_x] = pd.to_numeric(sub_df[metal_x], errors="coerce")
        sub_df[metal_y] = pd.to_numeric(sub_df[metal_y], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[(sub_df[metal_x] > 0) & (sub_df[metal_y] > 0)]

        fig, ax = plt.subplots(figsize=figsize)

        plot_x = np.log(sub_df[metal_x]) if log_scale else sub_df[metal_x]
        plot_y = np.log(sub_df[metal_y]) if log_scale else sub_df[metal_y]

        sns.regplot(
            x=plot_x,
            y=plot_y,
            ax=ax,
            color="#0284c7",
            scatter_kws={"s": 45, "alpha": 0.75, "edgecolor": "#0f172a", "linewidths": 0.6},
            line_kws={"linewidth": 1.6, "color": "#0369a1"}
        )

        lbl_x = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_x, self.get_label(metal_x)) + (" [ln]" if log_scale else "")
        lbl_y = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_y, self.get_label(metal_y)) + (" [ln]" if log_scale else "")

        ax.set_xlabel(lbl_x, fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_ylabel(lbl_y, fontsize=9.5, fontweight="medium", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        stat_subtitle = ""
        if show_stats and len(sub_df) >= 3:
            sp = spearman_correlation(sub_df[metal_x], sub_df[metal_y], n_boot=n_boot)
            rho_val = sp.get("rho", np.nan)
            p_val = sp.get("p_val", sp.get("p_value", np.nan))
            ci_l = sp.get("ci_low", np.nan)
            ci_h = sp.get("ci_high", np.nan)
            n_obs = sp.get("n_valid", len(sub_df))

            p_str = f"p = {p_val:.4f}" if (pd.notna(p_val) and p_val >= 0.0001) else "p < 0.0001"
            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""
            stat_subtitle = f"Spearman ρ = {rho_val:+.3f} (IC 95%: {ci_str}) | {p_str} | n = {n_obs}"

        main_title = f"{self.get_label(metal_y)} vs {self.get_label(metal_x)}"
        if stat_subtitle:
            ax.set_title(f"{main_title}\n{stat_subtitle}", fontsize=10.5, fontweight="bold", pad=12)
        else:
            ax.set_title(main_title, fontsize=10.5, fontweight="bold", pad=10)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def scatter_continuous(
        self,
        continuous_col: str,
        metal: str,
        log_scale: bool = False,
        show_stats: bool = True,
        n_boot: int = 2000,
        figsize: Tuple[float, float] = (6.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico para variables continuas vs metal."""
        return self.scatter_metals(
            metal_x=continuous_col,
            metal_y=metal,
            log_scale=log_scale,
            show_stats=show_stats,
            n_boot=n_boot,
            figsize=figsize,
            filepath=filepath,
        )

    def plot_continuous(
        self,
        metal: str,
        column: str,
        log_scale: bool = False,
        show_stats: bool = True,
        n_boot: int = 2000,
        figsize: Tuple[float, float] = (6.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico e intuitivo para variables continuas (ej. Edad, IMC, Peso, Score_Riesgo)."""
        return self.scatter_metals(
            metal_x=column,
            metal_y=metal,
            log_scale=log_scale,
            show_stats=show_stats,
            n_boot=n_boot,
            figsize=figsize,
            filepath=filepath,
        )

    # =========================================================================
    # 4. Tendencia Ordinal y Exposición Dietaria (Jonckheere-Terpstra)
    # =========================================================================

    def ordinal_trend_plot(
        self,
        ordinal_col: str,
        metal: str,
        ordinal_map: Optional[Dict[str, int]] = None,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: str = "mako",
        figsize: Tuple[float, float] = (7.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de tendencia ordinal (boxplots ordenados de menor a mayor frecuencia)
        para evaluar tendencias monótonas de exposición dietaria con la prueba de Jonckheere-Terpstra.
        """
        if ordinal_col not in self.df.columns or metal not in self.df.columns:
            raise ValueError(f"Las columnas '{ordinal_col}' o '{metal}' no se encuentran en el DataFrame.")

        sub_df = self.df[[ordinal_col, metal]].dropna().copy()
        sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[sub_df[metal] > 0]

        if ordinal_map is None and (ordinal_col.startswith("Alim_") or ordinal_col.startswith("Consumo_")):
            ordinal_map = DIET_ORDINAL_MAP

        if ordinal_map is not None:
            sub_df["_ord_val"] = sub_df[ordinal_col].apply(
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
            ordered_levels = sorted(sub_df[ordinal_col].unique(), key=lambda x: str(x))
            level_labels = [
                f"{str(lvl).title()} (n={(sub_df[ordinal_col] == lvl).sum()})"
                for lvl in ordered_levels
            ]

        fig, ax = plt.subplots(figsize=figsize)
        plot_x_col = "_ord_val" if ordinal_map is not None else ordinal_col

        sns.boxplot(
            data=sub_df,
            x=plot_x_col,
            y=metal,
            hue=plot_x_col,
            legend=False,
            order=ordered_levels,
            palette=palette,
            ax=ax,
            width=0.42,
            boxprops=dict(alpha=0.80, edgecolor="#334155", linewidth=1.1),
            medianprops=dict(color="#0f172a", linewidth=2.2),
            whiskerprops=dict(color="#475569", linewidth=1.1),
            capprops=dict(color="#475569", linewidth=1.1),
            showfliers=False
        )

        if show_points:
            sns.stripplot(
                data=sub_df,
                x=plot_x_col,
                y=metal,
                order=ordered_levels,
                color="#0f172a",
                alpha=0.65,
                size=6.5,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.5,
                ax=ax
            )

        limit_val = self._resolve_limit(metal, permissible_limit)
        if show_limit and limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85
            )
            ref_text = limit_label or DEFAULT_REFERENCE_LABELS.get(metal, f"Ref: {limit_val:g}")
            ax.text(
                len(ordered_levels) - 0.52,
                limit_val,
                f"{ref_text} ",
                color="#b91c1c",
                va="bottom",
                ha="right",
                fontsize=8.0,
                fontweight="semibold"
            )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_xticks(range(len(ordered_levels)))
        ax.set_xticklabels(level_labels, fontsize=9.0)

        var_lbl = self.get_label(ordinal_col)
        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
        if log_scale:
            metal_axis_lbl += " (Escala Log)"

        ax.set_title(f"Tendencia de {metal_axis_lbl.split('(')[0].strip()} según {var_lbl}", fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel("Frecuencia de Consumo", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_ylabel(metal_axis_lbl, fontsize=10, fontweight="medium", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_stats and len(ordered_levels) >= 2:
            groups_data = [sub_df[sub_df[plot_x_col] == lvl][metal].values for lvl in ordered_levels]
            jt = jonckheere_terpstra_test(groups_data)
            sp = spearman_correlation(sub_df[plot_x_col], sub_df[metal])

            jt_p = jt.get("p_val", jt.get("p_value", np.nan))
            jt_z = jt.get("z_stat", jt.get("z_statistic", np.nan))
            sp_p = sp.get("p_val", sp.get("p_value", np.nan))
            sp_rho = sp.get("rho", np.nan)

            jt_p_str = f"p = {jt_p:.3f}" if (pd.notna(jt_p) and jt_p >= 0.001) else "p < 0.001"
            sp_p_str = f"p = {sp_p:.3f}" if (pd.notna(sp_p) and sp_p >= 0.001) else "p < 0.001"

            stat_text = f"Jonckheere–Terpstra: {jt_p_str} (z = {jt_z:+.2f})  |  Spearman ρ = {sp_rho:+.2f} ({sp_p_str})"

            y_max_data = float(sub_df[metal].max()) if len(sub_df) > 0 else 1.0
            y_min_data = float(sub_df[metal].min()) if len(sub_df) > 0 else 0.0
            y_span = (y_max_data - y_min_data) if y_max_data > y_min_data else 1.0

            if log_scale and y_max_data > 0:
                y_bracket = y_max_data * 1.15
                h_bracket = y_bracket * 0.04
                y_top_limit = y_bracket * 1.35
            else:
                y_bracket = y_max_data + 0.08 * y_span
                h_bracket = 0.03 * y_span
                y_top_limit = y_bracket + 0.12 * y_span

            ax.plot([0, 0, len(ordered_levels)-1, len(ordered_levels)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
            ax.text(
                (len(ordered_levels) - 1) * 0.5,
                y_bracket + (0.01 * y_span if not log_scale else y_bracket * 0.02),
                stat_text,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="medium",
                color="#0f172a"
            )
            ax.set_ylim(top=y_top_limit)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_dietary(
        self,
        metal: str,
        dietary_col: str,
        ordinal_map: Optional[Dict[str, int]] = None,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: str = "mako",
        figsize: Tuple[float, float] = (7.0, 4.8),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico e intuitivo para variables dietarias y de escala ordinal."""
        return self.ordinal_trend_plot(
            ordinal_col=dietary_col,
            metal=metal,
            ordinal_map=ordinal_map,
            log_scale=log_scale,
            show_points=show_points,
            show_stats=show_stats,
            show_limit=show_limit,
            permissible_limit=permissible_limit,
            limit_label=limit_label,
            palette=palette,
            figsize=figsize,
            filepath=filepath,
        )

    # =========================================================================
    # 5. Cuadrícula Exploratoria Integrada (Grid)
    # =========================================================================

    def plot_grid(
        self,
        metal: str,
        group_cols: Optional[List[str]] = None,
        n_cols: int = 4,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        palette: Optional[Union[str, Sequence[str]]] = "crest",
        figsize_per_plot: Tuple[float, float] = (3.8, 3.4),
        filepath: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Genera una cuadrícula integrada (grid) con todos los boxplots posibles para un metal seleccionado.
        Detecta automáticamente las columnas candidatas (2 a 8 categorías válidas) y formatea
        cada recuadro con estética editorial sin obstrucciones.
        """
        cols_to_exclude = set(DEFAULT_PRIMARY_METALS + [
            "Muestra_Codificada", "Edad", "Peso_kg", "Altura_cm", "IMC", "Score_Riesgo"
        ])

        if group_cols is None:
            group_cols = []
            for col in self.df.columns:
                if col in cols_to_exclude or col.startswith("_"):
                    continue
                valid_unique = self.df[col].dropna().nunique()
                if 2 <= valid_unique <= 8:
                    group_cols.append(col)

        n_plots = len(group_cols)
        if n_plots == 0:
            warnings.warn(f"No se encontraron variables de agrupación válidas para {metal}.")
            return None

        n_rows = math.ceil(n_plots / n_cols)
        fig_width = n_cols * figsize_per_plot[0]
        fig_height = n_rows * figsize_per_plot[1]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten()

        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
        limit_val = self._resolve_limit(metal)

        for idx, col in enumerate(group_cols):
            ax = axes_flat[idx]
            sub_df = self.df[[col, metal]].dropna().copy()
            sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
            sub_df = sub_df.dropna()

            cats = sorted(sub_df[col].unique(), key=lambda x: str(x))
            if len(cats) < 2:
                ax.set_visible(False)
                continue

            counts = [int((sub_df[col] == c).sum()) for c in cats]
            xtick_labels = [self._format_category_label(col, c, n_c) for c, n_c in zip(cats, counts)]

            sns.boxplot(
                data=sub_df,
                x=col,
                y=metal,
                hue=col,
                legend=False,
                order=cats,
                palette=palette,
                ax=ax,
                width=0.42,
                boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.1),
                medianprops=dict(color="#0f172a", linewidth=2.0),
                whiskerprops=dict(color="#475569", linewidth=1.0),
                capprops=dict(color="#475569", linewidth=1.0),
                showfliers=False
            )

            if show_points:
                sns.stripplot(
                    data=sub_df,
                    x=col,
                    y=metal,
                    order=cats,
                    color="#0f172a",
                    alpha=0.65,
                    size=5,
                    jitter=0.15,
                    ax=ax
                )

            if show_limit and limit_val is not None:
                ax.axhline(
                    y=limit_val,
                    color="#dc2626",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.80
                )

            col_lbl = self.get_label(col)
            short_title = col_lbl if len(col_lbl) <= 28 else col_lbl[:25] + "..."
            ax.set_title(short_title, fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xlabel("", fontsize=8.5)
            ax.set_ylabel(metal_axis_lbl if idx % n_cols == 0 else "", fontsize=8.5, fontweight="medium")

            if any(len(str(c)) > 8 for c in xtick_labels) or len(cats) > 3:
                ax.tick_params(axis="x", rotation=30, labelsize=7.5)
            else:
                ax.tick_params(axis="x", labelsize=8.0)

            ax.tick_params(axis="y", labelsize=8.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if show_stats and len(cats) >= 2:
                try:
                    if len(cats) == 2:
                        g1 = sub_df[sub_df[col] == cats[0]][metal].values
                        g2 = sub_df[sub_df[col] == cats[1]][metal].values
                        mw = mann_whitney_test(g1, g2)
                        p_val = mw.get("p_val", mw.get("p_value", np.nan))
                        r_eff = mw.get("r_rb", mw.get("rank_biserial", np.nan))
                        p_txt = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        stat_txt = f"{p_txt} | r_rb={r_eff:+.2f}"
                    else:
                        groups_vals = [sub_df[sub_df[col] == c][metal].values for c in cats]
                        kw = kruskal_wallis_test(groups_vals)
                        p_val = kw.get("p_val", kw.get("p_value", np.nan))
                        p_txt = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        stat_txt = f"KW {p_txt} | ε²={kw['epsilon_sq']:.2f}"

                    is_sig = pd.notna(p_val) and p_val < 0.05
                    box_color = "#dcfce7" if is_sig else "#f8fafc"
                    edge_color = "#16a34a" if is_sig else "#cbd5e1"

                    ax.text(
                        0.96, 0.95,
                        stat_txt,
                        transform=ax.transAxes,
                        fontsize=7.2,
                        verticalalignment="top",
                        horizontalalignment="right",
                        fontweight="bold" if is_sig else "normal",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=box_color, edgecolor=edge_color, alpha=0.9)
                    )
                except Exception:
                    pass

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Exploración Bivariante de Boxplots: {metal_axis_lbl}", fontsize=13, fontweight="bold", y=1.002)
        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig

    # =========================================================================
    # 6. Panel de Validación del Algoritmo de Riesgo
    # =========================================================================

    def risk_algorithm_plots(
        self,
        metals: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12.0, 4.2),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, Any]:
        """
        Panel multi-gráfico con los 3 metales que contrasta el Score_Riesgo continuo vs concentraciones reales
        y dibuja las líneas de corte de referencia biológica CDC/OMS/EPA.
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        m_count = len(metals)
        fig, axes = plt.subplots(1, m_count, figsize=figsize, sharey=False)
        if m_count == 1:
            axes = [axes]

        for idx, metal in enumerate(metals):
            ax = axes[idx]
            metal_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
            ref_val = self._resolve_limit(metal)

            sub_df = self.df[["Score_Riesgo", metal]].dropna().copy()
            sub_df["Score_Riesgo"] = pd.to_numeric(sub_df["Score_Riesgo"], errors="coerce")
            sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
            sub_df = sub_df.dropna()

            sns.regplot(
                data=sub_df,
                x="Score_Riesgo",
                y=metal,
                scatter_kws=dict(color="#047857", alpha=0.75, s=50, edgecolor="#064e3b", linewidths=0.6),
                line_kws=dict(color="#059669", linewidth=1.8),
                ci=95,
                ax=ax
            )

            if ref_val is not None:
                ax.axhline(ref_val, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Ref. ({ref_val:g})")
                ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#ffffff")

            res_sp = spearman_correlation(sub_df["Score_Riesgo"], sub_df[metal])
            rho = res_sp.get("rho", np.nan)
            p_val = res_sp.get("p_val", res_sp.get("p_value", np.nan))
            ci_l = res_sp.get("ci_low", np.nan)
            ci_h = res_sp.get("ci_high", np.nan)

            p_str = f"p = {p_val:.4f}" if (pd.notna(p_val) and p_val >= 0.0001) else "p < 0.0001"
            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""

            ax.text(
                0.95, 0.95,
                f"ρ = {rho:+.3f}\n{p_str}\n{ci_str}",
                transform=ax.transAxes, fontsize=8.5, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92)
            )

            ax.set_title(metal_lbl.split('(')[0].strip(), fontsize=10.5, fontweight="bold", pad=8)
            ax.set_xlabel("Score Predictivo de Riesgo (0-10)", fontsize=9.0)
            ax.set_ylabel(metal_lbl, fontsize=9.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Validación Empírica del Algoritmo de Riesgo frente a Concentraciones Biológicas", fontsize=11.5, fontweight="bold", y=1.02)
        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_risk_algorithm(
        self,
        metals: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12.0, 4.2),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, Any]:
        """Alias para risk_algorithm_plots()."""
        return self.risk_algorithm_plots(metals=metals, figsize=figsize, filepath=filepath)

    # =========================================================================
    # 7. Volcano Plot y Forest Plot del Screening Bivariante (FDR)
    # =========================================================================

    def volcano_effect_plot(
        self,
        screening_df: Optional[pd.DataFrame] = None,
        metal: str = "Plomo_ug_dL",
        fdr_alpha: float = 0.10,
        p_alpha: float = 0.05,
        figsize: Tuple[float, float] = (8.0, 5.2),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de Volcano / Tamaño del Efecto para el tamizaje exploratorio de covariables,
        representando el Tamaño del Efecto en el eje X vs -log10(p-valor) en el eje Y,
        resaltando las variables que alcanzan descubrimiento significativo bajo control FDR.
        """
        if screening_df is None:
            from heavystats.bivariate.tables import BivariateTables
            bt = BivariateTables(self.df, labels_map=self.labels_map)
            rep = bt.master_association_matrix(metal=metal, fdr_alpha=fdr_alpha, p_alpha=p_alpha)
            screening_df = rep.df

        df_plot = screening_df.copy()

        def parse_p(val):
            s = str(val).replace("<strong>", "").replace("</strong>", "").replace("*", "").replace("<", "").strip()
            try:
                return float(s)
            except Exception:
                return np.nan

        def parse_eff(val):
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            if m:
                return float(m.group())
            return np.nan

        p_col = "p_Raw" if "p_Raw" in df_plot.columns else ("Valor p" if "Valor p" in df_plot.columns else "p sin ajustar")
        eff_col = "Efecto_Num" if "Efecto_Num" in df_plot.columns else ("Tamaño del Efecto" if "Tamaño del Efecto" in df_plot.columns else "Tamaño del Efecto [IC 95%]")
        var_col = "Variable" if "Variable" in df_plot.columns else ("Factor Exploratorio" if "Factor Exploratorio" in df_plot.columns else "Variable_Raw")

        if p_col in df_plot.columns:
            df_plot["p_num"] = df_plot[p_col].apply(parse_p)
        else:
            df_plot["p_num"] = 1.0

        if eff_col in df_plot.columns:
            df_plot["eff_num"] = df_plot[eff_col].apply(parse_eff)
        else:
            df_plot["eff_num"] = 0.0

        df_plot["log10_p"] = -np.log10(np.clip(df_plot["p_num"], 1e-6, 1.0))

        # Detectar significancia FDR
        if "p (FDR)" in df_plot.columns:
            df_plot["fdr_p"] = df_plot["p (FDR)"].apply(parse_p)
            disc_mask = df_plot["fdr_p"] < fdr_alpha
        elif "Nivel de Evidencia" in df_plot.columns:
            disc_mask = df_plot["Nivel de Evidencia"].str.contains("FDR", na=False)
        elif "Hallazgo Sugestivo" in df_plot.columns:
            disc_mask = df_plot["Hallazgo Sugestivo"].str.contains("Posible", na=False)
        else:
            disc_mask = df_plot["p_num"] < p_alpha

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            df_plot.loc[~disc_mask, "eff_num"],
            df_plot.loc[~disc_mask, "log10_p"],
            color="#94a3b8",
            alpha=0.6,
            s=45,
            label=f"Sin significación FDR (q ≥ {fdr_alpha:.2f})"
        )

        if disc_mask.any():
            ax.scatter(
                df_plot.loc[disc_mask, "eff_num"],
                df_plot.loc[disc_mask, "log10_p"],
                color="#e11d48",
                alpha=0.9,
                s=80,
                edgecolor="#881337",
                linewidths=1.2,
                label=f"Descubrimiento FDR (q < {fdr_alpha:.2f})"
            )

            for _, r in df_plot[disc_mask].iterrows():
                lbl = str(r[var_col]).replace("**", "").split("(")[0].strip()
                ax.annotate(
                    lbl,
                    (r["eff_num"], r["log10_p"]),
                    fontsize=8,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points"
                )

        ax.axhline(
            y=-np.log10(p_alpha),
            color="#f59e0b",
            linestyle="--",
            linewidth=1.1,
            label=f"Umbral nominal p = {p_alpha:.2f}"
        )

        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
        ax.set_xlabel("Magnitud del Tamaño del Efecto (r_rb / Spearman ρ / ε²)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_ylabel("-log10(p-valor crudo)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_title(f"Tamizaje Bivariante (Volcano Plot): {metal_axis_lbl}", fontsize=11, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.0)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_volcano(
        self,
        metal: str = "Plomo_ug_dL",
        screening_df: Optional[pd.DataFrame] = None,
        fdr_alpha: float = 0.10,
        p_alpha: float = 0.05,
        figsize: Tuple[float, float] = (8.0, 5.2),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para volcano_effect_plot()."""
        return self.volcano_effect_plot(
            screening_df=screening_df,
            metal=metal,
            fdr_alpha=fdr_alpha,
            p_alpha=p_alpha,
            figsize=figsize,
            filepath=filepath,
        )

    def screening_forest_plot(
        self,
        screening_df: Optional[pd.DataFrame] = None,
        metal: str = "Plomo_ug_dL",
        top_n: int = 15,
        figsize: Tuple[float, float] = (8.5, 6.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un Forest Plot con los principales tamaños de efecto e intervalos de confianza Bootstrap
        resultantes del screening bivariante de covariables frente a un metal.
        """
        if screening_df is None:
            from heavystats.bivariate.tables import BivariateTables
            bt = BivariateTables(self.df, labels_map=self.labels_map)
            rep = bt.master_association_matrix(metal=metal)
            screening_df = rep.df

        df_plot = screening_df.copy()

        # Extraer intervalos de confianza
        def parse_ci(val):
            m = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", str(val))
            if m:
                return float(m.group(1)), float(m.group(2))
            return np.nan, np.nan

        def parse_eff(val):
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            if m:
                return float(m.group())
            return np.nan

        eff_col = "Tamaño del Efecto [IC 95%]" if "Tamaño del Efecto [IC 95%]" in df_plot.columns else ("Tamaño del Efecto" if "Tamaño del Efecto" in df_plot.columns else "Efecto_Num")
        var_col = "Variable" if "Variable" in df_plot.columns else ("Factor Exploratorio" if "Factor Exploratorio" in df_plot.columns else "Variable_Raw")

        df_plot["_eff"] = df_plot[eff_col].apply(parse_eff)
        ci_tuples = df_plot[eff_col].apply(parse_ci)
        df_plot["_ci_low"] = [t[0] for t in ci_tuples]
        df_plot["_ci_high"] = [t[1] for t in ci_tuples]

        # Filtrar variables con IC válidos y ordenar por magnitud absoluta
        valid_df = df_plot.dropna(subset=["_eff", "_ci_low", "_ci_high"]).copy()
        valid_df["_abs_eff"] = valid_df["_eff"].abs()
        valid_df = valid_df.sort_values(by="_abs_eff", ascending=False).head(top_n).iloc[::-1]

        if valid_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Sin datos de intervalos de confianza suficientes", ha="center", va="center")
            return fig, ax

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(valid_df))
        labels = [str(r[var_col]).replace("**", "").strip() for _, r in valid_df.iterrows()]

        err_left = np.maximum(0.0, np.asarray(valid_df["_eff"] - valid_df["_ci_low"], dtype=float))
        err_right = np.maximum(0.0, np.asarray(valid_df["_ci_high"] - valid_df["_eff"], dtype=float))

        ax.errorbar(
            valid_df["_eff"].values,
            y_pos,
            xerr=[err_left, err_right],
            fmt="o",
            color="#0284c7",
            ecolor="#0369a1",
            elinewidth=1.8,
            capsize=4.0,
            markersize=6.5
        )

        ax.axvline(0.0, color="#64748b", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.0)

        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal, self.get_label(metal))
        ax.set_title(f"Top {len(valid_df)} Factores Asociados con {metal_axis_lbl}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Tamaño del Efecto e Intervalo de Confianza al 95% (Bootstrap)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_forest_effects(
        self,
        metal: str = "Plomo_ug_dL",
        screening_df: Optional[pd.DataFrame] = None,
        top_n: int = 15,
        figsize: Tuple[float, float] = (8.5, 6.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para screening_forest_plot()."""
        return self.screening_forest_plot(
            screening_df=screening_df,
            metal=metal,
            top_n=top_n,
            figsize=figsize,
            filepath=filepath,
        )


__all__ = [
    "BivariatePlots",
    "DEFAULT_BIOMEDICAL_METAL_LABELS",
    "DEFAULT_REFERENCE_LABELS",
]
