"""
Gráficos bivariantes con estética científica y calidad de publicación.
Implementa comparaciones de metales por grupos (boxplots con puntos y anotación estadística),
heatmaps de correlación con intervalos bootstrap, gráficos de dispersión con ajuste y gráficos de tendencia ordinal.
"""

import os
import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
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


class BivariatePlots:
    """
    Clase para construir gráficos bivariantes utilizando Seaborn y Matplotlib.
    Diseñada con estética editorial científica, anotaciones estadísticas automáticas
    e intervalos de confianza visuales para publicaciones biomédicas.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        palette: str = "crest",
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
            palette=self.palette,
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

    def metal_by_group(
        self,
        group_col: str,
        metal: str,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        permissible_limit: Optional[float] = None,
        palette: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 5.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de distribución comparativa (boxplot con puntos individuales superpuestos)
        para una concentración de metal entre las categorías de una variable cualitativa.
        Calcula e inserta automáticamente la prueba estadística (Mann-Whitney o Kruskal-Wallis)
        y el tamaño del efecto en una caja de anotación profesional.

        Parámetros
        ----------
        group_col : str
            Columna categórica de agrupación (ej. Sexo, Sector, Es_Expuesto).
        metal : str
            Columna continua del metal analizado.
        log_scale : bool
            Si True, utiliza escala logarítmica en el eje Y.
        show_points : bool
            Si True, superpone puntos individuales (stripplot / jitter).
        show_stats : bool
            Si True, incluye la caja con estadísticas y valor p.
        permissible_limit : float, opcional
            Límite permisible de referencia sanitaria para trazar línea guía.
        filepath : str, opcional
            Ruta para guardar la figura en PNG/PDF.

        Retorna
        -------
        Tuple[plt.Figure, plt.Axes]
        """
        sub_df = self.df[[group_col, metal]].dropna().copy()
        sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
        sub_df = sub_df.dropna()

        fig, ax = plt.subplots(figsize=figsize)
        active_palette = palette or self.palette

        # Ordenar niveles
        cats = sorted(sub_df[group_col].unique(), key=lambda x: str(x))

        # Boxplot base
        sns.boxplot(
            data=sub_df,
            x=group_col,
            y=metal,
            hue=group_col,
            legend=False,
            order=cats,
            palette=active_palette,
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.85, edgecolor="#334155", linewidth=1.2),
            medianprops=dict(color="#0f172a", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
            showfliers=False
        )

        # Puntos individuales
        if show_points:
            sns.stripplot(
                data=sub_df,
                x=group_col,
                y=metal,
                order=cats,
                color="#0f172a",
                alpha=0.65,
                size=7,
                jitter=0.18,
                ax=ax
            )

        # Límite permisible
        limit_val = self._resolve_limit(metal, permissible_limit)
        if limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.4,
                label=f"Límite CDC/OMS ({limit_val:g})"
            )
            ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)

        if log_scale:
            ax.set_yscale("log")

        var_lbl = self.get_label(group_col)
        metal_lbl = self.get_label(metal)

        ax.set_title(f"Distribución de {metal_lbl} según {var_lbl}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(var_lbl, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(metal_lbl + (" (Escala Log)" if log_scale else ""), fontsize=10, fontweight="bold", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Cálculo y caja de anotación estadística
        if show_stats and len(cats) >= 2:
            if len(cats) == 2:
                g1 = sub_df[sub_df[group_col] == cats[0]][metal].values
                g2 = sub_df[sub_df[group_col] == cats[1]][metal].values
                mw = mann_whitney_test(g1, g2)
                p_val = mw["p_value"]
                r_rb = mw["rank_biserial"]
                diff_med = mw["diff_medians"]
                stat_text = (
                    f"Mann-Whitney U: p = {p_val:.4f}\n"
                    f"Δ Medianas = {diff_med:+.2f}\n"
                    f"Efecto r_rb = {r_rb:+.2f}"
                )
            else:
                groups_vals = [sub_df[sub_df[group_col] == c][metal].values for c in cats]
                kw = kruskal_wallis_test(*groups_vals)
                stat_text = (
                    f"Kruskal-Wallis: p = {kw['p_value']:.4f}\n"
                    f"Estadístico H = {kw['h_statistic']:.2f}\n"
                    f"Efecto ε² = {kw['epsilon_squared']:.3f}"
                )

            ax.text(
                0.03, 0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
            )

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def correlation_matrix(
        self,
        metals: Optional[List[str]] = None,
        n_boot: int = 2000,
        cmap: str = "Blues",
        figsize: Tuple[float, float] = (7.0, 6.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un heatmap de correlación de Spearman entre metales (Pb, Hg, Cd)
        anotando en cada celda el coeficiente rho, el valor p y el intervalo de confianza Bootstrap al 95%.

        Parámetros
        ----------
        metals : List[str], opcional
            Metales a correlacionar.
        n_boot : int
            Réplicas bootstrap para intervalos de confianza.
        cmap : str
            Mapa de colores.
        filepath : str, opcional
            Ruta para guardar la figura.

        Retorna
        -------
        Tuple[plt.Figure, plt.Axes]
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        k = len(metals)
        rho_mat = np.ones((k, k))
        p_mat = np.zeros((k, k))
        annot_mat = np.empty((k, k), dtype=object)
        metal_labels = [self.get_label(m) for m in metals]

        for i in range(k):
            for j in range(k):
                if i == j:
                    annot_mat[i, j] = "1.00\n[Diag.]"
                else:
                    sp = spearman_correlation(self.df[metals[i]], self.df[metals[j]], n_boot=n_boot)
                    rho_mat[i, j] = sp["rho"]
                    p_mat[i, j] = sp["p_value"]
                    p_str = f"p={sp['p_value']:.3f}" if sp["p_value"] >= 0.001 else "p<0.001"
                    ci_str = f"[{sp['ci_low']:+.2f}, {sp['ci_high']:+.2f}]"
                    annot_mat[i, j] = f"ρ = {sp['rho']:+.2f}\n{p_str}\n{ci_str}"

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(rho_mat, dtype=bool), k=1)  # Mostrar triángulo inferior + diag

        sns.heatmap(
            rho_mat,
            annot=annot_mat,
            fmt="",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=1.5,
            linecolor="#ffffff",
            xticklabels=metal_labels,
            yticklabels=metal_labels,
            cbar_kws={"label": "Coeficiente de Spearman (ρ)", "shrink": 0.8},
            ax=ax,
            annot_kws={"fontsize": 8.5, "weight": "medium"}
        )

        ax.set_title("Matriz de Correlación de Spearman entre Metales Pesados\n(con Intervalos de Confianza Bootstrap al 95%)", fontsize=11, fontweight="bold", pad=14)
        plt.tight_layout()

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def scatter_metals(
        self,
        metal_x: str,
        metal_y: str,
        log_scale: bool = True,
        show_stats: bool = True,
        n_boot: int = 2000,
        figsize: Tuple[float, float] = (6.5, 5.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de dispersión bivariante entre 2 metales con recta de regresión
        y banda de confianza al 95%, incluyendo anotación de Spearman rho con intervalo bootstrap.

        Parámetros
        ----------
        metal_x : str
            Metal en el eje X.
        metal_y : str
            Metal en el eje Y.
        log_scale : bool
            Si True, utiliza escala logarítmica en ambos ejes.
        show_stats : bool
            Si True, inserta anotación con rho, p-valor e IC.
        filepath : str, opcional
            Ruta de guardado.

        Retorna
        -------
        Tuple[plt.Figure, plt.Axes]
        """
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
            scatter_kws={"s": 50, "alpha": 0.8, "edgecolor": "#0f172a"},
            line_kws={"linewidth": 1.8, "color": "#0369a1"}
        )

        lbl_x = self.get_label(metal_x) + (" [ln]" if log_scale else "")
        lbl_y = self.get_label(metal_y) + (" [ln]" if log_scale else "")

        ax.set_xlabel(lbl_x, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(lbl_y, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_title(f"Dispersión Bivariante: {self.get_label(metal_x)} vs {self.get_label(metal_y)}", fontsize=11, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_stats and len(sub_df) >= 3:
            sp = spearman_correlation(sub_df[metal_x], sub_df[metal_y], n_boot=n_boot)
            stat_text = (
                f"Spearman ρ = {sp['rho']:+.3f}\n"
                f"IC 95% Boot: [{sp['ci_low']:+.2f}, {sp['ci_high']:+.2f}]\n"
                f"Valor p = {sp['p_value']:.4f}\n"
                f"N = {sp['n']}"
            )
            ax.text(
                0.04, 0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
            )

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def ordinal_trend_plot(
        self,
        ordinal_col: str,
        metal: str,
        ordinal_map: Optional[Dict[str, int]] = None,
        log_scale: bool = False,
        show_stats: bool = True,
        figsize: Tuple[float, float] = (7.5, 5.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de tendencia ordinal (boxplots jerárquicos ordenados de menor a mayor frecuencia)
        para evaluar tendencias monótonas de exposición dietaria con la prueba de Jonckheere-Terpstra.

        Parámetros
        ----------
        ordinal_col : str
            Variable ordinal (ej. Alim_Pescado_mar).
        metal : str
            Metal analizado.
        ordinal_map : Dict[str, int], opcional
            Mapeo de niveles.
        filepath : str, opcional
            Ruta de guardado.

        Retorna
        -------
        Tuple[plt.Figure, plt.Axes]
        """
        sub_df = self.df[[ordinal_col, metal]].dropna().copy()
        sub_df[metal] = pd.to_numeric(sub_df[metal], errors="coerce")
        sub_df = sub_df.dropna()

        if ordinal_map is None and ordinal_col.startswith("Alim_"):
            ordinal_map = DIET_ORDINAL_MAP

        if ordinal_map is not None:
            sub_df["_ord_val"] = sub_df[ordinal_col].apply(
                lambda v: ordinal_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
            sub_df["_ord_val"] = pd.to_numeric(sub_df["_ord_val"], errors="coerce")
            sub_df = sub_df.dropna(subset=["_ord_val"])
            ordered_levels = sorted(sub_df["_ord_val"].unique())
            level_labels = [DIET_ORDINAL_LABELS.get(int(lvl), str(lvl)) for lvl in ordered_levels]
        else:
            ordered_levels = sorted(sub_df[ordinal_col].unique(), key=lambda x: str(x))
            level_labels = [str(lvl) for lvl in ordered_levels]

        fig, ax = plt.subplots(figsize=figsize)

        plot_x_col = "_ord_val" if ordinal_map is not None else ordinal_col

        sns.boxplot(
            data=sub_df,
            x=plot_x_col,
            y=metal,
            hue=plot_x_col,
            legend=False,
            order=ordered_levels,
            palette="mako",
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.85, edgecolor="#334155"),
            medianprops=dict(color="#0f172a", linewidth=2.0),
            showfliers=False
        )

        sns.stripplot(
            data=sub_df,
            x=plot_x_col,
            y=metal,
            order=ordered_levels,
            color="#0f172a",
            alpha=0.7,
            size=7,
            jitter=0.15,
            ax=ax
        )

        ax.set_xticks(range(len(ordered_levels)))
        ax.set_xticklabels(level_labels, fontsize=9.5)

        if log_scale:
            ax.set_yscale("log")

        var_lbl = self.get_label(ordinal_col)
        metal_lbl = self.get_label(metal)

        ax.set_title(f"Tendencia Ordinal: {metal_lbl} vs {var_lbl}", fontsize=11.5, fontweight="bold", pad=12)
        ax.set_xlabel(f"Nivel de Frecuencia ({var_lbl})", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(metal_lbl + (" (Escala Log)" if log_scale else ""), fontsize=10, fontweight="bold", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_stats and len(ordered_levels) >= 2:
            groups_data = [sub_df[sub_df[plot_x_col] == lvl][metal].values for lvl in ordered_levels]
            jt = jonckheere_terpstra_test(groups_data)
            sp = spearman_correlation(sub_df[plot_x_col], sub_df[metal])

            stat_text = (
                f"Jonckheere-Terpstra: p = {jt['p_value']:.4f}\n"
                f"Estadístico z = {jt['z_statistic']:+.2f}\n"
                f"Spearman ρ = {sp['rho']:+.3f} (p = {sp['p_value']:.4f})"
            )
            ax.text(
                0.03, 0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
            )

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def volcano_effect_plot(
        self,
        screening_df: pd.DataFrame,
        alpha_discovery: float = 0.10,
        figsize: Tuple[float, float] = (8.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un gráfico de Volcano / Tamaño del Efecto para el tamizaje exploratorio de covariables,
        representando el Tamaño del Efecto en el eje X vs -log10(p-valor) en el eje Y,
        resaltando las variables que alcanzan descubrimiento significativo bajo control FDR.

        Parámetros
        ----------
        screening_df : pd.DataFrame
            DataFrame generado por BivariateTables.exploratory_screening().to_dataframe().
        alpha_discovery : float
            Umbral de descubrimiento FDR (q-valor).
        filepath : str, opcional
            Ruta de guardado.

        Retorna
        -------
        Tuple[plt.Figure, plt.Axes]
        """
        df_plot = screening_df.copy()

        # Extraer p sin ajustar numérico
        def parse_p(val):
            s = str(val).replace("<strong>", "").replace("</strong>", "").replace("*", "").strip()
            try:
                return float(s)
            except Exception:
                return np.nan

        # Extraer tamaño del efecto numérico (r_rb o rho)
        def parse_eff(val):
            import re
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            if m:
                return float(m.group())
            return np.nan

        df_plot["p_num"] = df_plot["p sin ajustar"].apply(parse_p)
        df_plot["eff_num"] = df_plot["Tamaño del Efecto"].apply(parse_eff)
        df_plot["log10_p"] = -np.log10(np.clip(df_plot["p_num"], 1e-6, 1.0))

        # Hallazgo sugestivo
        disc_mask = df_plot["Hallazgo Sugestivo"].str.contains("Posible", na=False)

        fig, ax = plt.subplots(figsize=figsize)

        # No significativos
        ax.scatter(
            df_plot.loc[~disc_mask, "eff_num"],
            df_plot.loc[~disc_mask, "log10_p"],
            color="#94a3b8",
            alpha=0.6,
            s=45,
            label="Sin significación FDR"
        )

        # Significativos
        if disc_mask.any():
            ax.scatter(
                df_plot.loc[disc_mask, "eff_num"],
                df_plot.loc[disc_mask, "log10_p"],
                color="#e11d48",
                alpha=0.9,
                s=85,
                edgecolor="#881337",
                linewidth=1.2,
                label=f"Descubrimiento FDR (q < {alpha_discovery:.2f})"
            )

            # Etiquetar puntos destacados
            for _, r in df_plot[disc_mask].iterrows():
                lbl = f"{r['Factor Exploratorio']} ({r['Metal']})".replace("**", "")
                ax.annotate(
                    lbl,
                    (r["eff_num"], r["log10_p"]),
                    fontsize=8,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points"
                )

        # Línea de referencia p = 0.05
        ax.axhline(
            y=-np.log10(0.05),
            color="#f59e0b",
            linestyle="--",
            linewidth=1.2,
            label="Umbral nominal p = 0.05"
        )

        ax.set_xlabel("Magnitud del Tamaño del Efecto (r_rb / Spearman ρ)", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("-log10(p sin ajustar)", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_title("Tamizaje Exploratorio: Magnitud del Efecto vs Significancia Estadística", fontsize=11.5, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax
