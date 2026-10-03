"""
Gráficos bivariantes con estética científica y calidad de publicación editorial.
Estructurado según los 4 Pilares del Análisis Bivariante Epidemiológico y Toxicológico:
  1. Pilar 1: Variable Cualitativa vs. Variable Cualitativa (Tablas de contingencia, OR, RR, Fisher exacto, V de Cramér).
  2. Pilar 2: Variable Cuantitativa vs. Variable Cualitativa (Comparación de carga corporal: No Paramétrica vs. Paramétrica).
  3. Pilar 3: Variable Cuantitativa vs. Variable Cuantitativa (Gradientes continuos, correlación, regresión y co-exposición).
  4. Pilar 4: Transición al Análisis Multivariable (Tamizaje con control FDR Benjamini-Hochberg y diagnóstico de colinealidad).
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
    DEFAULT_METAL_LIMITS,
    DEFAULT_METAL_PAIRS,
    CDC_LEAD_REFERENCE_VALUE,
    EPA_MERCURY_REFERENCE_VALUE,
    OMS_CADMIUM_REFERENCE_VALUE,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)
from heavystats.bivariate.tests import (
    mann_whitney_test,
    kruskal_wallis_test,
    spearman_correlation,
    jonckheere_terpstra_test,
    qualitative_association_test,
    independent_t_test,
    anova_oneway_test,
    pearson_correlation,
    kendall_correlation,
    spearman_matrix,
    collinearity_matrix,
    adjust_pvalues,
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
    Estructurada rigurosamente en los 4 Pilares del Análisis Bivariante.
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

    def _resolve_column(self, col: str) -> str:
        """Resuelve el nombre exacto de una columna en el DataFrame admitiendo alias comunes y formato enriquecido."""
        clean_c = str(col).replace("**", "").replace("<strong>", "").replace("</strong>", "").strip()
        if clean_c in self.df.columns:
            return clean_c

        clean_underscore = clean_c.replace(" ", "_")
        if clean_underscore in self.df.columns:
            return clean_underscore

        alias_map = {
            "plomo": "Plomo_ug_dL",
            "pb": "Plomo_ug_dL",
            "mercurio": "Mercurio_ug_L",
            "hg": "Mercurio_ug_L",
            "cadmio": "Cadmio_ug_L",
            "cd": "Cadmio_ug_L",
            "edad": "Edad",
            "age": "Edad",
            "peso": "Peso_kg",
            "peso_kg": "Peso_kg",
            "altura": "Altura_cm",
            "talla": "Altura_cm",
            "altura_cm": "Altura_cm",
            "score": "Score_Riesgo",
            "score_riesgo": "Score_Riesgo",
            "riesgo": "Score_Riesgo",
            "sexo": "Sexo",
            "sector": "Sector",
            "institucion": "Institucion",
            "es_expuesto": "Es_Expuesto",
            "es expuesto": "Es_Expuesto",
        }
        lower_c = clean_underscore.lower()
        if lower_c in alias_map and alias_map[lower_c] in self.df.columns:
            return alias_map[lower_c]

        for c in self.df.columns:
            if c.lower() == lower_c or c.lower().replace("_", " ") == clean_c.lower():
                return c

        return clean_underscore

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
    # PILAR 1: Variable Cualitativa vs. Variable Cualitativa
    # (Asociación Epidemiológica, Contingencia y Cuantificación del Riesgo)
    # =========================================================================

    def qualitative_association(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        var_x: Optional[str] = None,
        var_y: Optional[str] = None,
        target_cutoff: Optional[float] = None,
        normalize: str = "index",
        palette: Optional[Union[str, Sequence[str]]] = None,
        show_stats: bool = True,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 5.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 1: Gráfico de asociación cualitativa (tabla de contingencia 2x2 o RxC).
        Renderiza gráfico de barras agrupadas con proporciones (%) y recuentos muestrales (n=...),
        anotando en un badge superior el contraste exacto de Fisher, Odds Ratio (OR con IC 95%),
        Riesgo Relativo (RR con IC 95%) y V de Cramér.
        Si la variable dependiente 'y' es continua (ej. Mercurio_ug_L), se binariza automáticamente
        mediante su umbral de referencia toxicológica (ej. ≥ 5.0 µg/L).
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).qualitative_association(
                x=x, y=y, var_x=var_x, var_y=var_y, target_cutoff=target_cutoff,
                normalize=normalize, palette=palette, show_stats=show_stats,
                title=title, figsize=figsize, filepath=filepath
            )

        col_x = x if x is not None else var_x
        col_y = y if y is not None else var_y
        if col_x is None or col_y is None:
            raise ValueError("Debe especificar las dos variables cualitativas a contrastar (x e y).")

        x_col = self._resolve_column(col_x)
        y_col = self._resolve_column(col_y)

        # Binarizar y si es cuantitativa (ej. concentración de metal)
        s_y = self.df[y_col]
        is_y_numeric = pd.to_numeric(s_y, errors="coerce").notna().sum() > (0.5 * len(s_y.dropna()))
        if is_y_numeric:
            cutoff = target_cutoff or DEFAULT_METAL_LIMITS.get(y_col, 5.0)
            y_clean = np.where(pd.to_numeric(s_y, errors="coerce") >= cutoff, f"≥ {cutoff:g} µg/L", f"< {cutoff:g} µg/L")
            y_name_lbl = f"{self.get_label(y_col)} (≥ {cutoff:g} µg/L)"
            y_series = pd.Series(y_clean, index=self.df.index, name=y_col)
        else:
            y_name_lbl = self.get_label(y_col)
            y_series = s_y

        sub_df = pd.DataFrame({"x": self.df[x_col], "y": y_series}).dropna()
        if sub_df.empty:
            raise ValueError(f"No hay observaciones válidas para las variables '{x_col}' y '{y_col}'.")

        stat_res = qualitative_association_test(sub_df["x"], sub_df["y"])

        cats_x = sorted(sub_df["x"].unique(), key=lambda v: str(v))
        cats_y = sorted(sub_df["y"].unique(), key=lambda v: str(v))

        ct_counts = pd.crosstab(sub_df["x"], sub_df["y"]).reindex(index=cats_x, columns=cats_y, fill_value=0)
        ct_pcts = pd.crosstab(sub_df["x"], sub_df["y"], normalize="index").reindex(index=cats_x, columns=cats_y, fill_value=0.0) * 100.0

        fig, ax = plt.subplots(figsize=figsize)
        active_palette = palette or (["#0284c7", "#e11d48"] if len(cats_y) == 2 else "crest")
        if isinstance(active_palette, str):
            colors = sns.color_palette(active_palette, n_colors=len(cats_y))
        else:
            colors = list(active_palette)[:len(cats_y)]

        x_indices = np.arange(len(cats_x))
        n_y = len(cats_y)
        bar_width = 0.70 / max(n_y, 1)

        counts_x = [int((sub_df["x"] == c).sum()) for c in cats_x]
        xticklabels = [self._format_category_label(x_col, c, n_c) for c, n_c in zip(cats_x, counts_x)]

        for j, cat_y in enumerate(cats_y):
            offsets = x_indices - (0.35 - bar_width / 2.0) + j * bar_width
            pcts = ct_pcts[cat_y].values
            cnts = ct_counts[cat_y].values

            bars = ax.bar(
                offsets,
                pcts,
                width=bar_width * 0.90,
                label=str(cat_y),
                color=colors[j % len(colors)],
                edgecolor="#334155",
                linewidth=1.0,
                alpha=0.85
            )

            for bar, pct, cnt in zip(bars, pcts, cnts):
                if pct > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 1.5,
                        f"{pct:.1f}%\n(n={cnt})",
                        ha="center",
                        va="bottom",
                        fontsize=8.0,
                        fontweight="bold",
                        color="#0f172a"
                    )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(xticklabels, fontsize=9.5)
        ax.set_ylabel("Proporción en cada grupo (%)", fontsize=10, fontweight="medium", labelpad=8)
        ax.set_ylim(0, 118)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_name_lbl = self.get_label(x_col)
        ax.set_xlabel(x_name_lbl, fontsize=10, fontweight="medium", labelpad=8)

        legend_title = y_name_lbl.split("(")[0].strip() if "(" in y_name_lbl else y_name_lbl
        ax.legend(title=legend_title, frameon=True, framealpha=0.9, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9, title_fontsize=9.5, loc="upper right")

        if show_stats:
            p_val = stat_res.get("fisher_p", np.nan)
            p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else ("p < 0.001" if pd.notna(p_val) else "p = N/D")
            v_val = stat_res.get("cramers_v", np.nan)
            v_str = f"V = {v_val:.2f}" if pd.notna(v_val) else ""

            or_val = stat_res.get("odds_ratio", np.nan)
            or_ci = stat_res.get("or_ci", (np.nan, np.nan))
            rr_val = stat_res.get("relative_risk", np.nan)
            rr_ci = stat_res.get("rr_ci", (np.nan, np.nan))

            stat_parts = [f"Fisher exacto: {p_str}"]
            if pd.notna(or_val):
                ci_txt = f" [{or_ci[0]:.2f}, {or_ci[1]:.2f}]" if (pd.notna(or_ci[0]) and pd.notna(or_ci[1])) else ""
                stat_parts.append(f"OR = {or_val:.2f}{ci_txt}")
            if pd.notna(rr_val):
                ci_txt = f" [{rr_ci[0]:.2f}, {rr_ci[1]:.2f}]" if (pd.notna(rr_ci[0]) and pd.notna(rr_ci[1])) else ""
                stat_parts.append(f"RR = {rr_val:.2f}{ci_txt}")
            if v_str:
                stat_parts.append(v_str)

            stat_badge = "  |  ".join(stat_parts)
            plot_title = title or f"Asociación: {y_name_lbl} según {x_name_lbl}"
            ax.set_title(f"{plot_title}\n{stat_badge}", fontsize=10.0, fontweight="bold", pad=12)
        else:
            plot_title = title or f"Asociación: {y_name_lbl} según {x_name_lbl}"
            ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=12)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_qualitative_association(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para qualitative_association()."""
        return self.qualitative_association(*args, **kwargs)

    def qualitative_summary(
        self,
        target: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        target_cutoff: Optional[float] = None,
        target_col: Optional[str] = None,
        candidates: Optional[Sequence[str]] = None,
        top_n: int = 15,
        figsize: Tuple[float, float] = (8.5, 6.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 1: Forest Plot de Odds Ratios para múltiples factores cualitativos
        frente al umbral clínico del biomarcador diana (ej. Mercurio ≥ 5.0 µg/L).
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).qualitative_summary(
                target=target, feature_cols=feature_cols, target_cutoff=target_cutoff,
                target_col=target_col, candidates=candidates, top_n=top_n,
                figsize=figsize, filepath=filepath
            )

        t_col = target or target_col or "Mercurio_ug_L"
        target_resolved = self._resolve_column(t_col)
        cutoff = target_cutoff or DEFAULT_METAL_LIMITS.get(target_resolved, 5.0)

        from heavystats.bivariate.tables import BivariateTables
        bt = BivariateTables(self.df, labels_map=self.labels_map)
        rep = bt.qualitative_summary(
            target=target_resolved,
            target_cutoff=cutoff,
            feature_cols=feature_cols or candidates
        )
        df_summary = rep.df.copy()

        def parse_ci(val):
            m = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", str(val))
            if m:
                return float(m.group(1)), float(m.group(2))
            return np.nan, np.nan

        def parse_num(val):
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            if m:
                return float(m.group())
            return np.nan

        or_col = "Odds Ratio [IC 95%]" if "Odds Ratio [IC 95%]" in df_summary.columns else "Odds Ratio"
        p_col = "Fisher (p)" if "Fisher (p)" in df_summary.columns else "p-valor"
        var_col = "Factor de Exposición" if "Factor de Exposición" in df_summary.columns else "Variable"

        df_summary["_or"] = df_summary[or_col].apply(parse_num)
        ci_tuples = df_summary[or_col].apply(parse_ci)
        df_summary["_ci_low"] = [t[0] for t in ci_tuples]
        df_summary["_ci_high"] = [t[1] for t in ci_tuples]
        df_summary["_p"] = df_summary[p_col].apply(parse_num)

        valid_df = df_summary.dropna(subset=["_or"]).copy()
        valid_df["_ci_low_plot"] = np.clip(valid_df["_ci_low"], 0.05, 50.0)
        valid_df["_ci_high_plot"] = np.clip(valid_df["_ci_high"], 0.05, 50.0)
        valid_df["_or_plot"] = np.clip(valid_df["_or"], 0.05, 50.0)

        valid_df = valid_df.sort_values(by="_or", ascending=False).head(top_n).iloc[::-1]

        fig, ax = plt.subplots(figsize=figsize)
        if valid_df.empty:
            ax.text(0.5, 0.5, "Sin factores cualitativos con OR calculable", ha="center", va="center")
            return fig, ax

        y_pos = np.arange(len(valid_df))
        labels = [str(r[var_col]).replace("**", "").strip() for _, r in valid_df.iterrows()]

        err_left = np.maximum(0.0, valid_df["_or_plot"] - valid_df["_ci_low_plot"].fillna(valid_df["_or_plot"]))
        err_right = np.maximum(0.0, valid_df["_ci_high_plot"].fillna(valid_df["_or_plot"]) - valid_df["_or_plot"])

        sig_mask = valid_df["_p"] < 0.05
        colors = ["#e11d48" if s else "#0284c7" for s in sig_mask]

        for i, (idx, row) in enumerate(valid_df.iterrows()):
            ax.errorbar(
                row["_or_plot"],
                y_pos[i],
                xerr=[[err_left.iloc[i]], [err_right.iloc[i]]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1.6,
                capsize=3.5,
                markersize=6.5
            )

        ax.axvline(1.0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.0)

        t_lbl = self.get_label(target_resolved)
        ax.set_title(f"Pilar 1: Odds Ratios de Factores Cualitativos vs {t_lbl} (≥ {cutoff:g} µg/L)", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Odds Ratio (OR) e Intervalo de Confianza al 95% [Escala Log]", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_qualitative_summary(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para qualitative_summary()."""
        return self.qualitative_summary(*args, **kwargs)

    # =========================================================================
    # PILAR 2: Variable Cuantitativa vs. Variable Cualitativa
    # (Comparación de Carga Corporal entre Grupos: Dual Paramétrica / No Paramétrica)
    # =========================================================================

    def compare_groups(
        self,
        quantitative: Optional[Union[str, Sequence[str]]] = None,
        group: Optional[str] = None,
        method: str = "nonparametric",
        continuous_col: Optional[str] = None,
        group_col: Optional[str] = None,
        metal: Optional[str] = None,
        log_scale: bool = False,
        show_points: bool = True,
        show_stats: bool = True,
        show_limit: bool = True,
        permissible_limit: Optional[float] = None,
        limit_label: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6.5, 5.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 2: Comparación de concentraciones o variables continuas entre categorías de un factor cualitativo.
        Soporta modo no paramétrico (Mann-Whitney U / Kruskal-Wallis con medianas [RIQ] y stripplot)
        y modo paramétrico (t de Welch / ANOVA con medias ± DE, Hedges' g / η² y marcadores de media).
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).compare_groups(
                quantitative=quantitative, group=group, method=method,
                continuous_col=continuous_col, group_col=group_col, metal=metal,
                log_scale=log_scale, show_points=show_points, show_stats=show_stats,
                show_limit=show_limit, permissible_limit=permissible_limit,
                limit_label=limit_label, palette=palette, title=title,
                figsize=figsize, filepath=filepath
            )

        q_name = quantitative or continuous_col or metal or "Mercurio_ug_L"
        g_name = group or group_col or "Es_Expuesto"

        q_col = self._resolve_column(q_name)
        g_col = self._resolve_column(g_name)

        if g_col not in self.df.columns or q_col not in self.df.columns:
            raise ValueError(f"Las columnas '{g_col}' o '{q_col}' no se encuentran en el DataFrame.")

        sub_df = self.df[[g_col, q_col]].dropna().copy()
        sub_df[q_col] = pd.to_numeric(sub_df[q_col], errors="coerce")
        sub_df = sub_df.dropna()

        if log_scale:
            sub_df = sub_df[sub_df[q_col] > 0]

        if sub_df.empty:
            raise ValueError(f"No hay observaciones numéricas válidas para '{q_col}' agrupado por '{g_col}'.")

        fig, ax = plt.subplots(figsize=figsize)
        cats = sorted(sub_df[g_col].unique(), key=lambda x: str(x))
        counts = [int((sub_df[g_col] == c).sum()) for c in cats]
        xtick_labels = [self._format_category_label(g_col, c, n_c) for c, n_c in zip(cats, counts)]

        is_param = method.lower().startswith("param")
        active_palette = palette or (["#0284c7", "#0f766e"] if len(cats) == 2 else self.palette)

        if is_param:
            sns.boxplot(
                data=sub_df,
                x=g_col,
                y=q_col,
                hue=g_col,
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
                showfliers=False
            )
        else:
            sns.boxplot(
                data=sub_df,
                x=g_col,
                y=q_col,
                hue=g_col,
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

        if show_points:
            sns.stripplot(
                data=sub_df,
                x=g_col,
                y=q_col,
                order=cats,
                color="#0f172a",
                alpha=0.65,
                size=6.5,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.5,
                ax=ax
            )

        limit_val = self._resolve_limit(q_col, permissible_limit)
        if show_limit and limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85
            )
            ref_text = limit_label or DEFAULT_REFERENCE_LABELS.get(q_col, f"Ref: {limit_val:g}")
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

        var_lbl = self.get_label(g_col)
        q_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(q_col, self.get_label(q_col))
        if log_scale:
            q_axis_lbl += " (Escala Log)"

        plot_title = title if title is not None else f"{q_axis_lbl.split('(')[0].strip()} según {var_lbl}"
        ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel("", fontsize=9.5)
        ax.set_ylabel(q_axis_lbl, fontsize=10, fontweight="medium", labelpad=8)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(xtick_labels, fontsize=9.5)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_stats and len(cats) >= 2:
            y_max_data = float(sub_df[q_col].max()) if len(sub_df) > 0 else 1.0
            y_min_data = float(sub_df[q_col].min()) if len(sub_df) > 0 else 0.0
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
                    g1 = sub_df[sub_df[g_col] == cats[0]][q_col].values
                    g2 = sub_df[sub_df[g_col] == cats[1]][q_col].values
                    tt = independent_t_test(g1, g2, equal_var=False)
                    p_val = tt.get("p_val", np.nan)
                    g_hedges = tt.get("hedges_g", np.nan)
                    diff_m = tt.get("diff_means", np.nan)

                    p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else ("p < 0.001" if pd.notna(p_val) else "p = N/D")
                    g_str = f"g = {g_hedges:+.2f}" if pd.notna(g_hedges) else ""
                    diff_str = f"ΔMedia = {diff_m:+.2f}" if pd.notna(diff_m) else ""

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
                        fontweight="medium",
                        color="#0f172a"
                    )
                else:
                    groups_vals = [sub_df[sub_df[g_col] == c][q_col].values for c in cats]
                    an = anova_oneway_test(groups_vals, group_names=[str(c) for c in cats], equal_var=False)
                    p_val = an.get("p_val", np.nan)
                    f_stat = an.get("f_stat", np.nan)
                    eta_sq = an.get("eta_squared", np.nan)

                    p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else ("p < 0.001" if pd.notna(p_val) else "p = N/D")
                    f_str = f"F = {f_stat:.2f}" if pd.notna(f_stat) else "F = N/D"
                    eta_str = f"η² = {eta_sq:.2f}" if pd.notna(eta_sq) else ""

                    stat_parts = [f"ANOVA F: {p_str}", f_str]
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
                        fontweight="medium",
                        color="#0f172a"
                    )
            else:
                if len(cats) == 2:
                    g1 = sub_df[sub_df[g_col] == cats[0]][q_col].values
                    g2 = sub_df[sub_df[g_col] == cats[1]][q_col].values
                    mw = mann_whitney_test(g1, g2)
                    p_val = mw.get("p_val", mw.get("p_value", np.nan))
                    r_rb = mw.get("r_rb", mw.get("rank_biserial", np.nan))
                    diff_med = mw.get("diff_medians", np.nan)

                    p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else ("p < 0.001" if pd.notna(p_val) else "p = N/D")
                    r_str = f"r_rb = {r_rb:+.2f}" if pd.notna(r_rb) else ""
                    diff_str = f"ΔMed = {diff_med:+.2f}" if pd.notna(diff_med) else ""

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
                        fontweight="medium",
                        color="#0f172a"
                    )
                else:
                    groups_vals = [sub_df[sub_df[g_col] == c][q_col].values for c in cats]
                    kw = kruskal_wallis_test(groups_vals)
                    p_val = kw.get("p_val", kw.get("p_value", np.nan))
                    h_stat = kw.get("h_stat", kw.get("h_statistic", np.nan))
                    eps_sq = kw.get("epsilon_sq", kw.get("epsilon_squared", np.nan))

                    p_str = f"p = {p_val:.3f}" if (pd.notna(p_val) and p_val >= 0.001) else ("p < 0.001" if pd.notna(p_val) else "p = N/D")
                    h_str = f"H = {h_stat:.2f}" if pd.notna(h_stat) else "H = N/D"
                    eps_str = f"ε² = {eps_sq:.2f}" if pd.notna(eps_sq) else ""

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
                        fontweight="medium",
                        color="#0f172a"
                    )

            ax.set_ylim(top=y_top_limit)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_compare_groups(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para compare_groups()."""
        return self.compare_groups(*args, **kwargs)

    def metal_by_group(
        self,
        group_col: str,
        metal: str,
        method: str = "nonparametric",
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper de compatibilidad para compare_groups()."""
        return self.compare_groups(quantitative=metal, group=group_col, method=method, **kwargs)

    def plot_binary(
        self,
        metal: str,
        group_col: str,
        method: str = "nonparametric",
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico para variables dicotómicas/binarias."""
        return self.compare_groups(quantitative=metal, group=group_col, method=method, **kwargs)

    def plot_categorical(
        self,
        metal: str,
        group_col: str,
        method: str = "nonparametric",
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico para variables categóricas/politómicas (>2 grupos)."""
        return self.compare_groups(quantitative=metal, group=group_col, method=method, **kwargs)

    # =========================================================================
    # PILAR 3: Variable Cuantitativa vs. Variable Cuantitativa
    # (Gradientes Continuos, Correlación, Regresión y Co-Exposición)
    # =========================================================================

    def correlation_analysis(
        self,
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
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 3: Dispersión y correlación entre dos variables cuantitativas continuas.
        Modo no paramétrico: Spearman rho con IC 95% Bootstrap y Kendall tau-b.
        Modo paramétrico: Pearson r con IC 95% Fisher z, recta de regresión OLS y R².
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).correlation_analysis(
                x=x, y=y, method=method, continuous_col=continuous_col,
                metal=metal, target=target, target_col=target_col,
                log_scale=log_scale, show_stats=show_stats, n_boot=n_boot,
                title=title, figsize=figsize, filepath=filepath
            )

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
            line_kws={"linewidth": 1.7, "color": "#0369a1"}
        )

        lbl_x = DEFAULT_BIOMEDICAL_METAL_LABELS.get(col_x, self.get_label(col_x)) + (" [ln]" if log_scale else "")
        lbl_y = DEFAULT_BIOMEDICAL_METAL_LABELS.get(col_y, self.get_label(col_y)) + (" [ln]" if log_scale else "")

        ax.set_xlabel(lbl_x, fontsize=10, fontweight="medium", labelpad=8)
        ax.set_ylabel(lbl_y, fontsize=10, fontweight="medium", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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

                p_str = f"p = {p_val:.4f}" if (pd.notna(p_val) and p_val >= 0.0001) else "p < 0.0001"
                ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""
                eq_str = f"y = {slp:.2f}x + {icpt:.2f}" if (pd.notna(slp) and pd.notna(icpt)) else ""

                parts = [f"Pearson r = {r_val:+.3f}"]
                if ci_str:
                    parts.append(f"IC 95%: {ci_str}")
                if pd.notna(r2):
                    parts.append(f"R² = {r2:.3f}")
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

                p_str = f"p = {p_val:.4f}" if (pd.notna(p_val) and p_val >= 0.0001) else "p < 0.0001"
                ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if (pd.notna(ci_l) and pd.notna(ci_h)) else ""

                parts = [f"Spearman ρ = {rho_val:+.3f}"]
                if ci_str:
                    parts.append(f"IC 95%: {ci_str}")
                if pd.notna(tau_val):
                    parts.append(f"Kendall τ_b = {tau_val:+.3f}")
                parts.extend([p_str, f"n = {n_obs}"])
                stat_subtitle = "  |  ".join(parts)

        main_title = title or f"{self.get_label(col_y)} vs {self.get_label(col_x)}"
        if stat_subtitle:
            ax.set_title(f"{main_title}\n{stat_subtitle}", fontsize=10.0, fontweight="bold", pad=12)
        else:
            ax.set_title(main_title, fontsize=11.0, fontweight="bold", pad=10)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_correlation_analysis(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para correlation_analysis()."""
        return self.correlation_analysis(*args, **kwargs)

    def scatter_continuous(self, continuous_col: str, metal: str, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper de compatibilidad hacia correlation_analysis."""
        return self.correlation_analysis(x=continuous_col, y=metal, **kwargs)

    def plot_continuous(self, metal: str, column: str, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper de compatibilidad hacia correlation_analysis."""
        return self.correlation_analysis(x=column, y=metal, **kwargs)

    def scatter_metals(self, metal_x: str, metal_y: str, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper de compatibilidad hacia correlation_analysis."""
        return self.correlation_analysis(x=metal_x, y=metal_y, **kwargs)

    def coexposure_matrix(
        self,
        metals: Optional[Sequence[str]] = None,
        variables: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        n_boot: int = 2000,
        cmap: str = "Blues",
        figsize: Tuple[float, float] = (6.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 3: Matriz de Co-Exposición Inter-Metálica (Heatmap con intervalos Bootstrap y significancia).
        Evalúa sincrónicamente Pb, Hg y Cd (o variables continuas especificadas) calculando
        correlaciones cruzadas no paramétricas o paramétricas.
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).coexposure_matrix(
                metals=metals, variables=variables, method=method, n_boot=n_boot, cmap=cmap,
                figsize=figsize, filepath=filepath
            )

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
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5, 0.5,
                "Se requieren al menos 2 variables cuantitativas en el DataFrame\npara la matriz de co-exposición (ej. Hg, Pb, Cd).",
                ha="center", va="center", fontsize=9.5, color="#64748b"
            )
            ax.axis("off")
            return fig, ax

        sub_df = self.df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
        k = len(valid_cols)
        corr_matrix = np.zeros((k, k))
        annot_matrix = np.empty((k, k), dtype=object)

        is_param = method.lower().startswith("param")

        for i in range(k):
            for j in range(k):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    annot_matrix[i, j] = "1.00\n—"
                else:
                    col_i = valid_cols[i]
                    col_j = valid_cols[j]
                    if is_param:
                        pr = pearson_correlation(sub_df[col_i], sub_df[col_j])
                        r_val = pr.get("r", 0.0)
                        p_val = pr.get("p_val", 1.0)
                        ci_l = pr.get("ci_low", np.nan)
                        ci_h = pr.get("ci_high", np.nan)
                        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                        corr_matrix[i, j] = r_val
                        annot_matrix[i, j] = f"{r_val:+.2f}{stars}\n[{ci_l:+.2f}, {ci_h:+.2f}]"
                    else:
                        sp = spearman_correlation(sub_df[col_i], sub_df[col_j], n_boot=n_boot)
                        rho_val = sp.get("rho", 0.0)
                        p_val = sp.get("p_val", 1.0)
                        ci_l = sp.get("ci_low", np.nan)
                        ci_h = sp.get("ci_high", np.nan)
                        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                        corr_matrix[i, j] = rho_val
                        annot_matrix[i, j] = f"{rho_val:+.2f}{stars}\n[{ci_l:+.2f}, {ci_h:+.2f}]"

        fig, ax = plt.subplots(figsize=figsize)
        labels = [self.get_label(m) for m in valid_cols]

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot_matrix,
            fmt="",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.75, "label": "Pearson r" if is_param else "Spearman ρ_s"},
            ax=ax,
            annot_kws={"size": 8.5, "fontweight": "medium"}
        )

        stat_name = "Pearson (Paramétrico)" if is_param else "Spearman (No Paramétrico)"
        ax.set_title(f"Co-Exposición Inter-Metales ({stat_name})\nn = {len(sub_df)} (* p<0.05, ** p<0.01, *** p<0.001)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_coexposure_matrix(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para coexposure_matrix()."""
        return self.coexposure_matrix(*args, **kwargs)

    def correlation_matrix(
        self,
        metals: Optional[List[str]] = None,
        n_boot: int = 2000,
        cmap: str = "Blues",
        figsize: Tuple[float, float] = (6.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Matriz de correlación inter-metales con intervalos Bootstrap."""
        return self.coexposure_matrix(metals=metals, method="nonparametric", n_boot=n_boot, cmap=cmap, figsize=figsize, filepath=filepath)

    def plot_metal_matrix(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para correlation_matrix()."""
        return self.correlation_matrix(*args, **kwargs)

    def pairplot_metals(
        self,
        metals: Optional[List[str]] = None,
        hue: Optional[str] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        filepath: Optional[str] = None
    ) -> sns.PairGrid:
        """Pairplot estilizado para evaluar relaciones multivariadas entre metales."""
        target_metals = metals or DEFAULT_PRIMARY_METALS
        valid_metals = [self._resolve_column(m) for m in target_metals if self._resolve_column(m) in self.df.columns]

        cols_to_use = list(valid_metals)
        if hue and hue in self.df.columns:
            cols_to_use.append(hue)

        sub_df = self.df[cols_to_use].dropna().copy()
        for m in valid_metals:
            sub_df[m] = pd.to_numeric(sub_df[m], errors="coerce")
        sub_df = sub_df.dropna()

        rename_map = {m: DEFAULT_BIOMEDICAL_METAL_LABELS.get(m, self.get_label(m)) for m in valid_metals}
        sub_df_renamed = sub_df.rename(columns=rename_map)

        grid = sns.pairplot(
            data=sub_df_renamed,
            vars=list(rename_map.values()),
            hue=hue,
            palette=palette or self.palette,
            corner=True,
            diag_kind="kde",
            plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "#0f172a", "linewidths": 0.6}
        )

        grid.fig.suptitle("Matriz de Dispersión Bivariante Inter-Metales", y=1.02, fontsize=12, fontweight="bold")
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            grid.savefig(filepath, dpi=300, bbox_inches="tight")

        return grid

    def plot_metal_pairplot(self, *args, **kwargs) -> sns.PairGrid:
        """Alias para pairplot_metals()."""
        return self.pairplot_metals(*args, **kwargs)

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
        Pilar 3: Gráfico de tendencia ordinal (boxplots ordenados de menor a mayor frecuencia)
        para evaluar tendencias monótonas de exposición dietaria con la prueba de Jonckheere-Terpstra.
        """
        ord_resolved = self._resolve_column(ordinal_col)
        metal_resolved = self._resolve_column(metal)

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

        sns.boxplot(
            data=sub_df,
            x=plot_x_col,
            y=metal_resolved,
            hue=plot_x_col,
            legend=False,
            order=ordered_levels,
            palette=palette,
            ax=ax,
            width=0.40,
            boxprops=dict(alpha=0.75, edgecolor="#334155", linewidth=1.2),
            medianprops=dict(color="#0f172a", linewidth=2.2),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
            showfliers=False
        )

        if show_points:
            sns.stripplot(
                data=sub_df,
                x=plot_x_col,
                y=metal_resolved,
                order=ordered_levels,
                color="#0f172a",
                alpha=0.65,
                size=6.0,
                jitter=0.15,
                edgecolor="#ffffff",
                linewidth=0.5,
                ax=ax
            )

        limit_val = self._resolve_limit(metal_resolved, permissible_limit)
        if show_limit and limit_val is not None:
            ax.axhline(
                y=limit_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85
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
                fontweight="semibold"
            )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        var_lbl = self.get_label(ord_resolved)
        metal_axis_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_resolved, self.get_label(metal_resolved))
        if log_scale:
            metal_axis_lbl += " (Escala Log)"

        ax.set_title(f"{metal_axis_lbl.split('(')[0].strip()} según {var_lbl}", fontsize=11, fontweight="bold", pad=14)
        ax.set_xlabel(f"Gradiente Ordinal de {var_lbl}", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_ylabel(metal_axis_lbl, fontsize=10, fontweight="medium", labelpad=8)

        ax.set_xticks(range(len(ordered_levels)))
        ax.set_xticklabels(level_labels, fontsize=9.0)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_stats and len(ordered_levels) >= 2:
            y_max_data = float(sub_df[metal_resolved].max()) if len(sub_df) > 0 else 1.0
            y_min_data = float(sub_df[metal_resolved].min()) if len(sub_df) > 0 else 0.0
            y_span = (y_max_data - y_min_data) if y_max_data > y_min_data else 1.0

            y_bracket = y_max_data + 0.08 * y_span
            h_bracket = 0.03 * y_span
            y_top_limit = y_bracket + 0.12 * y_span

            groups_vals = [sub_df[sub_df[plot_x_col] == lvl][metal_resolved].values for lvl in ordered_levels]
            jt = jonckheere_terpstra_test(groups_vals)
            p_val = jt.get("p_val", jt.get("p_value", np.nan))
            z_score = jt.get("z_stat", jt.get("z_score", np.nan))

            p_str = f"p = {p_val:.4f}" if (pd.notna(p_val) and p_val >= 0.0001) else "p < 0.0001"
            z_str = f"z = {z_score:+.2f}" if pd.notna(z_score) else ""
            stat_text = f"Tendencia Monótona (Jonckheere–Terpstra): {z_str}  |  {p_str}"

            ax.plot([0, 0, len(ordered_levels)-1, len(ordered_levels)-1], [y_bracket - h_bracket, y_bracket, y_bracket, y_bracket - h_bracket], lw=1.1, c="#334155")
            ax.text(
                (len(ordered_levels) - 1) * 0.5,
                y_bracket + 0.01 * y_span,
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

    def plot_dietary(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias semántico para ordinal_trend_plot()."""
        return self.ordinal_trend_plot(*args, **kwargs)

    def plot_grid(
        self,
        metal: str = "Plomo_ug_dL",
        binary_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        ncols: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Cuadrícula integrada de gráficos bivariantes para exploración general."""
        metal_resolved = self._resolve_column(metal)
        tasks = []
        if binary_cols:
            for c in binary_cols:
                if c in self.df.columns:
                    tasks.append(("binary", c))
        if categorical_cols:
            for c in categorical_cols:
                if c in self.df.columns:
                    tasks.append(("cat", c))
        if continuous_cols:
            for c in continuous_cols:
                if c in self.df.columns:
                    tasks.append(("cont", c))
        if ordinal_cols:
            for c in ordinal_cols:
                if c in self.df.columns:
                    tasks.append(("ord", c))

        n_plots = len(tasks)
        if n_plots == 0:
            raise ValueError("No se especificaron columnas válidas para la cuadrícula exploratoria.")

        nrows = math.ceil(n_plots / ncols)
        fig_h = nrows * 4.2 if figsize is None else figsize[1]
        fig_w = ncols * 5.5 if figsize is None else figsize[0]

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        flat_axes = np.array(axes).flatten()

        for idx, (kind, col) in enumerate(tasks):
            ax = flat_axes[idx]
            sub_df = self.df[[col, metal_resolved]].dropna().copy()
            sub_df[metal_resolved] = pd.to_numeric(sub_df[metal_resolved], errors="coerce")
            sub_df = sub_df.dropna()

            if kind in ("binary", "cat"):
                cats = sorted(sub_df[col].unique(), key=lambda x: str(x))
                sns.boxplot(data=sub_df, x=col, y=metal_resolved, order=cats, palette=self.palette, ax=ax, width=0.45)
                sns.stripplot(data=sub_df, x=col, y=metal_resolved, order=cats, color="#0f172a", alpha=0.55, size=5.0, ax=ax)
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)}", fontsize=10, fontweight="bold")
            elif kind == "cont":
                sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
                sub_df = sub_df.dropna()
                sns.regplot(data=sub_df, x=col, y=metal_resolved, ax=ax, color="#0284c7", scatter_kws={"s": 35, "alpha": 0.7})
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)}", fontsize=10, fontweight="bold")
            elif kind == "ord":
                cats = sorted(sub_df[col].unique(), key=lambda x: str(x))
                sns.boxplot(data=sub_df, x=col, y=metal_resolved, order=cats, palette="mako", ax=ax, width=0.45)
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)} (Ordinal)", fontsize=10, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(n_plots, len(flat_axes)):
            flat_axes[j].set_visible(False)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, axes

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

        valid_metals = [self._resolve_column(m) for m in metals if self._resolve_column(m) in self.df.columns]
        m_count = len(valid_metals)
        fig, axes = plt.subplots(1, m_count, figsize=figsize, sharey=False)
        if m_count == 1:
            axes = [axes]

        for idx, metal in enumerate(valid_metals):
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

    def plot_risk_algorithm(self, *args, **kwargs) -> Tuple[plt.Figure, Any]:
        """Alias para risk_algorithm_plots()."""
        return self.risk_algorithm_plots(*args, **kwargs)

    # =========================================================================
    # PILAR 4: Transición Hacia el Análisis Multivariable
    # (Tamizaje, FDR Benjamini-Hochberg y Diagnóstico de Colinealidad)
    # =========================================================================

    def multivariate_screening(
        self,
        target: Optional[str] = None,
        candidate_features: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        screening_p_threshold: float = 0.20,
        fdr_alpha: float = 0.10,
        screening_report: Optional[Any] = None,
        top_n: int = 12,
        figsize: Tuple[float, float] = (13.5, 5.5),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """
        Pilar 4: Dashboard editorial integrado de tamizaje bivariante (*screening*) multivariable.
        Panel A: Volcano Plot (-log10(p) vs Tamaño de Efecto) con umbrales de Hosmer-Lemeshow (p < 0.20)
        y control de FDR Benjamini-Hochberg (q < 0.10).
        Panel B: Forest Plot de predictores candidatos priorizados con intervalos de confianza
        y banderas de alerta de colinealidad (|rho| >= 0.70) para resguardar la parsimonia con n=20.
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).multivariate_screening(
                target=target, candidate_features=candidate_features, method=method,
                screening_p_threshold=screening_p_threshold, fdr_alpha=fdr_alpha,
                screening_report=screening_report, top_n=top_n, figsize=figsize, filepath=filepath
            )

        t_col = target or "Mercurio_ug_L"
        target_resolved = self._resolve_column(t_col)

        if screening_report is None:
            from heavystats.bivariate.tables import BivariateTables
            bt = BivariateTables(self.df, labels_map=self.labels_map)
            rep = bt.multivariate_screening(
                target=target_resolved,
                candidate_features=candidate_features,
                method=method,
                screening_p_threshold=screening_p_threshold,
                fdr_alpha=fdr_alpha
            )
        else:
            rep = screening_report

        df_screen = rep.df.copy()

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

        def parse_ci(val):
            m = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", str(val))
            if m:
                return float(m.group(1)), float(m.group(2))
            return np.nan, np.nan

        var_col = next((c for c in ["Variable Predictora", "Predictor Candidato", "Variable", "Factor Exploratorio", "Variable_Raw"] if c in df_screen.columns), df_screen.columns[0])
        p_col = next((c for c in ["p (Crudo)", "p-valor (crudo)", "p_Raw", "Valor p", "p sin ajustar"] if c in df_screen.columns), None)
        fdr_col = next((c for c in ["FDR p-valor (BH)", "p (FDR)", "fdr_p"] if c in df_screen.columns), None)
        eff_col = next((c for c in ["Tamaño Efecto", "Tamaño del Efecto [IC 95%]", "Tamaño del Efecto", "Efecto_Num"] if c in df_screen.columns), None)
        collin_col = "Colinealidad (rho max)" if "Colinealidad (rho max)" in df_screen.columns else None

        df_screen["_p_num"] = df_screen[p_col].apply(parse_p) if p_col else 1.0
        df_screen["_fdr_num"] = df_screen[fdr_col].apply(parse_p) if fdr_col else 1.0
        df_screen["_eff_num"] = df_screen[eff_col].apply(parse_eff) if eff_col else 0.0
        ci_tuples = df_screen[eff_col].apply(parse_ci) if eff_col else [(np.nan, np.nan)] * len(df_screen)
        df_screen["_ci_low"] = [t[0] for t in ci_tuples]
        df_screen["_ci_high"] = [t[1] for t in ci_tuples]
        df_screen["_log10_p"] = -np.log10(np.clip(df_screen["_p_num"], 1e-5, 1.0))

        fig, (ax_volcano, ax_forest) = plt.subplots(1, 2, figsize=figsize)

        # -------------------------------------------------------------
        # PANEL A: VOLCANO PLOT
        # -------------------------------------------------------------
        mask_fdr = df_screen["_fdr_num"] < fdr_alpha
        mask_screen = (df_screen["_p_num"] < screening_p_threshold) & (~mask_fdr)
        mask_none = ~mask_fdr & ~mask_screen

        ax_volcano.scatter(
            df_screen.loc[mask_none, "_eff_num"],
            df_screen.loc[mask_none, "_log10_p"],
            color="#94a3b8",
            alpha=0.60,
            s=45,
            edgecolors="none",
            label=f"Descartado (p ≥ {screening_p_threshold:.2f})"
        )
        if mask_screen.any():
            ax_volcano.scatter(
                df_screen.loc[mask_screen, "_eff_num"],
                df_screen.loc[mask_screen, "_log10_p"],
                color="#0284c7",
                alpha=0.85,
                s=70,
                edgecolor="#0f172a",
                linewidths=0.8,
                label=f"Candidato Hosmer (p < {screening_p_threshold:.2f})"
            )
        if mask_fdr.any():
            ax_volcano.scatter(
                df_screen.loc[mask_fdr, "_eff_num"],
                df_screen.loc[mask_fdr, "_log10_p"],
                color="#e11d48",
                alpha=0.95,
                s=90,
                edgecolor="#881337",
                linewidths=1.2,
                label=f"Descubrimiento FDR (q < {fdr_alpha:.2f})"
            )

        y_screen_line = -np.log10(screening_p_threshold)
        ax_volcano.axhline(y_screen_line, color="#0284c7", linestyle="--", linewidth=1.1, alpha=0.8, label=f"Corte Hosmer (p = {screening_p_threshold:.2f})")

        for _, r in df_screen[mask_screen | mask_fdr].iterrows():
            lbl = str(r[var_col]).replace("**", "").split("(")[0].strip()
            ax_volcano.annotate(
                lbl,
                (r["_eff_num"], r["_log10_p"]),
                fontsize=7.8,
                fontweight="semibold",
                xytext=(4, 4),
                textcoords="offset points"
            )

        ax_volcano.set_title("A. Tamizaje de Predictores (Volcano Plot)", fontsize=11, fontweight="bold", pad=12)
        ax_volcano.set_xlabel("Magnitud del Tamaño del Efecto (|r_rb|, |ρ_s|)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax_volcano.set_ylabel("-log₁₀(p-valor crudo)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax_volcano.legend(loc="upper left", fontsize=8, framealpha=0.9, facecolor="#f8fafc", edgecolor="#cbd5e1")
        ax_volcano.spines["top"].set_visible(False)
        ax_volcano.spines["right"].set_visible(False)

        # -------------------------------------------------------------
        # PANEL B: FOREST PLOT DE PREDICTORES PRIORIZADOS
        # -------------------------------------------------------------
        candidates_df = df_screen[df_screen["_p_num"] < screening_p_threshold].copy()
        if candidates_df.empty:
            candidates_df = df_screen.sort_values(by="_p_num").head(top_n).copy()
        else:
            candidates_df = candidates_df.sort_values(by="_p_num").head(top_n)

        candidates_df = candidates_df.iloc[::-1]

        y_pos = np.arange(len(candidates_df))
        labels_b = []
        point_colors = []

        for _, r in candidates_df.iterrows():
            lbl = str(r[var_col]).replace("**", "").strip()
            collin_val = parse_eff(r.get(collin_col, 0.0)) if collin_col else 0.0
            is_collinear = abs(collin_val) >= 0.70
            if is_collinear:
                lbl += f" [⚠ |ρ|={abs(collin_val):.2f}]"
                point_colors.append("#d97706")
            elif r["_fdr_num"] < fdr_alpha:
                point_colors.append("#e11d48")
            else:
                point_colors.append("#0284c7")
            labels_b.append(lbl)

        err_left = np.maximum(0.0, candidates_df["_eff_num"] - candidates_df["_ci_low"].fillna(candidates_df["_eff_num"]))
        err_right = np.maximum(0.0, candidates_df["_ci_high"].fillna(candidates_df["_eff_num"]) - candidates_df["_eff_num"])

        for i, (idx, row) in enumerate(candidates_df.iterrows()):
            ax_forest.errorbar(
                row["_eff_num"],
                y_pos[i],
                xerr=[[err_left.iloc[i]], [err_right.iloc[i]]],
                fmt="o",
                color=point_colors[i],
                ecolor=point_colors[i],
                elinewidth=1.6,
                capsize=3.5,
                markersize=6.5
            )

        ax_forest.axvline(0.0, color="#64748b", linestyle="--", linewidth=1.1, alpha=0.7)
        ax_forest.set_yticks(y_pos)
        ax_forest.set_yticklabels(labels_b, fontsize=8.5)
        ax_forest.set_title("B. Candidatos Priorizados Multivariables (Top)", fontsize=11, fontweight="bold", pad=12)
        ax_forest.set_xlabel("Tamaño del Efecto e Intervalo de Confianza al 95%", fontsize=9.5, fontweight="medium", labelpad=8)
        ax_forest.spines["top"].set_visible(False)
        ax_forest.spines["right"].set_visible(False)

        target_lbl = self.get_label(target_resolved)
        fig.suptitle(f"Pilar 4: Tamizaje Bivariante hacia el Análisis Multivariable frente a {target_lbl}", fontsize=12, fontweight="bold", y=1.02)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, (ax_volcano, ax_forest)

    def plot_multivariate_screening(self, *args, **kwargs) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Alias para multivariate_screening()."""
        return self.multivariate_screening(*args, **kwargs)

    def collinearity_heatmap(
        self,
        candidates: Optional[Sequence[str]] = None,
        screening_report: Optional[Any] = None,
        target: str = "Mercurio_ug_L",
        threshold: float = 0.70,
        cmap: str = "vlag",
        figsize: Tuple[float, float] = (7.0, 6.0),
        filepath: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pilar 4: Heatmap de Diagnóstico de Colinealidad entre Predictores Priorizados.
        Evalúa correlaciones inter-variables (|rho| >= 0.70) para descartar redundancias
        y preservar la regla de parsimonia (2-3 predictores con n=20).
        """
        if isinstance(self, pd.DataFrame):
            return BivariatePlots(self).collinearity_heatmap(
                candidates=candidates, screening_report=screening_report,
                target=target, threshold=threshold, cmap=cmap,
                figsize=figsize, filepath=filepath
            )

        if candidates is None:
            if screening_report is not None and hasattr(screening_report, "df"):
                df_s = screening_report.df
            else:
                from heavystats.bivariate.tables import BivariateTables
                bt = BivariateTables(self.df, labels_map=self.labels_map)
                rep = bt.multivariate_screening(target=target, screening_p_threshold=0.20)
                df_s = rep.df

            var_col = next((c for c in ["Variable Predictora", "Predictor Candidato", "Variable", "Factor Exploratorio"] if c in df_s.columns), df_s.columns[0])
            p_col = next((c for c in ["p (Crudo)", "p-valor (crudo)", "p_Raw", "Valor p"] if c in df_s.columns), None)

            def parse_p(v):
                try:
                    return float(str(v).replace("<strong>", "").replace("</strong>", "").replace("*", "").replace("<", "").strip())
                except Exception:
                    return 1.0

            if p_col:
                candidates = df_s[df_s[p_col].apply(parse_p) < 0.20][var_col].head(8).tolist()
            else:
                candidates = df_s[var_col].head(8).tolist()

        valid_cands = [self._resolve_column(c) for c in candidates if self._resolve_column(c) in self.df.columns]
        if len(valid_cands) < 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Menos de 2 predictores candidatos para evaluar colinealidad.", ha="center", va="center")
            return fig, ax

        sub_cands = self.df[valid_cands].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        corr_mat = sub_cands.corr(method="spearman").fillna(0.0)

        labels = [self.get_label(c) for c in valid_cands]
        k = len(valid_cands)
        annot_mat = np.empty((k, k), dtype=object)

        for i in range(k):
            for j in range(k):
                val = corr_mat.iloc[i, j]
                warn = " ⚠" if (i != j and abs(val) >= threshold) else ""
                annot_mat[i, j] = f"{val:+.2f}{warn}"

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

        sns.heatmap(
            corr_mat,
            mask=mask,
            annot=annot_mat,
            fmt="",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.75, "label": "Spearman ρ_s"},
            ax=ax,
            annot_kws={"size": 8.5, "fontweight": "medium"}
        )

        ax.set_title(f"Diagnóstico de Colinealidad entre Predictores Candidatos\n(Alerta ⚠ en |ρ| ≥ {threshold:.2f} para parsimonia con n=20)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_collinearity_heatmap(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para collinearity_heatmap()."""
        return self.collinearity_heatmap(*args, **kwargs)

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
        Genera un gráfico de Volcano / Tamaño del Efecto para el tamizaje exploratorio de covariables.
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
        ax.set_title(f"Tamizaje Bivariante (Volcano Plot): {metal_axis_lbl}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Tamaño del Efecto (diferencia o correlación)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.set_ylabel("-log₁₀(p)", fontsize=9.5, fontweight="medium", labelpad=8)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_volcano(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para volcano_effect_plot()."""
        return self.volcano_effect_plot(*args, **kwargs)

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

    def plot_forest_effects(self, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Alias para screening_forest_plot()."""
        return self.screening_forest_plot(*args, **kwargs)


# =============================================================================
# Funciones Modulares de Conveniencia a Nivel de Módulo
# =============================================================================

def qualitative_association_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).qualitative_association()."""
    return BivariatePlots(df).qualitative_association(*args, **kwargs)


def qualitative_summary_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).qualitative_summary()."""
    return BivariatePlots(df).qualitative_summary(*args, **kwargs)


def compare_groups_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).compare_groups()."""
    return BivariatePlots(df).compare_groups(*args, **kwargs)


def correlation_analysis_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).correlation_analysis()."""
    return BivariatePlots(df).correlation_analysis(*args, **kwargs)


def coexposure_matrix_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).coexposure_matrix()."""
    return BivariatePlots(df).coexposure_matrix(*args, **kwargs)


def multivariate_screening_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Función de conveniencia para BivariatePlots(df).multivariate_screening()."""
    return BivariatePlots(df).multivariate_screening(*args, **kwargs)


def collinearity_heatmap_plot(df: pd.DataFrame, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Función de conveniencia para BivariatePlots(df).collinearity_heatmap()."""
    return BivariatePlots(df).collinearity_heatmap(*args, **kwargs)


__all__ = [
    "BivariatePlots",
    "qualitative_association_plot",
    "qualitative_summary_plot",
    "compare_groups_plot",
    "correlation_analysis_plot",
    "coexposure_matrix_plot",
    "multivariate_screening_plot",
    "collinearity_heatmap_plot",
    "DEFAULT_BIOMEDICAL_METAL_LABELS",
    "DEFAULT_REFERENCE_LABELS",
]
