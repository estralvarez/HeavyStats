"""
Módulo de perfil de subcohortes, similitud correlacional y factores compartidos.
Implementa:
  1. Ranking de similitud / correlación de Spearman frente a biomarcadores metálicos.
  2. Contraste bioestadístico y toxicológico de subcohortes con anotación individual de IDs.
  3. Minería de factores compartidos y prevalencia diferencial.
  4. Mosaico editorial multipanel coordinado (Paneles A, B y C).
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, ttest_ind

from heavystats.bivariate.plots.base import (
    BivariateBasePlots,
    DEFAULT_BIOMEDICAL_METAL_LABELS,
)

DEFAULT_DIETARY_MAP: Dict[Any, str] = {
    0: "Nunca",
    1: "Rara vez",
    2: "A veces",
    3: "Frecuente",
    4: "Diario",
    "0": "Nunca",
    "1": "Rara vez",
    "2": "A veces",
    "3": "Frecuente",
    "4": "Diario",
}


class SubcohortPlotsMixin:
    """Mixin para análisis de subcohortes, rankings correlacionales y factores compartidos."""

    def similarity_ranking_plot(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        item_cols: Optional[List[str]] = None,
        highlight_item: Optional[str] = None,
        highlight_color: str = "#2b5c8f",
        sig_color: str = "#d95f02",
        nonsig_color: str = "#999999",
        method: str = "nonparametric",
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (8.0, 5.5),
        filepath: Optional[str] = None,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        """
        Gráfico A: Ranking horizontal de asociación correlacional de un conjunto
        de variables (e.g. dieta o ambientales) frente a la concentración biológica del metal.

        Parameters
        ----------
        target_metal : str
            Columna de concentración del metal.
        item_cols : list of str, optional
            Lista de columnas a evaluar. Si es None, selecciona todas las que inicien con 'Alim_'.
        highlight_item : str, optional
            Nombre de la columna a resaltar visualmente.
        highlight_color : str
            Color para el ítem resaltado.
        sig_color : str
            Color para ítems estadísticamente significativos (p < 0.05).
        nonsig_color : str
            Color para ítems no significativos.
        method : {"nonparametric", "parametric"}, default="nonparametric"
            Método de correlación: "nonparametric" (Spearman rho) o "parametric" (Pearson r).
        title : str, optional
            Título del gráfico.
        ax : plt.Axes, optional
            Eje de matplotlib donde renderizar. Si es None, se crea nueva figura.
        figsize : tuple
            Dimensiones de la figura (si ax es None).
        filepath : str, optional
            Ruta para exportar la figura.
        dpi : int
            Resolución en puntos por pulgada.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes, pd.DataFrame]
        """
        metal_resolved = self._resolve_column(target_metal)
        if metal_resolved not in self.df.columns:
            raise ValueError(f"Columna de metal '{metal_resolved}' no encontrada en el DataFrame.")

        if item_cols is None:
            item_cols = [c for c in self.df.columns if c.startswith("Alim_")]

        valid_items = [c for c in item_cols if c in self.df.columns]
        if not valid_items:
            raise ValueError("No se encontraron columnas válidas para calcular el ranking de similitud.")

        is_parametric = str(method).lower() in ["parametric", "paramétrico", "pearson"]
        res_correlacion = []
        for c in valid_items:
            sub = self.df[[c, metal_resolved]].dropna()
            if len(sub) < 3:
                continue
            if is_parametric:
                coef, pval = pearsonr(sub[c], sub[metal_resolved])
            else:
                coef, pval = spearmanr(sub[c], sub[metal_resolved])

            clean_name = c.replace("Alim_", "").replace("Exposicion_", "").replace("Salud_", "").replace("_", " ").title()
            coef_val = float(coef) if pd.notna(coef) else 0.0
            p_val = float(pval) if pd.notna(pval) else 1.0
            res_correlacion.append({
                "Variable": c,
                "Nombre": clean_name,
                "Coeficiente": coef_val,
                "Spearman_rho": coef_val,
                "Pearson_r": coef_val,
                "p_valor": p_val,
                "Significativo": "p < 0.05" if p_val < 0.05 else "n.s.",
                "Metodo": "Pearson" if is_parametric else "Spearman",
            })

        df_similitud = pd.DataFrame(res_correlacion).sort_values(by="Coeficiente", ascending=False).reset_index(drop=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        colores_barras = []
        for _, row in df_similitud.iterrows():
            if highlight_item and row["Variable"] == highlight_item:
                colores_barras.append(highlight_color)
            elif row["p_valor"] < 0.05:
                colores_barras.append(sig_color)
            else:
                colores_barras.append(nonsig_color)

        ax.barh(
            df_similitud["Nombre"],
            df_similitud["Coeficiente"],
            color=colores_barras,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        if is_parametric:
            ax.set_xlabel(r"Coeficiente de Pearson ($r$)", fontsize=9.5, fontweight="bold")
            default_title = "A. Similitud de Alimentos vs. Concentración Sanguínea (Pearson $r$)"
        else:
            ax.set_xlabel(r"Coeficiente de Spearman ($\rho_s$)", fontsize=9.5, fontweight="bold")
            default_title = "A. Similitud de Alimentos vs. Concentración Sanguínea (Spearman $\rho_s$)"
        
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        else:
            ax.set_title(default_title, fontsize=10.5, fontweight="bold", loc="left")

        ax.grid(axis="x", linestyle=":", alpha=0.6)
        self._clean_spines_and_ticks(ax)

        # Anotación textual para el ítem evaluado si aplica
        if highlight_item:
            for idx, row in df_similitud.iterrows():
                if row["Variable"] == highlight_item:
                    offset = 0.02 if row["Coeficiente"] >= 0 else -0.02
                    ha = "left" if row["Coeficiente"] >= 0 else "right"
                    ax.text(
                        row["Coeficiente"] + offset,
                        idx,
                        "Ítem Evaluado",
                        va="center",
                        ha=ha,
                        fontsize=8,
                        fontweight="bold",
                        color=highlight_color,
                    )

        if filepath:
            self._save_figure(fig, filepath=filepath, dpi=dpi)

        return fig, ax, df_similitud

    def subcohort_contrast_plot(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        subcohort_mask: Optional[Union[pd.Series, np.ndarray, Sequence[bool]]] = None,
        criterion_col: Optional[str] = None,
        criterion_val: Optional[Any] = None,
        comparison_op: str = "==",
        method: str = "nonparametric",
        subcohort_label: Optional[str] = None,
        rest_label: Optional[str] = None,
        id_col: Optional[str] = None,
        annotate_ids: bool = False,
        show_ids: bool = False,
        palette: Optional[Sequence[str]] = None,
        point_color: str = "#08519c",
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (6.5, 5.0),
        filepath: Optional[str] = None,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """
        Gráfico B: Contraste bioestadístico y toxicológico de distribución plasmática entre
        el subgrupo aislado (por ejemplo, la opción de consumo de mayor frecuencia) y el resto
        de la cohorte.

        Parameters
        ----------
        target_metal : str
            Columna de concentración del metal.
        subcohort_mask : boolean mask, optional
            Máscara booleana para seleccionar la subcohorte.
        criterion_col : str, optional
            Columna para definir el criterio de subcohorte si no se pasa mask.
        criterion_val : Any, optional
            Valor de corte para criterion_col. Si es None, selecciona automáticamente la opción
            con mayor cantidad de muestras (categoría modal).
        comparison_op : str, default="=="
            Operador de comparación ("==", ">=", "<=", ">", "<").
        method : {"nonparametric", "parametric"}, default="nonparametric"
            Método estadístico de contraste: "nonparametric" (Mann-Whitney U) o "parametric" (t de Welch).
        subcohort_label : str, optional
            Etiqueta del subgrupo. Si es None, se genera automáticamente basada en el criterio.
        rest_label : str, optional
            Etiqueta del resto de la cohorte (default: "Resto Cohorte").
        id_col : str, optional
            Columna con identificador del participante (si se desea anotar).
        annotate_ids : bool, default=False
            Si es True, anota los IDs de cada participante sobre el gráfico.
        show_ids : bool, default=False
            Alias de annotate_ids.
        palette : sequence of str, optional
            Colores para las cajas del boxplot.
        point_color : str, default="#08519c"
            Color para los puntos del stripplot.
        title : str, optional
            Título del gráfico.
        ax : plt.Axes, optional
            Eje de matplotlib donde renderizar.
        figsize : tuple, default=(6.5, 5.0)
            Dimensiones de la figura.
        filepath : str, optional
            Ruta para exportar.
        dpi : int, default=300
            Resolución en DPI.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes, Dict[str, Any]]
        """
        metal_resolved = self._resolve_column(target_metal)
        if metal_resolved not in self.df.columns:
            raise ValueError(f"Columna de metal '{metal_resolved}' no encontrada en el DataFrame.")

        # Construcción de la máscara si no fue suministrada directamente
        if subcohort_mask is None:
            if criterion_col is None:
                raise ValueError("Debe suministrar 'subcohort_mask' o 'criterion_col'.")
            col_crit = self._resolve_column(criterion_col)
            if col_crit not in self.df.columns:
                raise ValueError(f"Columna de criterio '{col_crit}' no encontrada en el DataFrame.")
            
            s = self.df[col_crit]
            
            # Si criterion_val no fue especificado, tomar la categoría con la mayor cantidad de muestras
            if criterion_val is None:
                counts = s.dropna().value_counts()
                if counts.empty:
                    raise ValueError(f"La columna '{col_crit}' no contiene valores válidos para definir la subcohorte.")
                criterion_val = counts.idxmax()

            if comparison_op == "==":
                mask = (s == criterion_val)
            elif comparison_op == ">=":
                mask = (s >= criterion_val)
            elif comparison_op == "<=":
                mask = (s <= criterion_val)
            elif comparison_op == ">":
                mask = (s > criterion_val)
            elif comparison_op == "<":
                mask = (s < criterion_val)
            else:
                raise ValueError(f"Operador '{comparison_op}' no reconocido.")
        else:
            mask = pd.Series(subcohort_mask, index=self.df.index).astype(bool)

        # Generar etiquetas descriptivas si no fueron provistas
        if subcohort_label is None or subcohort_label == "Subcohorte":
            if criterion_col is not None:
                clean_col = criterion_col.replace("Alim_", "").replace("Salud_", "").replace("Exposicion_", "").replace("_", " ").title()
                cat_desc = DEFAULT_DIETARY_MAP.get(criterion_val, str(criterion_val))
                if comparison_op == "==":
                    subcohort_label = f"{clean_col}: {cat_desc}"
                else:
                    subcohort_label = f"{clean_col} {comparison_op} {cat_desc}"
            else:
                subcohort_label = "Subcohorte"

        if rest_label is None:
            rest_label = "Resto Cohorte"

        subgrupo = self.df[mask].copy()
        resto = self.df[~mask].copy()

        n_sub = len(subgrupo)
        n_res = len(resto)

        if n_sub == 0 or n_res == 0:
            raise ValueError(f"División inválida: Subcohorte tiene n={n_sub} y Resto tiene n={n_res}.")

        hg_sub = pd.to_numeric(subgrupo[metal_resolved], errors="coerce").dropna()
        hg_res = pd.to_numeric(resto[metal_resolved], errors="coerce").dropna()

        is_parametric = str(method).lower() in ["parametric", "paramétrico", "welch", "t", "ttest"]

        if is_parametric:
            t_res = ttest_ind(hg_sub, hg_res, equal_var=False)
            t_stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)
            mean_sub = float(hg_sub.mean())
            std_sub = float(hg_sub.std())
            mean_res = float(hg_res.mean())
            std_res = float(hg_res.std())

            # Hedges' g
            n1, n2 = len(hg_sub), len(hg_res)
            diff_mean = mean_sub - mean_res
            denom = (n1 - 1) * (std_sub ** 2) + (n2 - 1) * (std_res ** 2)
            s_pooled = np.sqrt(denom / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
            d_val = diff_mean / s_pooled if s_pooled > 0 else 0.0
            correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2 > 2) else 1.0
            hedges_g = float(d_val * correction)

            stats_dict = {
                "method": "parametric",
                "n_subcohort": n_sub,
                "n_rest": n_res,
                "t_stat": t_stat,
                "p_value": p_val,
                "hedges_g": hedges_g,
                "subcohort_mean": mean_sub,
                "subcohort_std": std_sub,
                "rest_mean": mean_res,
                "rest_std": std_res,
                "criterion_col": criterion_col,
                "criterion_val": criterion_val,
            }
            test_name = "t de Welch"
        else:
            u_stat, u_pval = mannwhitneyu(hg_sub, hg_res, alternative="two-sided")
            q25_sub, med_sub, q75_sub = np.percentile(hg_sub, [25, 50, 75])
            q25_res, med_res, q75_res = np.percentile(hg_res, [25, 50, 75])

            r_rb = 1.0 - (2.0 * float(u_stat) / (n_sub * n_res)) if (n_sub * n_res) > 0 else 0.0
            p_val = float(u_pval)

            stats_dict = {
                "method": "nonparametric",
                "n_subcohort": n_sub,
                "n_rest": n_res,
                "u_stat": float(u_stat),
                "p_value": p_val,
                "rank_biserial_r": float(r_rb),
                "subcohort_median": float(med_sub),
                "subcohort_iqr": (float(q25_sub), float(q75_sub)),
                "rest_median": float(med_res),
                "rest_iqr": (float(q25_res), float(q75_res)),
                "criterion_col": criterion_col,
                "criterion_val": criterion_val,
            }
            test_name = "Mann–Whitney"

        lbl_sub = f"{subcohort_label}\n(n={n_sub})"
        lbl_res = f"{rest_label}\n(n={n_res})"

        df_plot = pd.DataFrame({
            "Grupo": [lbl_sub if m else lbl_res for m in mask],
            "Metal": pd.to_numeric(self.df[metal_resolved], errors="coerce"),
            "Is_Sub": mask.values,
        }).dropna(subset=["Metal"])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        box_palette = palette or ["#6baed6", "#bdd7e7"]

        sns.boxplot(
            data=df_plot,
            x="Grupo",
            y="Metal",
            hue="Grupo",
            legend=False,
            palette=box_palette,
            width=0.45,
            ax=ax,
            fliersize=0,
        )
        sns.stripplot(
            data=df_plot,
            x="Grupo",
            y="Metal",
            color=point_color,
            size=7,
            jitter=0.18,
            ax=ax,
            alpha=0.85,
        )

        # Anotación opcional de identificadores de participantes
        should_annotate = bool(annotate_ids or show_ids)
        if should_annotate and id_col and id_col in self.df.columns:
            for _, r in subgrupo.iterrows():
                val_m = float(r[metal_resolved])
                p_id = r[id_col]
                id_text = f"ID {int(p_id):02d}" if isinstance(p_id, (int, float, np.number)) else f"ID {p_id}"
                ax.text(
                    0.20,
                    val_m,
                    id_text,
                    fontsize=7.5,
                    va="center",
                    color=point_color,
                    fontweight="bold",
                )

        metal_lbl = DEFAULT_BIOMEDICAL_METAL_LABELS.get(metal_resolved, metal_resolved)
        ax.set_ylabel(metal_lbl, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        self._clean_spines_and_ticks(ax)

        p_str = self._format_p_val(p_val)
        sig_tag = "Significativo p < 0.05" if p_val < 0.05 else "n.s."
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        else:
            ax.set_title(
                f"B. Distribución de Carga Corporal ({test_name} {p_str}, {sig_tag})",
                fontsize=10.5,
                fontweight="bold",
                loc="left",
            )

        if filepath:
            self._save_figure(fig, filepath=filepath, dpi=dpi)

        return fig, ax, stats_dict

    def shared_factors_plot(
        self: BivariateBasePlots,
        subcohort_mask: Union[pd.Series, np.ndarray, Sequence[bool]],
        subcohort_label: str = "Subcohorte",
        rest_label: str = "Resto Cohorte",
        consensus_threshold_pct: float = 60.0,
        ignore_cols: Optional[List[str]] = None,
        max_factors: int = 10,
        subcohort_color: str = "#2171b5",
        rest_color: str = "#bdbdbd",
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10.0, 5.5),
        filepath: Optional[str] = None,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        """
        Gráfico C: Factores compartidos y prevalencia diferencial entre la subcohorte
        y el resto de la cohorte para variables con consenso superior al umbral.
        """
        mask = pd.Series(subcohort_mask, index=self.df.index).astype(bool)
        subgrupo = self.df[mask].copy()
        resto = self.df[~mask].copy()

        default_ignore = [
            "Muestra_Codificada", "ID", "Mercurio_ug_L", "Plomo_ug_dL", "Cadmio_ug_L",
            "Score_Riesgo", "Es_Expuesto"
        ]
        if ignore_cols:
            default_ignore.extend(ignore_cols)

        etiquetas_dieta = {
            0: "Nunca", 1: "Rara vez", 2: "A veces", 3: "Frecuentemente", 4: "Diario"
        }

        factores_compartidos = []
        for c in self.df.columns:
            if c in default_ignore:
                continue
            vc_sub = subgrupo[c].value_counts(normalize=True)
            if vc_sub.empty:
                continue
            cat_top = vc_sub.index[0]
            pct_sub = vc_sub.iloc[0] * 100.0

            if pct_sub >= consensus_threshold_pct:
                vc_res = resto[c].value_counts(normalize=True)
                pct_res = vc_res.get(cat_top, 0.0) * 100.0
                diff_pct = pct_sub - pct_res

                cat_repr = str(cat_top)
                if c.startswith("Alim_") and cat_top in etiquetas_dieta:
                    cat_repr = f"{cat_top} ({etiquetas_dieta[cat_top]})"

                clean_var = c.replace("Exposicion_", "Exp_").replace("Salud_", "")
                factores_compartidos.append({
                    "Variable": c,
                    "Etiqueta": f"{clean_var} = {cat_repr}",
                    "Prevalencia_Subgrupo_%": pct_sub,
                    "Prevalencia_Resto_%": pct_res,
                    "Diferencial_%": diff_pct,
                })

        df_compartidos = pd.DataFrame(factores_compartidos)
        if not df_compartidos.empty:
            df_compartidos = df_compartidos.sort_values(
                by=["Diferencial_%", "Prevalencia_Subgrupo_%"], ascending=[False, False]
            ).head(max_factors)
            # Ordenar ascendente para la gráfica horizontal
            df_plot = df_compartidos.sort_values(by="Diferencial_%", ascending=True)
        else:
            df_plot = pd.DataFrame(columns=["Etiqueta", "Prevalencia_Subgrupo_%", "Prevalencia_Resto_%"])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        if not df_plot.empty:
            y_pos = np.arange(len(df_plot))
            ax.barh(
                y_pos - 0.18,
                df_plot["Prevalencia_Subgrupo_%"],
                height=0.35,
                color=subcohort_color,
                label=subcohort_label,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.barh(
                y_pos + 0.18,
                df_plot["Prevalencia_Resto_%"],
                height=0.35,
                color=rest_color,
                label=rest_label,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_plot["Etiqueta"], fontsize=8.5)
            ax.legend(frameon=True, fontsize=8.5, loc="lower right")
        else:
            ax.text(0.5, 0.5, "No se identificaron factores compartidos que superen el umbral.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)

        ax.set_xlabel("Prevalencia en la Muestra (%)", fontsize=9.5, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        self._clean_spines_and_ticks(ax)

        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        else:
            ax.set_title(f"C. Factores Compartidos con Mayor Sobre-Representación (Consenso >= {consensus_threshold_pct:.0f}%)",
                         fontsize=10.5, fontweight="bold", loc="left")

        if filepath:
            self._save_figure(fig, filepath=filepath, dpi=dpi)

        return fig, ax, df_compartidos

    def subcohort_mosaic_plot(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        criterion_col: str = "Alim_Leguminosas",
        criterion_val: Optional[Any] = None,
        comparison_op: str = "==",
        method: str = "nonparametric",
        subcohort_label: Optional[str] = None,
        rest_label: Optional[str] = "Resto Cohorte",
        id_col: Optional[str] = None,
        annotate_ids: bool = False,
        show_ids: bool = False,
        consensus_threshold_pct: float = 60.0,
        include_shared_factors: bool = True,
        figsize: Tuple[float, float] = (16.0, 10.0),
        filepath: Optional[str] = None,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, Any]]:
        """
        Mosaico editorial completo para análisis de subcohorte:
        Combina en una cuadrícula multipanel:
          - Panel A: Ranking correlacional (Spearman rho o Pearson r).
          - Panel B: Contraste toxicológico del biomarcador.
          - Panel C (opcional): Factores compartidos y prevalencia diferencial.
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)

        col_crit = self._resolve_column(criterion_col)
        s = self.df[col_crit]

        # Si criterion_val es None, seleccionar la categoría modal (mayor número de muestras)
        if criterion_val is None:
            counts = s.dropna().value_counts()
            criterion_val = counts.idxmax() if not counts.empty else 2

        # Determinar etiqueta automática del subgrupo si no se suministra
        if subcohort_label is None:
            clean_crit = criterion_col.replace("Alim_", "").replace("Salud_", "").replace("Exposicion_", "").replace("_", " ").title()
            cat_desc = DEFAULT_DIETARY_MAP.get(criterion_val, str(criterion_val))
            op_sym = "=" if comparison_op == "==" else comparison_op
            subcohort_label = f"{clean_crit} {op_sym} {cat_desc}"

        # Construir máscara
        if comparison_op == "==":
            mask = (s == criterion_val)
        elif comparison_op == ">=":
            mask = (s >= criterion_val)
        elif comparison_op == "<=":
            mask = (s <= criterion_val)
        elif comparison_op == ">":
            mask = (s > criterion_val)
        elif comparison_op == "<":
            mask = (s < criterion_val)
        else:
            raise ValueError(f"Operador '{comparison_op}' no reconocido.")

        axes_dict: Dict[str, plt.Axes] = {}
        results_dict: Dict[str, Any] = {}

        if include_shared_factors:
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])

            _, _, df_sim = self.similarity_ranking_plot(
                target_metal=target_metal,
                highlight_item=criterion_col,
                method=method,
                ax=ax1,
            )
            _, _, stats_dict = self.subcohort_contrast_plot(
                target_metal=target_metal,
                subcohort_mask=mask,
                method=method,
                subcohort_label=subcohort_label,
                rest_label=rest_label or "Resto Cohorte",
                id_col=id_col,
                annotate_ids=annotate_ids,
                show_ids=show_ids,
                ax=ax2,
            )
            _, _, df_shared = self.shared_factors_plot(
                subcohort_mask=mask,
                subcohort_label=subcohort_label,
                rest_label=rest_label or "Resto Cohorte",
                consensus_threshold_pct=consensus_threshold_pct,
                ignore_cols=[criterion_col],
                ax=ax3,
            )

            axes_dict = {"ax_similarity": ax1, "ax_contrast": ax2, "ax_factors": ax3}
            results_dict = {
                "similarity_df": df_sim,
                "contrast_stats": stats_dict,
                "shared_factors_df": df_shared,
            }
        else:
            gs = fig.add_gridspec(1, 2, wspace=0.25)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])

            _, _, df_sim = self.similarity_ranking_plot(
                target_metal=target_metal,
                highlight_item=criterion_col,
                method=method,
                ax=ax1,
            )
            _, _, stats_dict = self.subcohort_contrast_plot(
                target_metal=target_metal,
                subcohort_mask=mask,
                method=method,
                subcohort_label=subcohort_label,
                rest_label=rest_label or "Resto Cohorte",
                id_col=id_col,
                annotate_ids=annotate_ids,
                show_ids=show_ids,
                ax=ax2,
            )

            axes_dict = {"ax_similarity": ax1, "ax_contrast": ax2}
            results_dict = {
                "similarity_df": df_sim,
                "contrast_stats": stats_dict,
            }

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout()
            except Exception:
                pass
        if filepath:
            self._save_figure(fig, filepath=filepath, dpi=dpi)

        return fig, axes_dict, results_dict

    def scan_subcohort_contrasts(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        candidate_cols: Optional[List[str]] = None,
        method: str = "nonparametric",
        alpha: float = 0.05,
        test_all_categories: bool = False,
        min_n: int = 2,
        plot_significant: bool = True,
        filepath_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Itera y evalúa contrastes toxicológicos de carga corporal a través de un conjunto
        de variables (e.g. dieta o exposición), identificando aquellas con diferencias
        estadísticamente significativas (p < alpha).

        Parameters
        ----------
        target_metal : str, default="Mercurio_ug_L"
            Columna con las concentraciones biológicas del metal.
        candidate_cols : list of str, optional
            Columnas a escanear. Por defecto, todas las columnas que inician con 'Alim_'.
        method : {"nonparametric", "parametric"}, default="nonparametric"
            Método estadístico: "nonparametric" (Mann-Whitney U) o "parametric" (t de Welch).
        alpha : float, default=0.05
            Nivel de significancia estadística umbral.
        test_all_categories : bool, default=False
            Si es False, evalúa únicamente la opción modal (mayor frecuencia) vs resto.
            Si es True, evalúa todas las opciones individuales de consumo con N >= min_n.
        min_n : int, default=2
            Tamaño muestral mínimo por subgrupo para ser evaluado.
        plot_significant : bool, default=True
            Si es True, genera y muestra automáticamente el subcohort_contrast_plot para
            cada variable o categoría que resulte estadísticamente significativa.
        filepath_prefix : str, optional
            Prefijo de ruta para guardar las figuras generadas en disco.

        Returns
        -------
        pd.DataFrame
            Tabla consolidada con todas las comparaciones ordenadas ascendentemente por p-valor.
        """
        metal_resolved = self._resolve_column(target_metal)
        if metal_resolved not in self.df.columns:
            raise ValueError(f"Columna de metal '{metal_resolved}' no encontrada en el DataFrame.")

        if candidate_cols is None:
            candidate_cols = [c for c in self.df.columns if c.startswith("Alim_")]

        is_parametric = str(method).lower() in ["parametric", "paramétrico", "welch", "t", "ttest"]
        metal_series = pd.to_numeric(self.df[metal_resolved], errors="coerce")

        records = []
        significant_items = []

        for col in candidate_cols:
            col_res = self._resolve_column(col)
            if col_res not in self.df.columns:
                continue
            s = self.df[col_res]
            counts = s.dropna().value_counts()
            if counts.empty:
                continue

            if test_all_categories:
                vals_to_test = [(val, "Moda vs Resto" if val == counts.idxmax() else f"Cat {val} vs Resto") for val in counts.index]
            else:
                vals_to_test = [(counts.idxmax(), "Moda vs Resto")]

            for val, tipo in vals_to_test:
                mask = (s == val)
                n_sub = int(mask.sum())
                n_res = len(self.df) - n_sub
                if n_sub < min_n or n_res < min_n:
                    continue

                y_sub = metal_series[mask].dropna()
                y_res = metal_series[~mask].dropna()
                if len(y_sub) < min_n or len(y_res) < min_n:
                    continue

                clean_name = col_res.replace("Alim_", "").replace("Salud_", "").replace("Exposicion_", "").replace("_", " ").title()
                cat_desc = DEFAULT_DIETARY_MAP.get(val, str(val))

                if is_parametric:
                    t_res = ttest_ind(y_sub, y_res, equal_var=False)
                    p_val = float(t_res.pvalue)
                    m_sub, s_sub = float(y_sub.mean()), float(y_sub.std())
                    m_res, s_res = float(y_res.mean()), float(y_res.std())

                    n1, n2 = len(y_sub), len(y_res)
                    diff_m = m_sub - m_res
                    denom = (n1 - 1) * (s_sub ** 2) + (n2 - 1) * (s_res ** 2)
                    s_pool = np.sqrt(denom / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
                    d_v = diff_m / s_pool if s_pool > 0 else 0.0
                    g_val = float(d_v * (1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)))) if (n1 + n2 > 2) else 0.0

                    rec = {
                        "Variable": col_res,
                        "Nombre": clean_name,
                        "Valor": val,
                        "Categoria": cat_desc,
                        "Tipo": tipo,
                        "N_Sub": n_sub,
                        "N_Rest": n_res,
                        "Media_Sub": m_sub,
                        "Media_Rest": m_res,
                        "Hedges_g": g_val,
                        "t_stat": float(t_res.statistic),
                        "p_valor": p_val,
                        "Significativo": bool(p_val < alpha),
                        "Metodo": "t de Welch",
                    }
                else:
                    u_stat, u_pval = mannwhitneyu(y_sub, y_res, alternative="two-sided")
                    p_val = float(u_pval)
                    med_sub = float(y_sub.median())
                    med_res = float(y_res.median())
                    r_rb = 1.0 - (2.0 * float(u_stat) / (n_sub * n_res)) if (n_sub * n_res) > 0 else 0.0

                    rec = {
                        "Variable": col_res,
                        "Nombre": clean_name,
                        "Valor": val,
                        "Categoria": cat_desc,
                        "Tipo": tipo,
                        "N_Sub": n_sub,
                        "N_Rest": n_res,
                        "Mediana_Sub": med_sub,
                        "Mediana_Rest": med_res,
                        "Rank_Biserial_r": float(r_rb),
                        "u_stat": float(u_stat),
                        "p_valor": p_val,
                        "Significativo": bool(p_val < alpha),
                        "Metodo": "Mann–Whitney",
                    }

                records.append(rec)
                if p_val < alpha:
                    significant_items.append((col_res, val, cat_desc, p_val))

        df_results = pd.DataFrame(records)
        if not df_results.empty:
            df_results = df_results.sort_values(by="p_valor", ascending=True).reset_index(drop=True)

        if plot_significant and significant_items:
            for item in significant_items:
                c_col, c_val, c_desc, p_v = item
                save_fp = f"{filepath_prefix}_{c_col}_val{c_val}.png" if filepath_prefix else None
                self.subcohort_contrast_plot(
                    target_metal=target_metal,
                    criterion_col=c_col,
                    criterion_val=c_val,
                    method=method,
                    annotate_ids=False,
                    filepath=save_fp,
                )
                plt.show()

        return df_results
