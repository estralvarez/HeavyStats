"""
Módulo de visualización de factores y determinantes de exposición toxicológica.
Orquesta figuras compuestas y mosaicos multipanel para el Capítulo 4 de la tesis.
"""

import warnings
from typing import Dict, Optional, Any, Tuple, Union, Sequence
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from heavystats.univariate.plots.base import BasePlots
from heavystats.cleaning import desaggregate_multiple_responses


class ExposicionPlotsMixin:
    """Mixin que agrupa los pipelines de orquestación para figuras de exposición ambiental y factores de riesgo."""

    def plot_factores(
        self: BasePlots,
        dimensions: Optional[Dict[str, str]] = None,
        sep: str = ";",
        exclude_patterns: Optional[Sequence[str]] = ("ningun", "ninguno", "ninguna", "nada"),
        dim_colors: Optional[Dict[str, Any]] = None,
        n_total: Optional[int] = None,
        figsize: Tuple[float, float] = (8.5, 4.8),
        title: Optional[str] = None,
        xlabel: str = "Frecuencia ($n$)",
        legend_title: str = "Dimensión de Exposición",
        legend_loc: str = "lower right",
        bar_height: float = 0.60,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Orquesta la Figura de factores de exposición ambiental agrupados
        por dimensiones analíticas (talleres, industrias, lugares de riesgo) con barras coloreadas.
        """
        if dimensions is None:
            dimensions = {
                "Exposicion_Lugares": "Lugares de Riesgo",
                "Exposicion_Talleres": "Talleres y Servicios",
                "Exposicion_Industrias": "Industrias Químicas/Metales",
            }

        total = n_total if n_total is not None else len(self.df)
        items = []

        # Utilizar desaggregate_multiple_responses para obtener opciones limpias
        cols_presentes = [self._resolve_column(c) for c in dimensions.keys() if self._resolve_column(c) in self.df.columns]
        if cols_presentes:
            df_desag = desaggregate_multiple_responses(self.df, columns=cols_presentes, separator=sep)
            excl = [p.lower() for p in exclude_patterns] if exclude_patterns else []

            for col_orig, dim in dimensions.items():
                real_col = self._resolve_column(col_orig)
                prefix = f"{real_col}_"
                for c in df_desag.columns:
                    if c.startswith(prefix):
                        factor_name = c[len(prefix):].replace("_", " ")
                        if excl and any(p in factor_name.lower() for p in excl):
                            continue
                        count_val = int(df_desag[c].sum(skipna=True))
                        if count_val > 0:
                            items.append({"Dimension": dim, "Factor": factor_name, "n": count_val})

        if not items:
            warnings.warn("No se encontraron factores válidos para las dimensiones especificadas.", UserWarning)
            if ax is not None:
                return ax.figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            return fig

        counts = pd.DataFrame(items)
        counts["pct"] = (counts["n"] / total) * 100
        counts = counts.sort_values(by=["Dimension", "n"], ascending=[True, True])

        unique_dims = list(dict.fromkeys(counts["Dimension"]))
        palette_colors = sns.color_palette(self.palette, max(8, len(unique_dims) * 2 + 2))

        if dim_colors is None:
            dim_colors = {
                dim: palette_colors[(2 * idx + 1) % len(palette_colors)]
                for idx, dim in enumerate(unique_dims)
            }

        counts["color"] = counts["Dimension"].map(dim_colors)

        if ax is not None:
            fig = ax.figure
            target_ax = ax
        else:
            fig, target_ax = self.canvas(figsize=figsize, dpi=dpi, bold=True)

        self._draw_bar_panel(
            ax=target_ax,
            categories=counts["Factor"],
            values=counts["n"],
            total_n=total,
            percentages=counts["pct"],
            color=counts["color"],
            title=title,
            xlabel=xlabel,
            bar_width=bar_height,
            alpha=0.90,
            orientation="horizontal",
            bold=True,
        )

        handles = [mpatches.Patch(color=dim_colors[d], label=d) for d in unique_dims if d in dim_colors]
        target_ax.legend(handles=handles, title=legend_title, loc=legend_loc, frameon=True, fontsize=9.5)

        if ax is None:
            fig.tight_layout()

        self._save_figure(
            fig=fig,
            base_name="Factores_Exposicion_agrupado_elegante",
            save_dir=save_dir,
            formats=save_format,
            dpi=dpi,
            close=close_after_save,
        )
        return fig

    # Nombres simplificados
    plot_factores_exposicion = plot_factores
    factores = plot_factores

