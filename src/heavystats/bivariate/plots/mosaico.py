"""
Módulo de mosaicos y cuadrículas exploratorias bivariantes.
Implementa paneles multipanel coordinados para resumir múltiples determinantes
(binarios, politómicos, continuos y ordinales) frente a un biomarcador de interés.
"""

import math
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.bivariate.plots.base import BivariateBasePlots


class MosaicoPlotsMixin:
    """Mixin para cuadrículas y mosaicos exploratorios multideterminantes."""

    def plot_grid(
        self: BivariateBasePlots,
        metal: str = "Plomo_ug_dL",
        binary_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        ncols: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        filepath: Optional[str] = None,
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
                sns.boxplot(data=sub_df, x=col, y=metal_resolved, hue=col, legend=False, order=cats, palette=self.palette, ax=ax, width=0.45)
                sns.stripplot(data=sub_df, x=col, y=metal_resolved, order=cats, color="#b91c1c", alpha=0.90, size=5.5, edgecolor="#ffffff", linewidth=0.5, ax=ax)
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)}", fontsize=10, fontweight="bold")
            elif kind == "cont":
                sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
                sub_df = sub_df.dropna()
                sns.regplot(data=sub_df, x=col, y=metal_resolved, ax=ax, color="#0284c7", scatter_kws={"s": 35, "alpha": 0.7})
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)}", fontsize=10, fontweight="bold")
            elif kind == "ord":
                cats = sorted(sub_df[col].unique(), key=lambda x: str(x))
                sns.boxplot(data=sub_df, x=col, y=metal_resolved, hue=col, legend=False, order=cats, palette="mako", ax=ax, width=0.45)
                sns.stripplot(data=sub_df, x=col, y=metal_resolved, order=cats, color="#b91c1c", alpha=0.90, size=5.0, edgecolor="#ffffff", linewidth=0.5, ax=ax)
                ax.set_title(f"{self.get_label(metal_resolved)} vs {self.get_label(col)} (Ordinal)", fontsize=10, fontweight="bold")

            ax.set_xlabel(self.get_label(col), fontsize=9.5, fontweight="bold", labelpad=6)
            ax.set_ylabel(self.get_label(metal_resolved), fontsize=9.5, fontweight="bold", labelpad=6)
            self._clean_spines_and_ticks(ax)

        for j in range(n_plots, len(flat_axes)):
            flat_axes[j].set_visible(False)

        self._save_figure(fig, filepath=filepath)
        return fig, axes
