"""
Módulo base de estilos, utilidades y renderizado gráfico univariante.
Define configuraciones de publicación en negrita (#0f172a), resolución de variables y límites.
"""

import os
import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)


class BasePlots:
    """
    Motor central de Canvas, estilizado editorial (#0f172a) y renderizado gráfico.
    Puede instanciarse sin datos para gestionar lienzos y gráficos atómicos,
    o con un DataFrame para análisis estadístico univariante.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        palette: str = "crest",
        style: str = "ticks",
        context: str = "notebook",
        labels_map: Optional[Dict[str, str]] = None,
        rc: Optional[Dict[str, Any]] = None,
    ):
        self.df = df.copy() if df is not None else None
        self.palette = palette
        self.style = style
        self.context = context
        self.labels_map = DEFAULT_LABELS_MAP.copy()
        if labels_map is not None:
            self.labels_map.update(labels_map)

        self.rc = DEFAULT_CUSTOM_PARAMS.copy()
        if rc is not None:
            self.rc.update(rc)

        # Configuración estética global
        sns.set_theme(
            style=self.style,
            palette=self.palette,
            rc=self.rc,
            context=self.context,
        )
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

    # =========================================================================
    # Creación y Gestión del Canvas (Lienzo)
    # =========================================================================

    def canvas(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
        sharex: bool = False,
        sharey: bool = False,
        gridspec_kw: Optional[Dict[str, Any]] = None,
        bold: bool = True,
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Crea el canvas (Figure y Axes) aplicando el estilizado editorial (#0f172a).
        Listo para usar con o sin datos asociados.
        """
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
        )
        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                self._style_axis(ax, bold=bold)
        else:
            self._style_axis(axes, bold=bold)
        return fig, axes

    crear_lienzo = canvas
    create_canvas = canvas

    def crear_mosaico(
        self,
        mosaic: Union[Sequence[Sequence[Any]], Sequence[Any], str],
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
        height_ratios: Optional[Sequence[float]] = None,
        width_ratios: Optional[Sequence[float]] = None,
        bold: bool = True,
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Crea un mosaico multipanel estructurado con estilo de publicación aplicado a cada panel.

        Formatos admitidos para `mosaic`:
        - Cuadrícula 2x2 (4 gráficos): ``[['a', 'b'], ['c', 'd']]`` o ``"a b; c d"``
        - 3 en 1 columna (3x1): ``[['a'], ['b'], ['c']]`` o ``"a; b; c"``
        - 3 en 1 fila (1x3): ``[['a', 'b', 'c']]`` o ``['a', 'b', 'c']`` o ``"a b c"``
        - Diseños asimétricos: ``[['a', 'a'], ['b', 'c']]``
        """
        layout = mosaic
        if isinstance(mosaic, (list, tuple)) and len(mosaic) > 0 and all(isinstance(x, str) for x in mosaic):
            # Lista 1D de strings -> 1 fila con N columnas
            layout = [list(mosaic)]

        mosaic_kws = {}
        if height_ratios is not None:
            mosaic_kws["height_ratios"] = list(height_ratios)
        if width_ratios is not None:
            mosaic_kws["width_ratios"] = list(width_ratios)

        fig, axd = plt.subplot_mosaic(
            layout,
            figsize=figsize,
            dpi=dpi,
            **mosaic_kws,
        )
        for ax in axd.values():
            self._style_axis(ax, bold=bold)
        return fig, axd

    create_mosaic = crear_mosaico
    mosaic = crear_mosaico
    mosaico = crear_mosaico

    # =========================================================================
    # Métodos Auxiliares Privados
    # =========================================================================

    def _resolve_column(self, col: str) -> str:
        """Resuelve el nombre exacto de la columna en el DataFrame a partir de nombres o alias comunes."""
        if self.df is None:
            return col
        if col in self.df.columns:
            return col
        col_clean = str(col).strip()
        if col_clean in self.df.columns:
            return col_clean

        alias_map = {
            "plomo": "Plomo_ug_dL",
            "pb": "Plomo_ug_dL",
            "lead": "Plomo_ug_dL",
            "plomo_ug_dl": "Plomo_ug_dL",
            "mercurio": "Mercurio_ug_L",
            "hg": "Mercurio_ug_L",
            "mercury": "Mercurio_ug_L",
            "mercurio_ug_l": "Mercurio_ug_L",
            "cadmio": "Cadmio_ug_L",
            "cd": "Cadmio_ug_L",
            "cadmium": "Cadmio_ug_L",
            "cadmio_ug_l": "Cadmio_ug_L",
            "edad": "Edad",
            "age": "Edad",
            "peso": "Peso_kg",
            "peso_kg": "Peso_kg",
            "weight": "Peso_kg",
            "altura": "Altura_cm",
            "talla": "Altura_cm",
            "altura_cm": "Altura_cm",
            "height": "Altura_cm",
            "score": "Score_Riesgo",
            "score_riesgo": "Score_Riesgo",
            "riesgo": "Score_Riesgo",
            "sexo": "Sexo",
            "gender": "Sexo",
            "sector": "Sector",
            "es_expuesto": "Es_Expuesto",
            "exposicion": "Es_Expuesto",
        }
        col_lower = col_clean.lower()
        if col_lower in alias_map and alias_map[col_lower] in self.df.columns:
            return alias_map[col_lower]

        for c in self.df.columns:
            if c.lower() == col_lower:
                return c

        return col

    def _normalize_columns(self, columns: Union[str, Sequence[str]]) -> List[str]:
        """Normaliza y resuelve el argumento de columnas para admitir un string único o secuencias."""
        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        return [self._resolve_column(c) for c in cols]

    def _clean_series(self, col: str, min_obs: int = 1) -> Optional[pd.Series]:
        """
        Extrae y limpia una columna numérica del DataFrame, convirtiendo a tipo numérico
        y descartando valores nulos.
        """
        real_col = self._resolve_column(col)
        if real_col not in self.df.columns:
            warnings.warn(
                f"La columna '{col}' no se encuentra en el DataFrame. Omitiendo gráfico.",
                UserWarning,
                stacklevel=3,
            )
            return None

        data_clean = pd.to_numeric(self.df[real_col], errors="coerce").dropna()
        if len(data_clean) < min_obs:
            warnings.warn(
                f"La columna '{col}' contiene menos de {min_obs} observaciones válidas. Omitiendo gráfico.",
                UserWarning,
                stacklevel=3,
            )
            return None

        return data_clean

    def _resolve_limit(
        self,
        col: str,
        permissible_limit: Optional[float] = None,
        permissible_limits: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Resuelve el límite permisible aplicable con precedencia jerárquica clara."""
        if permissible_limit is not None:
            return permissible_limit

        real_col = self._resolve_column(col)
        active_limits = DEFAULT_PERMISSIBLE_LIMITS.copy()
        if permissible_limits is not None:
            active_limits.update(permissible_limits)

        if real_col in active_limits:
            return active_limits[real_col]
        if col in active_limits:
            return active_limits[col]
        return active_limits.get(col.lower())

    @staticmethod
    def _style_axis(
        ax: plt.Axes,
        tick_length: float = 4.5,
        labelsize: float = 11.5,
        hide_top_right: bool = True,
        bold: bool = False,
    ) -> None:
        """Aplica la jerarquía visual de bordes (spines) y ticks de calidad de publicación."""
        ax.set_axisbelow(True)
        ax.grid(False)
        if hide_top_right:
            for s in ["top", "right"]:
                if s in ax.spines:
                    ax.spines[s].set_visible(False)
        spine_color = "#0f172a" if bold else "#222222"
        for s in ["left", "bottom"]:
            if s in ax.spines:
                ax.spines[s].set_color(spine_color)
                ax.spines[s].set_linewidth(1.3)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=labelsize,
            width=1.3,
            length=tick_length,
            colors=spine_color,
        )
        if bold:
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
                tick.set_color("#0f172a")
            for tick in ax.get_xticklabels():
                tick.set_fontweight("bold")
                tick.set_color("#0f172a")

    @staticmethod
    def _save_figure(
        fig: plt.Figure,
        base_name: str,
        save_dir: Optional[str] = None,
        formats: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close: bool = False,
    ) -> None:
        """Guarda la figura en formatos seleccionados y libera memoria si se solicita."""
        if not save_dir:
            return

        os.makedirs(save_dir, exist_ok=True)
        fmt_list = [formats] if isinstance(formats, str) else list(formats)
        for fmt in fmt_list:
            clean_fmt = fmt.lstrip(".").lower()
            filepath = os.path.join(save_dir, f"{base_name}.{clean_fmt}")
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

        if close:
            plt.close(fig)

    @staticmethod
    def close_all() -> None:
        """Cierra todas las figuras abiertas de matplotlib para liberar memoria."""
        plt.close("all")

    def get_label(self, col: str, labels_map: Optional[Dict[str, str]] = None) -> str:
        """Obtiene la etiqueta formal formateada para la variable."""
        active_map = self.labels_map.copy()
        if labels_map:
            active_map.update(labels_map)
        return get_label(col, active_map)

    @classmethod
    def dibujar_barras(
        cls,
        ax: plt.Axes,
        categories: Sequence[Any],
        values: Sequence[float],
        total_n: Optional[int] = None,
        percentages: Optional[Sequence[float]] = None,
        color: Any = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        orientation: str = "horizontal",
        bar_width: float = 0.55,
        alpha: float = 0.90,
        show_values: bool = True,
        as_percentage: bool = False,
        xlim_factor: float = 1.30,
        rotation: Optional[int] = None,
        bold: bool = True,
    ) -> plt.Axes:
        """
        Motor central de renderizado sobre canvas para paneles de barras horizontales o verticales.
        Estandariza spines, ticks, etiquetas y anotaciones de valor/porcentaje en negrita (#0f172a).
        """
        vals = np.asarray(values, dtype=float)
        cats = [str(c) for c in categories]
        tot = total_n if total_n is not None else (int(np.sum(vals)) if len(vals) > 0 else 0)
        pcts = percentages if percentages is not None else ((vals / tot * 100) if tot > 0 else np.zeros_like(vals))
        bar_lengths = pcts if as_percentage else vals
        bar_color = color if color is not None else "#2b6cb0"

        cls._style_axis(ax, tick_length=4.5, bold=bold)
        axis_color = "#0f172a" if bold else "#222222"
        font_weight = "bold" if bold else "normal"

        if orientation == "horizontal":
            bars = ax.barh(
                cats,
                bar_lengths,
                color=bar_color,
                edgecolor="white",
                linewidth=1.0,
                alpha=alpha,
                height=bar_width,
            )
            max_len = max(bar_lengths) if len(bar_lengths) > 0 else 1
            ax.set_xlim(0, max_len * xlim_factor)

            if show_values:
                for bar, n_val, pct_val in zip(bars, vals, pcts):
                    label_txt = f"{pct_val:.1f}%" if as_percentage else f"{n_val:g} ({pct_val:.1f}%)"
                    ax.annotate(
                        label_txt,
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left",
                        fontsize=9.5,
                        fontweight=font_weight,
                        color=axis_color,
                    )

            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=10.5 if title else 11.5, fontweight=font_weight, labelpad=6, color=axis_color)
            elif not title:
                ax.set_xlabel("Porcentaje (%)" if as_percentage else "Frecuencia ($n$)", fontsize=11.5, fontweight=font_weight, labelpad=6, color=axis_color)

            if ylabel:
                ax.set_ylabel(ylabel, fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            if title:
                ax.set_title(title, fontsize=11.5, fontweight=font_weight, pad=10, color=axis_color)

            ax.grid(axis="x", linestyle="--", alpha=0.35)

        else:  # vertical
            bars = ax.bar(
                cats,
                bar_lengths,
                color=bar_color,
                edgecolor="white",
                linewidth=1.0,
                alpha=alpha,
                width=bar_width,
            )
            max_len = max(bar_lengths) if len(bar_lengths) > 0 else 1
            ax.set_ylim(0, max_len * xlim_factor)

            if show_values:
                for bar, n_val, pct_val in zip(bars, vals, pcts):
                    label_txt = f"{pct_val:.1f}%" if as_percentage else f"{n_val:g}\n({pct_val:.1f}%)"
                    ax.annotate(
                        label_txt,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        fontweight=font_weight,
                        color=axis_color,
                    )

            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            elif not title:
                ax.set_ylabel("Porcentaje (%)" if as_percentage else "Frecuencia ($n$)", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)

            if xlabel:
                ax.set_xlabel(xlabel, fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            if title:
                ax.set_title(title, fontsize=11.5, fontweight=font_weight, pad=10, color=axis_color)

            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if rotation is not None:
                ax.tick_params(axis="x", rotation=rotation)

        return ax

    _draw_bar_panel = dibujar_barras

