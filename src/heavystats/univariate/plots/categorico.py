"""
Módulo de visualización para variables cualitativas y de selección múltiple.
Proporciona pipelines para diagramas de barras simples y multirrespuesta con frecuencias y porcentajes.
"""

import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.univariate.plots.base import BasePlots
from heavystats.cleaning import desaggregate_multiple_responses


def _capitalize_category(val: Any) -> str:
    """Asegura que la etiqueta categórica tenga la primera letra en mayúscula."""
    s = str(val).strip()
    if not s:
        return s
    if s.upper() in ("SI", "SÍ", "TRUE", "1", "1.0"):
        return "Sí"
    if s.upper() in ("NO", "FALSE", "0", "0.0"):
        return "No"
    return s[0].upper() + s[1:]


def _resolve_category_color(
    palette_name: str,
    color: Optional[Union[str, int, Sequence[Any]]],
    idx: int = 0,
    total_items: int = 1,
    categories: Optional[Sequence[Any]] = None,
) -> Any:
    """
    Resuelve el color para el gráfico categórico:
    - Si color es un entero (ej. color=0, color=1, color=2): selecciona el tono de la paleta activa.
    - Si color es None: genera automáticamente un tono contrastante de la paleta según el índice de columna.
    - Si color es una lista: lo asocia a categorías o columnas.
    - Si color es un nombre de paleta de seaborn/matplotlib: extrae el tono de esa paleta.
    - Si color es un string hex/color CSS (ej. "#2b6cb0", "teal"): lo utiliza directamente.
    """
    # 1. Índice numérico de paleta (ej. color=0, 1, 2)
    if isinstance(color, int):
        palette_colors = sns.color_palette(palette_name, max(8, color + 1))
        return palette_colors[color % len(palette_colors)]

    # 2. Sin color explícito: asignar tono contrastante según índice de columna
    if color is None:
        n_pal = max(8, total_items * 2 + 2)
        palette_colors = sns.color_palette(palette_name, n_pal)
        step_idx = (2 * idx + 1) % len(palette_colors)
        return palette_colors[step_idx]

    # 3. Lista o tupla de colores
    if isinstance(color, (list, tuple)):
        if categories is not None and len(color) == len(categories):
            return list(color)
        if total_items > 1 and len(color) >= total_items:
            return color[idx % len(color)]
        return color

    # 4. String: comprobar si es una paleta válida o un color CSS/Hex
    if isinstance(color, str):
        try:
            if not color.startswith("#") and color.lower() in plt.colormaps():
                pal = sns.color_palette(color, max(8, total_items * 2 + 2))
                return pal[(2 * idx + 1) % len(pal)]
        except Exception:
            pass
        return color

    return color


def desglosar_multirrespuesta(
    serie: pd.Series,
    sep: str = ";",
    exclude_patterns: Optional[Sequence[str]] = ("ningun", "ninguno", "ninguna", "nada"),
    ascending: bool = True,
) -> pd.Series:
    """
    Desagrega respuestas de opción múltiple separadas por un delimitador y calcula frecuencias.
    Preserva la capitalización original de las opciones y filtra términos de exclusión.
    """
    conteo: Dict[str, int] = {}
    excl = [p.lower() for p in exclude_patterns] if exclude_patterns else []
    for val in serie.dropna():
        opciones = [op.strip() for op in str(val).split(sep) if op.strip()]
        for op in opciones:
            if excl and any(p in op.lower() for p in excl):
                continue
            op_cap = _capitalize_category(op)
            conteo[op_cap] = conteo.get(op_cap, 0) + 1
    s = pd.Series(conteo, dtype=int)
    return s.sort_values(ascending=ascending)


class CategoricoPlotsMixin:
    """Mixin que agrupa los pipelines de visualización para variables categóricas y multirrespuesta."""

    def plot_categorico(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        orientation: str = "horizontal",
        show_values: bool = True,
        as_percentage: bool = False,
        order: Optional[Union[List[str], str]] = "frequency",
        figsize: Optional[Tuple[float, float]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        color: Optional[Union[str, int, Sequence[Any]]] = None,
        bar_width: float = 0.55,
        alpha: float = 0.90,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        rotation: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        bold_annotations: bool = True,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """Construye diagramas de barras con frecuencias absolutas y relativas para variables cualitativas."""
        cols = self._normalize_columns(columns)
        total_cols = len(cols)
        if ax is not None and total_cols > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []

        for idx, col in enumerate(cols):
            if col not in self.df.columns:
                warnings.warn(f"La columna '{col}' no se encuentra en el DataFrame. Omitiendo gráfico.", UserWarning, stacklevel=3)
                continue

            series_clean = self.df[col].dropna()
            total_n = len(series_clean)
            if total_n == 0:
                warnings.warn(f"La columna '{col}' no contiene observaciones no nulas. Omitiendo gráfico.", UserWarning, stacklevel=3)
                continue

            series_formatted = series_clean.astype(str).map(_capitalize_category)
            counts = series_formatted.value_counts()
            if order == "frequency":
                counts = counts.sort_values(ascending=(orientation == "horizontal"))
            elif order == "alpha":
                counts = counts.sort_index(ascending=(orientation != "horizontal"))
            elif isinstance(order, (list, tuple)):
                order_formatted = [_capitalize_category(cat) for cat in order]
                available = [cat for cat in order_formatted if cat in counts.index]
                missing = [cat for cat in counts.index if cat not in available]
                sorted_idx = available + missing
                if orientation == "horizontal":
                    sorted_idx = sorted_idx[::-1]
                counts = counts.reindex(sorted_idx).dropna()

            categories = [str(cat) for cat in counts.index]
            values = counts.values
            percentages = (values / total_n) * 100
            bar_lengths = percentages if as_percentage else values

            # Resolver color del gráfico (por índice, paleta activa o valor explícito)
            plot_color = _resolve_category_color(
                palette_name=self.palette,
                color=color,
                idx=idx,
                total_items=total_cols,
                categories=categories,
            )

            if figsize is None:
                if orientation == "horizontal":
                    calc_figsize = (7.0, max(3.2, len(categories) * 0.55 + 1.2))
                else:
                    calc_figsize = (max(5.5, len(categories) * 0.9 + 1.0), 5.0)
            else:
                calc_figsize = figsize

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = self.canvas(figsize=calc_figsize, dpi=dpi, bold=bold_annotations)

            var_label = self.get_label(col, labels_map)

            self._draw_bar_panel(
                ax=target_ax,
                categories=categories,
                values=values,
                total_n=total_n,
                percentages=percentages,
                color=plot_color,
                title=title,
                bar_width=bar_width,
                alpha=alpha,
                orientation=orientation,
                rotation=rotation,
                xlabel=xlabel if xlabel is not None else (None if orientation == "horizontal" else (var_label if title is None else None)),
                ylabel=ylabel if ylabel is not None else (var_label if orientation == "horizontal" and title is None else None),
                show_values=show_values,
                as_percentage=as_percentage,
                bold=bold_annotations,
            )

            if ax is None:
                fig.tight_layout()

            self._save_figure(fig=fig, base_name=f"{col}_categorico", save_dir=save_dir, formats=save_format, dpi=dpi, close=close_after_save)
            figures.append(fig)

        return figures

    def plot_multirrespuesta(
        self: BasePlots,
        column: str,
        replace_map: Optional[Dict[str, str]] = None,
        mapping: Optional[Dict[Any, str]] = None,
        desglosar: bool = False,
        sep: str = ";",
        exclude_patterns: Optional[Sequence[str]] = ("ningun", "ninguno", "ninguna", "nada"),
        order: Optional[Union[List[str], str]] = "frequency",
        color: Optional[Any] = None,
        bar_height: float = 0.55,
        figsize: Tuple[float, float] = (7.5, 3.8),
        title: Optional[str] = None,
        xlabel: str = "Frecuencia ($n$)",
        n_total: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Construye un gráfico de barras horizontales para variables de respuesta múltiple (ej. Salud_Agua)
        o variables ordinales reclasificadas (ej. Pescado), orquestado con el motor editorial _draw_bar_panel.
        """
        real_col = self._resolve_column(column)
        if real_col not in self.df.columns:
            raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

        total = n_total if n_total is not None else len(self.df)
        s = self.df[real_col].dropna()

        active_map = replace_map or mapping
        if active_map:
            s = s.replace(active_map) if not desglosar else s

        if desglosar or (s.astype(str).str.contains(sep).any() and not active_map):
            counts = desglosar_multirrespuesta(s, sep=sep, exclude_patterns=exclude_patterns, ascending=(order == "frequency"))
        else:
            counts = s.astype(str).value_counts()
            if order == "frequency":
                counts = counts.sort_values(ascending=True)
            elif order == "alpha":
                counts = counts.sort_index(ascending=False)
            elif isinstance(order, (list, tuple)):
                counts = counts.reindex(list(order)[::-1], fill_value=0)

        palette_colors = sns.color_palette(self.palette)
        bar_c = color if color is not None else (palette_colors[0] if palette_colors else "teal")

        if ax is not None:
            fig = ax.figure
            target_ax = ax
        else:
            fig, target_ax = self.crear_lienzo(figsize=figsize, dpi=dpi, bold=True)

        panel_title = title if title is not None else self.get_label(real_col)
        self._draw_bar_panel(
            ax=target_ax,
            categories=counts.index,
            values=counts.values,
            total_n=total,
            color=bar_c,
            title=panel_title,
            bar_width=bar_height,
            xlabel=xlabel,
            orientation="horizontal",
            bold=True,
        )

        if ax is None:
            fig.tight_layout()

        self._save_figure(fig=fig, base_name=f"{real_col}_frecuencia", save_dir=save_dir, formats=save_format, dpi=dpi, close=close_after_save)
        return fig

    # Nombres simplificados
    categorico = plot_categorico
    multirrespuesta = plot_multirrespuesta
    desglosar_multirrespuesta = staticmethod(desglosar_multirrespuesta)
