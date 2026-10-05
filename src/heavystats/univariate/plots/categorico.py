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


# Mapeo estándar de frecuencias dietarias en estudios de biomonitoreo
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
    "0.0": "Nunca",
    "1.0": "Rara vez",
    "2.0": "A veces",
    "3.0": "Frecuente",
    "4.0": "Diario",
}


def _is_binary_series(serie: pd.Series) -> bool:
    """Determina si una serie de datos es verdaderamente dicotómica / binaria (ej. Sí/No, True/False, 0/1)."""
    clean_vals = serie.dropna().unique()
    if len(clean_vals) == 0 or len(clean_vals) > 2:
        return False
    norm_vals = set()
    for v in clean_vals:
        s_str = str(v).strip().lower()
        if s_str.endswith(".0"):
            s_str = s_str[:-2]
        norm_vals.add(s_str)
    binary_tokens = {"0", "1", "true", "false", "si", "sí", "no", "yes"}
    return norm_vals.issubset(binary_tokens)


def _format_category_label(val: Any, is_dichotomous: bool = False) -> str:
    """
    Formatea la etiqueta de una categoría.
    Solo convierte 0 y 1 a 'No' y 'Sí' si la variable es rigurosamente dicotómica.
    Para datos no dicotómicos numéricos u ordinales, preserva su valor original limpio.
    """
    if pd.isna(val):
        return "N/D"
    s = str(val).strip()
    if not s:
        return s

    if is_dichotomous:
        s_upper = s.upper()
        if s_upper in ("SI", "SÍ", "TRUE", "1", "1.0", "YES"):
            return "Sí"
        if s_upper in ("NO", "FALSE", "0", "0.0"):
            return "No"

    # Si es un número decimal que representa entero (ej. "2.0", "3.0")
    try:
        f_val = float(s)
        if f_val.is_integer():
            return str(int(f_val))
    except (ValueError, TypeError):
        pass

    # Capitalización elegante para texto
    if len(s) > 0 and s[0].islower():
        return s[0].upper() + s[1:]
    return s


def _capitalize_category(val: Any) -> str:
    """Compatibilidad retrospectiva para capitalizar categorías."""
    return _format_category_label(val, is_dichotomous=False)


def _resolve_category_color(
    palette_name: str,
    color: Optional[Union[str, int, Sequence[Any]]],
    idx: int = 0,
    total_items: int = 1,
    categories: Optional[Sequence[Any]] = None,
    palette: Optional[str] = None,
    color_by_category: bool = False,
) -> Any:
    """
    Resuelve el color o lista de colores para el gráfico categórico:
    - Si color_by_category=True: asigna un color individual de la paleta a cada barra.
    - Si color es un nombre de paleta: extrae una gama para todas las categorías.
    - Si color es un entero: selecciona el tono correspondiente de la paleta.
    - Si color es una lista: asocia los colores a las categorías, ciclando si hay menos colores.
    - Si color es un string hex/CSS: lo utiliza directamente para las barras.
    - Si color es None: genera un tono contrastante elegante o una paleta por categoría.
    """
    n_cats = len(categories) if categories is not None else 1
    active_palette = palette or palette_name

    # 1. Si se solicita colorear cada categoría de forma independiente
    if color_by_category:
        pal_name = color if (isinstance(color, str) and not color.startswith("#") and color.lower() in plt.colormaps()) else active_palette
        return sns.color_palette(pal_name, max(1, n_cats))

    # 2. Índice numérico de paleta (ej. color=0, 1, 2)
    if isinstance(color, int):
        palette_colors = sns.color_palette(active_palette, max(8, color + 1))
        return palette_colors[color % len(palette_colors)]

    # 3. Sin color explícito
    if color is None:
        if palette is not None:
            return sns.color_palette(palette, max(1, n_cats))
        n_pal = max(8, total_items * 2 + 2)
        palette_colors = sns.color_palette(active_palette, n_pal)
        step_idx = (2 * idx + 1) % len(palette_colors)
        return palette_colors[step_idx]

    # 4. Lista o tupla de colores
    if isinstance(color, (list, tuple)):
        if n_cats > 0:
            if len(color) == n_cats:
                return list(color)
            if len(color) < n_cats:
                # Ciclar colores para evitar errores en matplotlib
                return [color[i % len(color)] for i in range(n_cats)]
            return list(color[:n_cats])
        return color

    # 5. String: comprobar si es una paleta válida o un color CSS/Hex
    if isinstance(color, str):
        try:
            if not color.startswith("#") and color.lower() in plt.colormaps():
                return sns.color_palette(color, max(1, n_cats))
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
            op_cap = _format_category_label(op, is_dichotomous=False)
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
        order: Optional[Union[List[str], Tuple[str, ...], Dict[Any, Any], str]] = "frequency",
        figsize: Optional[Tuple[float, float]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        color: Optional[Union[str, int, Sequence[Any]]] = None,
        palette: Optional[str] = None,
        color_by_category: bool = False,
        mapping: Optional[Dict[Any, str]] = None,
        replace_map: Optional[Dict[Any, str]] = None,
        desglosar: bool = False,
        sep: str = ";",
        exclude_patterns: Optional[Sequence[str]] = ("ningun", "ninguno", "ninguna", "nada"),
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
        """
        Construye diagramas de barras con frecuencias absolutas y relativas para variables cualitativas.
        Soporta variables dicotómicas, politómicas (no dicotómicas), escalas ordinales y respuestas múltiples.
        """
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

            # 1. Resolver mapeo de categorías (reemplazos explícitos o dict en order)
            active_map = replace_map or mapping
            if isinstance(order, dict) and active_map is None:
                active_map = order

            # Detección automática para frecuencias dietarias codificadas a enteros (ej. Alim_*)
            if active_map is None and col.startswith("Alim_"):
                unique_numeric = set(pd.to_numeric(series_clean, errors="coerce").dropna().unique())
                if len(unique_numeric) > 0 and unique_numeric.issubset({0, 1, 2, 3, 4, 0.0, 1.0, 2.0, 3.0, 4.0}):
                    active_map = DEFAULT_DIETARY_MAP

            # 2. Desglose de respuesta múltiple si se solicita o si hay delimitador
            is_delimited = series_clean.astype(str).str.contains(sep, regex=False).any()
            if desglosar or (is_delimited and not active_map):
                counts = desglosar_multirrespuesta(series_clean, sep=sep, exclude_patterns=exclude_patterns, ascending=(orientation == "horizontal"))
                is_binary = False
            else:
                if active_map:
                    def _apply_map(v):
                        if v in active_map:
                            return active_map[v]
                        try:
                            f_v = float(v)
                            if f_v in active_map:
                                return active_map[f_v]
                            if f_v.is_integer() and int(f_v) in active_map:
                                return active_map[int(f_v)]
                        except (ValueError, TypeError):
                            pass
                        s_v = str(v).strip()
                        return active_map.get(s_v, v)
                    series_mapped = series_clean.map(_apply_map)
                else:
                    series_mapped = series_clean

                is_binary = _is_binary_series(series_mapped)
                series_formatted = series_mapped.map(lambda v: _format_category_label(v, is_dichotomous=is_binary))
                counts = series_formatted.value_counts()

            # 3. Ordenamiento de categorías
            if isinstance(order, dict):
                target_order = list(dict.fromkeys(order.values()))
                target_formatted = [_format_category_label(cat, is_dichotomous=is_binary) for cat in target_order]
                available = [c for c in target_formatted if c in counts.index]
                missing = [c for c in counts.index if c not in available]
                sorted_idx = available + missing
                if orientation == "horizontal":
                    sorted_idx = sorted_idx[::-1]
                counts = counts.reindex(sorted_idx).fillna(0)
            elif order == "frequency":
                counts = counts.sort_values(ascending=(orientation == "horizontal"))
            elif order in ("alpha", "alphabetical"):
                counts = counts.sort_index(ascending=(orientation != "horizontal"))
            elif order in ("natural", "scale", "ordinal", "numeric"):
                scale_order = ["Nunca", "Rara vez", "A veces", "Frecuentemente", "Frecuente", "Diario"]
                if any(c in scale_order for c in counts.index):
                    matched = [c for c in scale_order if c in counts.index]
                    extra = [c for c in counts.index if c not in matched]
                    sorted_idx = matched + extra
                else:
                    try:
                        sorted_idx = sorted(counts.index, key=lambda x: (float(x), str(x)))
                    except Exception:
                        sorted_idx = sorted(counts.index, key=lambda x: str(x))
                if orientation == "horizontal":
                    sorted_idx = sorted_idx[::-1]
                counts = counts.reindex(sorted_idx).dropna()
            elif isinstance(order, (list, tuple)):
                order_formatted = [_format_category_label(cat, is_dichotomous=is_binary) for cat in order]
                idx_map = {str(c).lower(): c for c in counts.index}
                available = []
                for cat in order_formatted:
                    c_low = str(cat).lower()
                    if c_low in idx_map:
                        available.append(idx_map[c_low])
                    elif cat in counts.index:
                        available.append(cat)
                available = list(dict.fromkeys(available))

                if len(available) > 0:
                    missing = [c for c in counts.index if c not in available]
                    sorted_idx = available + missing
                    if orientation == "horizontal":
                        sorted_idx = sorted_idx[::-1]
                    counts = counts.reindex(sorted_idx).dropna()
                else:
                    warnings.warn(
                        f"Las categorías en 'order' {list(order)} no coinciden con los valores de la columna '{col}'. "
                        f"Se mantendrá el orden por frecuencia para datos no dicotómicos.",
                        UserWarning,
                        stacklevel=3
                    )
                    counts = counts.sort_values(ascending=(orientation == "horizontal"))

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
                palette=palette,
                color_by_category=color_by_category,
            )

            # Ajuste dinámico de dimensiones de lienzo según número de categorías
            if figsize is None:
                if orientation == "horizontal":
                    calc_figsize = (7.0, max(3.2, len(categories) * 0.55 + 1.2))
                else:
                    calc_figsize = (max(5.5, len(categories) * 0.95 + 1.0), 5.0)
            else:
                calc_figsize = figsize

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = self.canvas(figsize=calc_figsize, dpi=dpi, bold=bold_annotations)

            var_label = self.get_label(col, labels_map)

            # Rotación automática de etiquetas en orientación vertical si son extensas
            active_rotation = rotation
            if active_rotation is None and orientation == "vertical":
                max_cat_len = max((len(str(c)) for c in categories), default=0)
                if max_cat_len > 6 or len(categories) >= 4:
                    active_rotation = 25

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
                rotation=active_rotation,
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
