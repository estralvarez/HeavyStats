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


class UnivariatePlots:
    """
    Clase para construir gráficos descriptivos univariantes utilizando Seaborn y Matplotlib.
    Diseñada con estética científica y jerarquía tipográfica optimizada para publicaciones.
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
        
        # Configuración estética global
        sns.set_theme(
            style=self.style, 
            palette=self.palette, 
            rc=self.rc, 
            context=self.context
        )
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

    # =========================================================================
    # Métodos Auxiliares Privados (DRY)
    # =========================================================================

    def _resolve_column(self, col: str) -> str:
        """Resuelve el nombre exacto de la columna en el DataFrame a partir de nombres o alias comunes."""
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
            "exposicion": "Es_Expuesto"
        }
        col_lower = col_clean.lower()
        if col_lower in alias_map and alias_map[col_lower] in self.df.columns:
            return alias_map[col_lower]

        for c in self.df.columns:
            if c.lower() == col_lower:
                return c

        return col

    def _normalize_columns(self, columns: Union[str, Sequence[str]]) -> List[str]:
        """Normaliza y resuelve el argumento de columnas para admitir un string único, secuencias o alias."""
        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        return [self._resolve_column(c) for c in cols]

    def _clean_series(self, col: str, min_obs: int = 1) -> Optional[pd.Series]:
        """
        Extrae y limpia una columna numérica del DataFrame, convirtiendo a tipo numérico
        y descartando valores nulos. Emite advertencias explícitas en caso de datos faltantes.
        """
        real_col = self._resolve_column(col)
        if real_col not in self.df.columns:
            warnings.warn(
                f"La columna '{col}' no se encuentra en el DataFrame. Omitiendo gráfico.",
                UserWarning,
                stacklevel=3
            )
            return None

        data_clean = pd.to_numeric(self.df[real_col], errors="coerce").dropna()
        if len(data_clean) < min_obs:
            warnings.warn(
                f"La columna '{col}' contiene menos de {min_obs} observaciones válidas. Omitiendo gráfico.",
                UserWarning,
                stacklevel=3
            )
            return None

        return data_clean

    def _resolve_limit(
        self, 
        col: str, 
        permissible_limit: Optional[float] = None, 
        permissible_limits: Optional[Dict[str, float]] = None
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
        hide_top_right: bool = True
    ) -> None:
        """Aplica la jerarquía visual de bordes (spines) y ticks de calidad de publicación."""
        ax.set_axisbelow(True)
        ax.grid(False)
        if hide_top_right:
            for s in ["top", "right"]:
                if s in ax.spines:
                    ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            if s in ax.spines:
                ax.spines[s].set_color("#222222")
                ax.spines[s].set_linewidth(1.3)
        ax.tick_params(
            axis="both", 
            which="major", 
            labelsize=labelsize, 
            width=1.3, 
            length=tick_length, 
            colors="#222222"
        )

    @staticmethod
    def _save_figure(
        fig: plt.Figure, 
        base_name: str, 
        save_dir: Optional[str] = None, 
        formats: Union[str, Sequence[str]] = "png", 
        dpi: int = 300, 
        close: bool = False
    ) -> None:
        """Guarda la figura en uno o múltiples formatos vectoriales/raster y libera memoria si se solicita."""
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
        """Cierra todas las figuras abiertas de matplotlib para liberar recursos del sistema."""
        plt.close("all")

    def get_label(self, col: str, labels_map: Optional[Dict[str, str]] = None) -> str:
        """
        Obtiene la etiqueta legible para una columna, eliminando snake_case.
        Si se pasa labels_map o la columna está en self.labels_map, la retorna directamente.
        En caso contrario, aplica heurísticas automáticas de formato de unidades.
        """
        active_map = self.labels_map.copy()
        if labels_map:
            active_map.update(labels_map)
        return get_label(col, active_map)

    # =========================================================================
    # Gráficos Numéricos
    # =========================================================================

    def plot_histograms(
        self, 
        columns: Union[str, Sequence[str]], 
        kde: bool = False, 
        show_limit: bool = False,
        permissible_limit: Optional[float] = None,
        permissible_limits: Optional[Dict[str, float]] = None,
        bins: Optional[Any] = None,
        log_scale: Union[bool, float, Tuple[bool, bool]] = False,
        figsize: Tuple[float, float] = (7.0, 5.0),
        labels_map: Optional[Dict[str, str]] = None,
        xlabel: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any
    ) -> List[plt.Figure]:
        """
        Construye histogramas de calidad de publicación científica, con KDE desactivado
        por defecto para rigor estadístico sobre conteos enteros y límite permisible opcional.

        Parámetros
        ----------
        columns : Union[str, Sequence[str]]
            Columna o lista de columnas numéricas a graficar.
        kde : bool, opcional (por defecto False)
            Si True, traza la curva de estimación de densidad univariante.
        show_limit : bool, opcional (por defecto False)
            Si True, traza el límite permisible de referencia si está definido.
        permissible_limit : float, opcional
            Límite permisible personalizado para la columna.
        permissible_limits : Dict[str, float], opcional
            Diccionario de límites permisibles por columna para sobrescribir los predeterminados.
        bins : int o secuencia, opcional
            Definición explícita de bins para el histograma.
        log_scale : bool o float, opcional (por defecto False)
            Si True o base numérica, aplica transformación logarítmica al eje de la variable.
        figsize : Tuple[float, float], opcional (por defecto (7.0, 5.0))
            Tamaño de cada figura generada en pulgadas (ignorado si se pasa `ax`).
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables para los ejes.
        xlabel : str, opcional
            Etiqueta personalizada para el eje X.
        ax : plt.Axes, opcional
            Eje de Matplotlib existente donde dibujar. Solo aplicable al graficar una sola columna.
        save_dir : str, opcional
            Directorio donde guardar las figuras exportadas.
        save_format : Union[str, Sequence[str]], opcional (por defecto "png")
            Formato o lista de formatos de exportación (ej. "png", "pdf", ["png", "svg"]).
        dpi : int, opcional (por defecto 300)
            Resolución en puntos por pulgada de las figuras exportadas.
        close_after_save : bool, opcional (por defecto False)
            Si True, cierra cada figura tras guardarla en disco para liberar memoria.
        **kwargs : Any
            Argumentos adicionales pasados a `sns.histplot`.

        Retorna
        -------
        List[plt.Figure]
            Lista de figuras de Matplotlib generadas.
        """
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        color = palette_colors[0] if palette_colors else "teal"

        for col in cols:
            data_clean = self._clean_series(col, min_obs=1)
            if data_clean is None:
                continue

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = plt.subplots(figsize=figsize)

            hist_kwargs = kwargs.copy()
            if bins is not None:
                hist_kwargs["bins"] = bins

            sns.histplot(
                x=data_clean, 
                kde=kde, 
                color=color, 
                ax=target_ax,
                edgecolor="white", 
                linewidth=1.0, 
                alpha=0.70,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs
            )
            
            var_label = xlabel if xlabel is not None else self.get_label(col, labels_map)
            has_legend_entries = False
            
            # Límite permisible de metal en sangre (opcional, alpha=0.5)
            if show_limit:
                limit_val = self._resolve_limit(col, permissible_limit, permissible_limits)
                if limit_val is not None:
                    target_ax.axvline(
                        limit_val, 
                        color="#d73027", 
                        linestyle=":", 
                        linewidth=2.0, 
                        alpha=0.5, 
                        label=f"Límite permisible ({limit_val:g})", 
                        zorder=4
                    )
                    has_legend_entries = True

            if has_legend_entries:
                target_ax.legend(
                    frameon=True, 
                    facecolor="white", 
                    edgecolor="#cccccc", 
                    framealpha=0.92, 
                    fontsize=9.5, 
                    loc="upper right"
                )

            self._style_axis(target_ax, tick_length=4.5)

            if not kde and not log_scale:
                target_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                target_ax.set_ylabel("Frecuencia", fontsize=11.5, labelpad=8, color="#222222")
            elif not kde and log_scale:
                target_ax.set_ylabel("Frecuencia", fontsize=11.5, labelpad=8, color="#222222")
            else:
                target_ax.set_ylabel("Densidad", fontsize=11.5, labelpad=8, color="#222222")

            target_ax.set_xlabel(f"{var_label}", fontsize=11.5, labelpad=8, color="#222222")
            
            if ax is None:
                fig.tight_layout()

            self._save_figure(
                fig=fig, 
                base_name=f"{col}_histogram", 
                save_dir=save_dir, 
                formats=save_format, 
                dpi=dpi, 
                close=close_after_save
            )
            figures.append(fig)
            
        return figures

    def plot_qq(
        self, 
        columns: Union[str, Sequence[str]], 
        labels_map: Optional[Dict[str, str]] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[float, float] = (6.0, 5.5),
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any
    ) -> List[plt.Figure]:
        """
        Construye gráficos Q-Q de cuantiles observados vs teóricos de la Normal
        con estética y jerarquía tipográfica apta para publicación científica.

        Parámetros
        ----------
        columns : Union[str, Sequence[str]]
            Columna o lista de columnas numéricas a graficar.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables para los ejes.
        ylabel : str, opcional
            Etiqueta personalizada para el eje Y.
        figsize : Tuple[float, float], opcional (por defecto (6.0, 5.5))
            Tamaño de la figura en pulgadas (ignorado si se pasa `ax`).
        ax : plt.Axes, opcional
            Eje de Matplotlib existente donde dibujar. Solo aplicable al graficar una sola columna.
        save_dir : str, opcional
            Directorio donde guardar las figuras exportadas.
        save_format : Union[str, Sequence[str]], opcional (por defecto "png")
            Formato o lista de formatos de exportación (ej. "png", "pdf").
        dpi : int, opcional (por defecto 300)
            Resolución en DPI de las figuras exportadas.
        close_after_save : bool, opcional (por defecto False)
            Si True, cierra cada figura tras guardarla en disco para liberar memoria.

        Retorna
        -------
        List[plt.Figure]
            Lista de figuras de Matplotlib generadas.
        """
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        dot_color = palette_colors[0] if palette_colors else "blue"
        line_color = palette_colors[1] if len(palette_colors) > 1 else "#d73027"
        
        for col in cols:
            data_clean = self._clean_series(col, min_obs=3)
            if data_clean is None:
                continue
                
            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = plt.subplots(figsize=figsize)

            stats.probplot(data_clean, dist="norm", plot=target_ax)
            target_ax.set_title("")
            
            # Personalizar líneas generadas por scipy.stats.probplot
            if len(target_ax.lines) >= 2:
                target_ax.lines[0].set_color(dot_color)
                target_ax.lines[0].set_marker("o")
                target_ax.lines[0].set_markersize(6.5)
                target_ax.lines[0].set_markeredgewidth(0.8)
                target_ax.lines[0].set_markeredgecolor("white")
                target_ax.lines[0].set_alpha(0.75)
                
                target_ax.lines[1].set_color(line_color)
                target_ax.lines[1].set_linestyle("--")
                target_ax.lines[1].set_linewidth(1.8)
                
            var_label = ylabel if ylabel is not None else self.get_label(col, labels_map)
            target_ax.set_xlabel("Cuantiles Teóricos", fontsize=11.5, labelpad=8, color="#222222")
            target_ax.set_ylabel(f"Cuantiles Observados ({var_label})", fontsize=11.5, labelpad=8, color="#222222")
            
            self._style_axis(target_ax, tick_length=4.5)
            
            if ax is None:
                fig.tight_layout()
            
            self._save_figure(
                fig=fig, 
                base_name=f"{col}_qqplot", 
                save_dir=save_dir, 
                formats=save_format, 
                dpi=dpi, 
                close=close_after_save
            )
            figures.append(fig)
            
        return figures

    def plot_boxplots(
        self, 
        columns: Union[str, Sequence[str]], 
        overlay_points: bool = True, 
        show_limit: bool = False,
        permissible_limit: Optional[float] = None,
        permissible_limits: Optional[Dict[str, float]] = None,
        log_scale: bool = False,
        labels_map: Optional[Dict[str, str]] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 6.0),
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any
    ) -> List[plt.Figure]:
        """
        Construye diagramas de caja con puntos individuales superpuestos (stripplot)
        y estética refinada de publicación científica.

        Parámetros
        ----------
        columns : Union[str, Sequence[str]]
            Columna o lista de columnas numéricas a graficar.
        overlay_points : bool, opcional (por defecto True)
            Si True, superpone puntos individuales con jittering horizontal.
        show_limit : bool, opcional (por defecto False)
            Si True, traza el límite permisible de referencia en sangre si está definido.
        permissible_limit : float, opcional
            Límite permisible personalizado para la columna.
        permissible_limits : Dict[str, float], opcional
            Diccionario de límites permisibles para sobrescribir los predeterminados.
        log_scale : bool, opcional (por defecto False)
            Si True, aplica escala logarítmica en el eje Y.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables para los ejes.
        ylabel : str, opcional
            Etiqueta personalizada para el eje Y.
        figsize : Tuple[float, float], opcional (por defecto (5.5, 6.0))
            Tamaño de la figura en pulgadas (ignorado si se pasa `ax`).
        ax : plt.Axes, opcional
            Eje de Matplotlib existente donde dibujar. Solo aplicable al graficar una sola columna.
        save_dir : str, opcional
            Directorio donde guardar las figuras exportadas.
        save_format : Union[str, Sequence[str]], opcional (por defecto "png")
            Formato o lista de formatos de exportación (ej. "png", "pdf").
        dpi : int, opcional (por defecto 300)
            Resolución en DPI de las figuras exportadas.
        close_after_save : bool, opcional (por defecto False)
            Si True, cierra cada figura tras guardarla en disco para liberar memoria.

        Retorna
        -------
        List[plt.Figure]
            Lista de figuras de Matplotlib generadas.
        """
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        main_color = palette_colors[0] if palette_colors else "teal"

        for col in cols:
            data_clean = self._clean_series(col, min_obs=1)
            if data_clean is None:
                continue

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = plt.subplots(figsize=figsize)
            
            # Dibujar caja con estilo editorial
            sns.boxplot(
                y=data_clean, 
                ax=target_ax, 
                width=0.42, 
                color=main_color,
                boxprops=dict(alpha=0.35, edgecolor="#2c3e50", linewidth=1.3),
                whiskerprops=dict(color="#2c3e50", linewidth=1.3),
                capprops=dict(color="#2c3e50", linewidth=1.3),
                medianprops=dict(color="#16a085", linewidth=2.2),
                fliersize=0  # ocultar outliers para no duplicar con stripplot
            )
            
            if overlay_points:
                sns.stripplot(
                    y=data_clean, 
                    ax=target_ax, 
                    color=main_color, 
                    size=6.0, 
                    jitter=0.25, 
                    alpha=0.75, 
                    linewidth=0.8, 
                    edgecolor="white",
                    zorder=4
                )
                
            has_legend_entries = False
                
            if show_limit:
                limit_val = self._resolve_limit(col, permissible_limit, permissible_limits)
                if limit_val is not None:
                    target_ax.axhline(
                        limit_val, 
                        color="#d73027", 
                        linestyle=":", 
                        linewidth=2.0, 
                        alpha=0.5, 
                        label=f"Límite permisible ({limit_val:g})", 
                        zorder=3
                    )
                    has_legend_entries = True
                    
            if has_legend_entries:
                target_ax.legend(
                    frameon=True, 
                    facecolor="white", 
                    edgecolor="#cccccc", 
                    framealpha=0.92, 
                    fontsize=9.5, 
                    loc="upper right"
                )
                
            if log_scale:
                target_ax.set_yscale("log")

            var_label = ylabel if ylabel is not None else self.get_label(col, labels_map)
            target_ax.set_ylabel(f"{var_label}", fontsize=11.5, labelpad=8, color="#222222")
            target_ax.set_xlabel("", fontsize=11.5)
            
            self._style_axis(target_ax, tick_length=4.5)
            
            if ax is None:
                fig.tight_layout()
            
            self._save_figure(
                fig=fig, 
                base_name=f"{col}_boxplot", 
                save_dir=save_dir, 
                formats=save_format, 
                dpi=dpi, 
                close=close_after_save
            )
            figures.append(fig)

        return figures

    def plot_box_histograms(
        self, 
        columns: Union[str, Sequence[str]], 
        kde: bool = False, 
        overlay_points: bool = True, 
        show_limit: bool = False,
        permissible_limit: Optional[float] = None,
        permissible_limits: Optional[Dict[str, float]] = None,
        bins: Optional[Any] = None,
        log_scale: Union[bool, float, Tuple[bool, bool]] = False,
        figsize: Tuple[float, float] = (7.2, 5.2),
        labels_map: Optional[Dict[str, str]] = None,
        xlabel: Optional[str] = None,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any
    ) -> List[plt.Figure]:
        """
        Construye gráficos combinados que integran un diagrama de caja horizontal
        en la parte superior con un histograma en la parte inferior, compartiendo el mismo
        eje X coordinado con calidad de publicación científica.

        Parámetros
        ----------
        columns : Union[str, Sequence[str]]
            Columna o lista de columnas numéricas a graficar.
        kde : bool, opcional (por defecto False)
            Si True, traza la curva de estimación de densidad empírica en el histograma.
        overlay_points : bool, opcional (por defecto True)
            Si True, superpone puntos individuales (stripplot) en el boxplot con jittering vertical.
        show_limit : bool, opcional (por defecto False)
            Si True, traza el límite permisible coordinado en ambos subplots con opacidad 0.5.
        permissible_limit : float, opcional
            Límite permisible personalizado para la columna.
        permissible_limits : Dict[str, float], opcional
            Diccionario de límites permisibles por columna para sobrescribir los predeterminados.
        bins : int o secuencia, opcional
            Definición explícita de bins para el histograma.
        log_scale : bool o float, opcional (por defecto False)
            Si True o base numérica, aplica transformación logarítmica coordinada a ambos paneles.
        figsize : Tuple[float, float], opcional (por defecto (7.2, 5.2))
            Tamaño de cada figura generada en pulgadas (ignorado si se pasa `axes`).
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables para los ejes.
        xlabel : str, opcional
            Etiqueta personalizada para el eje X.
        axes : Tuple[plt.Axes, plt.Axes], opcional
            Tupla (ax_box, ax_hist) de ejes preexistentes donde incrustar el gráfico.
        save_dir : str, opcional
            Directorio donde guardar las figuras exportadas.
        save_format : Union[str, Sequence[str]], opcional (por defecto "png")
            Formato o lista de formatos de exportación (ej. "png", "pdf").
        dpi : int, opcional (por defecto 300)
            Resolución en DPI de las figuras exportadas.
        close_after_save : bool, opcional (por defecto False)
            Si True, cierra cada figura tras guardarla en disco para liberar memoria.
        **kwargs : Any
            Argumentos adicionales pasados a `sns.histplot`.

        Retorna
        -------
        List[plt.Figure]
            Lista de figuras de Matplotlib generadas.
        """
        cols = self._normalize_columns(columns)
        if axes is not None and len(cols) > 1:
            raise ValueError("El parámetro 'axes' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        color = palette_colors[0] if palette_colors else "teal"

        for col in cols:
            data_clean = self._clean_series(col, min_obs=1)
            if data_clean is None:
                continue

            if axes is not None:
                ax_box, ax_hist = axes
                fig = ax_box.figure
            else:
                # Separación mínima vertical entre paneles integrados
                fig, (ax_box, ax_hist) = plt.subplots(
                    2, 1, 
                    figsize=figsize, 
                    sharex=True, 
                    gridspec_kw={"height_ratios": [0.18, 0.82], "hspace": 0.02}
                )

            # 1. Boxplot superior (horizontal) con diseño nítido
            sns.boxplot(
                x=data_clean, 
                ax=ax_box, 
                color=color, 
                width=0.45, 
                boxprops=dict(alpha=0.35, edgecolor="#2c3e50", linewidth=1.3),
                whiskerprops=dict(color="#2c3e50", linewidth=1.3),
                capprops=dict(color="#2c3e50", linewidth=1.3),
                medianprops=dict(color="#16a085", linewidth=2.4),
                fliersize=0 if overlay_points else 3
            )
            if overlay_points:
                sns.stripplot(
                    x=data_clean, 
                    ax=ax_box, 
                    color=color, 
                    size=6.0, 
                    jitter=0.28, 
                    alpha=0.75, 
                    linewidth=0.8, 
                    edgecolor="white",
                    zorder=4
                )

            ax_box.set(xlabel="", yticks=[])
            ax_box.tick_params(bottom=False, labelbottom=False)
            for spine in ["top", "right", "left", "bottom"]:
                if spine in ax_box.spines:
                    ax_box.spines[spine].set_visible(False)
            ax_box.grid(False)
            ax_box.set_ylim(-0.48, 0.48)

            # 2. Histograma inferior
            hist_kwargs = kwargs.copy()
            if bins is not None:
                hist_kwargs["bins"] = bins

            sns.histplot(
                x=data_clean, 
                kde=kde, 
                ax=ax_hist, 
                color=color, 
                edgecolor="white", 
                linewidth=1.0, 
                alpha=0.70,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs
            )
            
            # Coordinar escala logarítmica si aplica
            if log_scale:
                ax_box.set_xscale("log")
                ax_hist.set_xscale("log")

            self._style_axis(ax_hist, tick_length=5.0)

            has_legend_entries = False

            # 3. Límite permisible coordinado en ambos paneles (alpha=0.5)
            if show_limit:
                limit_val = self._resolve_limit(col, permissible_limit, permissible_limits)
                if limit_val is not None:
                    ax_box.axvline(limit_val, color="#d73027", linestyle=":", linewidth=2.0, alpha=0.5, zorder=3)
                    ax_hist.axvline(
                        limit_val, 
                        color="#d73027", 
                        linestyle=":", 
                        linewidth=2.0, 
                        alpha=0.5, 
                        label=f"Límite permisible ({limit_val:g})", 
                        zorder=3
                    )
                    has_legend_entries = True

            if has_legend_entries:
                ax_hist.legend(
                    frameon=True, 
                    facecolor="white", 
                    edgecolor="#cccccc", 
                    framealpha=0.92, 
                    fontsize=9.5, 
                    loc="upper right"
                )

            var_label = xlabel if xlabel is not None else self.get_label(col, labels_map)
            
            if not kde and not log_scale:
                ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax_hist.set_ylabel("Frecuencia", fontsize=11.5, labelpad=8, color="#222222")
            elif not kde and log_scale:
                ax_hist.set_ylabel("Frecuencia", fontsize=11.5, labelpad=8, color="#222222")
            else:
                ax_hist.set_ylabel("Densidad", fontsize=11.5, labelpad=8, color="#222222")

            ax_hist.set_xlabel(f"{var_label}", fontsize=11.5, labelpad=8, color="#222222")
            
            if axes is None:
                fig.subplots_adjust(hspace=0.02)

            self._save_figure(
                fig=fig, 
                base_name=f"{col}_box_histogram", 
                save_dir=save_dir, 
                formats=save_format, 
                dpi=dpi, 
                close=close_after_save
            )
            figures.append(fig)

        return figures

    # =========================================================================
    # Gráficos Categóricos
    # =========================================================================

    def plot_categorical(
        self, 
        columns: Union[str, Sequence[str]], 
        orientation: str = "horizontal",
        show_values: bool = True,
        as_percentage: bool = False,
        order: Optional[Union[List[str], str]] = "frequency",
        figsize: Optional[Tuple[float, float]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        **kwargs: Any
    ) -> List[plt.Figure]:
        """
        Construye diagramas de barras con frecuencias absolutas y/o relativas
        para variables categóricas o discretas (ej. Sexo, Sector, Nivel de Riesgo).

        Parámetros
        ----------
        columns : Union[str, Sequence[str]]
            Columna o lista de columnas categóricas a graficar.
        orientation : str, opcional (por defecto "horizontal")
            Orientación de las barras: "horizontal" o "vertical".
        show_values : bool, opcional (por defecto True)
            Si True, anota los valores numéricos ($n$ y/o %) al extremo de cada barra.
        as_percentage : bool, opcional (por defecto False)
            Si True, la longitud de las barras representa porcentaje (%) en vez de conteo absoluto ($n$).
        order : Union[List[str], str], opcional (por defecto "frequency")
            Criterio de ordenación: "frequency" (por frecuencia descendente), "alpha" (alfabético)
            o lista explícita de categorías.
        figsize : Tuple[float, float], opcional
            Tamaño de la figura. Si es None, se calcula automáticamente según el número de categorías.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables para los títulos y ejes.
        xlabel : str, opcional
            Etiqueta personalizada para el eje X.
        ylabel : str, opcional
            Etiqueta personalizada para el eje Y.
        ax : plt.Axes, opcional
            Eje de Matplotlib existente donde dibujar. Solo aplicable al graficar una sola columna.
        save_dir : str, opcional
            Directorio donde guardar las figuras exportadas.
        save_format : Union[str, Sequence[str]], opcional (por defecto "png")
            Formato o lista de formatos de exportación (ej. "png", "pdf").
        dpi : int, opcional (por defecto 300)
            Resolución en DPI de las figuras exportadas.
        close_after_save : bool, opcional (por defecto False)
            Si True, cierra cada figura tras guardarla en disco para liberar memoria.

        Retorna
        -------
        List[plt.Figure]
            Lista de figuras de Matplotlib generadas.
        """
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        main_color = palette_colors[0] if palette_colors else "teal"

        for col in cols:
            if col not in self.df.columns:
                warnings.warn(
                    f"La columna '{col}' no se encuentra en el DataFrame. Omitiendo gráfico.",
                    UserWarning,
                    stacklevel=3
                )
                continue

            series_clean = self.df[col].dropna()
            total_n = len(series_clean)
            if total_n == 0:
                warnings.warn(
                    f"La columna '{col}' no contiene observaciones no nulas. Omitiendo gráfico.",
                    UserWarning,
                    stacklevel=3
                )
                continue

            counts = series_clean.astype(str).value_counts()
            
            # Ordenamiento de categorías
            if order == "frequency":
                counts = counts.sort_values(ascending=(orientation == "horizontal"))
            elif order == "alpha":
                counts = counts.sort_index(ascending=(orientation != "horizontal"))
            elif isinstance(order, (list, tuple)):
                available = [cat for cat in order if cat in counts.index]
                missing = [cat for cat in counts.index if cat not in available]
                sorted_idx = available + missing
                if orientation == "horizontal":
                    sorted_idx = sorted_idx[::-1]
                counts = counts.reindex(sorted_idx).dropna()

            categories = [str(cat) for cat in counts.index]
            values = counts.values
            percentages = (values / total_n) * 100

            bar_lengths = percentages if as_percentage else values

            # Dimensiones automáticas si no se especifican
            if figsize is None:
                if orientation == "horizontal":
                    calculated_figsize = (7.0, max(3.2, len(categories) * 0.55 + 1.2))
                else:
                    calculated_figsize = (max(5.5, len(categories) * 0.9 + 1.0), 5.0)
            else:
                calculated_figsize = figsize

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = plt.subplots(figsize=calculated_figsize)

            var_label = self.get_label(col, labels_map)

            if orientation == "horizontal":
                bars = target_ax.barh(
                    categories, 
                    bar_lengths, 
                    color=main_color, 
                    edgecolor="white", 
                    linewidth=1.0, 
                    alpha=0.82, 
                    height=0.55
                )
                self._style_axis(target_ax, tick_length=4.5)
                
                # Anotación numérica al final de cada barra
                if show_values:
                    max_len = max(bar_lengths) if len(bar_lengths) > 0 else 1
                    for bar, n_val, pct_val in zip(bars, values, percentages):
                        label_txt = f"{pct_val:.1f}%" if as_percentage else f"{n_val:g} ({pct_val:.1f}%)"
                        target_ax.annotate(
                            label_txt,
                            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0),
                            textcoords="offset points",
                            va="center",
                            ha="left",
                            fontsize=10.0,
                            color="#222222"
                        )
                    target_ax.set_xlim(0, max_len * 1.18)

                target_ax.set_xlabel(
                    xlabel if xlabel is not None else ("Porcentaje (%)" if as_percentage else "Frecuencia ($n$)"),
                    fontsize=11.5,
                    labelpad=8,
                    color="#222222"
                )
                target_ax.set_ylabel(
                    ylabel if ylabel is not None else var_label,
                    fontsize=11.5,
                    labelpad=8,
                    color="#222222"
                )

            else:  # Vertical
                bars = target_ax.bar(
                    categories, 
                    bar_lengths, 
                    color=main_color, 
                    edgecolor="white", 
                    linewidth=1.0, 
                    alpha=0.82, 
                    width=0.55
                )
                self._style_axis(target_ax, tick_length=4.5)

                if show_values:
                    max_len = max(bar_lengths) if len(bar_lengths) > 0 else 1
                    for bar, n_val, pct_val in zip(bars, values, percentages):
                        label_txt = f"{pct_val:.1f}%" if as_percentage else f"{n_val:g}\n({pct_val:.1f}%)"
                        target_ax.annotate(
                            label_txt,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9.5,
                            color="#222222"
                        )
                    target_ax.set_ylim(0, max_len * 1.18)

                target_ax.set_ylabel(
                    ylabel if ylabel is not None else ("Porcentaje (%)" if as_percentage else "Frecuencia ($n$)"),
                    fontsize=11.5,
                    labelpad=8,
                    color="#222222"
                )
                target_ax.set_xlabel(
                    xlabel if xlabel is not None else var_label,
                    fontsize=11.5,
                    labelpad=8,
                    color="#222222"
                )
                
                # Rotar etiquetas si son extensas
                max_str_len = max((len(c) for c in categories), default=0)
                if max_str_len > 6:
                    target_ax.tick_params(axis="x", rotation=25)

            if ax is None:
                fig.tight_layout()

            self._save_figure(
                fig=fig, 
                base_name=f"{col}_categorical", 
                save_dir=save_dir, 
                formats=save_format, 
                dpi=dpi, 
                close=close_after_save
            )
            figures.append(fig)

        return figures

    # Alias convenientes
    plot_hist_box = plot_box_histograms
    plot_combined = plot_box_histograms
    plot_bars = plot_categorical
    plot_frequencies = plot_categorical
