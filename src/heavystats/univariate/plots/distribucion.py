"""
Módulo de visualización de distribuciones cuantitativas y evaluación de normalidad.
Proporciona pipelines para gráficos combinados (boxplot + histograma), histogramas, boxplots y Q-Q.
"""

from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from heavystats.univariate.plots.base import BasePlots
from heavystats.univariate.constants import DEFAULT_PERMISSIBLE_LIMITS


def draw_boxplot(
    ax: plt.Axes,
    data: pd.Series,
    color: Any = "teal",
    edge_color: str = "#0f172a",
    orientation: str = "horizontal",
    overlay_points: bool = True,
) -> None:
    """Dibuja un boxplot editorial con puntos individuales (stripplot) superpuestos."""
    is_horiz = orientation == "horizontal"
    box_kwargs = {
        ("x" if is_horiz else "y"): data,
        "ax": ax,
        "color": color,
        "width": 0.45 if is_horiz else 0.42,
        "boxprops": dict(alpha=0.35, edgecolor=edge_color, linewidth=1.3),
        "whiskerprops": dict(color=edge_color, linewidth=1.3),
        "capprops": dict(color=edge_color, linewidth=1.3),
        "medianprops": dict(color="#16a085", linewidth=2.4),
        "fliersize": 0 if overlay_points else 3,
    }
    sns.boxplot(**box_kwargs)

    if overlay_points:
        strip_kwargs = {
            ("x" if is_horiz else "y"): data,
            "ax": ax,
            "color": color,
            "size": 6.0,
            "jitter": 0.28 if is_horiz else 0.25,
            "alpha": 0.75,
            "linewidth": 0.8,
            "edgecolor": "white",
            "zorder": 4,
        }
        sns.stripplot(**strip_kwargs)


def draw_histogram(
    ax: plt.Axes,
    data: Union[pd.Series, pd.DataFrame],
    x: Optional[str] = None,
    hue: Optional[str] = None,
    color: Any = "teal",
    palette: Optional[Any] = None,
    multiple: str = "layer",
    shrink: Optional[float] = None,
    kde: bool = False,
    log_scale: Union[bool, float, Tuple[bool, bool]] = False,
    bins: Optional[Any] = None,
    bold: bool = True,
    **kwargs: Any,
) -> None:
    """
    Dibuja un histograma con estimación de densidad opcional y soporte de estratificación por factor (hue).
    Permite barras apiladas (multiple='stack'), espaciado entre barras (shrink), y estilización editorial de leyendas.
    """
    hist_kwargs = kwargs.copy()
    if bins is not None:
        hist_kwargs["bins"] = bins

    if hue is not None:
        hist_kwargs["multiple"] = multiple
        if shrink is not None and multiple in ("stack", "dodge", "fill"):
            hist_kwargs["shrink"] = shrink
        elif shrink is not None and "shrink" not in hist_kwargs:
            hist_kwargs["shrink"] = shrink
        if palette is not None:
            hist_kwargs["palette"] = palette

        if isinstance(data, pd.DataFrame) and x is not None:
            sns.histplot(
                data=data,
                x=x,
                hue=hue,
                kde=kde,
                ax=ax,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.80,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs,
            )
        else:
            df_temp = pd.DataFrame({"_x": data, "_hue": hue})
            sns.histplot(
                data=df_temp,
                x="_x",
                hue="_hue",
                kde=kde,
                ax=ax,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.80,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs,
            )

        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            if bold:
                if leg.get_title() is not None:
                    leg.get_title().set_fontweight("bold")
                    leg.get_title().set_color("#0f172a")
                    leg.get_title().set_fontsize(11.0)
                for text in leg.get_texts():
                    text.set_fontweight("bold")
                    text.set_color("#0f172a")
                    txt = text.get_text()
                    if txt and isinstance(txt, str) and txt[0].islower():
                        text.set_text(txt.capitalize())
    else:
        if shrink is not None and "shrink" not in hist_kwargs:
            hist_kwargs["shrink"] = shrink
        if isinstance(data, pd.DataFrame) and x is not None:
            sns.histplot(
                data=data,
                x=x,
                kde=kde,
                color=color,
                ax=ax,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.75,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs,
            )
        else:
            sns.histplot(
                x=data,
                kde=kde,
                color=color,
                ax=ax,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.75,
                log_scale=log_scale,
                line_kws={"linewidth": 2.2} if kde else None,
                **hist_kwargs,
            )


def draw_qq(
    ax: plt.Axes,
    data: pd.Series,
    dot_color: Any = "teal",
    line_color: Any = "#d73027",
    var_label: str = "",
    bold: bool = True,
) -> None:
    """Dibuja un gráfico de cuantiles teóricos vs observados de la Normal."""
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title("")

    if len(ax.lines) >= 2:
        ax.lines[0].set_color(dot_color)
        ax.lines[0].set_marker("o")
        ax.lines[0].set_markersize(6.5)
        ax.lines[0].set_markeredgewidth(0.8)
        ax.lines[0].set_markeredgecolor("white")
        ax.lines[0].set_alpha(0.80)

        ax.lines[1].set_color(line_color)
        ax.lines[1].set_linestyle("--")
        ax.lines[1].set_linewidth(2.0)

    axis_color = "#0f172a" if bold else "#222222"
    font_weight = "bold" if bold else "normal"
    ax.set_xlabel("Cuantiles Teóricos", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
    ax.set_ylabel(f"Cuantiles Observados ({var_label})", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)


class DistribucionPlotsMixin:
    """Mixin que agrupa los pipelines de visualización para variables cuantitativas."""

    def plot_distribucion(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        color: Optional[Any] = None,
        box_color: Optional[Any] = None,
        hist_color: Optional[Any] = None,
        kde: bool = False,
        overlay_points: bool = True,
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
        bold_annotations: bool = True,
        percentile_mask: Union[bool, Tuple[float, float]] = False,
        show_limit: bool = False,
        limit: Optional[float] = None,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """
        Construye el gráfico integrado univariante por excelencia de la investigación:
        diagrama de caja superior (con stripplot de dispersión individual) e histograma inferior
        con eje común, escala logarítmica opcional, máscara de estratos percentilares y límites de referencia.
        """
        cols = self._normalize_columns(columns)
        if axes is not None and len(cols) > 1:
            raise ValueError("El parámetro 'axes' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        if color is None and "color" in kwargs:
            color = kwargs.pop("color")
        elif "color" in kwargs:
            kwargs.pop("color")

        if color is None:
            plot_color = palette_colors[0] if palette_colors else "teal"
        elif isinstance(color, int):
            plot_color = palette_colors[color % len(palette_colors)]
        else:
            plot_color = color

        final_box_color = box_color if box_color is not None else plot_color
        final_hist_color = hist_color if hist_color is not None else plot_color
        edge_c = "#0f172a" if bold_annotations else "#2c3e50"

        # Manejo de aliases para máscara percentilar y límite de referencia
        show_mask = percentile_mask
        if "mask_percentiles" in kwargs:
            show_mask = kwargs.pop("mask_percentiles")
        elif "show_percentiles" in kwargs:
            show_mask = kwargs.pop("show_percentiles")
        elif "mascara_percentiles" in kwargs:
            show_mask = kwargs.pop("mascara_percentiles")

        if "show_limit" in kwargs:
            show_limit = kwargs.pop("show_limit")
        if "limit" in kwargs:
            limit = kwargs.pop("limit")

        for col in cols:
            data_clean = self._clean_series(col, min_obs=1)
            if data_clean is None:
                continue

            if axes is not None:
                ax_box, ax_hist = axes
                fig = ax_box.figure
            else:
                fig, (ax_box, ax_hist) = self.crear_lienzo(
                    nrows=2,
                    ncols=1,
                    figsize=figsize,
                    dpi=dpi,
                    sharex=True,
                    gridspec_kw={"height_ratios": [0.18, 0.82], "hspace": 0.02},
                    bold=bold_annotations,
                )

            # 1. Pipeline Boxplot superior
            draw_boxplot(ax_box, data_clean, color=final_box_color, edge_color=edge_c, orientation="horizontal", overlay_points=overlay_points)
            ax_box.set(xlabel="", yticks=[])
            ax_box.tick_params(bottom=False, labelbottom=False)
            for spine in ["top", "right", "left", "bottom"]:
                if spine in ax_box.spines:
                    ax_box.spines[spine].set_visible(False)
            ax_box.grid(False)
            ax_box.set_ylim(-0.48, 0.48)

            # 2. Pipeline Histograma inferior
            draw_histogram(
                ax_hist,
                data_clean,
                color=final_hist_color,
                kde=kde,
                log_scale=log_scale,
                bins=bins,
                bold=bold_annotations,
                **kwargs,
            )

            # Coordinar escala logarítmica
            if log_scale:
                ax_box.set_xscale("log")
                ax_hist.set_xscale("log")

            self._style_axis(ax_hist, tick_length=5.0, bold=bold_annotations)

            # Formateo de etiquetas de ejes
            is_log = bool(log_scale[0] if isinstance(log_scale, (tuple, list)) else log_scale)
            if xlabel is not None:
                var_label = xlabel
            elif is_log:
                base_label = self.get_label(col, labels_map)
                var_label = f"ln [{base_label}]" if not base_label.lower().startswith(("ln", "log")) else base_label
            else:
                var_label = self.get_label(col, labels_map)

            axis_color = "#0f172a" if bold_annotations else "#222222"
            font_weight = "bold" if bold_annotations else "normal"

            if not kde:
                if not log_scale:
                    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax_hist.set_ylabel("Frecuencia ($n$)", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            else:
                ax_hist.set_ylabel("Densidad", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)

            ax_hist.set_xlabel(f"{var_label}", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)

            # Límite permisible de referencia toxicológica opcional
            if show_limit or limit is not None:
                lim_val = limit if limit is not None else DEFAULT_PERMISSIBLE_LIMITS.get(col)
                if lim_val is not None:
                    ax_hist.axvline(lim_val, color="#d95f02", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)
                    ax_box.axvline(lim_val, color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

            # Máscara percentilar (<= P25, entre > P25 y < P75, y >= P75)
            if show_mask:
                if isinstance(show_mask, (tuple, list)) and len(show_mask) == 2:
                    q_low, q_high = float(show_mask[0]), float(show_mask[1])
                else:
                    q_low, q_high = 25.0, 75.0

                p_low = float(np.percentile(data_clean, q_low))
                p_high = float(np.percentile(data_clean, q_high))

                n_tot = len(data_clean)
                n_low = int((data_clean <= p_low).sum())
                n_mid = int(((data_clean > p_low) & (data_clean < p_high)).sum())
                n_high = int((data_clean >= p_high).sum())

                pct_low = (n_low / n_tot) * 100.0 if n_tot > 0 else 0.0
                pct_mid = (n_mid / n_tot) * 100.0 if n_tot > 0 else 0.0
                pct_high = (n_high / n_tot) * 100.0 if n_tot > 0 else 0.0

                x_min, x_max = ax_hist.get_xlim()

                # Margen superior suficiente para ubicar las tarjetas informativas
                curr_ymin, curr_ymax = ax_hist.get_ylim()
                ax_hist.set_ylim(curr_ymin, curr_ymax * 1.25)

                # 1. Franjas sombreadas de fondo
                ax_hist.axvspan(x_min, p_low, color="#e0f2fe", alpha=0.40, zorder=0)
                ax_hist.axvspan(p_low, p_high, color="#f8fafc", alpha=0.55, zorder=0)
                ax_hist.axvspan(p_high, x_max, color="#fee2e2", alpha=0.40, zorder=0)

                ax_box.axvspan(x_min, p_low, color="#e0f2fe", alpha=0.25, zorder=0)
                ax_box.axvspan(p_low, p_high, color="#f8fafc", alpha=0.35, zorder=0)
                ax_box.axvspan(p_high, x_max, color="#fee2e2", alpha=0.25, zorder=0)

                # 2. Líneas divisorias de percentiles
                ax_hist.axvline(p_low, color="#0284c7", linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)
                ax_hist.axvline(p_high, color="#e11d48", linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)

                ax_box.axvline(p_low, color="#0284c7", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
                ax_box.axvline(p_high, color="#e11d48", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)

                ax_hist.set_xlim(x_min, x_max)
                ax_box.set_xlim(x_min, x_max)

                # 3. Posicionamiento geométrico de las tarjetas (badges)
                if is_log:
                    center_low = 10 ** ((np.log10(x_min) + np.log10(p_low)) / 2) if x_min > 0 and p_low > 0 else (x_min + p_low) / 2
                    center_mid = 10 ** ((np.log10(p_low) + np.log10(p_high)) / 2) if p_low > 0 and p_high > 0 else (p_low + p_high) / 2
                    center_high = 10 ** ((np.log10(p_high) + np.log10(x_max)) / 2) if p_high > 0 and x_max > 0 else (p_high + x_max) / 2
                else:
                    center_low = (x_min + p_low) / 2
                    center_mid = (p_low + p_high) / 2
                    center_high = (p_high + x_max) / 2

                bbox_low = dict(boxstyle="round,pad=0.32", fc="#e0f2fe", ec="#0284c7", lw=1.1, alpha=0.95)
                bbox_mid = dict(boxstyle="round,pad=0.32", fc="#ffffff", ec="#64748b", lw=1.1, alpha=0.95)
                bbox_high = dict(boxstyle="round,pad=0.32", fc="#fee2e2", ec="#e11d48", lw=1.1, alpha=0.95)

                q_low_str = f"P_{{{int(q_low)}}}" if q_low.is_integer() else f"P_{{{q_low}}}"
                q_high_str = f"P_{{{int(q_high)}}}" if q_high.is_integer() else f"P_{{{q_high}}}"

                ax_hist.text(
                    center_low, 0.90,
                    f"$\\leq {q_low_str}$ ({p_low:.2f})\n{pct_low:.1f}% ($n={n_low}$)",
                    transform=ax_hist.get_xaxis_transform(),
                    ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0369a1", bbox=bbox_low, zorder=5
                )
                ax_hist.text(
                    center_mid, 0.90,
                    f"$> {q_low_str}$ y $< {q_high_str}$\n{pct_mid:.1f}% ($n={n_mid}$)",
                    transform=ax_hist.get_xaxis_transform(),
                    ha="center", va="center", fontsize=9.0, fontweight="bold", color="#334155", bbox=bbox_mid, zorder=5
                )
                ax_hist.text(
                    center_high, 0.90,
                    f"$\\geq {q_high_str}$ ({p_high:.2f})\n{pct_high:.1f}% ($n={n_high}$)",
                    transform=ax_hist.get_xaxis_transform(),
                    ha="center", va="center", fontsize=9.0, fontweight="bold", color="#be123c", bbox=bbox_high, zorder=5
                )

            if bold_annotations:
                for tick in ax_hist.get_yticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                for tick in ax_hist.get_xticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                ax_hist.grid(axis="y", linestyle="--", alpha=0.35)

            if axes is None:
                fig.subplots_adjust(hspace=0.02)

            self._save_figure(
                fig=fig,
                base_name=f"{col}_distribucion",
                save_dir=save_dir,
                formats=save_format,
                dpi=dpi,
                close=close_after_save,
            )
            figures.append(fig)

        return figures

    def obtener_mascara_percentiles(
        self: BasePlots,
        column: str,
        q_low: float = 25.0,
        q_high: float = 75.0,
    ) -> Dict[str, Any]:
        """
        Calcula la máscara booleana y estadísticas de estratificación percentilar
        para una variable cuantitativa (<= P_low, entre > P_low y < P_high, y >= P_high).

        Retorna un diccionario con:
        - p_low: valor numérico del percentil inferior
        - p_high: valor numérico del percentil superior
        - n_total: total de observaciones válidas
        - estratos: diccionario con máscaras booleanas, recuentos (n) y porcentajes (%)
        - resumen: DataFrame estructurado con los 3 estratos
        """
        real_col = self._resolve_column(column)
        data_clean = self._clean_series(column, min_obs=1)
        if data_clean is None:
            raise ValueError(f"No hay observaciones válidas para la columna '{column}'.")

        p_low = float(np.percentile(data_clean, q_low))
        p_high = float(np.percentile(data_clean, q_high))

        mask_low = data_clean <= p_low
        mask_mid = (data_clean > p_low) & (data_clean < p_high)
        mask_high = data_clean >= p_high

        n_tot = len(data_clean)
        n_low = int(mask_low.sum())
        n_mid = int(mask_mid.sum())
        n_high = int(mask_high.sum())

        pct_low = (n_low / n_tot) * 100.0 if n_tot > 0 else 0.0
        pct_mid = (n_mid / n_tot) * 100.0 if n_tot > 0 else 0.0
        pct_high = (n_high / n_tot) * 100.0 if n_tot > 0 else 0.0

        q_low_label = f"P{int(q_low) if q_low.is_integer() else q_low}"
        q_high_label = f"P{int(q_high) if q_high.is_integer() else q_high}"

        resumen = pd.DataFrame([
            {"Estrato": f"<= {q_low_label}", "Rango": f"<= {p_low:.2f}", "n": n_low, "Porcentaje (%)": round(pct_low, 2)},
            {"Estrato": f"> {q_low_label} y < {q_high_label}", "Rango": f"({p_low:.2f}, {p_high:.2f})", "n": n_mid, "Porcentaje (%)": round(pct_mid, 2)},
            {"Estrato": f">= {q_high_label}", "Rango": f">= {p_high:.2f}", "n": n_high, "Porcentaje (%)": round(pct_high, 2)},
        ])

        return {
            "column": real_col,
            "q_low": q_low,
            "q_high": q_high,
            "p_low": p_low,
            "p_high": p_high,
            "p_lower": p_low,
            "p_upper": p_high,
            "n_total": n_tot,
            "counts": {
                f"<={q_low_label}": n_low,
                f">{q_low_label} y <{q_high_label}": n_mid,
                f">={q_high_label}": n_high,
            },
            "percentages": {
                f"<={q_low_label}": pct_low,
                f">{q_low_label} y <{q_high_label}": pct_mid,
                f">={q_high_label}": pct_high,
            },
            "estratos": {
                f"<={q_low_label}": {"n": n_low, "pct": pct_low, "mask": mask_low},
                f">{q_low_label} y <{q_high_label}": {"n": n_mid, "pct": pct_mid, "mask": mask_mid},
                f">={q_high_label}": {"n": n_high, "pct": pct_high, "mask": mask_high},
            },
            "resumen": resumen,
        }

    get_percentile_mask = obtener_mascara_percentiles
    mascara_percentiles = obtener_mascara_percentiles

    def plot_qq(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        color: Optional[Any] = None,
        line_color: Optional[Any] = None,
        labels_map: Optional[Dict[str, str]] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[float, float] = (6.0, 5.5),
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        bold_annotations: bool = True,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """Construye gráficos Q-Q de cuantiles observados vs teóricos de la Normal."""
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        if color is None and "color" in kwargs:
            color = kwargs.pop("color")
        elif "color" in kwargs:
            kwargs.pop("color")

        dot_color = color if color is not None else (palette_colors[0] if palette_colors else "teal")
        active_line_color = line_color if line_color is not None else (palette_colors[1] if len(palette_colors) > 1 else "#d73027")

        for col in cols:
            data_clean = self._clean_series(col, min_obs=3)
            if data_clean is None:
                continue

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = self.crear_lienzo(figsize=figsize, dpi=dpi, bold=bold_annotations)

            var_label = ylabel if ylabel is not None else self.get_label(col, labels_map)
            draw_qq(target_ax, data_clean, dot_color, active_line_color, var_label, bold=bold_annotations)
            self._style_axis(target_ax, tick_length=4.5, bold=bold_annotations)

            if bold_annotations:
                for tick in target_ax.get_yticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                for tick in target_ax.get_xticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                target_ax.grid(True, linestyle="--", alpha=0.35)

            if ax is None:
                fig.tight_layout()

            self._save_figure(
                fig=fig,
                base_name=f"{col}_qqplot",
                save_dir=save_dir,
                formats=save_format,
                dpi=dpi,
                close=close_after_save,
            )
            figures.append(fig)

        return figures

    def plot_histograma(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        color: Optional[Any] = None,
        hue: Optional[str] = None,
        multiple: str = "layer",
        shrink: Optional[float] = None,
        palette: Optional[Any] = None,
        kde: bool = False,
        bins: Optional[Any] = None,
        log_scale: Union[bool, float, Tuple[bool, bool]] = False,
        figsize: Tuple[float, float] = (7.0, 5.0),
        labels_map: Optional[Dict[str, str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        bold_annotations: bool = True,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """
        Construye un histograma individual o estratificado (por hue) con estimación de densidad opcional.
        """
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        if color is None and "color" in kwargs:
            color = kwargs.pop("color")
        elif "color" in kwargs:
            kwargs.pop("color")

        if color is None:
            plot_color = palette_colors[0] if palette_colors else "teal"
        elif isinstance(color, int):
            plot_color = palette_colors[color % len(palette_colors)]
        else:
            plot_color = color
        active_palette = palette if palette is not None else self.palette

        for col in cols:
            real_col = self._resolve_column(col)
            real_hue = self._resolve_column(hue) if hue is not None else None

            if hue is not None:
                if self.df is None or real_col not in self.df.columns or real_hue not in self.df.columns:
                    warnings.warn(
                        f"Las columnas '{col}' o '{hue}' no se encuentran en el DataFrame. Omitiendo gráfico.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                sub_df = self.df[[real_col, real_hue]].dropna().copy()
                sub_df[real_col] = pd.to_numeric(sub_df[real_col], errors="coerce")
                sub_df = sub_df.dropna(subset=[real_col])
                if len(sub_df) < 1:
                    warnings.warn(
                        f"La columna '{col}' no contiene observaciones válidas tras filtrar por '{hue}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
            else:
                data_clean = self._clean_series(col, min_obs=1)
                if data_clean is None:
                    continue

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = self.crear_lienzo(figsize=figsize, dpi=dpi, bold=bold_annotations)

            if hue is not None:
                draw_histogram(
                    target_ax,
                    data=sub_df,
                    x=real_col,
                    hue=real_hue,
                    palette=active_palette,
                    multiple=multiple,
                    shrink=shrink,
                    kde=kde,
                    log_scale=log_scale,
                    bins=bins,
                    bold=bold_annotations,
                    **kwargs,
                )
            else:
                draw_histogram(
                    target_ax,
                    data=data_clean,
                    color=plot_color,
                    kde=kde,
                    log_scale=log_scale,
                    bins=bins,
                    shrink=shrink,
                    bold=bold_annotations,
                    **kwargs,
                )

            is_log = bool(log_scale[0] if isinstance(log_scale, (tuple, list)) else log_scale)
            if xlabel is not None:
                var_label = xlabel
            elif is_log:
                base_label = self.get_label(col, labels_map)
                var_label = f"ln [{base_label}]" if not base_label.lower().startswith(("ln", "log")) else base_label
            else:
                var_label = self.get_label(col, labels_map)

            self._style_axis(target_ax, tick_length=4.5, bold=bold_annotations)
            axis_color = "#0f172a" if bold_annotations else "#222222"
            font_weight = "bold" if bold_annotations else "normal"

            if ylabel is not None:
                target_ax.set_ylabel(ylabel, fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            elif kwargs.get("stat") == "density":
                target_ax.set_ylabel("Densidad", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            elif kwargs.get("stat") == "probability" or multiple == "fill":
                target_ax.set_ylabel("Proporción", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            elif multiple == "percent" or kwargs.get("stat") == "percent":
                target_ax.set_ylabel("Porcentaje (%)", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            else:
                if not log_scale:
                    target_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                target_ax.set_ylabel("Frecuencia ($n$)", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)

            target_ax.set_xlabel(f"{var_label}", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)

            if title is not None:
                target_ax.set_title(title, fontsize=12.0, fontweight=font_weight, pad=10, color=axis_color)

            if bold_annotations:
                for tick in target_ax.get_yticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                for tick in target_ax.get_xticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                target_ax.grid(axis="y", linestyle="--", alpha=0.35)

            if hue is not None and target_ax.get_legend() is not None:
                leg = target_ax.get_legend()
                hue_title = self.get_label(real_hue, labels_map)
                leg.set_title(hue_title)
                if bold_annotations:
                    if leg.get_title() is not None:
                        leg.get_title().set_fontweight("bold")
                        leg.get_title().set_color("#0f172a")
                        leg.get_title().set_fontsize(11.0)
                    for text in leg.get_texts():
                        text.set_fontweight("bold")
                        text.set_color("#0f172a")
                        txt = text.get_text()
                        if txt and isinstance(txt, str) and txt[0].islower():
                            text.set_text(txt.capitalize())

            if ax is None:
                fig.tight_layout()

            base_name = f"{col}_histograma_por_{real_hue.lower()}" if hue is not None else f"{col}_histograma"
            self._save_figure(
                fig=fig,
                base_name=base_name,
                save_dir=save_dir,
                formats=save_format,
                dpi=dpi,
                close=close_after_save,
            )
            figures.append(fig)

        return figures

    def plot_histograma_estratificado(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        hue: str = "Sexo",
        multiple: str = "stack",
        shrink: float = 0.8,
        palette: Optional[Any] = None,
        kde: bool = True,
        bins: Optional[Any] = None,
        log_scale: Union[bool, float, Tuple[bool, bool]] = False,
        figsize: Tuple[float, float] = (7.5, 5.0),
        labels_map: Optional[Dict[str, str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        bold_annotations: bool = True,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """
        Construye un histograma estratificado por una covariable categórica (hue)
        con barras apiladas (multiple="stack"), separación visual (shrink) y estimación de densidad (KDE).
        """
        active_palette = palette if palette is not None else self.palette
        return self.plot_histograma(
            columns=columns,
            hue=hue,
            multiple=multiple,
            shrink=shrink,
            palette=active_palette,
            kde=kde,
            bins=bins,
            log_scale=log_scale,
            figsize=figsize,
            labels_map=labels_map,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ax=ax,
            save_dir=save_dir,
            save_format=save_format,
            dpi=dpi,
            close_after_save=close_after_save,
            bold_annotations=bold_annotations,
            **kwargs,
        )

    def plot_boxplot(
        self: BasePlots,
        columns: Union[str, Sequence[str]],
        color: Optional[Any] = None,
        overlay_points: bool = True,
        log_scale: bool = False,
        labels_map: Optional[Dict[str, str]] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 6.0),
        ax: Optional[plt.Axes] = None,
        save_dir: Optional[str] = None,
        save_format: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close_after_save: bool = False,
        bold_annotations: bool = True,
        **kwargs: Any,
    ) -> List[plt.Figure]:
        """Construye un diagrama de caja individual."""
        cols = self._normalize_columns(columns)
        if ax is not None and len(cols) > 1:
            raise ValueError("El parámetro 'ax' solo se puede utilizar cuando se proporciona una sola columna.")

        figures = []
        palette_colors = sns.color_palette(self.palette)
        if color is None and "color" in kwargs:
            color = kwargs.pop("color")
        elif "color" in kwargs:
            kwargs.pop("color")

        if color is None:
            plot_color = palette_colors[0] if palette_colors else "teal"
        elif isinstance(color, int):
            plot_color = palette_colors[color % len(palette_colors)]
        else:
            plot_color = color
        edge_c = "#0f172a" if bold_annotations else "#2c3e50"

        for col in cols:
            data_clean = self._clean_series(col, min_obs=1)
            if data_clean is None:
                continue

            if ax is not None:
                fig = ax.figure
                target_ax = ax
            else:
                fig, target_ax = self.crear_lienzo(figsize=figsize, dpi=dpi, bold=bold_annotations)

            draw_boxplot(target_ax, data_clean, color=plot_color, edge_color=edge_c, orientation="vertical", overlay_points=overlay_points)

            if log_scale:
                target_ax.set_yscale("log")

            if ylabel is not None:
                var_label = ylabel
            elif log_scale:
                base_label = self.get_label(col, labels_map)
                var_label = f"ln [{base_label}]" if not base_label.lower().startswith(("ln", "log")) else base_label
            else:
                var_label = self.get_label(col, labels_map)

            axis_color = "#0f172a" if bold_annotations else "#222222"
            font_weight = "bold" if bold_annotations else "normal"

            target_ax.set_ylabel(f"{var_label}", fontsize=11.5, fontweight=font_weight, labelpad=8, color=axis_color)
            target_ax.set_xlabel("", fontsize=11.5)
            target_ax.set_xticks([])
            target_ax.tick_params(axis="x", bottom=False, labelbottom=False)

            self._style_axis(target_ax, tick_length=4.5, bold=bold_annotations)

            if bold_annotations:
                for tick in target_ax.get_yticklabels():
                    tick.set_fontweight("bold")
                    tick.set_color("#0f172a")
                target_ax.grid(axis="y", linestyle="--", alpha=0.35)

            if ax is None:
                fig.tight_layout()

            self._save_figure(
                fig=fig,
                base_name=f"{col}_boxplot",
                save_dir=save_dir,
                formats=save_format,
                dpi=dpi,
                close=close_after_save,
            )
            figures.append(fig)

        return figures

    # Nombres simplificados
    distribucion = plot_distribucion
    histograma = plot_histograma
    histograma_estratificado = plot_histograma_estratificado
    plot_histograma_apilado = plot_histograma_estratificado
    histograma_apilado = plot_histograma_estratificado
    boxplot = plot_boxplot
    qq = plot_qq


def obtener_mascara_percentiles(
    df: pd.DataFrame,
    column: str,
    q_low: float = 25.0,
    q_high: float = 75.0,
) -> Dict[str, Any]:
    """
    Función funcional directa para calcular la máscara booleana y estadísticas de estratificación
    percentilar de una variable cuantitativa (<= P_low, entre > P_low y < P_high, y >= P_high).
    """
    base = BasePlots(df)
    real_col = base._resolve_column(column)
    if real_col not in df.columns:
        raise ValueError(f"Columna '{column}' no encontrada en el DataFrame.")

    data_clean = pd.to_numeric(df[real_col], errors="coerce").dropna()
    if len(data_clean) == 0:
        raise ValueError(f"No hay observaciones válidas para la columna '{column}'.")

    p_low = float(np.percentile(data_clean, q_low))
    p_high = float(np.percentile(data_clean, q_high))

    mask_low = data_clean <= p_low
    mask_mid = (data_clean > p_low) & (data_clean < p_high)
    mask_high = data_clean >= p_high

    n_tot = len(data_clean)
    n_low = int(mask_low.sum())
    n_mid = int(mask_mid.sum())
    n_high = int(mask_high.sum())

    pct_low = (n_low / n_tot) * 100.0 if n_tot > 0 else 0.0
    pct_mid = (n_mid / n_tot) * 100.0 if n_tot > 0 else 0.0
    pct_high = (n_high / n_tot) * 100.0 if n_tot > 0 else 0.0

    q_low_label = f"P{int(q_low) if q_low.is_integer() else q_low}"
    q_high_label = f"P{int(q_high) if q_high.is_integer() else q_high}"

    resumen = pd.DataFrame([
        {"Estrato": f"<= {q_low_label}", "Rango": f"<= {p_low:.2f}", "n": n_low, "Porcentaje (%)": round(pct_low, 2)},
        {"Estrato": f"> {q_low_label} y < {q_high_label}", "Rango": f"({p_low:.2f}, {p_high:.2f})", "n": n_mid, "Porcentaje (%)": round(pct_mid, 2)},
        {"Estrato": f">= {q_high_label}", "Rango": f">= {p_high:.2f}", "n": n_high, "Porcentaje (%)": round(pct_high, 2)},
    ])

    return {
        "column": real_col,
        "q_low": q_low,
        "q_high": q_high,
        "p_low": p_low,
        "p_high": p_high,
        "p_lower": p_low,
        "p_upper": p_high,
        "n_total": n_tot,
        "counts": {
            f"<={q_low_label}": n_low,
            f">{q_low_label} y <{q_high_label}": n_mid,
            f">={q_high_label}": n_high,
        },
        "percentages": {
            f"<={q_low_label}": pct_low,
            f">{q_low_label} y <{q_high_label}": pct_mid,
            f">={q_high_label}": pct_high,
        },
        "estratos": {
            f"<={q_low_label}": {"n": n_low, "pct": pct_low, "mask": mask_low},
            f">{q_low_label} y <{q_high_label}": {"n": n_mid, "pct": pct_mid, "mask": mask_mid},
            f">={q_high_label}": {"n": n_high, "pct": pct_high, "mask": mask_high},
        },
        "resumen": resumen,
    }


get_percentile_mask = obtener_mascara_percentiles
mascara_percentiles = obtener_mascara_percentiles


