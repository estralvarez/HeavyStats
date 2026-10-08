"""
Módulo de radiografía epidemiológica y mapas de calor temáticos de la encuesta.
Implementa:
  1. Renderizado de mapas de calor por bloques temáticos con estratificación por percentiles.
  2. Normalización de totalizaciones por reactivos [0.0 - 1.0].
  3. Matriz ejecutiva de síntesis global multidominio.
  4. Generación masiva automatizada de los 7 paneles temáticos a 300 DPI (PNG y PDF).
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.bivariate.plots.base import BivariateBasePlots


class RadiografiaPlotsMixin:
    """Mixin para mapas de calor temáticos y radiografía epidemiológica de cuestionarios."""

    def survey_block_heatmap(
        self: BivariateBasePlots,
        mat_num: pd.DataFrame,
        mat_annot: pd.DataFrame,
        xlabel_text: str = "",
        header_total: str = "TOTAL\nBLOQUE",
        target_metal: str = "Mercurio_ug_L",
        id_col: str = "Muestra_Codificada",
        p25_val: Optional[float] = None,
        p75_val: Optional[float] = None,
        corte_bajo_idx: Optional[int] = None,
        corte_medio_idx: Optional[int] = None,
        row_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10.0, 8.5),
        rot_x: int = 45,
        cbar_pad: float = 0.12,
        filepath: Optional[str] = None,
        dpi: int = 300,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Dibuja un mapa de calor temático con columnas normalizadas [0.0 - 1.0],
        separador visual para la columna totalizadora, líneas de división por percentiles
        y etiquetas verticales de estrato en el margen derecho sin advertencias de fuentes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        mat_num = mat_num.astype(float)
        n_cols = mat_num.shape[1]
        n_filas = len(mat_num)

        sns.heatmap(
            mat_num,
            cmap="YlGnBu",
            annot=mat_annot,
            fmt="",
            vmin=0.0,
            vmax=1.0,
            cbar=True,
            cbar_kws={
                "label": "Escala Normalizada [0.0 - 1.0]",
                "shrink": 0.75,
                "pad": cbar_pad,
            },
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            yticklabels=row_labels if row_labels is not None else True,
        )

        metal_str = str(target_metal).lower() if target_metal else ""
        if "mercurio" in metal_str or "hg" in metal_str or "mercury" in metal_str:
            ylabel_text = "Participante (Orden Ascendente de Hg Plasmático)"
        elif "plomo" in metal_str or "lead" in metal_str or "pb" in metal_str:
            ylabel_text = "Participante (Orden Ascendente de Pb en Sangre)"
        elif "cadmio" in metal_str or "cd" in metal_str or "cadmium" in metal_str:
            ylabel_text = "Participante (Orden Ascendente de Cd en Sangre)"
        elif target_metal:
            ylabel_text = f"Participante (Orden Ascendente de {target_metal})"
        else:
            ylabel_text = "Participante (Orden Ascendente de Biomarcador)"
        ax.set_ylabel(ylabel_text, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel_text, fontsize=10, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=rot_x, ha="right", rotation_mode="anchor", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)

        # Línea vertical divisoria para la columna totalizadora (última columna)
        ax.axvline(n_cols - 1, color="#d95f02", linestyle="-", linewidth=2.2, alpha=0.95)
        ax.text(n_cols - 0.5, -0.65, header_total, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#d95f02")

        # Líneas horizontales de estratificación si se definieron los índices
        if corte_bajo_idx is not None and corte_medio_idx is not None:
            ax.axhline(corte_bajo_idx, color="#d95f02", linestyle="-", linewidth=2.0, alpha=0.9)
            ax.axhline(corte_medio_idx, color="#d95f02", linestyle="-", linewidth=2.0, alpha=0.9)

            n_bajo = corte_bajo_idx
            n_medio = corte_medio_idx - corte_bajo_idx
            n_alto = n_filas - corte_medio_idx

            # Rótulos de estratos en orientación vertical (270°) en el margen derecho (ASCII seguro)
            x_text = n_cols + (cbar_pad * n_cols * 0.45)
            lbl_b = f"ESTRATO BAJO (<= P25, n={n_bajo})" if p25_val is None else f"ESTRATO BAJO (<= P25: {p25_val:.2f}, n={n_bajo})"
            lbl_m = f"ESTRATO MEDIO (P25 - P75, n={n_medio})" if (p25_val is None or p75_val is None) else f"ESTRATO MEDIO ({p25_val:.2f} - {p75_val:.2f}, n={n_medio})"
            lbl_a = f"ESTRATO ALTO (>= P75, n={n_alto})" if p75_val is None else f"ESTRATO ALTO (>= P75: {p75_val:.2f}, n={n_alto})"

            ax.text(x_text, corte_bajo_idx / 2.0, lbl_b,
                    va="center", ha="center", rotation=270, fontsize=8.5, fontweight="bold", color="#2b83ba")
            ax.text(x_text, corte_bajo_idx + (corte_medio_idx - corte_bajo_idx) / 2.0, lbl_m,
                    va="center", ha="center", rotation=270, fontsize=8.5, fontweight="bold", color="#238b45")
            ax.text(x_text, corte_medio_idx + (n_filas - corte_medio_idx) / 2.0, lbl_a,
                    va="center", ha="center", rotation=270, fontsize=8.5, fontweight="bold", color="#d7191c")

        if filepath:
            self._save_figure(fig, filepath=filepath, dpi=dpi)

        return fig, ax

    def plot_survey_radiography(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        id_col: str = "Muestra_Codificada",
        p25_quantile: float = 0.25,
        p75_quantile: float = 0.75,
        blocks: Optional[Sequence[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Genera la suite completa de 7 mapas de calor temáticos de la encuesta
        (6 bloques temáticos + 1 síntesis de exposición global) con totalizaciones
        normalizadas en [0.0 - 1.0].

        Parameters
        ----------
        target_metal : str, default="Mercurio_ug_L"
            Columna del biomarcador para ordenamiento y estratificación percentilar
            ('Mercurio_ug_L', 'Plomo_ug_dL', 'Cadmio_ug_L', o alias como 'mercurio', 'plomo', 'cadmio', 'cd').
        id_col : str
            Columna de identificación de los participantes.
        p25_quantile : float
            Cuantil inferior (por defecto 0.25).
        p75_quantile : float
            Cuantil superior (por defecto 0.75).
        output_dir : str or Path, optional
            Directorio donde guardar las figuras en PNG y PDF.
        show : bool
            Si True, ejecuta plt.show() para cada figura.
        dpi : int
            Resolución gráfica.

        Returns
        -------
        Dict[str, Tuple[plt.Figure, plt.Axes]]
            Diccionario con las claves 'bloque1_dieta', ..., 'bloque7_exposicion_total'.
        """
        metal_resolved = self._resolve_column(target_metal)
        if metal_resolved not in self.df.columns:
            raise ValueError(f"Columna '{metal_resolved}' no encontrada en el DataFrame.")

        # Ordenar ascendente por concentración de biomarcador
        df_sorted = self.df.sort_values(by=metal_resolved).reset_index(drop=True).copy()

        p25 = float(np.percentile(df_sorted[metal_resolved], p25_quantile * 100))
        p75 = float(np.percentile(df_sorted[metal_resolved], p75_quantile * 100))

        corte_bajo = int((df_sorted[metal_resolved] <= p25).sum())
        corte_medio = int((df_sorted[metal_resolved] < p75).sum())

        # Rótulos de fila para las figuras
        metal_lower = metal_resolved.lower()
        unit_str = "ug/dL" if ("plomo" in metal_lower or "pb" in metal_lower or "lead" in metal_lower) else "ug/L"
        row_labels = [
            f"ID {int(r[id_col]):02d}  ({r[metal_resolved]:.2f} {unit_str})"
            if (id_col in df_sorted.columns and pd.notna(r[id_col]))
            else f"Obs {idx:02d}  ({r[metal_resolved]:.2f} {unit_str})"
            for idx, r in df_sorted.iterrows()
        ]

        # Mapeo editorial de variables en español
        nombres_map = {
            "Alim_Cereales_norm": "Cereales", "Alim_Leguminosas_norm": "Leguminosas",
            "Alim_Tuberculos_norm": "Tubérculos", "Alim_Carnes_norm": "Carnes",
            "Alim_Pescados_norm": "Pescados", "Alim_Bebidas_norm": "Bebidas",
            "Alim_Huevos_norm": "Huevos", "Alim_Lacteos_norm": "Lácteos",
            "Alim_Frutas_norm": "Frutas", "Alim_Vegetales_norm": "Vegetales",
            "Alim_Azucar_norm": "Azúcar", "Alim_Grasas_norm": "Grasas",
            "Alim_Chocolate_norm": "Chocolate",
            "Salud_Fuma_bin": "Tabaquismo", "Salud_Actividad_bin": "Actividad Física",
            "Salud_Bombillos_bin": "Bombillos Fluor.", "Salud_Techo_bin": "Techo Riesgo",
            "Salud_Joyeria_bin": "Uso Joyería",
            "Salud_Transporte_caminar": "Caminar", "Salud_Transporte_publico": "Transp. Público",
            "Salud_Transporte_vehiculo": "Vehículo Propio",
            "Salud_Agua_pozo_profundo": "Pozo Profundo", "Salud_Agua_filtrada": "Agua Filtrada",
            "Salud_Agua_mineral_embotellada": "Agua Embotellada",
            "Exposicion_Talleres_carpinteria": "Carpintería", "Exposicion_Talleres_latoneria": "Latonería",
            "Exposicion_Talleres_mecanico": "Taller Mecánico", "Exposicion_Lugares_canales": "Canales Drenaje",
            "Exposicion_Lugares_estacion_gasolina": "Estación Servicio", "Exposicion_Lugares_llenadora_gas_natural": "Llenadora Gas",
            "Exposicion_Lugares_rectificadora_motores": "Rectificadora Mot.", "Exposicion_Lugares_rios": "Cercanía a Ríos",
            "Exposicion_Industrias_fabrica_metales": "Fábrica Metales", "Exposicion_Industrias_fabrica_productos_quimicos": "Fábrica Químicos",
            "Exposicion_Cualquier_Taller": "Cualquier Taller", "Exposicion_Cualquier_Industria": "Cualquier Industria",
            "Exposicion_Cualquier_Lugar_Riesgo": "Cualquier Lugar Riesgo", "Salud_Cualquier_Agua_Riesgo": "Cualquier Agua Riesgo",
        }

        # Detección de columnas
        cols_df = list(df_sorted.columns)
        cols_alim = [c for c in cols_df if c.startswith("Alim_")]
        cols_hab = [c for c in ["Salud_Fuma", "Salud_Actividad", "Salud_Bombillos", "Salud_Techo", "Salud_Joyeria"] if c in cols_df]
        cols_mov = [c for c in ["Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo"] if c in cols_df]
        cols_agua = [c for c in ["Salud_Agua_pozo_profundo", "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada"] if c in cols_df]
        cols_antropica = [c for c in [
            "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
            "Exposicion_Lugares_canales", "Exposicion_Lugares_estacion_gasolina", "Exposicion_Lugares_llenadora_gas_natural",
            "Exposicion_Lugares_rectificadora_motores", "Exposicion_Lugares_rios", "Exposicion_Industrias_fabrica_metales",
            "Exposicion_Industrias_fabrica_productos_quimicos"
        ] if c in cols_df]
        cols_riesgo = [c for c in [
            "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria",
            "Exposicion_Cualquier_Lugar_Riesgo", "Salud_Cualquier_Agua_Riesgo"
        ] if c in cols_df]

        # Normalización por reactivo
        for c in cols_alim:
            df_sorted[f"{c}_norm"] = pd.to_numeric(df_sorted[c], errors="coerce").fillna(0) / 4.0
        for c in cols_hab:
            df_sorted[f"{c}_bin"] = (df_sorted[c].astype(str).str.upper() == "SI").astype(int)

        # Totales normalizados en [0.0 - 1.0]
        n_alim = max(len(cols_alim), 1)
        n_hab = max(len(cols_hab), 1)
        n_mov = max(len(cols_mov), 1)
        n_agua = max(len(cols_agua), 1)
        n_ant = max(len(cols_antropica), 1)
        n_rie = max(len(cols_riesgo), 1)

        df_sorted["Tot_Dieta_Norm"] = df_sorted[[f"{c}_norm" for c in cols_alim]].sum(axis=1) / float(n_alim)
        df_sorted["Tot_Habitos_Norm"] = df_sorted[[f"{c}_bin" for c in cols_hab]].sum(axis=1) / float(n_hab)
        df_sorted["Tot_Movilidad_Norm"] = df_sorted[cols_mov].sum(axis=1) / float(n_mov)
        df_sorted["Tot_Agua_Norm"] = df_sorted[cols_agua].sum(axis=1) / float(n_agua)
        df_sorted["Tot_Exp_Antropica_Norm"] = df_sorted[cols_antropica].sum(axis=1) / float(n_ant)
        df_sorted["Tot_Riesgo_Norm"] = df_sorted[cols_riesgo].sum(axis=1) / float(n_rie)

        # Exposición total global normalizada
        cols_todas = [f"{c}_norm" for c in cols_alim] + [f"{c}_bin" for c in cols_hab] + cols_mov + cols_agua + cols_antropica + cols_riesgo
        n_todas = max(len(cols_todas), 1)
        df_sorted["Exposicion_Total_Norm"] = df_sorted[cols_todas].sum(axis=1) / float(n_todas)

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        resultados: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

        # Definición de configuraciones de los bloques
        bloques_cfg = [
            {
                "key": "bloque1_dieta",
                "cols": [f"{c}_norm" for c in cols_alim],
                "col_tot": "Tot_Dieta_Norm",
                "nom_tot": "Dieta Norm.",
                "fmt_cols": lambda v: f"{v:g}",
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque I: Hábitos Dietarios (Escala 0.0 - 1.0) y Frecuencia Global Normalizada",
                "header_tot": "TOTAL\nDIETA (0-1)",
                "figsize": (12.5, 9.0),
                "rot_x": 50,
                "pad": 0.08,
            },
            {
                "key": "bloque2_habitos_hogar",
                "cols": [f"{c}_bin" for c in cols_hab],
                "col_tot": "Tot_Habitos_Norm",
                "nom_tot": "Hábitos Norm.",
                "fmt_cols": lambda v: str(int(v)),
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque II: Hábitos y Hogar (Presencia 1/0) y Fracción Activa Normalizada",
                "header_tot": "TOTAL\nHÁBITOS (0-1)",
                "figsize": (8.5, 9.0),
                "rot_x": 45,
                "pad": 0.12,
            },
            {
                "key": "bloque3_movilidad",
                "cols": cols_mov,
                "col_tot": "Tot_Movilidad_Norm",
                "nom_tot": "Movilidad Norm.",
                "fmt_cols": lambda v: str(int(v)),
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque III: Movilidad Urbana (Presencia 1/0) y Proporción Modal Normalizada",
                "header_tot": "TOTAL\nMOVILIDAD (0-1)",
                "figsize": (7.0, 9.0),
                "rot_x": 35,
                "pad": 0.15,
            },
            {
                "key": "bloque4_agua",
                "cols": cols_agua,
                "col_tot": "Tot_Agua_Norm",
                "nom_tot": "Agua Norm.",
                "fmt_cols": lambda v: str(int(v)),
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque IV: Fuentes Hídricas (Presencia 1/0) y Proporción Hídrica Normalizada",
                "header_tot": "TOTAL\nAGUA (0-1)",
                "figsize": (7.0, 9.0),
                "rot_x": 35,
                "pad": 0.15,
            },
            {
                "key": "bloque5_exposicion_antropica",
                "cols": cols_antropica,
                "col_tot": "Tot_Exp_Antropica_Norm",
                "nom_tot": "Exp. Antrópica Norm.",
                "fmt_cols": lambda v: str(int(v)),
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque V: Zonas Físicas y Antrópicas (Presencia 1/0) e Índice Antrópico Normalizado",
                "header_tot": "TOTAL\nANTRÓPICO (0-1)",
                "figsize": (11.0, 9.0),
                "rot_x": 55,
                "pad": 0.09,
            },
            {
                "key": "bloque6_riesgo_consolidado",
                "cols": cols_riesgo,
                "col_tot": "Tot_Riesgo_Norm",
                "nom_tot": "Riesgo Norm.",
                "fmt_cols": lambda v: str(int(v)),
                "fmt_tot": lambda v: f"{v:.2f}",
                "xlabel": "Bloque VI: Indicadores Agregados de Riesgo e Índice de Riesgo Normalizado",
                "header_tot": "TOTAL\nRIESGO (0-1)",
                "figsize": (8.0, 9.0),
                "rot_x": 45,
                "pad": 0.12,
            },
        ]

        # Renderizar bloques 1 al 6
        id_series = df_sorted[id_col] if id_col in df_sorted.columns else df_sorted.index
        for b in bloques_cfg:
            if blocks is not None and b["key"] not in blocks and "todos" not in blocks:
                continue

            mat_n = df_sorted[b["cols"]].copy()
            mat_n.index = id_series
            mat_n = mat_n.rename(columns=nombres_map)
            mat_n[b["nom_tot"]] = pd.to_numeric(df_sorted[b["col_tot"]], errors="coerce").fillna(0.0).values
            mat_n = mat_n.astype(float)

            mat_t = pd.DataFrame(index=mat_n.index, columns=mat_n.columns)
            for idx, r in df_sorted.iterrows():
                p_id = id_series.iloc[idx]
                for c in b["cols"]:
                    col_name = nombres_map.get(c, c)
                    mat_t.loc[p_id, col_name] = b["fmt_cols"](r[c])
                mat_t.loc[p_id, b["nom_tot"]] = b["fmt_tot"](r[b["col_tot"]])

            fp_png = str(out_path / f"radiografia_{b['key']}.png") if out_path else None
            fp_pdf = str(out_path / f"radiografia_{b['key']}.pdf") if out_path else None

            fig_b, ax_b = self.survey_block_heatmap(
                mat_num=mat_n,
                mat_annot=mat_t,
                xlabel_text=b["xlabel"],
                header_total=b["header_tot"],
                target_metal=metal_resolved,
                p25_val=p25,
                p75_val=p75,
                corte_bajo_idx=corte_bajo,
                corte_medio_idx=corte_medio,
                row_labels=row_labels,
                figsize=b["figsize"],
                rot_x=b["rot_x"],
                cbar_pad=b["pad"],
                filepath=fp_png,
                dpi=dpi,
            )
            if fp_pdf:
                fig_b.savefig(fp_pdf, dpi=dpi, bbox_inches="tight")

            resultados[b["key"]] = (fig_b, ax_b)
            if show:
                plt.show()

        # Renderizar Bloque 7: Síntesis Global
        render_b7 = (blocks is None or "bloque7_exposicion_total" in blocks or "sintesis" in blocks or "todos" in blocks)
        if render_b7:
            cols_b7 = [
                "Tot_Dieta_Norm", "Tot_Habitos_Norm", "Tot_Movilidad_Norm",
                "Tot_Agua_Norm", "Tot_Exp_Antropica_Norm", "Tot_Riesgo_Norm", "Exposicion_Total_Norm"
            ]
            nombres_b7 = {
                "Tot_Dieta_Norm": "Dieta Norm. (0-1)",
                "Tot_Habitos_Norm": "Hábitos Norm. (0-1)",
                "Tot_Movilidad_Norm": "Movilidad Norm. (0-1)",
                "Tot_Agua_Norm": "Agua Norm. (0-1)",
                "Tot_Exp_Antropica_Norm": "Antrópico Norm. (0-1)",
                "Tot_Riesgo_Norm": "Riesgo Norm. (0-1)",
                "Exposicion_Total_Norm": "EXPOSICIÓN TOTAL (0-1)",
            }
            mat_n7 = pd.DataFrame(index=id_series)
            mat_t7 = pd.DataFrame(index=id_series)
            for c in cols_b7:
                nom_c = nombres_b7[c]
                mat_n7[nom_c] = pd.to_numeric(df_sorted[c], errors="coerce").fillna(0.0).values
                mat_t7[nom_c] = [f"{v:.2f}" for v in df_sorted[c].values]
            mat_n7 = mat_n7.astype(float)

            fp7_png = str(out_path / "radiografia_bloque7_exposicion_total.png") if out_path else None
            fp7_pdf = str(out_path / "radiografia_bloque7_exposicion_total.pdf") if out_path else None

            fig7, ax7 = self.survey_block_heatmap(
                mat_num=mat_n7,
                mat_annot=mat_t7,
                xlabel_text="Síntesis Epidemiológica: Cargas Normalizadas por Dominio y Exposición Global (0.0 - 1.0)",
                header_total="EXPOSICIÓN\nGLOBAL (0-1)",
                target_metal=metal_resolved,
                p25_val=p25,
                p75_val=p75,
                corte_bajo_idx=corte_bajo,
                corte_medio_idx=corte_medio,
                row_labels=row_labels,
                figsize=(10.5, 9.0),
                rot_x=45,
                cbar_pad=0.10,
                filepath=fp7_png,
                dpi=dpi,
            )
            if fp7_pdf:
                fig7.savefig(fp7_pdf, dpi=dpi, bbox_inches="tight")

            resultados["bloque7_exposicion_total"] = (fig7, ax7)
            if show:
                plt.show()

        return resultados

    def plot_survey_synthesis(
        self: BivariateBasePlots,
        target_metal: str = "Mercurio_ug_L",
        id_col: str = "Muestra_Codificada",
        p25_quantile: float = 0.25,
        p75_quantile: float = 0.75,
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera y retorna exclusivamente la matriz de síntesis epidemiológica global (Bloque 7).

        Parameters
        ----------
        target_metal : str, default="Mercurio_ug_L"
            Columna o biomarcador ('Mercurio_ug_L', 'Plomo_ug_dL', 'Cadmio_ug_L', o alias 'cadmio', 'cd', 'plomo', etc.).
        """
        res = self.plot_survey_radiography(
            target_metal=target_metal,
            id_col=id_col,
            p25_quantile=p25_quantile,
            p75_quantile=p75_quantile,
            blocks=["bloque7_exposicion_total"],
            output_dir=output_dir,
            show=show,
            dpi=dpi,
        )
        return res["bloque7_exposicion_total"]

