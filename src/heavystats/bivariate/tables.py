"""
Generación de tablas científicas y reportes para análisis bivariantes.
Produce reportes de alta legibilidad en Jupyter mediante HTML responsivo (estilo Booktabs)
y exportación a Excel (.xlsx), CSV y DataFrames estructurados de Pandas.
"""

import os
import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats

from heavystats.html_utils import render_html_table, BaseReport
from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    get_label,
)
from heavystats.bivariate.constants import (
    DEFAULT_PRIMARY_BINARY_VARS,
    DEFAULT_PRIMARY_METALS,
    DEFAULT_EXPLORATORY_GROUPS,
    DIET_ORDINAL_MAP,
)
from heavystats.bivariate.tests import (
    mann_whitney_test,
    welch_ttest_log,
    kruskal_wallis_test,
    dunn_posthoc_test,
    jonckheere_terpstra_test,
    spearman_correlation,
    spearman_matrix,
    fisher_chi2_test,
    adjust_pvalues,
)


class BivariateTableReport(BaseReport):
    """
    Reporte de tabla bivariante con datos estructurados en un DataFrame
    y renderizado interactivo en HTML de calidad de publicación (estilo Booktabs).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "",
        subtitle: Optional[str] = None,
        notes: Optional[List[str]] = None,
        filepath: Optional[str] = None,
        cols: Optional[List[str]] = None,
        column_alignments: Optional[Dict[str, str]] = None,
        spanners: Optional[List[Dict[str, Any]]] = None,
        group_col: Optional[str] = None,
        posthoc_df: Optional[pd.DataFrame] = None,
        divider_borders: bool = True
    ):
        self._df = df.copy()
        self.title = title
        self.subtitle = subtitle
        self.notes = notes if notes is not None else []
        self.cols = cols if cols is not None else []
        self.column_alignments = column_alignments if column_alignments is not None else {}
        self.spanners = spanners if spanners is not None else []
        self.group_col = group_col
        self._posthoc_df = posthoc_df.copy() if posthoc_df is not None else None
        self.divider_borders = divider_borders

        if filepath:
            self.to_html(filepath)

    @property
    def df(self) -> pd.DataFrame:
        """Devuelve el DataFrame principal estructurado."""
        if self.group_col and self.group_col in self._df.columns:
            return self._df.drop(columns=[self.group_col]).copy()
        return self._df.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve una copia limpia del DataFrame de resultados."""
        return self.df

    def get_posthoc_dataframe(self) -> Optional[pd.DataFrame]:
        """Devuelve el DataFrame con los contrastes post-hoc pareados, si existen."""
        if self._posthoc_df is not None:
            return self._posthoc_df.copy()
        return None

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Exporta los resultados a un archivo CSV."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.df.to_csv(filepath, index=kwargs.get("index", False), **kwargs)

    def to_excel(self, filepath: str, sheet_name: str = "Bivariante", **kwargs: Any) -> None:
        """Exporta los resultados a un libro de Excel (.xlsx), incluyendo hoja de post-hoc si aplica."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            with pd.ExcelWriter(filepath) as writer:
                self.df.to_excel(writer, sheet_name=sheet_name, index=kwargs.get("index", False), **kwargs)
                if self._posthoc_df is not None and len(self._posthoc_df) > 0:
                    self._posthoc_df.to_excel(writer, sheet_name="PostHoc_Dunn", index=False)
        except Exception as e:
            alt_csv = os.path.splitext(filepath)[0] + ".csv"
            self.df.to_csv(alt_csv, index=kwargs.get("index", False))
            import warnings
            warnings.warn(f"No se pudo escribir en formato Excel ({e}). Se exportó como CSV en: '{alt_csv}'")

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Genera una tabla HTML responsiva con calidad de publicación (estilo Booktabs)."""
        html_code = render_html_table(
            df=self._df,
            title=self.title,
            subtitle=self.subtitle,
            notes=self.notes,
            column_alignments=self.column_alignments,
            spanners=self.spanners,
            group_col=self.group_col,
            full_page=full_page,
            divider_borders=self.divider_borders
        )

        if self._posthoc_df is not None and len(self._posthoc_df) > 0:
            posthoc_html = render_html_table(
                df=self._posthoc_df,
                title="Contrastes Pareados Post-Hoc (Prueba de Dunn)",
                subtitle="Ajuste de multiplicidad de valores p mediante el método de Holm-Bonferroni",
                column_alignments={"Contraste": "l", "Diferencia Rangos": "c", "Estadístico z": "c", "p sin ajustar": "c", "p ajustado (Holm)": "c", "¿Significativo?": "c"},
                full_page=False,
                divider_borders=True
            )
            html_code += "\n<div style='margin-top: 24px;'></div>\n" + posthoc_html

        if filepath:
            self._write_file(filepath, html_code)
        return html_code

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Devuelve los resultados en texto plano estructurado."""
        report_str = self.df.to_string(index=False)
        if self._posthoc_df is not None and len(self._posthoc_df) > 0:
            report_str += "\n\n--- Contrastes Post-Hoc (Dunn-Holm) ---\n" + self._posthoc_df.to_string(index=False)
        if filepath:
            self._write_file(filepath, report_str)
        return report_str

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<BivariateTableReport shape={self.df.shape} title='{self.title}'>"


class BivariateTables:
    """
    Clase para el cálculo de estadísticas bivariantes y generación de tablas científicas.
    Implementa comparaciones de dos grupos (Mann-Whitney y Welch t-test log), multivariables (Kruskal-Wallis y Dunn-Holm),
    tendencias ordinales (Jonckheere-Terpstra y Spearman), matrices de correlación inter-metales con bootstrap,
    asociaciones categóricas con Fisher exacto y tamizaje exploratorio con corrección FDR.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        labels_map: Optional[Dict[str, str]] = None
    ):
        self.df = df.copy()
        self.labels_map = DEFAULT_LABELS_MAP.copy()
        if labels_map is not None:
            self.labels_map.update(labels_map)

    def get_label(self, col: str, labels_map: Optional[Dict[str, str]] = None) -> str:
        """Obtiene la etiqueta legible para una columna."""
        active_map = self.labels_map.copy()
        if labels_map:
            active_map.update(labels_map)
        return get_label(col, active_map)

    def compare_binary(
        self,
        binary_col: str,
        metals: Optional[List[str]] = None,
        parametric_welch: bool = True,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Compara concentraciones de metales entre dos grupos independientes (ej. Sexo, Sector, Es_Expuesto).
        Calcula Mann-Whitney U con Diferencia de Medianas [IC 95% Bootstrap], estimador de Hodges-Lehmann,
        y prueba t de Welch sobre log-concentraciones con Razón de Medias Geométricas (GMR) [IC 95%].

        Parámetros
        ----------
        binary_col : str
            Nombre de la columna dicotómica o binaria (2 categorías).
        metals : List[str], opcional
            Lista de metales a comparar (por defecto Pb, Hg, Cd).
        parametric_welch : bool
            Si True, incluye las columnas paramétricas de Welch sobre escala logarítmica (GMR).
        n_boot : int
            Número de réplicas bootstrap para los intervalos de confianza.
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas personalizadas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        if binary_col not in self.df.columns:
            raise ValueError(f"La columna '{binary_col}' no se encuentra en el DataFrame.")

        # Obtener los 2 niveles no nulos
        unique_levels = list(self.df[binary_col].dropna().unique())
        if len(unique_levels) != 2:
            raise ValueError(
                f"La columna '{binary_col}' debe tener exactamente 2 categorías para compare_binary. Niveles encontrados: {unique_levels}"
            )

        lvl1, lvl2 = unique_levels[0], unique_levels[1]
        lvl1_str, lvl2_str = str(lvl1).strip(), str(lvl2).strip()

        records = []
        for metal in metals:
            if metal not in self.df.columns:
                continue

            sub_df = self.df.dropna(subset=[binary_col, metal])
            g1 = pd.to_numeric(sub_df[sub_df[binary_col] == lvl1][metal], errors="coerce").dropna().values
            g2 = pd.to_numeric(sub_df[sub_df[binary_col] == lvl2][metal], errors="coerce").dropna().values

            n1, n2 = len(g1), len(g2)
            if n1 == 0 or n2 == 0:
                continue

            # 1. Mann-Whitney U
            mw_res = mann_whitney_test(g1, g2, n_boot=n_boot)

            # 2. Welch t-test log
            pos_g1 = g1[g1 > 0]
            pos_g2 = g2[g2 > 0]
            if len(pos_g1) >= 2 and len(pos_g2) >= 2 and parametric_welch:
                welch_res = welch_ttest_log(pos_g1, pos_g2)
            else:
                welch_res = None

            metal_label = self.get_label(metal, labels_map)

            # Resumen de grupo 1 y grupo 2
            g1_med_str = f"{mw_res['median1']:.2f} [{mw_res['q25_1']:.2f} - {mw_res['q75_1']:.2f}]"
            g2_med_str = f"{mw_res['median2']:.2f} [{mw_res['q25_2']:.2f} - {mw_res['q75_2']:.2f}]"

            diff_med_ci = mw_res["diff_medians_ci"]
            diff_med_str = f"{mw_res['diff_medians']:+.2f} [{diff_med_ci[0]:+.2f}, {diff_med_ci[1]:+.2f}]"

            hl_ci = mw_res["hodges_lehmann_ci"]
            hl_str = f"{mw_res['hodges_lehmann']:+.2f} [{hl_ci[0]:+.2f}, {hl_ci[1]:+.2f}]"

            r_rb = mw_res["rank_biserial"]
            cles = mw_res["cles"]

            p_mw = mw_res["p_value"]
            p_mw_str = f"<strong>{p_mw:.4f}*</strong>" if p_mw < 0.05 else f"{p_mw:.4f}"

            row = {
                "Metal": f"**{metal_label}**",
                f"{lvl1_str} (n={n1}) Mediana [RIQ]": g1_med_str,
                f"{lvl2_str} (n={n2}) Mediana [RIQ]": g2_med_str,
                "Dif. Medianas [IC 95% Boot]": diff_med_str,
                "Hodges-Lehmann [IC 95%]": hl_str,
                "Efecto r_rb": f"{r_rb:+.3f}",
                "CLES (P[X>Y])": f"{cles:.2f}",
                "Mann-Whitney (p-val)": p_mw_str,
            }

            if parametric_welch and welch_res is not None:
                gmr = welch_res["gmr"]
                gmr_ci = welch_res["gmr_ci"]
                gmr_str = f"{gmr:.2f} [{gmr_ci[0]:.2f}, {gmr_ci[1]:.2f}]"
                p_welch = welch_res["p_value"]
                p_welch_str = f"<strong>{p_welch:.4f}*</strong>" if p_welch < 0.05 else f"{p_welch:.4f}"

                row["Razón Medias Geom. [IC 95%]"] = gmr_str
                row["Cohen d (log)"] = f"{welch_res['cohen_d_log']:+.2f}"
                row["Welch t log (p-val)"] = p_welch_str

            records.append(row)

        res_df = pd.DataFrame(records)

        var_label = self.get_label(binary_col, labels_map)
        title = f"Comparación Bivariante de Metales según {var_label}"
        subtitle = f"Grupos comparados: {lvl1_str} vs {lvl2_str} — Estimadores de localización, razones geométricas y tamaños del efecto"

        notes = [
            "Mediana [RIQ]: Mediana con rango intercuartílico [Q1 - Q3].",
            "Dif. Medianas [IC 95% Boot]: Diferencia de medianas obtenida mediante 2,000 remuestreos Bootstrap.",
            "Hodges-Lehmann [IC 95%]: Estimador no paramétrico de desplazamiento pareado con intervalo exacto.",
            "Efecto r_rb: Correlación biserial por rangos (-1 a +1); CLES: Common Language Effect Size / Probabilidad de Superioridad.",
            "Razón Medias Geom. [IC 95%]: Razón de Medias Geométricas (GMR) con intervalo analítico al 95% bajo escala logarítmica.",
            "* Indica valor p < 0.05 estadísticamente significativo."
        ]

        spanners = [
            {"label": "Metal", "columns": ["Metal"]},
            {"label": f"Distribución por {var_label}", "columns": [f"{lvl1_str} (n={n1}) Mediana [RIQ]", f"{lvl2_str} (n={n2}) Mediana [RIQ]"]},
            {"label": "Medidas de Efecto No Paramétricas", "columns": ["Dif. Medianas [IC 95% Boot]", "Hodges-Lehmann [IC 95%]", "Efecto r_rb", "CLES (P[X>Y])", "Mann-Whitney (p-val)"]},
        ]
        if parametric_welch:
            spanners.append({
                "label": "Parámetros Logarítmicos (Welch)",
                "columns": ["Razón Medias Geom. [IC 95%]", "Cohen d (log)", "Welch t log (p-val)"]
            })

        alignments = {c: "l" if c == "Metal" else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments,
            spanners=spanners
        )

    def compare_multigroup(
        self,
        cat_col: str,
        metals: Optional[List[str]] = None,
        posthoc: bool = True,
        method_p_adjust: str = "holm",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Aplica la prueba de Kruskal-Wallis para comparar metales entre 3 o más categorías independientes.
        Calcula el tamaño del efecto Epsilon al cuadrado (epsilon^2) y aplica la prueba post-hoc pareada
        de Dunn con ajuste de multiplicidad (Holm-Bonferroni) solo tras confirmación de significación global.

        Parámetros
        ----------
        cat_col : str
            Columna categórica (>= 3 niveles).
        metals : List[str], opcional
            Lista de metales a comparar.
        posthoc : bool
            Si True, calcula los contrastes pareados de Dunn-Holm para los metales con p global < 0.05.
        method_p_adjust : str
            Método de ajuste de multiplicidad ('holm' o 'fdr_bh').
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas personalizadas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        if cat_col not in self.df.columns:
            raise ValueError(f"La columna '{cat_col}' no se encuentra en el DataFrame.")

        unique_cats = [c for c in self.df[cat_col].dropna().unique()]
        if len(unique_cats) < 2:
            raise ValueError(f"La columna '{cat_col}' debe tener al menos 2 categorías distintas.")

        records = []
        posthoc_records = []

        for metal in metals:
            if metal not in self.df.columns:
                continue

            sub_df = self.df.dropna(subset=[cat_col, metal])
            groups_data = []
            group_names = []

            for cat in sorted(unique_cats, key=lambda x: str(x)):
                cat_vals = pd.to_numeric(sub_df[sub_df[cat_col] == cat][metal], errors="coerce").dropna().values
                if len(cat_vals) > 0:
                    groups_data.append(cat_vals)
                    group_names.append(str(cat))

            if len(groups_data) < 2:
                continue

            # Prueba de Kruskal-Wallis
            kw_res = kruskal_wallis_test(*groups_data, group_names=group_names)
            metal_label = self.get_label(metal, labels_map)

            # Resumen formateado por categoría
            cat_desc_list = []
            for s in kw_res["group_summaries"]:
                cat_desc_list.append(f"{s['name']} (n={s['n']}): {s['median']:.2f} [{s['q25']:.2f} - {s['q75']:.2f}]")
            cat_desc_str = " | ".join(cat_desc_list)

            p_kw = kw_res["p_value"]
            p_kw_str = f"<strong>{p_kw:.4f}*</strong>" if p_kw < 0.05 else f"{p_kw:.4f}"

            records.append({
                "Metal": f"**{metal_label}**",
                "Categorías Evaluadas": f"{kw_res['k_groups']} grupos",
                "Distribución Mediana [RIQ] por Categoría": cat_desc_str,
                "Estadístico H": f"{kw_res['h_statistic']:.2f}",
                "gl": kw_res["df"],
                "Kruskal-Wallis (p-val)": p_kw_str,
                "Tamaño Efecto (eps2)": f"{kw_res['epsilon_squared']:.3f}",
            })

            # Contrastes post-hoc de Dunn-Holm si aplica
            if posthoc and (p_kw < 0.05 or len(metals) == 1):
                dunn_comps = dunn_posthoc_test(*groups_data, group_names=group_names, method=method_p_adjust)
                for comp in dunn_comps:
                    p_adj = comp["p_adj"]
                    p_adj_str = f"<strong>{p_adj:.4f}*</strong>" if p_adj < 0.05 else f"{p_adj:.4f}"
                    posthoc_records.append({
                        "Metal": metal_label,
                        "Contraste": comp["contrast"],
                        "Diferencia Rangos": f"{comp['mean_rank_diff']:+.2f}",
                        "Estadístico z": f"{comp['z_statistic']:+.2f}",
                        "p sin ajustar": f"{comp['p_raw']:.4f}",
                        f"p ajustado ({method_p_adjust})": p_adj_str,
                        "¿Significativo?": "Sí" if comp["significant"] else "No"
                    })

        res_df = pd.DataFrame(records)
        posthoc_df = pd.DataFrame(posthoc_records) if posthoc_records else None

        var_label = self.get_label(cat_col, labels_map)
        title = f"Comparación Multi-Grupo de Metales según {var_label}"
        subtitle = f"Prueba global no paramétrica de Kruskal-Wallis y tamaño del efecto Epsilon al cuadrado (ε²)"

        notes = [
            "Mediana [RIQ]: Mediana con rango intercuartílico [Q1 - Q3] reportado para cada categoría.",
            "Estadístico H: Prueba de Kruskal-Wallis con aproximación Chi-cuadrado con (k-1) grados de libertad (gl).",
            "Tamaño Efecto (ε²): Epsilon al cuadrado, indicador normalizado de varianza explicada en rangos (0 a 1).",
            "* Indica valor p < 0.05 estadísticamente significativo."
        ]

        alignments = {c: "l" if c in ("Metal", "Distribución Mediana [RIQ] por Categoría") else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments,
            posthoc_df=posthoc_df
        )

    def ordinal_trend(
        self,
        ordinal_col: str,
        metals: Optional[List[str]] = None,
        ordinal_map: Optional[Dict[str, int]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa la tendencia ordinal entre una variable cualitativa ordenada (ej. consumo dietario de alimentos)
        y las concentraciones de metales mediante la prueba de tendencia de Jonckheere-Terpstra
        y el coeficiente de correlación de rangos de Spearman con intervalo Bootstrap al 95%.

        Parámetros
        ----------
        ordinal_col : str
            Columna de la variable ordinal (o frecuencias Alim_*).
        metals : List[str], opcional
            Lista de metales a evaluar.
        ordinal_map : Dict[str, int], opcional
            Mapeo de categorías a enteros ordinales.
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas legibles.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        if ordinal_col not in self.df.columns:
            raise ValueError(f"La columna '{ordinal_col}' no se encuentra en el DataFrame.")

        # Preparar columna ordinal codificada
        series_ord = self.df[ordinal_col].copy()
        if ordinal_map is None and ordinal_col.startswith("Alim_"):
            ordinal_map = DIET_ORDINAL_MAP

        if ordinal_map is not None:
            series_num = series_ord.apply(
                lambda v: ordinal_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
            series_num = pd.to_numeric(series_num, errors="coerce")
        else:
            series_num = pd.to_numeric(series_ord, errors="coerce")

        records = []
        for metal in metals:
            if metal not in self.df.columns:
                continue

            sub_df = pd.DataFrame({"ord": series_num, "val": pd.to_numeric(self.df[metal], errors="coerce")}).dropna()
            if len(sub_df) < 3:
                continue

            # 1. Correlación de Spearman con Bootstrap
            spearman_res = spearman_correlation(sub_df["ord"], sub_df["val"])

            # 2. Jonckheere-Terpstra
            ordered_levels = sorted(sub_df["ord"].unique())
            groups = [sub_df[sub_df["ord"] == lvl]["val"].values for lvl in ordered_levels]

            if len(groups) >= 2:
                jt_res = jonckheere_terpstra_test(groups)
                j_stat = f"{jt_res['j_statistic']:.1f}"
                z_stat = f"{jt_res['z_statistic']:+.2f}"
                p_jt = jt_res["p_value"]
                p_jt_str = f"<strong>{p_jt:.4f}*</strong>" if p_jt < 0.05 else f"{p_jt:.4f}"
            else:
                j_stat, z_stat, p_jt_str = "N/A", "N/A", "N/A"

            metal_label = self.get_label(metal, labels_map)
            rho = spearman_res["rho"]
            ci_low, ci_high = spearman_res["ci_low"], spearman_res["ci_high"]
            p_rho = spearman_res["p_value"]
            p_rho_str = f"<strong>{p_rho:.4f}*</strong>" if p_rho < 0.05 else f"{p_rho:.4f}"

            # Dirección de la tendencia
            if rho > 0.1 and p_rho < 0.1:
                trend_direction = "Ascendente (+)"
            elif rho < -0.1 and p_rho < 0.1:
                trend_direction = "Descendente (-)"
            else:
                trend_direction = "Sin tendencia clara"

            records.append({
                "Metal": f"**{metal_label}**",
                "N": len(sub_df),
                "Niveles Ordinales": f"{len(ordered_levels)} niveles",
                "Spearman (rho)": f"{rho:+.3f}",
                "IC 95% Boot (rho)": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
                "Spearman (p-val)": p_rho_str,
                "Jonckheere J": j_stat,
                "Jonckheere z": z_stat,
                "Jonckheere (p-val)": p_jt_str,
                "Interpretación de Tendencia": trend_direction
            })

        res_df = pd.DataFrame(records)

        var_label = self.get_label(ordinal_col, labels_map)
        title = f"Evaluación de Tendencias Ordinales para {var_label}"
        subtitle = f"Pruebas de correlación no paramétrica de Spearman (con IC Bootstrap) y tendencia monótona de Jonckheere-Terpstra"

        notes = [
            "Spearman (rho) [IC 95% Boot]: Coeficiente de correlación de rangos con intervalo de confianza al 95% vía Bootstrap (2,000 réplicas).",
            "Jonckheere-Terpstra: Prueba no paramétrica de hipótesis de tendencia monótona ordenada con corrección rigurosa de empates.",
            "* Indica valor p < 0.05 estadísticamente significativo."
        ]

        spanners = [
            {"label": "Metal e Información Muestral", "columns": ["Metal", "N", "Niveles Ordinales"]},
            {"label": "Correlación de Spearman", "columns": ["Spearman (rho)", "IC 95% Boot (rho)", "Spearman (p-val)"]},
            {"label": "Tendencia Jonckheere-Terpstra", "columns": ["Jonckheere J", "Jonckheere z", "Jonckheere (p-val)", "Interpretación de Tendencia"]}
        ]

        alignments = {c: "l" if c in ("Metal", "Interpretación de Tendencia") else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments,
            spanners=spanners
        )

    def metal_correlations(
        self,
        metals: Optional[List[str]] = None,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Calcula las correlaciones de Spearman pareadas entre Pb, Hg y Cd,
        obteniendo intervalos de confianza Bootstrap no paramétricos al 95%.

        Parámetros
        ----------
        metals : List[str], opcional
            Lista de metales a correlacionar (por defecto Plomo, Mercurio, Cadmio).
        n_boot : int
            Número de réplicas bootstrap.
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas personalizadas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        records = []
        k = len(metals)

        for i in range(k):
            for j in range(i + 1, k):
                m1, m2 = metals[i], metals[j]
                res = spearman_correlation(
                    self.df[m1], self.df[m2], n_boot=n_boot, confidence=0.95
                )

                m1_lbl = self.get_label(m1, labels_map)
                m2_lbl = self.get_label(m2, labels_map)

                rho = res["rho"]
                p_val = res["p_value"]
                p_str = f"<strong>{p_val:.4f}*</strong>" if p_val < 0.05 else f"{p_val:.4f}"
                ci_str = f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]"

                # Magnitud de correlación
                abs_rho = abs(rho)
                if abs_rho >= 0.7:
                    strength = "Fuerte"
                elif abs_rho >= 0.4:
                    strength = "Moderada"
                elif abs_rho >= 0.2:
                    strength = "Débil"
                else:
                    strength = "Muy débil / Nula"

                records.append({
                    "Par de Metales": f"**{m1_lbl}** vs **{m2_lbl}**",
                    "N": res["n"],
                    "Spearman (rho)": f"{rho:+.3f}",
                    "IC 95% Bootstrap": ci_str,
                    "Valor p": p_str,
                    "Magnitud de Asociación": strength,
                    "Dirección": "Positiva (+)" if rho > 0 else "Negativa (-)"
                })

        res_df = pd.DataFrame(records)

        title = "Matriz de Correlación de Spearman entre Metales Pesados"
        subtitle = "Asociaciones bivariantes pareadas entre Plomo, Mercurio y Cadmio con Intervalos de Confianza Bootstrap al 95%"

        notes = [
            "Spearman (rho): Coeficiente de correlación por rangos, robusto ante valores atípicos y distribuciones asimétricas.",
            "IC 95% Bootstrap: Intervalo de confianza no paramétrico obtenido mediante 2,000 remuestreos con reemplazo.",
            "* Indica valor p < 0.05 estadísticamente significativo."
        ]

        alignments = {c: "l" if c == "Par de Metales" else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments
        )

    def categorical_association(
        self,
        var1: str,
        var2: str,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa la asociación entre 2 variables categóricas cualitativas (ej. Es_Expuesto vs Riesgo_Pb).
        Calcula la tabla de contingencia cruzada, frecuencias observadas y esperadas,
        y aplica la Prueba Exacta de Fisher (o Chi-cuadrado) con advertencia automática si existen frecuencias esperadas < 5.

        Parámetros
        ----------
        var1 : str
            Primera variable categórica (filas).
        var2 : str
            Segunda variable categórica (columnas).
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas legibles.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if var1 not in self.df.columns or var2 not in self.df.columns:
            raise ValueError(f"Una o ambas columnas ('{var1}', '{var2}') no existen en el DataFrame.")

        sub_df = self.df[[var1, var2]].dropna()
        if len(sub_df) < 2:
            raise ValueError("Datos insuficientes para la tabla de contingencia.")

        # Tabla cruzada de contingencia
        ct = pd.crosstab(sub_df[var1], sub_df[var2], margins=True, margins_name="Total")
        ct_raw = pd.crosstab(sub_df[var1], sub_df[var2])

        # Test estadístico
        test_res = fisher_chi2_test(ct_raw.values)

        v1_lbl = self.get_label(var1, labels_map)
        v2_lbl = self.get_label(var2, labels_map)

        # Construir filas de la tabla formateada
        records = []
        for r_idx in ct.index:
            r_str = str(r_idx)
            row_dict = {f"{v1_lbl} \\ {v2_lbl}": f"**{r_str}**" if r_str == "Total" else r_str}
            row_total = ct.loc[r_idx, "Total"]
            for c_idx in ct.columns:
                c_str = str(c_idx)
                cnt = ct.loc[r_idx, c_idx]
                if r_str == "Total" or c_str == "Total":
                    pct_str = f"{cnt}"
                else:
                    pct = (cnt / row_total) * 100 if row_total > 0 else 0.0
                    pct_str = f"{cnt} ({pct:.1f}%)"
                row_dict[c_str] = pct_str
            records.append(row_dict)

        res_df = pd.DataFrame(records)

        title = f"Tabla de Contingencia: {v1_lbl} vs {v2_lbl}"
        p_rec = test_res["recommended_p"]
        p_rec_str = f"<strong>{p_rec:.4f}*</strong>" if p_rec < 0.05 else f"{p_rec:.4f}"

        subtitle = f"Prueba recomendada: <strong>{test_res['test']}</strong> (p={p_rec_str}) — {test_res['warning_msg']}"

        notes = [
            f"Total de observaciones evaluadas: N = {test_res['total_n']:.0f}.",
            f"Frecuencias esperadas pequeñas (<5): {test_res['cells_below_5']} celdas ({test_res['pct_below_5']:.0f}%).",
            f"Chi-cuadrado: Chi2 = {test_res['chi2_statistic']:.3f} (gl={test_res['chi2_df']}, p={test_res['chi2_p']:.4f}); V de Cramer = {test_res['cramer_v']:.3f}."
        ]

        if test_res["is_2x2"] and pd.notna(test_res["odds_ratio"]):
            or_ci = test_res["odds_ratio_ci"]
            notes.append(f"Prueba Exacta de Fisher (p={test_res['fisher_p']:.4f}); Odds Ratio (OR) = {test_res['odds_ratio']:.2f} [IC 95%: {or_ci[0]:.2f}, {or_ci[1]:.2f}].")
            if pd.notna(test_res["risk_ratio"]):
                rr_ci = test_res["risk_ratio_ci"]
                notes.append(f"Riesgo Relativo (RR) = {test_res['risk_ratio']:.2f} [IC 95%: {rr_ci[0]:.2f}, {rr_ci[1]:.2f}].")

        alignments = {c: "l" if c == f"{v1_lbl} \\ {v2_lbl}" else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments
        )

    def primary_analysis_summary(
        self,
        metals: Optional[List[str]] = None,
        primary_vars: Optional[List[str]] = None,
        method_p_adjust: str = "holm",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Consolida el Análisis Bivariante Primario (hipótesis a priori clave)
        contrastando los 3 metales (Pb, Hg, Cd) frente a los factores principales
        (`Es_Expuesto`, `Sector`, `Sexo`, `Score_Riesgo`), aplicando Mann-Whitney (2 grupos),
        Kruskal-Wallis (>=3 grupos) o correlación de Spearman (variables continuas)
        e incorporando ajuste de multiplicidad por Holm-Bonferroni.

        Parámetros
        ----------
        metals : List[str], opcional
            Metales primarios (Pb, Hg, Cd).
        primary_vars : List[str], opcional
            Variables primarias de agrupación (por defecto: Es_Expuesto, Sector, Sexo, Score_Riesgo).
        method_p_adjust : str
            Método de corrección de comparaciones múltiples ('holm' o 'fdr_bh').
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas personalizadas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        if primary_vars is None:
            primary_vars = [c for c in ["Es_Expuesto", "Sector", "Sexo", "Score_Riesgo"] if c in self.df.columns]

        records = []
        raw_pvalues = []

        for var in primary_vars:
            if var not in self.df.columns:
                continue

            var_lbl = self.get_label(var, labels_map)

            # Caso 1: Variable continua (ej. Score_Riesgo)
            if pd.api.types.is_numeric_dtype(self.df[var]) and len(self.df[var].dropna().unique()) > 4:
                for metal in metals:
                    if metal not in self.df.columns:
                        continue
                    sub_df = self.df[[var, metal]].dropna()
                    if len(sub_df) >= 3:
                        sp = spearman_correlation(sub_df[var], sub_df[metal])
                        metal_lbl = self.get_label(metal, labels_map)
                        raw_p = sp["p_value"]
                        raw_pvalues.append(raw_p)

                        records.append({
                            "Hipótesis / Factor": f"**{var_lbl}**",
                            "Metal": f"**{metal_lbl}**",
                            "Prueba / Contraste": "Correlación de Spearman (rho)",
                            "Resumen Grupo 1": f"N={sp['n']}",
                            "Resumen Grupo 2 / Categorías": "-",
                            "Diferencia / Efecto": f"rho = {sp['rho']:+.3f}",
                            "IC 95% Bootstrap": f"[{sp['ci_low']:+.3f}, {sp['ci_high']:+.3f}]",
                            "p sin ajustar": raw_p,
                        })
                continue

            # Caso 2: Variable categórica
            unique_lvls = [c for c in self.df[var].dropna().unique()]
            if len(unique_lvls) == 2:
                # Dicotómica: Mann-Whitney
                lvl1, lvl2 = unique_lvls[0], unique_lvls[1]
                for metal in metals:
                    if metal not in self.df.columns:
                        continue
                    sub_df = self.df.dropna(subset=[var, metal])
                    g1 = pd.to_numeric(sub_df[sub_df[var] == lvl1][metal], errors="coerce").dropna().values
                    g2 = pd.to_numeric(sub_df[sub_df[var] == lvl2][metal], errors="coerce").dropna().values
                    if len(g1) == 0 or len(g2) == 0:
                        continue

                    mw = mann_whitney_test(g1, g2)
                    metal_lbl = self.get_label(metal, labels_map)
                    diff_ci = mw["diff_medians_ci"]
                    hl_ci = mw["hodges_lehmann_ci"]
                    raw_p = mw["p_value"]
                    raw_pvalues.append(raw_p)

                    records.append({
                        "Hipótesis / Factor": f"**{var_lbl}**",
                        "Metal": f"**{metal_lbl}**",
                        "Prueba / Contraste": f"Mann-Whitney ({lvl1} vs {lvl2})",
                        "Resumen Grupo 1": f"{lvl1} (n={mw['n1']}): {mw['median1']:.2f} [{mw['q25_1']:.2f} - {mw['q75_1']:.2f}]",
                        "Resumen Grupo 2 / Categorías": f"{lvl2} (n={mw['n2']}): {mw['median2']:.2f} [{mw['q25_2']:.2f} - {mw['q75_2']:.2f}]",
                        "Diferencia / Efecto": f"Dif.Med = {mw['diff_medians']:+.2f} (r_rb = {mw['rank_biserial']:+.2f})",
                        "IC 95% Bootstrap": f"[{diff_ci[0]:+.2f}, {diff_ci[1]:+.2f}]",
                        "p sin ajustar": raw_p,
                    })

            elif len(unique_lvls) > 2:
                # Multi-grupo: Kruskal-Wallis
                for metal in metals:
                    if metal not in self.df.columns:
                        continue
                    sub_df = self.df.dropna(subset=[var, metal])
                    groups_data = []
                    group_names = []
                    for cat in sorted(unique_lvls, key=lambda x: str(x)):
                        cat_vals = pd.to_numeric(sub_df[sub_df[var] == cat][metal], errors="coerce").dropna().values
                        if len(cat_vals) > 0:
                            groups_data.append(cat_vals)
                            group_names.append(str(cat))

                    if len(groups_data) >= 2:
                        kw = kruskal_wallis_test(*groups_data, group_names=group_names)
                        metal_lbl = self.get_label(metal, labels_map)
                        raw_p = kw["p_value"]
                        raw_pvalues.append(raw_p)

                        cat_summs = [f"{s['name']}: {s['median']:.2f}" for s in kw["group_summaries"]]
                        records.append({
                            "Hipótesis / Factor": f"**{var_lbl}**",
                            "Metal": f"**{metal_lbl}**",
                            "Prueba / Contraste": f"Kruskal-Wallis ({kw['k_groups']} grupos)",
                            "Resumen Grupo 1": f"N total = {kw['total_n']}",
                            "Resumen Grupo 2 / Categorías": ", ".join(cat_summs),
                            "Diferencia / Efecto": f"H = {kw['h_statistic']:.2f} (eps2 = {kw['epsilon_squared']:.3f})",
                            "IC 95% Bootstrap": f"gl = {kw['df']}",
                            "p sin ajustar": raw_p,
                        })

        # Ajuste de multiplicidad
        adj_pvalues = adjust_pvalues(raw_pvalues, method=method_p_adjust)
        for r, p_adj in zip(records, adj_pvalues):
            p_raw = r["p sin ajustar"]
            r["p sin ajustar"] = f"<strong>{p_raw:.4f}*</strong>" if p_raw < 0.05 else f"{p_raw:.4f}"
            r[f"p ajustado ({method_p_adjust})"] = f"<strong>{p_adj:.4f}*</strong>" if p_adj < 0.05 else f"{p_adj:.4f}"

        res_df = pd.DataFrame(records)

        title = "Resumen Consolidado del Análisis Bivariante Primario"
        subtitle = f"Evaluación de hipótesis a priori fundamentales con corrección de multiplicidad por {method_p_adjust.upper()}"

        notes = [
            "Análisis Primario: Incluye únicamente predictores con hipótesis toxicológicas a priori (Exposición, Sector, Sexo, Score de Riesgo).",
            f"p ajustado ({method_p_adjust}): Corrección de comparaciones múltiples mediante el procedimiento de Holm-Bonferroni (control estricto de FWER).",
            "* Indica significación estadística (p < 0.05)."
        ]

        spanners = [
            {"label": "Definición del Contraste Primario", "columns": ["Hipótesis / Factor", "Metal", "Prueba / Contraste"]},
            {"label": "Valores Observados", "columns": ["Resumen Grupo 1", "Resumen Grupo 2 / Categorías"]},
            {"label": "Estimación del Efecto y CIs", "columns": ["Diferencia / Efecto", "IC 95% Bootstrap"]},
            {"label": "Inferencia Estadística", "columns": ["p sin ajustar", f"p ajustado ({method_p_adjust})"]}
        ]

        alignments = {c: "l" if c in ("Hipótesis / Factor", "Metal", "Resumen Grupo 2 / Categorías") else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments,
            spanners=spanners
        )

    def exploratory_screening(
        self,
        predictor_cols: Optional[List[str]] = None,
        metals: Optional[List[str]] = None,
        method_p_adjust: str = "fdr_bh",
        alpha_discovery: float = 0.10,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Realiza un tamizaje exploratorio sistemático de múltiples factores de riesgo
        (frecuencias dietarias Alim_*, fuentes de agua, transporte, talleres, industrias, síntomas)
        frente a las concentraciones de metales pesados, aplicando control de tasa de falso descubrimiento (FDR de Benjamini-Hochberg).

        Parámetros
        ----------
        predictor_cols : List[str], opcional
            Lista de predictores a evaluar (por defecto todos los factores exploratorios).
        metals : List[str], opcional
            Metales a evaluar (Pb, Hg, Cd).
        method_p_adjust : str
            Método de ajuste de multiplicidad (por defecto 'fdr_bh').
        alpha_discovery : float
            Umbral de descubrimiento de FDR (por defecto q < 0.10 o q < 0.05).
        labels_map : Dict[str, str], opcional
            Mapeo de etiquetas personalizadas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte HTML (.html).

        Retorna
        -------
        BivariateTableReport
        """
        if metals is None:
            metals = [c for c in DEFAULT_PRIMARY_METALS if c in self.df.columns]

        if predictor_cols is None:
            # Detectar todas las variables exploratorias
            detected_cols = []
            for col in self.df.columns:
                if col.startswith("Alim_") or col.startswith("Salud_") or col.startswith("Exposicion_"):
                    # Evitar columnas base que fueron desglosadas
                    if col not in ("Salud_Transporte", "Salud_Agua", "Exposicion_Talleres", "Exposicion_Lugares", "Exposicion_Industrias", "Salud_Suplementos"):
                        detected_cols.append(col)
            predictor_cols = detected_cols

        records = []
        raw_pvalues = []

        for col in predictor_cols:
            if col not in self.df.columns:
                continue

            var_lbl = self.get_label(col, labels_map)
            is_diet = col.startswith("Alim_")

            # Determinar si es binaria o ordinal
            unique_vals = set(self.df[col].dropna().unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "si", "no", "sí", "yes", "SI", "NO"}) and len(unique_vals) <= 2

            for metal in metals:
                if metal not in self.df.columns:
                    continue

                sub_df = self.df[[col, metal]].dropna()
                if len(sub_df) < 3:
                    continue

                metal_lbl = self.get_label(metal, labels_map)

                if is_diet or (not is_binary and len(unique_vals) > 2):
                    # Variable ordinal -> Jonckheere-Terpstra + Spearman
                    if is_diet:
                        ord_num = sub_df[col].apply(lambda v: DIET_ORDINAL_MAP.get(str(v).strip().lower(), v) if pd.notna(v) else v)
                        ord_num = pd.to_numeric(ord_num, errors="coerce")
                    else:
                        ord_num = pd.to_numeric(sub_df[col], errors="coerce")

                    clean_pair = pd.DataFrame({"ord": ord_num, "val": pd.to_numeric(sub_df[metal], errors="coerce")}).dropna()
                    if len(clean_pair) < 3:
                        continue

                    sp = spearman_correlation(clean_pair["ord"], clean_pair["val"])
                    rho = sp["rho"]
                    raw_p = sp["p_value"]
                    raw_pvalues.append(raw_p)

                    records.append({
                        "Factor Exploratorio": f"**{var_lbl}**",
                        "Metal": f"**{metal_lbl}**",
                        "Tipo Variable": "Ordinal (Dieta)",
                        "Prueba Aplicada": "Spearman / Trend",
                        "Tamaño del Efecto": f"rho = {rho:+.3f}",
                        "IC 95% Bootstrap": f"[{sp['ci_low']:+.3f}, {sp['ci_high']:+.3f}]",
                        "p sin ajustar": raw_p,
                    })

                elif is_binary and len(unique_vals) == 2:
                    # Variable binaria -> Mann-Whitney
                    lvls = list(unique_vals)
                    g1 = pd.to_numeric(sub_df[sub_df[col] == lvls[0]][metal], errors="coerce").dropna().values
                    g2 = pd.to_numeric(sub_df[sub_df[col] == lvls[1]][metal], errors="coerce").dropna().values

                    if len(g1) == 0 or len(g2) == 0:
                        continue

                    mw = mann_whitney_test(g1, g2)
                    raw_p = mw["p_value"]
                    raw_pvalues.append(raw_p)

                    r_rb = mw["rank_biserial"]
                    diff_med = mw["diff_medians"]

                    records.append({
                        "Factor Exploratorio": f"**{var_lbl}**",
                        "Metal": f"**{metal_lbl}**",
                        "Tipo Variable": "Dicotómica (SI/NO)",
                        "Prueba Aplicada": "Mann-Whitney U",
                        "Tamaño del Efecto": f"r_rb = {r_rb:+.3f} (Dif.Med={diff_med:+.2f})",
                        "IC 95% Bootstrap": f"[{mw['diff_medians_ci'][0]:+.2f}, {mw['diff_medians_ci'][1]:+.2f}]",
                        "p sin ajustar": raw_p,
                    })

        # Ajuste de multiplicidad FDR
        adj_pvalues = adjust_pvalues(raw_pvalues, method=method_p_adjust)
        for r, p_adj in zip(records, adj_pvalues):
            p_raw = r["p sin ajustar"]
            r["p sin ajustar"] = f"<strong>{p_raw:.4f}*</strong>" if p_raw < 0.05 else f"{p_raw:.4f}"
            is_disc = p_adj < alpha_discovery
            r[f"q-valor ({method_p_adjust})"] = f"<strong>{p_adj:.4f}*</strong>" if is_disc else f"{p_adj:.4f}"
            r["Hallazgo Sugestivo"] = "Posible Asociacion" if is_disc else "No significativo"

        res_df = pd.DataFrame(records)
        # Ordenar por q-valor ascendente si hay datos
        if not res_df.empty and f"q-valor ({method_p_adjust})" in res_df.columns:
            res_df["_sort_q"] = adj_pvalues
            res_df = res_df.sort_values(by="_sort_q").drop(columns=["_sort_q"]).reset_index(drop=True)

        title = "Tamizaje Exploratorio de Factores de Exposición y Hábitos"
        subtitle = f"Evaluación masiva de covariables con ajuste de tasa de falso descubrimiento ({method_p_adjust.upper()} q < {alpha_discovery:.2f})"

        notes = [
            "Análisis Exploratorio: Diseñado para generar hipótesis secundarias. Debe interpretarse con cautela por el riesgo de error tipo I.",
            f"q-valor ({method_p_adjust}): Tasa de Falso Descubrimiento (FDR) de Benjamini-Hochberg.",
            f"Posible Asociacion: Factores que mantienen significación tras corrección por multiplicidad (q < {alpha_discovery:.2f})."
        ]

        spanners = [
            {"label": "Variable y Metal", "columns": ["Factor Exploratorio", "Metal", "Tipo Variable"]},
            {"label": "Magnitud del Efecto", "columns": ["Prueba Aplicada", "Tamaño del Efecto", "IC 95% Bootstrap"]},
            {"label": "Inferencia con Control FDR", "columns": ["p sin ajustar", f"q-valor ({method_p_adjust})", "Hallazgo Sugestivo"]}
        ]

        alignments = {c: "l" if c in ("Factor Exploratorio", "Metal") else "c" for c in res_df.columns}

        return BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=alignments,
            spanners=spanners
        )
