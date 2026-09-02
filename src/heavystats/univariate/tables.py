import os
import re
import warnings
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats

from heavystats.univariate.constants import (
    CDC_BMI_REFERENCE,
    DEFAULT_LABELS_MAP,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)
from heavystats.html_utils import render_html_table, BaseReport


class UnivariateTableReport(BaseReport):
    """
    Reporte de tabla univariante que contiene los datos estructurados en un DataFrame
    y permite renderizado interactivo en HTML de calidad de publicación (estilo Booktabs),
    y exportación a archivos Excel/CSV/HTML.
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
        classified_df: Optional[pd.DataFrame] = None,
        divider_borders: bool = False
    ):
        self._df = df.copy()
        self.title = title
        self.subtitle = subtitle
        self.notes = notes if notes is not None else []
        self.cols = cols if cols is not None else []
        self.column_alignments = column_alignments if column_alignments is not None else {}
        self.spanners = spanners if spanners is not None else []
        self.group_col = group_col
        self._classified_df = classified_df.copy() if classified_df is not None else None
        self.divider_borders = divider_borders

        if filepath:
            self.to_html(filepath)

    @property
    def df(self) -> pd.DataFrame:
        """Devuelve el DataFrame estructurado de la tabla (sin metadatos internos de grupo)."""
        if self.group_col and self.group_col in self._df.columns:
            return self._df.drop(columns=[self.group_col]).copy()
        return self._df.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve una copia limpia del DataFrame de datos estadísticos subyacente."""
        return self.df

    def get_classified_df(self) -> Optional[pd.DataFrame]:
        """Devuelve el DataFrame con los datos enriquecidos por paciente (ej. IMC), si aplica."""
        if self._classified_df is not None:
            return self._classified_df.copy()
        return None

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Exporta los datos estructurados a un archivo CSV."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.df.to_csv(filepath, index=kwargs.get("index", False), **kwargs)

    def to_excel(self, filepath: str, sheet_name: str = "Resumen", **kwargs: Any) -> None:
        """Exporta los datos estructurados a un libro de Excel (.xlsx)."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.df.to_excel(filepath, sheet_name=sheet_name, index=kwargs.get("index", False), **kwargs)

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera una tabla HTML responsiva con calidad de publicación (estilo Booktabs),
        optimizada con tema claro forzado para lectura perfecta en fondos oscuros o claros.
        """
        html_code = render_html_table(
            df=self._df,
            title=self.title,
            subtitle=self.subtitle,
            notes=self.notes,
            column_alignments=self.column_alignments,
            spanners=self.spanners,
            group_col=self.group_col,
            classified_df=self._classified_df,
            full_page=full_page,
            divider_borders=self.divider_borders
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Devuelve la tabla en texto plano estructurado."""
        report_str = self.df.to_string(index=False)
        if filepath:
            self._write_file(filepath, report_str)
        return report_str

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<UnivariateTableReport shape={self.df.shape} title='{self.title}'>"


class UnivariateTables:
    """
    Clase para el cálculo de estadísticas univariantes y generación de tablas científicas.
    Produce reportes de alta legibilidad en Jupyter mediante HTML responsivo y exportación a Excel/CSV/HTML.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        df_total: Optional[pd.DataFrame] = None,
        labels_map: Optional[Dict[str, str]] = None
    ):
        self.df = df.copy()
        self.df_total = df_total.copy() if df_total is not None else self.df.copy()
        self.labels_map = DEFAULT_LABELS_MAP.copy()
        if labels_map is not None:
            self.labels_map.update(labels_map)

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

    @property
    def default_categorical_cols(self) -> List[str]:
        """
        Devuelve la lista detectada dinámicamente de columnas categóricas,
        evitando cortes rígidos por número de columna.
        """
        base_cols = ["Sexo", "Sector", "Institucion", "Es_Expuesto", "Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
        found_base = [c for c in base_cols if c in self.df.columns]

        # Columnas detectadas dinámicamente: tipos object, category, boolean o prefijos estándar
        detected = []
        for col in self.df.columns:
            if col in found_base:
                continue
            # Columnas booleanas desglosadas o categorizadas
            if (
                col.startswith("Exposicion_")
                or col.startswith("Salud_")
                or col.startswith("Alim_")
                or str(self.df[col].dtype) in ("object", "category", "bool")
            ):
                detected.append(col)
            else:
                # Si los valores únicos no nulos son un subconjunto binario {0, 1}
                uvals = set(self.df[col].dropna().unique())
                if len(uvals) > 0 and uvals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "si", "no", "sí"}):
                    detected.append(col)

        # Filtrar las columnas de respuesta múltiple originales para evitar duplicación
        multi_bases = {"Salud_Transporte", "Salud_Agua", "Exposicion_Talleres", "Exposicion_Lugares", "Exposicion_Industrias", "Salud_Suplementos"}
        all_cols = found_base + detected
        # Excluir lugares de exposición y datos de alimentación por completo
        return [c for c in all_cols if c not in multi_bases and not c.startswith("Exposicion_Lugares") and not c.startswith("Alim_")]

    def categorical_summary(
        self, 
        columns: Optional[List[str]] = None, 
        groups: Optional[Dict[str, List[str]]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> UnivariateTableReport:
        """
        Calcula frecuencias y porcentajes para variables categóricas.
        Genera una tabla agrupada temáticamente y formateada profesionalmente,
        comparando la muestra seleccionada n (%) con la población total N (%).

        Parámetros
        ----------
        columns : List[str], opcional
            Lista de columnas categóricas. Si es None, utiliza default_categorical_cols.
        groups : Dict[str, List[str]], opcional
            Diccionario de agrupación conceptual para organizar las variables.
        labels_map : Dict[str, str], opcional
            Diccionario personalizado para etiquetas de variables.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        UnivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        if columns is None:
            columns = self.default_categorical_cols

        # Grupos conceptuales definidos por defecto
        if groups is None:
            groups = {
                "Variables Sociodemográficas": [
                    "Sexo", "Sector", "Institucion"
                ],
                "Variables de Exposición y Factores de Riesgo": [
                    "Es_Expuesto", "Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd",
                    "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
                    "Exposicion_Industrias_fabrica_metales", "Exposicion_Industrias_fabrica_productos_quimicos"
                ],
                "Hábitos y Fuentes de Agua": [
                    "Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo",
                    "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada", "Salud_Agua_pozo_profundo",
                    "Salud_Fuma", "Salud_Actividad", "Salud_Bombillos", "Salud_Techo", "Salud_Joyeria",
                    "Salud_Suplementos_multivitaminico", "Salud_Suplementos_proteico"
                ]
            }

        grouped_cols: Dict[str, List[str]] = {}
        ungrouped_cols = list(columns)

        for g_name, g_cols in groups.items():
            matched = [c for c in columns if c in g_cols]
            if matched:
                grouped_cols[g_name] = matched
                for c in matched:
                    if c in ungrouped_cols:
                        ungrouped_cols.remove(c)

        if ungrouped_cols:
            grouped_cols["Otras Variables"] = ungrouped_cols

        def format_val_str(val: Any) -> str:
            val_str = str(val).strip()
            if val_str.lower() in ("si", "sí", "yes", "true", "1", "1.0"):
                return "Sí"
            if val_str.lower() in ("no", "false", "0", "0.0"):
                return "No"
            return val_str[0].upper() + val_str[1:] if len(val_str) > 0 else val_str

        def get_binary_positive_count(val_counts: pd.Series) -> int:
            pos_count = 0
            for val, count in val_counts.items():
                try:
                    if float(val) == 1.0:
                        pos_count += count
                except (ValueError, TypeError):
                    if str(val).strip().lower() in ("true", "1", "1.0", "si", "sí", "yes"):
                        pos_count += count
            return pos_count

        records = []
        for group_name, cols in grouped_cols.items():
            # Encabezado visual de grupo
            records.append({
                "Característica / Variable": f"**{group_name}**",
                "Categoría": "",
                "n (%)": "",
                "N (%)": "",
                "_IS_GROUP": "__GROUP_HEADER__"
            })

            for col in cols:
                if col not in self.df.columns:
                    continue

                val_counts_n = self.df[col].value_counts(dropna=True)
                total_n = val_counts_n.sum()

                if col in self.df_total.columns:
                    val_counts_N = self.df_total[col].value_counts(dropna=True)
                    total_N = val_counts_N.sum()
                else:
                    val_counts_N = val_counts_n
                    total_N = total_n

                unique_vals = set(self.df_total[col].dropna().unique()) if col in self.df_total.columns else set(self.df[col].dropna().unique())
                is_binary = (
                    unique_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "0.0", "1.0", "si", "no", "sí", "yes"})
                    and len(unique_vals) > 0
                )

                var_label = self.get_label(col, labels_map)

                if is_binary:
                    pos_n = get_binary_positive_count(val_counts_n)
                    pos_N = get_binary_positive_count(val_counts_N)

                    pct_n = (pos_n / total_n) * 100 if total_n > 0 else 0.0
                    pct_N = (pos_N / total_N) * 100 if total_N > 0 else 0.0

                    records.append({
                        "Característica / Variable": f"**{var_label}**",
                        "Categoría": "Sí",
                        "n (%)": f"{pos_n} ({pct_n:.1f}%)",
                        "N (%)": f"{pos_N} ({pct_N:.1f}%)",
                        "_IS_GROUP": ""
                    })
                else:
                    categories = list(unique_vals)
                    categories.sort(key=lambda x: (-val_counts_N.get(x, 0), str(x)))

                    first = True
                    for cat in categories:
                        row_var_label = f"**{var_label}**" if first else ""
                        cat_label = format_val_str(cat)

                        count_n = val_counts_n.get(cat, 0)
                        count_N = val_counts_N.get(cat, 0)

                        pct_n = (count_n / total_n) * 100 if total_n > 0 else 0.0
                        pct_N = (count_N / total_N) * 100 if total_N > 0 else 0.0

                        records.append({
                            "Característica / Variable": row_var_label,
                            "Categoría": cat_label,
                            "n (%)": f"{count_n} ({pct_n:.1f}%)",
                            "N (%)": f"{count_N} ({pct_N:.1f}%)",
                            "_IS_GROUP": ""
                        })
                        first = False

        res_df = pd.DataFrame(records)

        spanners = [
            {"label": "Variable y Categoría", "columns": ["Característica / Variable", "Categoría"]},
            {"label": f"Muestra (n={len(self.df)})", "columns": ["n (%)"]},
            {"label": f"Población Total (N={len(self.df_total)})", "columns": ["N (%)"]}
        ]

        report = UnivariateTableReport(
            df=res_df,
            title="Resumen de Variables Categóricas",
            subtitle=f"Comparación entre la muestra de estudio (n={len(self.df)}) y la población total (N={len(self.df_total)})",
            notes=["Valores expresados como frecuencia absoluta y porcentaje: n (%)."],
            filepath=filepath,
            cols=columns,
            column_alignments={
                "Característica / Variable": "l",
                "Categoría": "l",
                "n (%)": "c",
                "N (%)": "c"
            },
            spanners=spanners,
            group_col="_IS_GROUP",
            divider_borders=True
        )
        return report

    def numerical_summary(
        self, 
        columns: Optional[List[str]] = None, 
        skewed_columns: Optional[List[str]] = None,
        iqr_format: str = "range",
        normality_criterion: str = "shapiro",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> UnivariateTableReport:
        """
        Calcula estadísticas descriptivas para variables numéricas según estándares ICMJE/STROBE.
        Reporta Media (DE) para variables aproximadamente simétricas y Mediana [Q1 - Q3] para variables sesgadas.
        También calcula valores mín-máx, asimetría, curtosis y la prueba de Shapiro-Wilk.

        Parámetros
        ----------
        columns : List[str], opcional
            Lista de columnas numéricas (por defecto: Edad, Peso_kg, Altura_cm, Score_Riesgo).
        skewed_columns : List[str], opcional
            Lista explícita de columnas a tratar como sesgadas.
        iqr_format : str, opcional (por defecto "range")
            Formato del rango intercuartílico: "range" -> [Q1 - Q3] o "width" -> [RIQ].
        normality_criterion : str, opcional (por defecto "shapiro")
            Criterio para clasificar la distribución:
            - "shapiro": Shapiro-Wilk p < 0.05 indica distribución sesgada.
            - "skewness": |Asimetría| > 0.5 indica distribución sesgada.
            - "both": Si p < 0.05 o |Asimetría| > 0.5.
            - "manual": Solo las especificadas en skewed_columns.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles para los encabezados.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        UnivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        if columns is None:
            columns = ["Edad", "Peso_kg", "Altura_cm", "Score_Riesgo"]

        if skewed_columns is None:
            skewed_columns = []

        records = []
        for col in columns:
            if col not in self.df.columns:
                continue

            data_raw = self.df[col]
            n_total = len(data_raw)
            data_clean = pd.to_numeric(data_raw, errors="coerce").dropna()
            n_valid = len(data_clean)
            if n_valid == 0:
                continue

            n_miss = n_total - n_valid
            n_str = f"{n_valid}" if n_miss == 0 else f"{n_valid} ({n_miss} perd.)"

            mean = float(data_clean.mean())
            std = float(data_clean.std())
            median = float(data_clean.median())
            q25 = float(data_clean.quantile(0.25))
            q75 = float(data_clean.quantile(0.75))
            iqr = q75 - q25
            val_min = float(data_clean.min())
            val_max = float(data_clean.max())

            skew = float(stats.skew(data_clean))
            kurt = float(stats.kurtosis(data_clean))

            if n_valid >= 3:
                shapiro_w, shapiro_p = stats.shapiro(data_clean)
                shapiro_str = f"{shapiro_p:.4f}"
                p_val = float(shapiro_p)
            else:
                shapiro_str = "N/A"
                p_val = 1.0

            # Evaluación de normalidad
            if normality_criterion == "shapiro":
                is_skewed = (col in skewed_columns) or (p_val < 0.05)
            elif normality_criterion == "skewness":
                is_skewed = (col in skewed_columns) or (abs(skew) > 0.5)
            elif normality_criterion == "both":
                is_skewed = (col in skewed_columns) or (p_val < 0.05) or (abs(skew) > 0.5)
            else:  # manual
                is_skewed = col in skewed_columns

            dist_type = "Sesgada" if is_skewed else "Simétrica"

            # Formato de medidas
            mean_str = f"{mean:.2f} ({std:.2f})"
            if iqr_format == "range":
                median_str = f"{median:.2f} [{q25:.2f} - {q75:.2f}]"
            else:
                median_str = f"{median:.2f} [{iqr:.2f}]"

            # Resaltar la medida de tendencia central recomendada en negrita
            if is_skewed:
                median_display = f"**{median_str}**"
                mean_display = mean_str
            else:
                mean_display = f"**{mean_str}**"
                median_display = median_str

            var_label = self.get_label(col, labels_map)

            records.append({
                "Variable": f"**{var_label}**",
                "Distribución": dist_type,
                "N": n_str,
                "Media (DE)": mean_display,
                "Mediana [RIQ]": median_display,
                "Mín - Máx": f"{val_min:.2f} - {val_max:.2f}",
                "Asimetría": f"{skew:+.3f}",
                "Curtosis": f"{kurt:+.3f}",
                "Shapiro-Wilk (p-val)": shapiro_str
            })

        res_df = pd.DataFrame(records)

        notes = [
            "DE: Desviación Estándar; RIQ: Rango Intercuartílico [Q1 - Q3]",
            "En negrita se resalta la medida de tendencia central recomendada según la simetría y la prueba de Shapiro-Wilk (p < 0.05 indica distribución sesgada/no paramétrica)."
        ]

        report = UnivariateTableReport(
            df=res_df,
            title="Resumen de Variables Numéricas",
            subtitle="Parámetros de tendencia central, dispersión y pruebas de normalidad",
            notes=notes,
            filepath=filepath,
            cols=columns,
            column_alignments={
                "Variable": "l",
                "Distribución": "c",
                "N": "c",
                "Media (DE)": "c",
                "Mediana [RIQ]": "c",
                "Mín - Máx": "c",
                "Asimetría": "c",
                "Curtosis": "c",
                "Shapiro-Wilk (p-val)": "c"
            }
        )
        return report

    def metal_summary(
        self, 
        columns: Optional[List[str]] = None, 
        lods: Optional[Dict[str, float]] = None,
        permissible_limits: Optional[Dict[str, float]] = None,
        percentiles: Optional[List[int]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> UnivariateTableReport:
        """
        Calcula estadísticas descriptivas y toxicológicas para concentraciones de metales pesados en sangre.
        Incluye porcentaje detectable (>=LOD), porcentaje que supera el límite permisible de referencia,
        media geométrica con su desviación estándar geométrica (GSD), y percentiles poblacionales.

        Parámetros
        ----------
        columns : List[str], opcional
            Metales a analizar (por defecto: Plomo_ug_dL, Mercurio_ug_L, Cadmio_ug_L).
        lods : Dict[str, float], opcional
            Límites de detección de los instrumentos analíticos.
        permissible_limits : Dict[str, float], opcional
            Límites permisibles de referencia toxicológica (CDC/OMS/EPA).
        percentiles : List[int], opcional
            Percentiles a reportar (por defecto: 5, 10, 25, 50, 75, 90, 95).
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles para los ejes y tablas.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        UnivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        if columns is None:
            columns = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L"]

        if lods is None:
            lods = {
                "Plomo_ug_dL": 0.1,
                "Mercurio_ug_L": 0.1,
                "Cadmio_ug_L": 0.05,
                "Plomo": 0.1,
                "Mercurio": 0.1,
                "Cadmio": 0.05
            }

        active_limits = DEFAULT_PERMISSIBLE_LIMITS.copy()
        if permissible_limits is not None:
            active_limits.update(permissible_limits)

        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]

        records = []
        for col in columns:
            if col not in self.df.columns:
                continue

            data_clean = pd.to_numeric(self.df[col], errors="coerce").dropna()
            n = len(data_clean)
            if n == 0:
                continue

            lod = lods.get(col, 0.0)
            detectable_count = int((data_clean >= lod).sum())
            pct_detectable = (detectable_count / n) * 100 if n > 0 else 0.0

            # Porcentaje que supera el límite permisible
            limit_val = active_limits.get(col)
            if limit_val is not None:
                exceed_count = int((data_clean >= limit_val).sum())
                pct_exceed = (exceed_count / n) * 100 if n > 0 else 0.0
                exceed_str = f"{pct_exceed:.1f}% ({limit_val:g})"
            else:
                exceed_str = "N/D"

            val_min = float(data_clean.min())
            val_max = float(data_clean.max())

            # Media geométrica y desviación estándar geométrica (GSD)
            if (data_clean > 0).all():
                log_vals = np.log(data_clean)
                geo_mean = float(np.exp(log_vals.mean()))
                gsd = float(np.exp(log_vals.std()))
                geo_mean_str = f"{geo_mean:.2f} ({gsd:.2f})"
            else:
                geo_mean_str = "No calc."

            perc_vals = np.percentile(data_clean, percentiles)

            var_label = self.get_label(col, labels_map)

            row_dict = {
                "Metal": f"**{var_label}**",
                "N": f"{n}",
                "% Detectable": f"{pct_detectable:.1f}%",
                "% > Límite": exceed_str,
                "Mín - Máx": f"{val_min:.2f} - {val_max:.2f}",
                "Media Geom. (GSD)": geo_mean_str
            }

            for p, val in zip(percentiles, perc_vals):
                row_dict[f"p{p}"] = f"{val:.2f}"

            records.append(row_dict)

        res_df = pd.DataFrame(records)

        # Spanners de columnas para LaTeX
        perc_cols = [f"p{p}" for p in percentiles]
        spanners = [
            {"label": "Identificación y Muestra", "columns": ["Metal", "N"]},
            {"label": "Criterios Analíticos y Sanitarios", "columns": ["% Detectable", "% > Límite"]},
            {"label": "Concentración Global", "columns": ["Mín - Máx", "Media Geom. (GSD)"]},
            {"label": "Percentiles de Concentración", "columns": perc_cols}
        ]

        notes = [
            "LOD: Límite de Detección analítico; GSD: Desviación Estándar Geométrica.",
            "% > Límite: Porcentaje de la muestra que iguala o supera el límite permisible de referencia en sangre (CDC/OMS/EPA)."
        ]

        report = UnivariateTableReport(
            df=res_df,
            title="Análisis Descriptivo de Metales Pesados en Sangre",
            subtitle="Métricas analíticas, valores de referencia internacional y distribución por percentiles",
            notes=notes,
            filepath=filepath,
            cols=columns,
            column_alignments={c: "l" if c == "Metal" else "c" for c in res_df.columns},
            spanners=spanners
        )
        return report

    def log_transform_evaluation(
        self, 
        columns: Optional[List[str]] = None, 
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> UnivariateTableReport:
        """
        Evalúa la transformación logarítmica (ln) de cada metal comparando
        sus estadísticos de asimetría, curtosis y prueba de normalidad de Shapiro-Wilk,
        aportando una recomendación metodológica clara.

        Parámetros
        ----------
        columns : List[str], opcional
            Metales a evaluar.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        UnivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        if columns is None:
            columns = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L"]

        records = []
        for col in columns:
            if col not in self.df.columns:
                continue

            data_orig = pd.to_numeric(self.df[col], errors="coerce").dropna()
            n = len(data_orig)
            if n < 3:
                continue

            var_label = self.get_label(col, labels_map)

            # 1. Variable Original
            skew_orig = float(stats.skew(data_orig))
            kurt_orig = float(stats.kurtosis(data_orig))
            w_orig, p_orig = stats.shapiro(data_orig)
            norm_orig = "Sí" if p_orig >= 0.05 else "No"

            # 2. Variable Logarítmica (ln)
            positive_data = data_orig[data_orig > 0]
            if len(positive_data) >= 3:
                data_log = np.log(positive_data)
                skew_log = float(stats.skew(data_log))
                kurt_log = float(stats.kurtosis(data_log))
                w_log, p_log = stats.shapiro(data_log)
                norm_log = "Sí" if p_log >= 0.05 else "No"

                # Recomendación
                if p_log > p_orig and abs(skew_log) < abs(skew_orig):
                    verdict = "Recomendada"
                else:
                    verdict = "Original preferible"
            else:
                skew_log = np.nan
                kurt_log = np.nan
                w_log, p_log = np.nan, np.nan
                norm_log = "N/A"
                verdict = "No aplicable"

            records.append({
                "Metal": f"**{var_label}**",
                "Transformación": "Original",
                "Asimetría": f"{skew_orig:+.3f}",
                "Curtosis": f"{kurt_orig:+.3f}",
                "Shapiro-Wilk (W)": f"{w_orig:.4f}",
                "Shapiro-Wilk (p-val)": f"{p_orig:.4f}",
                "¿Normal?": norm_orig,
                "Conclusión": "Base de comparación"
            })

            log_w_str = f"{w_log:.4f}" if pd.notna(w_log) else "N/A"
            log_p_str = f"{p_log:.4f}" if pd.notna(p_log) else "N/A"
            log_sk_str = f"{skew_log:+.3f}" if pd.notna(skew_log) else "N/A"
            log_ku_str = f"{kurt_log:+.3f}" if pd.notna(kurt_log) else "N/A"

            records.append({
                "Metal": "",
                "Transformación": "Logarítmico (ln)",
                "Asimetría": log_sk_str,
                "Curtosis": log_ku_str,
                "Shapiro-Wilk (W)": log_w_str,
                "Shapiro-Wilk (p-val)": log_p_str,
                "¿Normal?": norm_log,
                "Conclusión": verdict
            })

        res_df = pd.DataFrame(records)

        notes = [
            "Prueba de normalidad de Shapiro-Wilk: p >= 0.05 indica concordancia con la distribución normal.",
            "La transformación logarítmica (ln) es estándar en toxicología para estabilizar varianzas en biomarcadores asimétricos."
        ]

        report = UnivariateTableReport(
            df=res_df,
            title="Evaluación de la Transformación Logarítmica",
            subtitle="Comparación de simetría y normalidad entre escala natural y logarítmica",
            notes=notes,
            filepath=filepath,
            cols=columns,
            column_alignments={
                "Metal": "l",
                "Transformación": "l",
                "Asimetría": "c",
                "Curtosis": "c",
                "Shapiro-Wilk (W)": "c",
                "Shapiro-Wilk (p-val)": "c",
                "¿Normal?": "c",
                "Conclusión": "c"
            }
        )
        return report

    def bmi_summary(
        self,
        weight_col: str = "Peso_kg",
        height_col: str = "Altura_cm",
        age_col: str = "Edad",
        sex_col: str = "Sexo",
        id_col: str = "Muestra_Codificada",
        include_patient_details: bool = True,
        max_patient_rows: Optional[int] = 20,
        hide_zero_categories: bool = False,
        filepath: Optional[str] = None
    ) -> UnivariateTableReport:
        """
        Calcula el IMC pediátrico e interpreta la clasificación nutricional
        según las referencias oficiales de la CDC (2000/2022 Extendida) para edades de 5 a 10 años.
        
        Permite recuperar el DataFrame enriquecido a nivel de paciente mediante report.get_classified_df().

        Parámetros
        ----------
        weight_col : str, opcional (por defecto "Peso_kg")
            Nombre de la columna con el peso en kilogramos.
        height_col : str, opcional (por defecto "Altura_cm")
            Nombre de la columna con la talla en centímetros.
        age_col : str, opcional (por defecto "Edad")
            Nombre de la columna con la edad en años.
        sex_col : str, opcional (por defecto "Sexo")
            Nombre de la columna con el sexo del paciente.
        id_col : str, opcional (por defecto "Muestra_Codificada")
            Nombre de la columna identificadora de paciente.
        include_patient_details : bool, opcional (por defecto True)
            Si True, incluye la tabla secundaria de desglose por paciente.
        max_patient_rows : int, opcional (por defecto 20)
            Número máximo de filas a renderizar en la tabla de detalle por paciente.
        hide_zero_categories : bool, opcional (por defecto False)
            Si True, oculta categorías de la CDC con 0 pacientes.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        UnivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y acceso a los datos completos.
        """
        df_bmi = self.df.dropna(subset=[weight_col, height_col]).copy()

        if len(df_bmi) == 0:
            empty_df = pd.DataFrame(columns=["Clasificación", "N", "Porcentaje (%)"])
            return UnivariateTableReport(
                df=empty_df,
                title="Reporte de Índice de Masa Corporal (IMC) Pediátrico",
                subtitle="Sin datos suficientes de Peso y Altura.",
                filepath=filepath
            )

        df_bmi["Calculated_IMC"] = df_bmi[weight_col] / ((df_bmi[height_col] / 100) ** 2)

        classifications = []
        for _, row in df_bmi.iterrows():
            age = row.get(age_col)
            sex = row.get(sex_col)
            imc = row["Calculated_IMC"]

            if pd.isna(age) or pd.isna(sex) or pd.isna(imc):
                classifications.append("Sin datos completos")
                continue

            sex_str = str(sex).strip().lower()
            if "fem" in sex_str or sex_str in ("f", "femenino", "mujer", "niña"):
                gender = "femenino"
            elif "masc" in sex_str or sex_str in ("m", "masculino", "hombre", "niño"):
                gender = "masculino"
            else:
                classifications.append("Sexo no reconocido")
                continue

            try:
                age_int = int(round(float(age)))
            except (ValueError, TypeError):
                classifications.append("Edad inválida")
                continue

            ref_gender = CDC_BMI_REFERENCE.get(gender, {})
            ref_age = ref_gender.get(age_int)

            if not ref_age:
                classifications.append("Edad fuera de referencia (5-10 años)")
                continue

            p5 = ref_age["P5"]
            p85 = ref_age["P85"]
            p95 = ref_age["P95"]

            if imc < p5:
                classifications.append("Bajo peso")
            elif imc < p85:
                classifications.append("Normopeso")
            elif imc < p95:
                classifications.append("Sobrepeso")
            else:
                if imc >= 1.2 * p95 or imc >= 35.0:
                    classifications.append("Obesidad Severa")
                else:
                    classifications.append("Obesidad")

        df_bmi["Clasificacion_IMC"] = classifications

        counts = df_bmi["Clasificacion_IMC"].value_counts()
        cdc_categories = [
            "Bajo peso", "Normopeso", "Sobrepeso", "Obesidad", "Obesidad Severa"
        ]
        total_valid = sum(counts.get(cat, 0) for cat in cdc_categories)

        summary_records = []
        for cat in cdc_categories:
            count = counts.get(cat, 0)
            if hide_zero_categories and count == 0:
                continue
            pct = (count / total_valid) * 100 if total_valid > 0 else 0.0
            summary_records.append({
                "Clasificación Nutricional": cat,
                "N": count,
                "Porcentaje (%)": f"{pct:.1f}%"
            })

        summary_df = pd.DataFrame(summary_records)

        notes = [
            f"Pacientes con antropometría completa evaluados: N={total_valid}.",
            "Puntos de corte CDC: Bajo peso (<P5), Normopeso (P5 a <P85), Sobrepeso (P85 a <P95), Obesidad (P95 a <120% P95), Obesidad Severa (>=120% P95 o IMC >=35)."
        ]

        report = UnivariateTableReport(
            df=summary_df,
            title="Clasificación del Índice de Masa Corporal (IMC) Pediátrico",
            subtitle="Criterios y puntos de corte oficiales de la CDC para niños de 5 a 10 años",
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Clasificación Nutricional": "l",
                "N": "c",
                "Porcentaje (%)": "c"
            },
            classified_df=df_bmi
        )
        return report
