"""
Generación de tablas científicas y reportes HTML (calidad Booktabs) para análisis bivariantes.
Implementa el protocolo de análisis bivariante centrado en el principio de metal individual
(con soporte opcional para secuencias de metales) y co-exposición inter-metales:
- Metal vs Variables Cuantitativas (Spearman rho, IC Bootstrap 95%, p, n).
- Metal vs Variables Binarias / Factores de Exposición / Hábitos (Mann-Whitney U, r_rb, Hodges-Lehmann, medianas).
- Metal vs Variables Categóricas Politómicas (Kruskal-Wallis H, epsilon^2, prueba post-hoc de Dunn con ajuste).
- Metal vs Hábitos Dietarios (Variables ordinales 0-4 con Spearman).
- Metal vs Algoritmo de Riesgo (Validación y calibración del Score de Riesgo e indicador específico).
- Metal vs Metal (Pares de co-exposición Pb, Hg, Cd).
- Clasificación de Niveles de Metal (Bajo, Medio, Alto) y relación con Score y Factores.
- Matriz Maestra de Asociaciones con control de FDR (Benjamini-Hochberg).
- Ranking Multicriterio de Variables (Prioridad Alta, Intermedia, Baja).
- Diagnóstico de Colinealidad y Redundancia.
- Preparación formal para PCA y PLS con recomendaciones de parsimonia (n=20).
"""

import os
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import pandas as pd
import numpy as np
from scipy import stats

from heavystats.html_utils import BaseReport, render_html_table, format_html_str
from heavystats.bivariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_PRIMARY_METALS,
    DEFAULT_METAL_LIMITS,
    DEFAULT_METAL_LODS,
    DEFAULT_METAL_CUTOFFS,
    DEFAULT_METAL_PAIRS,
    DEFAULT_BIVARIATE_GROUPS,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
    get_label,
)
from heavystats.bivariate.tests import (
    mann_whitney_test,
    independent_t_test,
    kruskal_wallis_test,
    anova_oneway_test,
    dunn_posthoc_test,
    jonckheere_terpstra_test,
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    spearman_matrix,
    adjust_pvalues,
    rank_bivariate_associations,
    collinearity_matrix,
    fisher_chi2_test,
    qualitative_association_test,
)


class BivariateTableReport(BaseReport):
    """
    Reporte para tablas bivariantes con soporte para renderizado HTML interactivo (Booktabs),
    exportación multi-hoja a Excel (.xlsx), CSV, DataFrame y texto formateado.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "",
        subtitle: Optional[str] = None,
        notes: Optional[List[str]] = None,
        column_alignments: Optional[Dict[str, str]] = None,
        spanners: Optional[List[Dict[str, Any]]] = None,
        group_col: Optional[str] = None,
        divider_borders: bool = False,
        posthoc_df: Optional[pd.DataFrame] = None,
        extra_sheets: Optional[Dict[str, pd.DataFrame]] = None,
        filepath: Optional[str] = None,
    ):
        self._df = df.copy()
        self.title = title
        self.subtitle = subtitle
        self.notes = notes if notes is not None else []
        self.column_alignments = column_alignments if column_alignments is not None else {}
        self.spanners = spanners if spanners is not None else []
        self.group_col = group_col
        self.divider_borders = divider_borders
        self._posthoc_df = posthoc_df.copy() if posthoc_df is not None else None
        self._extra_sheets = {k: v.copy() for k, v in extra_sheets.items()} if extra_sheets else {}

        if self._posthoc_df is not None and not self._posthoc_df.empty:
            self._extra_sheets["PostHoc_Dunn"] = self._posthoc_df

        if filepath:
            self.to_html(filepath)

    @property
    def df(self) -> pd.DataFrame:
        """Devuelve el DataFrame estructurado de la tabla (sin columnas internas de grupo o metadatos)."""
        drop_cols = [c for c in self._df.columns if c == self.group_col or str(c).startswith("_")]
        if drop_cols:
            return self._df.drop(columns=drop_cols).copy()
        return self._df.copy()

    @property
    def posthoc_df(self) -> Optional[pd.DataFrame]:
        """Devuelve el DataFrame de comparaciones post-hoc si aplica."""
        if self._posthoc_df is not None:
            return self._posthoc_df.copy()
        return None

    @property
    def extra_sheets(self) -> Dict[str, pd.DataFrame]:
        return self._extra_sheets

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve una copia limpia del DataFrame de resultados."""
        return self.df

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Exporta los resultados a un archivo CSV."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        kwargs.setdefault("encoding", "utf-8")
        self.df.to_csv(filepath, index=kwargs.get("index", False), **kwargs)

    def to_excel(self, filepath: str, sheet_name: str = "Bivariante", **kwargs: Any) -> None:
        """Exporta los resultados a un libro de Excel (.xlsx) con hojas múltiples si aplica."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                self.df.to_excel(writer, sheet_name=sheet_name[:31], index=kwargs.get("index", False), **kwargs)
                for s_name, s_df in self._extra_sheets.items():
                    if s_df is not None and not s_df.empty:
                        s_df.to_excel(writer, sheet_name=s_name[:31], index=False)
        except Exception:
            with pd.ExcelWriter(filepath) as writer:
                self.df.to_excel(writer, sheet_name=sheet_name[:31], index=kwargs.get("index", False), **kwargs)

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera el código HTML con calidad de publicación (estilo Booktabs),
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
        return f"<BivariateTableReport shape={self.df.shape} title='{self.title}'>"


class BivariateTables:
    """
    Clase para el cálculo de estadísticas bivariantes y generación de tablas científicas.
    Opera bajo el principio de análisis por metal individual sobre el tamaño muestral
    efectivo disponible (n=20), con soporte opcional para secuencias de metales.
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

    # -------------------------------------------------------------------------
    # Métodos Auxiliares Privados
    # -------------------------------------------------------------------------

    def _resolve_metal(self, metal: str) -> str:
        """Resuelve el nombre exacto de la columna del metal a partir de nombres o alias comunes."""
        if metal in self.df.columns:
            return metal
        clean_m = str(metal).strip()
        if clean_m in self.df.columns:
            return clean_m

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
        }
        lower_m = clean_m.lower()
        if lower_m in alias_map and alias_map[lower_m] in self.df.columns:
            return alias_map[lower_m]

        for c in self.df.columns:
            if c.lower() == lower_m:
                return c

        return metal

    def _resolve_column(self, col: str) -> str:
        """Resuelve el nombre exacto de una columna en el DataFrame."""
        if col in self.df.columns:
            return col
        clean_c = str(col).strip()
        if clean_c in self.df.columns:
            return clean_c

        alias_map = {
            "plomo": "Plomo_ug_dL",
            "pb": "Plomo_ug_dL",
            "mercurio": "Mercurio_ug_L",
            "hg": "Mercurio_ug_L",
            "cadmio": "Cadmio_ug_L",
            "cd": "Cadmio_ug_L",
            "edad": "Edad",
            "age": "Edad",
            "peso": "Peso_kg",
            "peso_kg": "Peso_kg",
            "altura": "Altura_cm",
            "talla": "Altura_cm",
            "altura_cm": "Altura_cm",
            "score": "Score_Riesgo",
            "score_riesgo": "Score_Riesgo",
            "riesgo": "Score_Riesgo",
            "sexo": "Sexo",
            "sector": "Sector",
            "institucion": "Institucion",
            "es_expuesto": "Es_Expuesto",
        }
        lower_c = clean_c.lower()
        if lower_c in alias_map and alias_map[lower_c] in self.df.columns:
            return alias_map[lower_c]

        for c in self.df.columns:
            if c.lower() == lower_c:
                return c

        return col

    def _normalize_metals(self, metals: Union[str, Sequence[str]]) -> List[str]:
        """Normaliza el argumento de metales a una lista de columnas válidas."""
        if isinstance(metals, str):
            m_list = [metals]
        else:
            m_list = list(metals)
        return [self._resolve_metal(m) for m in m_list]

    def _normalize_columns(self, columns: Union[str, Sequence[str]]) -> List[str]:
        """Normaliza el argumento de columnas a una lista de nombres válidos."""
        if isinstance(columns, str):
            c_list = [columns]
        else:
            c_list = list(columns)
        return [self._resolve_column(c) for c in c_list]

    def _get_metal_sample(self, metal_col: str) -> pd.DataFrame:
        """Extrae el subconjunto de participantes con valor válido y positivo para el metal."""
        if metal_col not in self.df.columns:
            return pd.DataFrame()
        s = pd.to_numeric(self.df[metal_col], errors="coerce")
        return self.df[s.notna() & (s > 0)].copy()

    def get_label(self, col: str, labels_map: Optional[Dict[str, str]] = None) -> str:
        """Obtiene la etiqueta legible para una columna."""
        active_map = self.labels_map.copy()
        if labels_map:
            active_map.update(labels_map)
        return get_label(col, active_map)

    # =========================================================================
    # Etapa 4: Metal vs Variables Cuantitativas / Antropométricas
    # =========================================================================

    def continuous_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        columns: Optional[Union[str, Sequence[str]]] = None,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Calcula la correlación no paramétrica de Spearman (rho_s) entre la concentración
        del metal seleccionado y las variables cuantitativas (Edad, Peso, Altura, IMC, Score de Riesgo).
        Incluye intervalo de confianza al 95% obtenido por remuestreo Bootstrap (n_boot réplicas).

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL'). Admite alias ('Plomo', 'pb', etc.) o lista.
        columns : Union[str, Sequence[str]], opcional
            Variables cuantitativas a contrastar (por defecto: Edad, Peso_kg, Altura_cm, IMC, Score_Riesgo).
        n_boot : int, opcional (por defecto 2000)
            Número de réplicas Bootstrap para el intervalo de confianza al 95%.
        labels_map : Dict[str, str], opcional
            Mapeo personalizado de nombres de variables.
        filepath : str, opcional
            Ruta de archivo para guardar el reporte en HTML (.html).

        Retorna
        -------
        BivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        metal_cols = self._normalize_metals(metal)
        if columns is None:
            raw_cols = ["Edad", "Peso_kg", "Altura_cm", "IMC", "Score_Riesgo"]
            var_cols = [self._resolve_column(c) for c in raw_cols if self._resolve_column(c) in self.df.columns]
        else:
            var_cols = [self._resolve_column(c) for c in self._normalize_columns(columns) if self._resolve_column(c) in self.df.columns]

        records = []
        is_multi_metal = len(metal_cols) > 1

        for m_col in metal_cols:
            df_m = self._get_metal_sample(m_col)
            n_metal = len(df_m)
            if n_metal == 0:
                continue

            metal_label = self.get_label(m_col, labels_map)

            if is_multi_metal:
                records.append({
                    "Variable": f"**{metal_label}**",
                    "n Válido": "",
                    "Spearman (ρₛ)": "",
                    "IC 95% Bootstrap": "",
                    "Valor p": "",
                    "Evidencia": "",
                    "_IS_GROUP": "__GROUP_HEADER__"
                })

            for var_col in var_cols:
                if var_col == m_col or var_col not in df_m.columns:
                    continue

                res = spearman_correlation(df_m[var_col], df_m[m_col], n_boot=n_boot)
                rho = res["rho"]
                p_val = res["p_val"]
                ci_l = res["ci_low"]
                ci_h = res["ci_high"]
                n_val = res["n_valid"]

                if np.isnan(rho):
                    rho_str = "N/D"
                    ci_str = "N/D"
                    p_str = "N/D"
                    evid_str = "Datos insuficientes"
                else:
                    rho_str = f"{rho:+.3f}"
                    if not np.isnan(ci_l) and not np.isnan(ci_h):
                        ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]"
                    else:
                        ci_str = "N/D"
                    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

                    if p_val < 0.05:
                        evid_str = "Asociación estadísticamente significativa"
                        rho_str = f"**{rho_str}**"
                        p_str = f"**{p_str}**"
                    elif p_val < 0.10:
                        evid_str = "Tendencia marginal / exploratoria"
                    else:
                        evid_str = "Sin evidencia suficiente"

                var_label = self.get_label(var_col, labels_map)
                rec = {
                    "Variable": f"**{var_label}**" if not is_multi_metal else var_label,
                    "n Válido": str(n_val),
                    "Spearman (ρₛ)": rho_str,
                    "IC 95% Bootstrap": ci_str,
                    "Valor p": p_str,
                    "Evidencia": evid_str,
                }
                if is_multi_metal:
                    rec["_IS_GROUP"] = ""
                records.append(rec)

        res_df = pd.DataFrame(records)

        target_title = f"Asociación entre Biomarcador y Variables Cuantitativas" if is_multi_metal else f"Asociación de {self.get_label(metal_cols[0], labels_map)} con Variables Cuantitativas"
        subtitle = f"Correlación no paramétrica de Spearman con intervalos de confianza al 95% (Bootstrap, B={n_boot})"

        spanners = [
            {"label": "Variable", "columns": ["Variable", "n Válido"]},
            {"label": "Correlación no Paramétrica", "columns": ["Spearman (ρₛ)", "IC 95% Bootstrap", "Valor p"]},
            {"label": "Inferencia", "columns": ["Evidencia"]}
        ]

        notes = [
            "ρₛ: Coeficiente de correlación de Spearman; IC 95% Bootstrap: Intervalo de confianza por remuestreo no paramétrico (2,000 réplicas).",
            "En negrita se resaltan las asociaciones estadísticamente significativas con p < 0.05.",
            "La correlación describe dependencia monotónica y no implica causalidad."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=target_title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Variable": "l",
                "n Válido": "c",
                "Spearman (ρₛ)": "c",
                "IC 95% Bootstrap": "c",
                "Valor p": "c",
                "Evidencia": "l"
            },
            spanners=spanners,
            group_col="_IS_GROUP" if is_multi_metal else None,
            divider_borders=is_multi_metal
        )
        return report

    # =========================================================================
    # Etapa 5, 8 y 9: Metal vs Variables Binarias (Exposición, Hábitos, Síntomas)
    # =========================================================================

    def binary_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        columns: Optional[Union[str, Sequence[str]]] = None,
        groups: Optional[Dict[str, List[str]]] = None,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Contrasta las concentraciones del metal seleccionado según factores dicotómicos / binarios
        (ej. Sexo, Condición de Exposición, Talleres, Industrias, Fuentes de Agua, Síntomas).
        Aplica la prueba no paramétrica de Mann-Whitney U y reporta medianas, IQR,
        tamaño de efecto biserial por rangos (r_rb con IC 95%) y estimador de Hodges-Lehmann.

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL'). Admite alias o lista.
        columns : Union[str, Sequence[str]], opcional
            Variables binarias a analizar. Si es None, utiliza las variables binarias estándar.
        groups : Dict[str, List[str]], opcional
            Diccionario de agrupación conceptual (ej. Exposición Ambiental, Hábitos, etc.).
        n_boot : int, opcional (por defecto 2000)
            Número de réplicas Bootstrap para intervalos de confianza.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres de variables.
        filepath : str, opcional
            Ruta para guardar el reporte en HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con DataFrame, renderizado interactivo en HTML y exportación a Excel/CSV.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        if columns is None:
            # Lista estándar de variables binarias
            cand_cols = [
                "Sexo", "Es_Expuesto",
                "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
                "Exposicion_Industrias_fabrica_metales", "Exposicion_Industrias_fabrica_productos_quimicos",
                "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria", "Exposicion_Cualquier_Lugar_Riesgo",
                "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada", "Salud_Agua_pozo_profundo",
                "Salud_Fuma", "Salud_Actividad", "Salud_Bombillos", "Salud_Techo", "Salud_Joyeria",
                "Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo",
            ]
            cand_cols = [c for c in cand_cols if c in df_m.columns]
        else:
            cand_cols = [self._resolve_column(c) for c in self._normalize_columns(columns) if self._resolve_column(c) in df_m.columns]

        if groups is None:
            groups = {
                "Factores Demográficos y Condición Global": [
                    "Sexo", "Es_Expuesto"
                ],
                "Fuentes de Exposición Ambiental y Talleres": [
                    "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
                    "Exposicion_Industrias_fabrica_metales", "Exposicion_Industrias_fabrica_productos_quimicos",
                    "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria", "Exposicion_Cualquier_Lugar_Riesgo",
                ],
                "Hábitos de Salud y Fuentes de Agua": [
                    "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada", "Salud_Agua_pozo_profundo",
                    "Salud_Fuma", "Salud_Actividad", "Salud_Bombillos", "Salud_Techo", "Salud_Joyeria",
                    "Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo",
                ]
            }

        grouped_cols: Dict[str, List[str]] = {}
        ungrouped = list(cand_cols)

        for g_name, g_list in groups.items():
            matched = [c for c in cand_cols if c in g_list]
            if matched:
                grouped_cols[g_name] = matched
                for c in matched:
                    if c in ungrouped:
                        ungrouped.remove(c)

        if ungrouped:
            grouped_cols["Otras Variables Dicótomas"] = ungrouped

        def parse_binary_groups(series: pd.Series) -> Tuple[pd.Series, pd.Series, str, str]:
            """
            Separa una serie en grupo positivo (Sí/1/Masculino) y grupo de referencia (No/0/Femenino).
            Retorna (pos_cases, neg_cases, label_pos, label_neg).
            """
            s_clean = series.dropna()
            if s_clean.empty:
                return pd.Series(dtype=float), pd.Series(dtype=float), "Sí", "No"

            # 1. Verificar si es variable de género / sexo
            str_vals = set(s_clean.astype(str).str.strip().str.lower())
            if any(v in ("masculino", "m", "hombre", "niño", "femenino", "f", "mujer", "niña") for v in str_vals):
                pos_mask = series.astype(str).str.strip().str.lower().isin(["masculino", "m", "hombre", "niño"])
                neg_mask = series.astype(str).str.strip().str.lower().isin(["femenino", "f", "mujer", "niña"])
                return series[pos_mask], series[neg_mask], "Masculino", "Femenino"

            # 2. Caso general binario estándar (Sí / No, 1 / 0, True / False)
            pos_mask = series.astype(str).str.strip().str.lower().isin(["si", "sí", "1", "1.0", "true", "yes", "s"])
            neg_mask = series.astype(str).str.strip().str.lower().isin(["no", "0", "0.0", "false", "n"])

            # Si la serie coincide con representaciones estándar de Sí/No o 1/0
            if pos_mask.any() or neg_mask.any():
                return series[pos_mask], series[neg_mask], "Sí", "No"

            # 3. Codificaciones dicotómicas arbitrarias
            uvals = sorted(list(s_clean.unique()))
            if len(uvals) == 2:
                pos_val, neg_val = uvals[1], uvals[0]
                return series[series == pos_val], series[series == neg_val], str(pos_val), str(neg_val)
            elif len(uvals) == 1:
                return series[series == uvals[0]], pd.Series(dtype=float), str(uvals[0]), "Sin casos"

            return pd.Series(dtype=float), pd.Series(dtype=float), "Sí", "No"

        records = []
        for g_name, cols in grouped_cols.items():
            records.append({
                "Factor / Variable": f"**{g_name}**",
                "n (%) Sí": "",
                "Mediana [RIQ] (Sí)": "",
                "Mediana [RIQ] (No)": "",
                "Efecto r<sub>rb</sub> [IC 95%]": "",
                "Hodges-Lehmann": "",
                "Mann-Whitney (p)": "",
                "_IS_GROUP": "__GROUP_HEADER__"
            })

            for col in cols:
                if col not in df_m.columns:
                    continue

                col_data = df_m[col]
                pos_cases, neg_cases, label_pos, label_neg = parse_binary_groups(col_data)

                n_pos = len(pos_cases)
                n_neg = len(neg_cases)
                n_tot = n_pos + n_neg
                if n_tot == 0:
                    continue

                pct_pos = (n_pos / n_tot) * 100 if n_tot > 0 else 0.0
                n_str = f"{n_pos} ({pct_pos:.1f}%)"

                # Extracción de valores del metal en cada grupo
                pos_idx = pos_cases.index
                neg_idx = neg_cases.index

                metal_pos = df_m.loc[pos_idx, m_col].dropna().to_numpy()
                metal_neg = df_m.loc[neg_idx, m_col].dropna().to_numpy()

                res_mw = mann_whitney_test(metal_pos, metal_neg, n_boot=n_boot)

                med_pos = res_mw["median1"]
                q25_pos, q75_pos = res_mw["iqr1"]
                med_neg = res_mw["median0"]
                q25_neg, q75_neg = res_mw["iqr0"]

                med_pos_str = f"{med_pos:.2f} [{q25_pos:.2f} - {q75_pos:.2f}]" if pd.notna(med_pos) else "N/D"
                med_neg_str = f"{med_neg:.2f} [{q25_neg:.2f} - {q75_neg:.2f}]" if pd.notna(med_neg) else "N/D"

                r_rb = res_mw["r_rb"]
                r_ci_l, r_ci_h = res_mw["r_rb_ci"]
                p_val = res_mw["p_val"]
                hl = res_mw["hl_shift"]

                if pd.notna(r_rb) and pd.notna(r_ci_l) and pd.notna(r_ci_h):
                    r_str = f"{r_rb:+.2f} [{r_ci_l:+.2f}, {r_ci_h:+.2f}]"
                elif pd.notna(r_rb):
                    r_str = f"{r_rb:+.2f} [—]"
                else:
                    r_str = "N/D"

                hl_str = f"{hl:+.2f}" if pd.notna(hl) else "N/D"
                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else ("<0.0001" if pd.notna(p_val) else "N/D")

                if pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"
                    r_str = f"**{r_str}**"

                # Marca de advertencia si n_pos < 2
                var_label = self.get_label(col, labels_map)
                if n_pos < 2:
                    var_display = f"{var_label} <sup>†</sup>"
                else:
                    var_display = var_label

                records.append({
                    "Factor / Variable": var_display,
                    "n (%) Sí": n_str,
                    "Mediana [RIQ] (Sí)": med_pos_str,
                    "Mediana [RIQ] (No)": med_neg_str,
                    "Efecto r<sub>rb</sub> [IC 95%]": r_str,
                    "Hodges-Lehmann": hl_str,
                    "Mann-Whitney (p)": p_str,
                    "_IS_GROUP": ""
                })

        res_df = pd.DataFrame(records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Comparación de {metal_name} según Factores Binarios y de Exposición"
        subtitle = f"Prueba no paramétrica de Mann-Whitney U, tamaño de efecto biserial por rangos ($r_{{rb}}$) y estimador de Hodges-Lehmann (n={n_metal})"

        spanners = [
            {"label": "Variable y Frecuencia", "columns": ["Factor / Variable", "n (%) Sí"]},
            {"label": f"Distribución de {metal_name}", "columns": ["Mediana [RIQ] (Sí)", "Mediana [RIQ] (No)"]},
            {"label": "Tamaño del Efecto e Inferencia", "columns": ["Efecto r<sub>rb</sub> [IC 95%]", "Hodges-Lehmann", "Mann-Whitney (p)"]}
        ]

        notes = [
            "n (%) Sí: Frecuencia absoluta y porcentaje relativo de participantes que presentan la condición o factor de riesgo (categoría 'Sí'/1) dentro de la cohorte analítica con determinaciones del metal.",
            "Variables compuestas de entorno: 'Cualquier Taller en Entorno', 'Cualquier Industria en Entorno' y 'Cualquier Punto Crítico Ambiental' son indicadores agregados calculados mediante disyunción lógica (máximo/OR) sobre las categorías específicas desagregadas, representando la exposición a al menos una de las fuentes del grupo.",
            "Mediana [RIQ]: Mediana y rango intercuartílico [Q1 - Q3] en escala natural.",
            "r<sub>rb</sub>: Correlación biserial por rangos (Rank Biserial Correlation, acotada en [-1, +1]); r<sub>rb</sub> > 0 indica concentraciones mayores en el grupo expuesto/Sí.",
            "Hodges-Lehmann: Estimador no paramétrico de desplazamiento de pseudomediana entre expuestos y no expuestos.",
            "N/D (No Determinado): El parámetro no puede calcularse debido a la ausencia de un grupo de contraste (ej. 100% de la cohorte pertenece a la misma categoría, n_No = 0).",
            "[—] (IC no disponible): Intervalo de confianza bootstrap no computable debido a tamaño muestral unitario en el subgrupo (n < 2).",
            "† Indica categorías con n < 2 observaciones positivas (baja potencia inferencial e IC no estimable; interpretar con cautela exploratoria).",
            "En negrita se resaltan contrastes estadísticamente significativos con p < 0.05."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Factor / Variable": "l",
                "n (%) Sí": "c",
                "Mediana [RIQ] (Sí)": "c",
                "Mediana [RIQ] (No)": "c",
                "Efecto r<sub>rb</sub> [IC 95%]": "c",
                "Hodges-Lehmann": "c",
                "Mann-Whitney (p)": "c"
            },
            spanners=spanners,
            group_col="_IS_GROUP",
            divider_borders=True
        )
        return report

    # =========================================================================
    # Etapa 6: Metal vs Variables Categóricas Politómicas (Sector, Institución)
    # =========================================================================

    def categorical_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        columns: Optional[Union[str, Sequence[str]]] = None,
        p_adjust: str = "holm",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Contrasta las concentraciones del metal seleccionado según variables categóricas
        nominales con más de dos grupos (ej. Sector residencial, Institución).
        Aplica la prueba de Kruskal-Wallis H, tamaño de efecto Epsilon Cuadrado (epsilon^2)
        y ejecuta la prueba post-hoc de comparaciones múltiples de Dunn.

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL').
        columns : Union[str, Sequence[str]], opcional
            Variables politómicas a contrastar (por defecto: Sector, Institucion).
        p_adjust : str, opcional (por defecto 'holm')
            Método de corrección para las comparaciones múltiples de Dunn.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla resumen de Kruskal-Wallis y acceso a comparaciones Dunn en posthoc_df.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        if columns is None:
            cand_cols = ["Sector", "Institucion"]
            target_cols = [c for c in cand_cols if c in df_m.columns]
        else:
            target_cols = [self._resolve_column(c) for c in self._normalize_columns(columns) if self._resolve_column(c) in df_m.columns]

        records = []
        all_dunn_dfs = []

        for col in target_cols:
            series_col = df_m[col].dropna()
            categories = sorted(series_col.unique(), key=lambda x: str(x))
            k = len(categories)
            if k < 2:
                continue

            groups_data = []
            group_names = []
            for cat in categories:
                sub_vals = df_m[df_m[col] == cat][m_col].dropna().to_numpy()
                groups_data.append(sub_vals)
                group_names.append(str(cat))

            res_kw = kruskal_wallis_test(groups_data, group_names)
            h_stat = res_kw["h_stat"]
            p_val = res_kw["p_val"]
            eps_sq = res_kw["epsilon_sq"]

            # Prueba post-hoc de Dunn
            df_dunn = dunn_posthoc_test(groups_data, group_names, p_adjust=p_adjust)
            if not df_dunn.empty:
                df_dunn.insert(0, "Variable", self.get_label(col, labels_map))
                all_dunn_dfs.append(df_dunn)

            var_label = self.get_label(col, labels_map)

            first = True
            for g_stat in res_kw["group_stats"]:
                c_name = g_stat["group"]
                n_g = g_stat["n"]
                pct_g = (n_g / n_metal) * 100 if n_metal > 0 else 0.0
                med_g = g_stat["median"]
                q25_g = g_stat["q25"]
                q75_g = g_stat["q75"]

                med_str = f"{med_g:.2f} [{q25_g:.2f} - {q75_g:.2f}]"
                h_str = f"{h_stat:.3f}" if first and pd.notna(h_stat) else ""
                eps_str = f"{eps_sq:.3f}" if first and pd.notna(eps_sq) else ""
                p_str = (f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001") if first and pd.notna(p_val) else ""

                if first and pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"
                    eps_str = f"**{eps_str}**"

                records.append({
                    "Variable": f"**{var_label}**" if first else "",
                    "Categoría / Grupo": str(c_name),
                    "n (%)": f"{n_g} ({pct_g:.1f}%)",
                    "Mediana [RIQ]": med_str,
                    "Kruskal-Wallis (H)": h_str,
                    "Efecto (ε²)": eps_str,
                    "Valor p": p_str
                })
                first = False

        res_df = pd.DataFrame(records)
        combined_dunn_df = pd.concat(all_dunn_dfs, ignore_index=True) if all_dunn_dfs else pd.DataFrame()

        metal_name = self.get_label(m_col, labels_map)
        title = f"Comparación de {metal_name} según Variables Categóricas Politómicas"
        subtitle = f"Prueba no paramétrica de Kruskal-Wallis H y tamaño de efecto Epsilon Cuadrado (ε²) (n={n_metal})"

        spanners = [
            {"label": "Variable y Grupos", "columns": ["Variable", "Categoría / Grupo", "n (%)"]},
            {"label": f"Distribución de {metal_name}", "columns": ["Mediana [RIQ]"]},
            {"label": "Estadística de Kruskal-Wallis", "columns": ["Kruskal-Wallis (H)", "Efecto (ε²)", "Valor p"]}
        ]

        notes = [
            "Mediana [RIQ]: Mediana y rango intercuartílico [Q1 - Q3] en escala natural.",
            "ε²: Epsilon Cuadrado (tamaño de efecto no paramétrico acotado en [0, 1]).",
            f"Comparaciones post-hoc realizadas con la prueba de Dunn y ajuste por multiplicidad de {p_adjust.upper()} (ver posthoc_df o pestaña PostHoc_Dunn en Excel).",
            "En negrita se resaltan contrastes estadísticamente significativos con p < 0.05."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Variable": "l",
                "Categoría / Grupo": "l",
                "n (%)": "c",
                "Mediana [RIQ]": "c",
                "Kruskal-Wallis (H)": "c",
                "Efecto (ε²)": "c",
                "Valor p": "c"
            },
            spanners=spanners,
            posthoc_df=combined_dunn_df,
            divider_borders=True
        )
        return report

    # =========================================================================
    # Etapa 7: Metal vs Hábitos Alimenticios (Variables Ordinales 0 a 4)
    # =========================================================================

    def dietary_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        columns: Optional[Union[str, Sequence[str]]] = None,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Analiza la asociación entre la frecuencia de consumo de grupos alimentarios
        (codificados ordinalmente de 0=Nunca a 4=Diario) y la concentración del metal seleccionado.
        Evalúa específicamente hipótesis biológicas (ej. Pescado -> Hg).

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL').
        columns : Union[str, Sequence[str]], opcional
            Variables dietarias a contrastar (por defecto: Alim_*).
        n_boot : int, opcional (por defecto 2000)
            Número de réplicas Bootstrap para el intervalo de confianza.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con correlaciones ordinales de Spearman e intervalos Bootstrap.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        if columns is None:
            cand_cols = [c for c in df_m.columns if c.startswith("Alim_")]
        else:
            cand_cols = [self._resolve_column(c) for c in self._normalize_columns(columns) if self._resolve_column(c) in df_m.columns]

        records = []
        for col in cand_cols:
            s_diet = df_m[col].dropna()
            if len(s_diet) < 3:
                continue

            # Conversión ordinal si contiene texto
            if pd.api.types.is_string_dtype(s_diet) or pd.api.types.is_object_dtype(s_diet):
                diet_num = s_diet.map(lambda v: DIET_ORDINAL_MAP.get(str(v).strip().lower(), np.nan))
            else:
                diet_num = pd.to_numeric(s_diet, errors="coerce")

            res_sp = spearman_correlation(diet_num, df_m.loc[diet_num.index, m_col], n_boot=n_boot)

            rho = res_sp["rho"]
            ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
            p_val = res_sp["p_val"]
            n_val = res_sp["n_valid"]

            if np.isnan(rho):
                continue

            rho_str = f"{rho:+.3f}"
            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else "N/D"
            p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else "<0.0001"

            # Relevancia biológica a priori
            bio_note = "Plausibilidad estándar"
            if "pescado" in col.lower() and "hg" in m_col.lower():
                bio_note = "Hipótesis primaria toxicológica (Metilmercurio en pesca)"
            elif "cereal" in col.lower() and "cd" in m_col.lower():
                bio_note = "Hipótesis primaria (Absorción de Cadmio en granos)"
            elif "agua" in col.lower() and "pb" in m_col.lower():
                bio_note = "Hipótesis primaria (Tuberías de Plomo)"

            if pd.notna(p_val) and p_val < 0.05:
                rho_str = f"**{rho_str}**"
                p_str = f"**{p_str}**"

            var_label = self.get_label(col, labels_map)
            records.append({
                "Alimento / Grupo Dietario": f"**{var_label}**",
                "Escala Ordinal": "0 a 4 (Nunca - Diario)",
                "n Válido": str(n_val),
                "Spearman (ρₛ)": rho_str,
                "IC 95% Bootstrap": ci_str,
                "Valor p": p_str,
                "Relevancia Toxicológica": bio_note
            })

        res_df = pd.DataFrame(records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Asociación entre Frecuencia de Consumo Dietario y {metal_name}"
        subtitle = f"Correlación ordinal no paramétrica de Spearman e intervalos de confianza Bootstrap (n={n_metal})"

        spanners = [
            {"label": "Variable Dietaria", "columns": ["Alimento / Grupo Dietario", "Escala Ordinal", "n Válido"]},
            {"label": "Correlación con Biomarcador", "columns": ["Spearman (ρₛ)", "IC 95% Bootstrap", "Valor p"]},
            {"label": "Interpretación", "columns": ["Relevancia Toxicológica"]}
        ]

        notes = [
            "Escala Ordinal: 0: Nunca, 1: Rara vez, 2: A veces, 3: Frecuentemente, 4: Diario.",
            "ρₛ: Coeficiente de correlación de Spearman; IC 95% Bootstrap: Intervalo por remuestreo no paramétrico (2,000 réplicas).",
            "En negrita se resaltan correlaciones estadísticamente significativas con p < 0.05."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Alimento / Grupo Dietario": "l",
                "Escala Ordinal": "c",
                "n Válido": "c",
                "Spearman (ρₛ)": "c",
                "IC 95% Bootstrap": "c",
                "Valor p": "c",
                "Relevancia Toxicológica": "l"
            },
            spanners=spanners
        )
        return report

    # =========================================================================
    # Etapa 10: Metal vs Score de Riesgo e Indicadores Específicos
    # =========================================================================

    def risk_score_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa la calibración empírica y concordancia del Algoritmo de Riesgo frente al biomarcador:
        1. Score Global de Riesgo (Score_Riesgo) vs Concentración del Metal (Spearman).
        2. Indicador Específico de Riesgo (ej. Riesgo_Pb para Pb) vs Concentración del Metal (Mann-Whitney).

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL').
        n_boot : int, opcional (por defecto 2000)
            Número de réplicas Bootstrap.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con parámetros de asociación del Score de Riesgo.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        # Mapeo de riesgo específico
        risk_ind_map = {
            "Plomo_ug_dL": "Riesgo_Pb",
            "Mercurio_ug_L": "Riesgo_Hg",
            "Cadmio_ug_L": "Riesgo_Cd"
        }
        specific_risk_col = risk_ind_map.get(m_col)

        records = []

        # 1. Score Global de Riesgo (Spearman)
        if "Score_Riesgo" in df_m.columns:
            res_sp = spearman_correlation(df_m["Score_Riesgo"], df_m[m_col], n_boot=n_boot)
            rho = res_sp["rho"]
            ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
            p_val = res_sp["p_val"]

            rho_str = f"{rho:+.3f}" if pd.notna(rho) else "N/D"
            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else "N/D"
            p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else "<0.0001"

            if pd.notna(p_val) and p_val < 0.05:
                rho_str = f"**{rho_str}**"
                p_str = f"**{p_str}**"

            records.append({
                "Indicador de Riesgo": "**Score Global de Riesgo (Score_Riesgo)**",
                "Tipo de Métrica": "Cuantitativa / Ordinal",
                "Método Estadístico": "Correlación de Spearman",
                "Tamaño del Efecto": f"ρₛ = {rho_str}",
                "IC 95% Bootstrap": ci_str,
                "Valor p": p_str,
                "Conclusión": "Asociación monotónica" if (pd.notna(p_val) and p_val < 0.05) else "Sin asociación lineal/monotónica estadísticamente demostrada"
            })

        # 2. Indicador Específico de Riesgo (Mann-Whitney)
        if specific_risk_col and specific_risk_col in df_m.columns:
            risk_s = df_m[specific_risk_col].dropna()
            pos_mask = risk_s.astype(str).str.strip().str.lower().isin(["si", "sí", "1", "1.0", "true"])

            vals_pos = df_m.loc[risk_s[pos_mask].index, m_col].dropna().to_numpy()
            vals_neg = df_m.loc[risk_s[~pos_mask].index, m_col].dropna().to_numpy()

            res_mw = mann_whitney_test(vals_pos, vals_neg, n_boot=n_boot)
            r_rb = res_mw["r_rb"]
            ci_l, ci_h = res_mw["r_rb_ci"]
            p_val = res_mw["p_val"]
            med_pos = res_mw["median1"]
            med_neg = res_mw["median0"]

            r_str = f"{r_rb:+.2f}" if pd.notna(r_rb) else "N/D"
            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else "N/D"
            p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else "<0.0001"

            if pd.notna(p_val) and p_val < 0.05:
                r_str = f"**{r_str}**"
                p_str = f"**{p_str}**"

            spec_label = self.get_label(specific_risk_col, labels_map)
            records.append({
                "Indicador de Riesgo": f"**Indicador Específico ({spec_label})**",
                "Tipo de Métrica": "Dicotómica (Alto vs Bajo Riesgo)",
                "Método Estadístico": "Mann-Whitney U",
                "Tamaño del Efecto": f"r_rb = {r_str} (Med: {med_pos:.2f} vs {med_neg:.2f})",
                "IC 95% Bootstrap": ci_str,
                "Valor p": p_str,
                "Conclusión": "Discriminación válida" if (pd.notna(p_val) and p_val < 0.05) else "Discriminación insuficiente en esta muestra"
            })

        res_df = pd.DataFrame(records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Validación Empírica del Algoritmo de Riesgo frente a {metal_name}"
        subtitle = f"Evaluación de calibración y discriminación diagnóstica (n={n_metal})"

        notes = [
            "Score_Riesgo: Índice cuantitativo continuo de factores acumulados.",
            "r_rb: Correlación biserial por rangos; ρₛ: Coeficiente de Spearman; IC 95% Bootstrap: 2,000 réplicas.",
            "Un valor p < 0.05 indica que los niños clasificados con mayor riesgo presentan niveles biológicos significativamente más elevados."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Indicador de Riesgo": "l",
                "Tipo de Métrica": "c",
                "Método Estadístico": "c",
                "Tamaño del Efecto": "c",
                "IC 95% Bootstrap": "c",
                "Valor p": "c",
                "Conclusión": "l"
            }
        )
        return report

    # =========================================================================
    # Etapa 11: Metal vs Metal (Co-Exposición Inter-Metálica)
    # =========================================================================

    def metal_correlations(
        self,
        metals: Optional[Sequence[str]] = None,
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Analiza las correlaciones de co-exposición inter-metales (Pb vs Hg, Pb vs Cd, Hg vs Cd).
        Reporta el coeficiente no paramétrico de Spearman, intervalos de confianza Bootstrap al 95%
        y tamaños muestrales efectivos de cada par analizado.

        Parámetros
        ----------
        metals : Sequence[str], opcional
            Lista de metales a evaluar (por defecto: Plomo_ug_dL, Mercurio_ug_L, Cadmio_ug_L).
        n_boot : int, opcional (por defecto 2000)
            Número de réplicas Bootstrap.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla de co-exposición inter-metales.
        """
        if metals is None:
            resolved_metals = [self._resolve_metal(m) for m in DEFAULT_PRIMARY_METALS if self._resolve_metal(m) in self.df.columns]
        else:
            resolved_metals = [self._resolve_metal(m) for m in self._normalize_metals(metals) if self._resolve_metal(m) in self.df.columns]

        k = len(resolved_metals)
        records = []

        for i in range(k):
            m1 = resolved_metals[i]
            lbl1 = self.get_label(m1, labels_map)
            for j in range(i + 1, k):
                m2 = resolved_metals[j]
                lbl2 = self.get_label(m2, labels_map)

                sub_df = self.df[[m1, m2]].dropna()
                sub_df = sub_df[(sub_df[m1] > 0) & (sub_df[m2] > 0)]
                n_pair = len(sub_df)

                if n_pair < 3:
                    continue

                res_sp = spearman_correlation(sub_df[m1], sub_df[m2], n_boot=n_boot)
                rho = res_sp["rho"]
                ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
                p_val = res_sp["p_val"]

                rho_str = f"{rho:+.3f}" if pd.notna(rho) else "N/D"
                ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else "N/D"
                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else "<0.0001"

                if pd.notna(p_val) and p_val < 0.05:
                    rho_str = f"**{rho_str}**"
                    p_str = f"**{p_str}**"
                    interp = "Evidencia de co-exposición o fuente común"
                elif pd.notna(p_val) and p_val < 0.10:
                    interp = "Tendencia marginal de co-exposición"
                else:
                    interp = "Sin correlación aparente"

                records.append({
                    "Par de Co-Exposición": f"**{lbl1}** ↔ **{lbl2}**",
                    "n Válido": str(n_pair),
                    "Spearman (ρₛ)": rho_str,
                    "IC 95% Bootstrap": ci_str,
                    "Valor p": p_str,
                    "Interpretación": interp
                })

        res_df = pd.DataFrame(records)

        title = "Matriz de Co-Exposición y Correlación Inter-Metales (Metal ↔ Metal)"
        subtitle = "Evaluación de fuentes comunes de contaminación y co-ocurrencia toxicológica (Spearman ρₛ con IC Bootstrap)"

        spanners = [
            {"label": "Pares Evaluados", "columns": ["Par de Co-Exposición", "n Válido"]},
            {"label": "Correlación de Co-Exposición", "columns": ["Spearman (ρₛ)", "IC 95% Bootstrap", "Valor p"]},
            {"label": "Inferencia Toxicológica", "columns": ["Interpretación"]}
        ]

        notes = [
            "ρₛ: Coeficiente de correlación de Spearman; IC 95% Bootstrap: Intervalo por remuestreo no paramétrico (2,000 réplicas).",
            "La correlación entre metales refleja co-exposición ambiental/dietaria o vías de transporte comunes, no causalidad entre biomarcadores."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Par de Co-Exposición": "l",
                "n Válido": "c",
                "Spearman (ρₛ)": "c",
                "IC 95% Bootstrap": "c",
                "Valor p": "c",
                "Interpretación": "l"
            },
            spanners=spanners
        )
        return report

    # =========================================================================
    # Etapa 12 y 13: Clasificación de Niveles de Metal (Bajo, Medio, Alto)
    # =========================================================================

    def metal_levels_summary(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        cutoffs: Optional[Tuple[float, float]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Clasifica la concentración del metal en tres niveles toxicológicamente justificados:
        - Bajo: Concentración menor al primer umbral (< corte 1).
        - Medio: Concentración entre el primer y segundo umbral (corte 1 a < corte 2).
        - Alto: Concentración igual o superior al valor de referencia internacional (>= corte 2).

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a clasificar (por defecto: 'Plomo_ug_dL').
        cutoffs : Tuple[float, float], opcional
            Tupla (corte_1, corte_2). Si es None, utiliza DEFAULT_METAL_CUTOFFS.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con desglose de frecuencias y concentraciones por nivel.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        if cutoffs is None:
            c1, c2 = DEFAULT_METAL_CUTOFFS.get(m_col, (1.0, 3.5))
        else:
            c1, c2 = cutoffs

        vals = df_m[m_col].dropna()

        # Clasificación en niveles
        low_vals = vals[vals < c1]
        med_vals = vals[(vals >= c1) & (vals < c2)]
        high_vals = vals[vals >= c2]

        categories = [
            ("Bajo", f"< {c1:g}", low_vals),
            ("Medio", f"{c1:g} a < {c2:g}", med_vals),
            ("Alto", f"≥ {c2:g} (Límite CDC/OMS/EPA)", high_vals),
        ]

        records = []
        for cat_name, cut_desc, subset in categories:
            n_cat = len(subset)
            pct_cat = (n_cat / n_metal) * 100 if n_metal > 0 else 0.0

            if n_cat > 0:
                v_min, v_max = float(subset.min()), float(subset.max())
                med = float(subset.median())
                q25, q75 = float(subset.quantile(0.25)), float(subset.quantile(0.75))
                range_str = f"{v_min:.2f} - {v_max:.2f}"
                med_str = f"{med:.2f} [{q25:.2f} - {q75:.2f}]"
            else:
                range_str = "N/D"
                med_str = "N/D"

            records.append({
                "Nivel Toxicológico": f"**{cat_name}**",
                "Criterio de Corte": cut_desc,
                "n (%)": f"{n_cat} ({pct_cat:.1f}%)",
                "Rango Observado": range_str,
                "Mediana [RIQ]": med_str
            })

        res_df = pd.DataFrame(records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Estratificación Toxicológica de {metal_name} en Niveles (Bajo / Medio / Alto)"
        subtitle = f"Puntos de corte normativos y toxicológicos internacionales (n={n_metal})"

        notes = [
            f"Límite superior de referencia toxicológica para {metal_name}: {c2:g} (CDC/EPA/OMS).",
            "La categorización en 3 niveles permite modelar el gradiente biológico de exposición previo al análisis multivariante."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Nivel Toxicológico": "l",
                "Criterio de Corte": "l",
                "n (%)": "c",
                "Rango Observado": "c",
                "Mediana [RIQ]": "c"
            }
        )
        return report

    def risk_by_metal_level(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        cutoffs: Optional[Tuple[float, float]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa si el Score de Riesgo aumenta de manera monotónica a través de los niveles
        del metal (Bajo < Medio < Alto) aplicando Kruskal-Wallis y tendencia de Jonckheere-Terpstra.

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a analizar (por defecto: 'Plomo_ug_dL').
        cutoffs : Tuple[float, float], opcional
            Puntos de corte toxicológicos.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con gradiente del Score de Riesgo por nivel de metal.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        if "Score_Riesgo" not in df_m.columns:
            return BivariateTableReport(df=pd.DataFrame(), title="Score de Riesgo no disponible")

        if cutoffs is None:
            c1, c2 = DEFAULT_METAL_CUTOFFS.get(m_col, (1.0, 3.5))
        else:
            c1, c2 = cutoffs

        def get_level(v):
            if pd.isna(v):
                return np.nan
            if v < c1:
                return "Bajo"
            if v < c2:
                return "Medio"
            return "Alto"

        df_m["_Nivel_Temp"] = df_m[m_col].apply(get_level)

        score_low = df_m[df_m["_Nivel_Temp"] == "Bajo"]["Score_Riesgo"].dropna().to_numpy()
        score_med = df_m[df_m["_Nivel_Temp"] == "Medio"]["Score_Riesgo"].dropna().to_numpy()
        score_high = df_m[df_m["_Nivel_Temp"] == "Alto"]["Score_Riesgo"].dropna().to_numpy()

        res_kw = kruskal_wallis_test([score_low, score_med, score_high], ["Bajo", "Medio", "Alto"])
        res_jt = jonckheere_terpstra_test([score_low, score_med, score_high], alternative="increasing")

        records = []
        levels_info = [("Bajo", score_low), ("Medio", score_med), ("Alto", score_high)]

        first = True
        for lvl_name, s_arr in levels_info:
            n_l = len(s_arr)
            pct_l = (n_l / n_metal) * 100 if n_metal > 0 else 0.0

            if n_l > 0:
                med = float(np.median(s_arr))
                q25, q75 = float(np.percentile(s_arr, 25)), float(np.percentile(s_arr, 75))
                med_str = f"{med:.2f} [{q25:.2f} - {q75:.2f}]"
            else:
                med_str = "N/D"

            kw_p_str = f"{res_kw['p_val']:.4f}" if first and pd.notna(res_kw['p_val']) else ""
            jt_p_str = f"{res_jt['p_val']:.4f}" if first and pd.notna(res_jt['p_val']) else ""

            records.append({
                "Nivel del Metal": f"**{lvl_name}**",
                "n (%)": f"{n_l} ({pct_l:.1f}%)",
                "Score de Riesgo Mediana [RIQ]": med_str,
                "Kruskal-Wallis (p)": kw_p_str,
                "Tendencia Jonckheere (p)": jt_p_str
            })
            first = False

        res_df = pd.DataFrame(records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Score de Riesgo a través de los Niveles de {metal_name} (Bajo < Medio < Alto)"
        subtitle = f"Evaluación de gradiente biológico y tendencia monotónica (n={n_metal})"

        notes = [
            "Tendencia Jonckheere (p): Prueba de Jonckheere-Terpstra para hipótesis de ordenamiento creciente estricto (Bajo < Medio < Alto).",
            "Un valor p < 0.05 en Jonckheere confirma que el score de riesgo se incrementa progresivamente con el nivel toxicológico del biomarcador."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Nivel del Metal": "l",
                "n (%)": "c",
                "Score de Riesgo Mediana [RIQ]": "c",
                "Kruskal-Wallis (p)": "c",
                "Tendencia Jonckheere (p)": "c"
            }
        )
        return report

    # =========================================================================
    # Etapa 17 y 18: Matriz Maestra de Asociaciones y Ranking de Variables
    # =========================================================================

    def master_association_matrix(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        fdr_method: str = "fdr_bh",
        fdr_alpha: float = 0.10,
        p_alpha: float = 0.05,
        n_boot: int = 1000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Construye la Matriz Maestra de Tamizaje Bivariante exhaustivo frente al metal seleccionado (Etapa 17).
        Calcula tamaños de efecto (r_rb o rho_s con IC 95%), valor p sin ajustar,
        valor p ajustado por FDR (Benjamini-Hochberg) y clasifica el nivel de evidencia.

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal analizado (por defecto: 'Plomo_ug_dL').
        fdr_method : str, opcional (por defecto 'fdr_bh')
            Método de control de FDR ('fdr_bh', 'holm', 'bonferroni').
        fdr_alpha : float, opcional (por defecto 0.10)
            Umbral de significancia FDR.
        p_alpha : float, opcional (por defecto 0.05)
            Umbral de significancia cruda.
        n_boot : int, opcional (por defecto 1000)
            Réplicas Bootstrap para intervalos de confianza.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con matriz maestra completa.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)
        n_metal = len(df_m)

        # Recopilar todas las variables candidatas no metales
        metal_all = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L", "Muestra_Codificada"]
        candidate_cols = [c for c in df_m.columns if c not in metal_all and not c.startswith("_")]

        raw_records = []
        p_values = []

        for col in candidate_cols:
            s_col = df_m[col].dropna()
            n_val = len(s_col)
            if n_val < 3:
                continue

            uvals = set(s_col.unique())
            is_binary = uvals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "si", "no", "sí", "yes"}) and len(uvals) <= 2
            is_dietary = col.startswith("Alim_")

            # 1. Variable Binaria
            if is_binary:
                pos_mask = s_col.astype(str).str.strip().str.lower().isin(["si", "sí", "1", "1.0", "true"])
                g_pos = df_m.loc[s_col[pos_mask].index, m_col].dropna().to_numpy()
                g_neg = df_m.loc[s_col[~pos_mask].index, m_col].dropna().to_numpy()
                n_pos = len(g_pos)

                res_mw = mann_whitney_test(g_pos, g_neg, n_boot=n_boot)
                eff = res_mw["r_rb"]
                ci_l, ci_h = res_mw["r_rb_ci"]
                p_val = res_mw["p_val"]

                tipo = "Binaria"
                metodo = "Mann-Whitney U"
                eff_label = f"r_rb = {eff:+.2f}" if pd.notna(eff) else "N/D"

            # 2. Variable Dietaria / Ordinal
            elif is_dietary:
                diet_num = s_col.map(lambda v: DIET_ORDINAL_MAP.get(str(v).strip().lower(), v))
                diet_num = pd.to_numeric(diet_num, errors="coerce")
                n_pos = int((diet_num > 0).sum())

                res_sp = spearman_correlation(diet_num, df_m.loc[diet_num.index, m_col], n_boot=n_boot)
                eff = res_sp["rho"]
                ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
                p_val = res_sp["p_val"]

                tipo = "Ordinal (Dieta)"
                metodo = "Spearman"
                eff_label = f"ρₛ = {eff:+.2f}" if pd.notna(eff) else "N/D"

            # 3. Variable Cuantitativa Continua
            elif pd.api.types.is_numeric_dtype(s_col) and len(uvals) > 5:
                num_s = pd.to_numeric(s_col, errors="coerce")
                n_pos = len(num_s)

                res_sp = spearman_correlation(num_s, df_m.loc[num_s.index, m_col], n_boot=n_boot)
                eff = res_sp["rho"]
                ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
                p_val = res_sp["p_val"]

                tipo = "Cuantitativa"
                metodo = "Spearman"
                eff_label = f"ρₛ = {eff:+.2f}" if pd.notna(eff) else "N/D"

            # 4. Variable Categórica Politómica (>2 grupos)
            else:
                cat_names = [str(c) for c in uvals]
                groups_arr = [df_m[df_m[col] == c][m_col].dropna().to_numpy() for c in uvals]
                res_kw = kruskal_wallis_test(groups_arr, cat_names)
                eff = res_kw["epsilon_sq"]
                ci_l, ci_h = np.nan, np.nan
                p_val = res_kw["p_val"]
                n_pos = n_val

                tipo = "Politómica"
                metodo = "Kruskal-Wallis"
                eff_label = f"ε² = {eff:.2f}" if pd.notna(eff) else "N/D"

            if pd.isna(p_val):
                continue

            ci_str = f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) and pd.notna(ci_h) else "N/D"

            raw_records.append({
                "Variable_Raw": col,
                "Variable": self.get_label(col, labels_map),
                "Tipo": tipo,
                "n_Positivo": n_pos,
                "Método": metodo,
                "Efecto_Num": eff if pd.notna(eff) else 0.0,
                "Efecto [IC 95%]": f"{eff_label} {ci_str}" if ci_str != "N/D" else eff_label,
                "p_Raw": p_val,
                "p_FDR": np.nan,
            })
            p_values.append(p_val)

        if not raw_records:
            return BivariateTableReport(df=pd.DataFrame(), title="Sin datos suficientes")

        # Ajuste Benjamini-Hochberg FDR
        adj_p = adjust_pvalues(p_values, method=fdr_method)
        for rec, padj in zip(raw_records, adj_p):
            rec["p_FDR"] = float(padj)

        df_full = pd.DataFrame(raw_records)

        # Clasificación de Evidencia
        evid_list = []
        for _, row in df_full.iterrows():
            p_r = row["p_Raw"]
            p_f = row["p_FDR"]
            n_p = row["n_Positivo"]
            eff = abs(row["Efecto_Num"])

            if n_p < 2:
                evid_list.append("Categoría rara (n<2)")
            elif p_f < fdr_alpha:
                evid_list.append("Asociación significativa (FDR < 0.10)")
            elif p_r < p_alpha and eff >= 0.30:
                evid_list.append("Asociación potencial (Efecto fuerte, p < 0.05)")
            elif p_r < p_alpha:
                evid_list.append("Asociación exploratoria (p < 0.05)")
            elif eff >= 0.30:
                evid_list.append("Efecto fuerte con alta incertidumbre")
            else:
                evid_list.append("Sin evidencia")

        df_full["Nivel de Evidencia"] = evid_list

        # Formato de presentación
        display_records = []
        for _, row in df_full.iterrows():
            p_r = row["p_Raw"]
            p_f = row["p_FDR"]
            p_r_str = f"{p_r:.4f}" if p_r >= 0.0001 else "<0.0001"
            p_f_str = f"{p_f:.4f}" if p_f >= 0.0001 else "<0.0001"

            if p_f < fdr_alpha:
                p_f_str = f"**{p_f_str}**"
                p_r_str = f"**{p_r_str}**"
            elif p_r < p_alpha:
                p_r_str = f"**{p_r_str}**"

            display_records.append({
                "Variable": f"**{row['Variable']}**",
                "Tipo": row["Tipo"],
                "n Exp": str(row["n_Positivo"]),
                "Método": row["Método"],
                "Tamaño del Efecto [IC 95%]": row["Efecto [IC 95%]"],
                "Valor p": p_r_str,
                "p (FDR)": p_f_str,
                "Nivel de Evidencia": row["Nivel de Evidencia"]
            })

        res_df = pd.DataFrame(display_records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Matriz Maestra de Screening Bivariante frente a {metal_name}"
        subtitle = f"Evaluación exhaustiva con control de multiplicidad de Benjamini-Hochberg (FDR) e intervalos Bootstrap (n={n_metal})"

        spanners = [
            {"label": "Variable", "columns": ["Variable", "Tipo", "n Exp"]},
            {"label": "Prueba y Magnitud", "columns": ["Método", "Tamaño del Efecto [IC 95%]"]},
            {"label": "Significancia y Multiplicidad", "columns": ["Valor p", "p (FDR)"]},
            {"label": "Interpretación", "columns": ["Nivel de Evidencia"]}
        ]

        notes = [
            "r_rb: Correlación biserial por rangos; ρₛ: Coeficiente de Spearman; ε²: Epsilon Cuadrado.",
            "p (FDR): Valor p ajustado mediante el procedimiento de Benjamini-Hochberg para controlar la tasa de falsos descubrimientos.",
            "En negrita se destacan asociaciones con significancia estadística formal tras ajuste por comparaciones múltiples (FDR < 0.10)."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Variable": "l",
                "Tipo": "c",
                "n Exp": "c",
                "Método": "c",
                "Tamaño del Efecto [IC 95%]": "c",
                "Valor p": "c",
                "p (FDR)": "c",
                "Nivel de Evidencia": "l"
            },
            spanners=spanners
        )
        return report

    def variable_ranking(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        fdr_method: str = "fdr_bh",
        fdr_alpha: float = 0.10,
        p_alpha: float = 0.05,
        min_effect: float = 0.30,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Genera el Ranking de Variables Multicriterio (Etapa 18) clasificando los factores
        en Prioridad Alta, Prioridad Intermedia y Prioridad Baja para su selección en PCA y PLS.

        Parámetros
        ----------
        metal : Union[str, Sequence[str]], opcional
            Metal a clasificar (por defecto: 'Plomo_ug_dL').
        fdr_method : str, opcional (por defecto 'fdr_bh')
            Método de ajuste por multiplicidad.
        fdr_alpha : float, opcional (por defecto 0.10)
            Umbral FDR.
        p_alpha : float, opcional (por defecto 0.05)
            Umbral de valor p crudo.
        min_effect : float, opcional (por defecto 0.30)
            Magnitud mínima de efecto para alta prioridad.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla jerárquica de variables rankeadas y justificadas.
        """
        master_rep = self.master_association_matrix(metal=metal, fdr_method=fdr_method, n_boot=500, labels_map=labels_map)
        df_master = master_rep.df
        if df_master.empty:
            return BivariateTableReport(df=pd.DataFrame(), title="Sin datos")

        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]
        df_m = self._get_metal_sample(m_col)

        # Construir matriz de ranking con tests.py
        candidate_cols = [c for c in df_m.columns if c not in ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L", "Muestra_Codificada"] and not c.startswith("_")]

        raw_rows = []
        p_vals = []
        for col in candidate_cols:
            s = df_m[col].dropna()
            if len(s) < 3:
                continue

            uvals = set(s.unique())
            is_bin = uvals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "si", "no", "sí", "yes"}) and len(uvals) <= 2
            is_diet = col.startswith("Alim_")

            if is_bin:
                pos_m = s.astype(str).str.strip().str.lower().isin(["si", "sí", "1", "1.0", "true"])
                g1 = df_m.loc[s[pos_m].index, m_col].dropna().to_numpy()
                g0 = df_m.loc[s[~pos_m].index, m_col].dropna().to_numpy()
                res = mann_whitney_test(g1, g0, n_boot=200)
                eff = res["r_rb"]
                p = res["p_val"]
                n_pos = len(g1)
            elif is_diet:
                dnum = s.map(lambda v: DIET_ORDINAL_MAP.get(str(v).strip().lower(), v))
                dnum = pd.to_numeric(dnum, errors="coerce")
                res = spearman_correlation(dnum, df_m.loc[dnum.index, m_col], n_boot=200)
                eff = res["rho"]
                p = res["p_val"]
                n_pos = int((dnum > 0).sum())
            elif pd.api.types.is_numeric_dtype(s) and len(uvals) > 5:
                res = spearman_correlation(s, df_m.loc[s.index, m_col], n_boot=200)
                eff = res["rho"]
                p = res["p_val"]
                n_pos = len(s)
            else:
                g_arr = [df_m[df_m[col] == c][m_col].dropna().to_numpy() for c in uvals]
                res = kruskal_wallis_test(g_arr)
                eff = res["epsilon_sq"]
                p = res["p_val"]
                n_pos = len(s)

            if pd.isna(p):
                continue

            raw_rows.append({
                "Variable_Raw": col,
                "Variable": self.get_label(col, labels_map),
                "n_Positivo": n_pos,
                "Efecto_Num": eff if pd.notna(eff) else 0.0,
                "p_Raw": p,
            })
            p_vals.append(p)

        df_rank_base = pd.DataFrame(raw_rows)
        df_rank_base["p_FDR"] = adjust_pvalues(p_vals, method=fdr_method)

        # Aplicar motor de ranking multicriterio
        df_ranked = rank_bivariate_associations(
            df_rank_base,
            effect_col="Efecto_Num",
            p_raw_col="p_Raw",
            p_fdr_col="p_FDR",
            n_pos_col="n_Positivo",
            min_effect_high=min_effect,
            fdr_alpha=fdr_alpha,
            p_alpha=p_alpha,
            min_n_pos=2
        )

        display_records = []
        for _, row in df_ranked.iterrows():
            eff_val = row["Efecto_Num"]
            p_r = row["p_Raw"]
            p_f = row["p_FDR"]
            prio = row["Prioridad"]

            prio_badge = f"**{prio}**" if prio == "Prioridad Alta" else prio
            display_records.append({
                "Variable Candidata": f"**{row['Variable']}**",
                "Prioridad": prio_badge,
                "Efecto (|E|)": f"{abs(eff_val):.2f}",
                "n Expuestos": str(row["n_Positivo"]),
                "Valor p": f"{p_r:.4f}" if p_r >= 0.0001 else "<0.0001",
                "p (FDR)": f"{p_f:.4f}" if p_f >= 0.0001 else "<0.0001",
                "Justificación Metodológica": row["Justificacion"]
            })

        res_df = pd.DataFrame(display_records)

        metal_name = self.get_label(m_col, labels_map)
        title = f"Ranking Multicriterio de Variables para Modelado Multivariante ({metal_name})"
        subtitle = "Clasificación de predictores en Prioridad Alta, Intermedia y Baja según magnitud, incertidumbre y FDR (Etapa 18)"

        spanners = [
            {"label": "Identificación de Predictor", "columns": ["Variable Candidata", "Prioridad"]},
            {"label": "Métricas Bioestadísticas", "columns": ["Efecto (|E|)", "n Expuestos", "Valor p", "p (FDR)"]},
            {"label": "Criterio de Selección", "columns": ["Justificación Metodológica"]}
        ]

        notes = [
            "Prioridad Alta: Magnitud sustancial (|E| ≥ 0.30), frecuencia representativa (n ≥ 2) y evidencia estadística (FDR < 0.10 o p < 0.05).",
            "Prioridad Intermedia: Efecto moderado o evidencia exploratoria sin significancia estricta.",
            "Prioridad Baja: Frecuencia insuficiente (n < 2) o efecto despreciable.",
            "Las variables de Prioridad Alta e Intermedia son los insumos primarios recomendados para PCA y PLS."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Variable Candidata": "l",
                "Prioridad": "c",
                "Efecto (|E|)": "c",
                "n Expuestos": "c",
                "Valor p": "c",
                "p (FDR)": "c",
                "Justificación Metodológica": "l"
            },
            spanners=spanners
        )
        return report

    # =========================================================================
    # Etapa 19, 20 y 21: Colinealidad, Preparación PCA y PLS
    # =========================================================================

    def collinearity_summary(
        self,
        columns: Optional[Sequence[str]] = None,
        threshold: float = 0.65,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa la colinealidad y redundancia entre pares de predictores (Etapa 19)
        para evitar sobreajuste y multicolinealidad en modelos multivariantes posteriores.

        Parámetros
        ----------
        columns : Sequence[str], opcional
            Variables a evaluar.
        threshold : float, opcional (por defecto 0.65)
            Umbral de correlación de Spearman para alertar sobre colinealidad.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte de pares redundantes o colineales.
        """
        metal_all = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L", "Muestra_Codificada"]
        if columns is None:
            cand_cols = [c for c in self.df.columns if c not in metal_all and not c.startswith("_")]
        else:
            cand_cols = [self._resolve_column(c) for c in self._normalize_columns(columns) if self._resolve_column(c) in self.df.columns]

        corr_mat = collinearity_matrix(self.df, cand_cols, method="spearman")
        cols = list(corr_mat.columns)
        k = len(cols)

        records = []
        for i in range(k):
            c1 = cols[i]
            lbl1 = self.get_label(c1, labels_map)
            for j in range(i + 1, k):
                c2 = cols[j]
                lbl2 = self.get_label(c2, labels_map)
                val = corr_mat.loc[c1, c2]

                if pd.notna(val) and abs(val) >= threshold:
                    # Detección de redundancia conceptual
                    if ("Es_Expuesto" in (c1, c2) and "Score_Riesgo" in (c1, c2)):
                        diag = "Redundancia conceptual directa (Es_Expuesto vs Score_Riesgo)"
                        rec = "No incluir ambos en el mismo modelo; preferir Score_Riesgo"
                    elif c1.startswith("Exposicion_Talleres") and c2.startswith("Exposicion_Talleres"):
                        diag = "Colinealidad entre talleres del mismo sector"
                        rec = "Utilizar indicador compuesto 'Exposicion_Cualquier_Taller'"
                    elif c1.startswith("Exposicion_Industrias") and c2.startswith("Exposicion_Industrias"):
                        diag = "Colinealidad entre industrias del mismo sector"
                        rec = "Utilizar indicador compuesto 'Exposicion_Cualquier_Industria'"
                    else:
                        diag = f"Alta colinealidad (|ρ| = {abs(val):.2f} ≥ {threshold})"
                        rec = "Evaluar reducción dimensional con PCA antes de regresión"

                    records.append({
                        "Variable 1": f"**{lbl1}**",
                        "Variable 2": f"**{lbl2}**",
                        "Correlación (ρₛ)": f"{val:+.3f}",
                        "Diagnóstico": diag,
                        "Recomendación Metodológica": rec
                    })

        if not records:
            records.append({
                "Variable 1": "Sin alertas",
                "Variable 2": "Sin alertas",
                "Correlación (ρₛ)": "—",
                "Diagnóstico": f"No se detectaron pares con |ρ| ≥ {threshold}",
                "Recomendación Metodológica": "Predictores libres de colinealidad severa"
            })

        res_df = pd.DataFrame(records)

        title = "Evaluación de Colinealidad y Redundancia entre Predictores"
        subtitle = f"Detección de dependencias estructurales y redundancias conceptuales (|ρₛ| ≥ {threshold})"

        notes = [
            "Introducir variables redundantes en modelos multivariables con n=20 produce inestabilidad numérica e inflación de varianza.",
            "Recomendación estándar: Consolidar indicadores en variables compuestas o seleccionar el predictor de mayor relevancia biológica."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Variable 1": "l",
                "Variable 2": "l",
                "Correlación (ρₛ)": "c",
                "Diagnóstico": "l",
                "Recomendación Metodológica": "l"
            }
        )
        return report

    def pca_candidates(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Genera la lista estructurada de variables candidatas para PCA (Etapa 20),
        organizadas por bloques conceptuales homogéneos.
        """
        metal_cols = self._normalize_metals(metal)
        m_col = metal_cols[0]

        records = [
            {"Bloque Conceptual": "**Bloque 1: Fuentes y Factores de Exposición**", "Variables Recomendadas": "Es_Expuesto, Exposicion_Cualquier_Taller, Exposicion_Cualquier_Industria, Salud_Agua_pozo_profundo", "Objetivo del PCA": "Identificar perfiles ambientales y ejes latentes de co-contaminación."},
            {"Bloque Conceptual": "**Bloque 2: Hábitos y Frecuencia Dietaria**", "Variables Recomendadas": "Alim_Pescados, Alim_Carnes, Alim_Tuberculos, Alim_Cereales, Alim_Vegetales", "Objetivo del PCA": "Extraer patrones dietéticos principales en la población infantil."},
            {"Bloque Conceptual": "**Bloque 3: Biomarcadores de Metales Pesados**", "Variables Recomendadas": "Plomo_ug_dL, Mercurio_ug_L, Cadmio_ug_L", "Objetivo del PCA": "Analizar la estructura de co-exposición y correlación multimetálica."},
            {"Bloque Conceptual": "**Bloque 4: Características Clínicas y Antropométricas**", "Variables Recomendadas": "Edad, Peso_kg, Altura_cm, IMC, Score_Riesgo", "Objetivo del PCA": "Evaluar el gradiente de desarrollo y vulnerabilidad física."}
        ]

        res_df = pd.DataFrame(records)

        title = "Preparación y Selección de Bloques para Análisis de Componentes Principales (PCA)"
        subtitle = "Estructura modular recomendada para evitar mezclas indiscriminadas de variables heterogéneas (Etapa 20)"

        notes = [
            "Con n=20, no se debe ejecutar un PCA global con todas las variables juntas.",
            "Se debe ejecutar un PCA enfocado por bloque temático o sobre los biomarcadores metálicos."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Bloque Conceptual": "l",
                "Variables Recomendadas": "l",
                "Objetivo del PCA": "l"
            }
        )
        return report

    def pls_candidates(
        self,
        metal: Union[str, Sequence[str]] = "Plomo_ug_dL",
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Define la formulación formal del modelo PLS (Partial Least Squares, Etapa 21):
        Bloque X (factores de exposición y hábitos seleccionados) vs Bloque Y (biomarcadores de metales).
        """
        records = [
            {"Matriz": "**Matriz X (Predictores)**", "Variables Seleccionadas": "Edad, Score_Riesgo, Exposicion_Cualquier_Taller, Exposicion_Cualquier_Industria, Salud_Agua_pozo_profundo, Alim_Pescados", "Dimensión Sugerida": "n=20 × p=6 (Parsimonioso)", "Criterio": "Variables de Prioridad Alta/Media sin colinealidad extrema"},
            {"Matriz": "**Matriz Y (Respuesta)**", "Variables Seleccionadas": "Plomo_ug_dL, Mercurio_ug_L, Cadmio_ug_L", "Dimensión Sugerida": "n=20 × q=3", "Criterio": "Biomarcadores toxicológicos en sangre"}
        ]

        res_df = pd.DataFrame(records)

        title = "Formulación y Preparación del Modelo PLS (Partial Least Squares)"
        subtitle = "Definición de matrices X (Exposiciones) e Y (Biomarcadores) con restricción de parsimonia (Etapa 21)"

        notes = [
            "Regla de parsimonia para n=20: Limitar la matriz X a un máximo de 5 a 6 componentes/predictores para evitar sobreajuste.",
            "El preprocesamiento debe incluir centrado y escalado de varianza unitaria (autoscaling) antes de ajustar PLS."
        ]

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={
                "Matriz": "l",
                "Variables Seleccionadas": "l",
                "Dimensión Sugerida": "c",
                "Criterio": "l"
            }
        )
        return report

    # =========================================================================
    # Pilar 1: Cualitativa vs Cualitativa (Asociación Epidemiológica y Riesgo)
    # =========================================================================

    def qualitative_association(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        var_x: Optional[str] = None,
        var_y: Optional[str] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Calcula la asociación bivariante completa entre dos variables cualitativas / categóricas (2x2 o RxC).
        Aplica la prueba exacta de Fisher, Chi-cuadrado de independencia (con corrección de Yates para 2x2),
        Odds Ratio (OR con IC 95%), Riesgo Relativo (RR con IC 95%) y V de Cramér / Coeficiente Phi.

        Parámetros
        ----------
        x / var_x : str
            Variable independiente / factor de exposición (ej. 'Salud_Agua_pozo_profundo', 'Es_Expuesto').
        y / var_y : str
            Variable dependiente / desenlace (ej. 'Excede_Umbral_Hg', 'Es_Expuesto', 'Sexo').
        labels_map : Dict[str, str], opcional
            Mapeo personalizado de nombres de variables.
        filepath : str, opcional
            Ruta para guardar el reporte en HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla de contingencia formateada y métricas de asociación y riesgo.
        """
        if isinstance(self, pd.DataFrame):
            return BivariateTables(self).qualitative_association(
                x=x, y=y, var_x=var_x, var_y=var_y, labels_map=labels_map, filepath=filepath
            )
        col_x = x if x is not None else var_x
        col_y = y if y is not None else var_y
        if col_x is None or col_y is None:
            raise ValueError("Debe especificar las dos variables cualitativas a contrastar (x e y).")

        x_col = self._resolve_column(col_x)
        y_col = self._resolve_column(col_y)

        # Si y es un biomarcador continuo (ej. Mercurio_ug_L), binarizar por umbral clínico (ej. 5.0)
        s_y = self.df[y_col]
        is_y_numeric = pd.to_numeric(s_y, errors="coerce").notna().sum() > (0.5 * len(s_y.dropna()))
        if is_y_numeric:
            cutoff = DEFAULT_METAL_LIMITS.get(y_col, 5.0)
            y_data = np.where(pd.to_numeric(s_y, errors="coerce") >= cutoff, f"≥ {cutoff:g}", f"< {cutoff:g}")
            y_label_full = f"{self.get_label(y_col, labels_map)} (≥ {cutoff:g} µg/L)"
        else:
            y_data = s_y
            y_label_full = self.get_label(y_col, labels_map)

        x_data = self.df[x_col]
        x_label_full = self.get_label(x_col, labels_map)

        res = qualitative_association_test(x_data, y_data)
        ctab = res["contingency_table"]

        if ctab.empty:
            return BivariateTableReport(
                df=pd.DataFrame(),
                title="Asociación Cualitativa no disponible",
                subtitle="Datos insuficientes para construir tabla de contingencia.",
                filepath=filepath
            )

        row_totals = ctab.sum(axis=1)
        records = []
        for row_val, row_series in ctab.iterrows():
            row_tot = row_totals[row_val]
            rec = {
                x_label_full: str(row_val),
            }
            for col_val in ctab.columns:
                cnt = row_series[col_val]
                pct = (cnt / row_tot) * 100.0 if row_tot > 0 else 0.0
                rec[f"{col_val} n (%)"] = f"{cnt} ({pct:.1f}%)"
            rec["Total"] = f"{row_tot} (100.0%)"
            records.append(rec)

        res_df = pd.DataFrame(records)

        notes = []
        if res["is_2x2"]:
            or_val = res["odds_ratio"]
            or_ci = res["or_ci"]
            rr_val = res["relative_risk"]
            rr_ci = res["rr_ci"]
            p_fish = res["fisher_p"]
            p_chi = res["chi2_yates_p"]
            v_cramer = res["cramers_v"]

            or_str = f"{or_val:.2f} [{or_ci[0]:.2f}, {or_ci[1]:.2f}]" if pd.notna(or_val) else "N/D"
            rr_str = f"{rr_val:.2f} [{rr_ci[0]:.2f}, {rr_ci[1]:.2f}]" if pd.notna(rr_val) else "N/D"
            p_fish_str = f"{p_fish:.4f}" if pd.notna(p_fish) and p_fish >= 0.0001 else ("<0.0001" if pd.notna(p_fish) else "N/D")
            p_chi_str = f"{p_chi:.4f}" if pd.notna(p_chi) and p_chi >= 0.0001 else ("<0.0001" if pd.notna(p_chi) else "N/D")
            v_str = f"{v_cramer:.3f}" if pd.notna(v_cramer) else "N/D"

            notes.append(f"Odds Ratio (OR exacto): {or_str}; Riesgo Relativo (RR): {rr_str}.")
            notes.append(f"Prueba Exacta de Fisher (bilateral): p = {p_fish_str}; Chi-cuadrado (Yates): p = {p_chi_str}; V de Cramér (ϕ): {v_str}.")
        else:
            chi_val = res["chi2_stat"]
            p_chi = res["chi2_p"]
            v_cramer = res["cramers_v"]
            notes.append(f"Chi-cuadrado de Pearson (RxC): χ² = {chi_val:.2f}, p = {p_chi:.4f}; V de Cramér: {v_cramer:.3f}.")

        notes.append("Frecuencias expresadas como recuento absoluto y porcentaje por fila: n (%).")

        col_alignments = {c: "c" for c in res_df.columns}
        col_alignments[x_label_full] = "l"

        title = f"Tabla de Contingencia: {x_label_full} vs. {y_label_full}"
        subtitle = "Evaluación de asociación epidemiológica, riesgo relativo y significancia exacta"

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=col_alignments
        )
        return report

    def qualitative_summary(
        self,
        target: Optional[str] = None,
        factors: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        candidate_features: Optional[Sequence[str]] = None,
        target_cutoff: Optional[float] = None,
        target_positive: Optional[Any] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Calcula una matriz resumen de factores de riesgo cualitativos frente a un desenlace binario
        (ej. superación del umbral de mercurio ≥ 5.0 µg/L o condición de expuesto).
        Reporta para cada factor: casos n (%), controles n (%), OR [IC 95%], RR [IC 95%], Fisher p y V de Cramér.

        Parámetros
        ----------
        target / target_col : str, opcional (por defecto 'Mercurio_ug_L')
            Desenlace binario o biomarcador cuantitativo a binarizar mediante corte de referencia.
        factors / feature_cols / candidate_features : Sequence[str], opcional
            Lista de factores de riesgo a contrastar.
        target_cutoff : float, opcional
            Punto de corte si target es cuantitativo (por defecto: 5.0 para Hg, 3.5 para Pb, 1.0 para Cd).
        target_positive : Any, opcional
            Valor específico que define la categoría positiva / casos.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte en HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla estructurada de cribado de riesgo cualitativo.
        """
        if isinstance(self, pd.DataFrame):
            return BivariateTables(self).qualitative_summary(
                target=target, factors=factors, target_col=target_col,
                feature_cols=feature_cols, candidate_features=candidate_features,
                target_cutoff=target_cutoff, target_positive=target_positive,
                labels_map=labels_map, filepath=filepath
            )
        t_col = target or target_col or "Mercurio_ug_L"
        target_col = self._resolve_column(t_col)
        fact_cols = factors if factors is not None else (feature_cols if feature_cols is not None else candidate_features)
        s_tar = self.df[target_col]
        is_num = pd.to_numeric(s_tar, errors="coerce").notna().sum() > (0.5 * len(s_tar.dropna()))

        if target_positive is not None:
            binary_tar = (s_tar.astype(str).str.strip().str.lower() == str(target_positive).strip().lower()).astype(int)
            target_label = f"{self.get_label(target_col, labels_map)} = {target_positive}"
        elif is_num:
            cutoff = target_cutoff if target_cutoff is not None else DEFAULT_METAL_LIMITS.get(target_col, 5.0)
            binary_tar = (pd.to_numeric(s_tar, errors="coerce") >= cutoff).astype(int)
            target_label = f"{self.get_label(target_col, labels_map)} ≥ {cutoff:g}"
        else:
            u_tar = sorted(list(s_tar.dropna().unique()))
            if len(u_tar) == 2:
                binary_tar = (s_tar == u_tar[1]).astype(int)
                target_label = f"{self.get_label(target_col, labels_map)} ({u_tar[1]})"
            else:
                binary_tar = (s_tar.astype(str).str.lower().isin(["si", "sí", "1", "true", "expuesto", "alto"])).astype(int)
                target_label = self.get_label(target_col, labels_map)

        if fact_cols is None:
            cand_factors = [
                "Sexo", "Es_Expuesto",
                "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria", "Exposicion_Cualquier_Lugar_Riesgo",
                "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
                "Exposicion_Industrias_fabrica_metales", "Exposicion_Industrias_fabrica_productos_quimicos",
                "Salud_Agua_pozo_profundo", "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada",
                "Salud_Bombillos", "Salud_Fuma", "Salud_Techo", "Salud_Joyeria",
                "Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo"
            ]
            cand_factors = [c for c in cand_factors if c in self.df.columns and c != target_col]
        else:
            cand_factors = [self._resolve_column(c) for c in self._normalize_columns(fact_cols) if self._resolve_column(c) in self.df.columns and self._resolve_column(c) != target_col]


        records = []
        for f_col in cand_factors:
            s_f = self.df[f_col]
            f_clean = s_f.dropna()
            if f_clean.empty:
                continue

            str_vals = f_clean.astype(str).str.strip().str.lower()
            if any(v in ("si", "sí", "1", "1.0", "true", "yes", "s") for v in str_vals):
                pos_mask = str_vals.isin(["si", "sí", "1", "1.0", "true", "yes", "s"])
                pos_label = "Sí"
            elif any(v in ("masculino", "m", "hombre", "niño") for v in str_vals):
                pos_mask = str_vals.isin(["masculino", "m", "hombre", "niño"])
                pos_label = "Masculino"
            else:
                u = sorted(list(f_clean.unique()))
                if len(u) == 2:
                    pos_mask = (s_f == u[1])
                    pos_label = str(u[1])
                else:
                    continue

            bin_f = pos_mask.astype(int)
            res_test = qualitative_association_test(bin_f, binary_tar)

            if not res_test["is_2x2"]:
                continue

            ctab = res_test["contingency_table"]
            n_exp_cases = int(ctab.loc[1, 1]) if 1 in ctab.index and 1 in ctab.columns else 0
            n_exp_ctrl = int(ctab.loc[1, 0]) if 1 in ctab.index and 0 in ctab.columns else 0
            n_noexp_cases = int(ctab.loc[0, 1]) if 0 in ctab.index and 1 in ctab.columns else 0
            n_noexp_ctrl = int(ctab.loc[0, 0]) if 0 in ctab.index and 0 in ctab.columns else 0

            tot_cases = n_exp_cases + n_noexp_cases
            tot_ctrl = n_exp_ctrl + n_noexp_ctrl

            pct_cases = (n_exp_cases / tot_cases) * 100 if tot_cases > 0 else 0.0
            pct_ctrl = (n_exp_ctrl / tot_ctrl) * 100 if tot_ctrl > 0 else 0.0

            or_val = res_test["odds_ratio"]
            or_ci = res_test["or_ci"]
            rr_val = res_test["relative_risk"]
            rr_ci = res_test["rr_ci"]
            p_fish = res_test["fisher_p"]
            p_chi = res_test["chi2_yates_p"]
            v_cramer = res_test["cramers_v"]

            or_str = f"{or_val:.2f} [{or_ci[0]:.2f}, {or_ci[1]:.2f}]" if pd.notna(or_val) else "N/D"
            rr_str = f"{rr_val:.2f} [{rr_ci[0]:.2f}, {rr_ci[1]:.2f}]" if pd.notna(rr_val) else "N/D"
            p_fish_str = f"{p_fish:.4f}" if pd.notna(p_fish) and p_fish >= 0.0001 else ("<0.0001" if pd.notna(p_fish) else "N/D")
            p_chi_str = f"{p_chi:.4f}" if pd.notna(p_chi) and p_chi >= 0.0001 else ("<0.0001" if pd.notna(p_chi) else "N/D")
            v_str = f"{v_cramer:.3f}" if pd.notna(v_cramer) else "N/D"

            if p_fish < 0.05 or (pd.notna(p_chi) and p_chi < 0.05):
                p_fish_str = f"**{p_fish_str}**"
                or_str = f"**{or_str}**"

            var_label = self.get_label(f_col, labels_map)
            records.append({
                "Factor de Exposición": f"**{var_label}**",
                "Categoría": pos_label,
                "Casos n (%)": f"{n_exp_cases} ({pct_cases:.1f}%)",
                "Controles n (%)": f"{n_exp_ctrl} ({pct_ctrl:.1f}%)",
                "Odds Ratio [IC 95%]": or_str,
                "Riesgo Relativo [IC 95%]": rr_str,
                "Fisher (p)": p_fish_str,
                "Chi² Yates (p)": p_chi_str,
                "V de Cramér": v_str
            })

        res_df = pd.DataFrame(records)

        spanners = [
            {"label": "Factor y Condición", "columns": ["Factor de Exposición", "Categoría"]},
            {"label": f"Frecuencia en {target_label}", "columns": ["Casos n (%)", "Controles n (%)"]},
            {"label": "Estimación de Riesgo Epidemiológico", "columns": ["Odds Ratio [IC 95%]", "Riesgo Relativo [IC 95%]"]},
            {"label": "Significancia y Asociación", "columns": ["Fisher (p)", "Chi² Yates (p)", "V de Cramér"]}
        ]

        notes = [
            "Casos: Participantes que presentan el desenlace ('Sí' o sobre el umbral de referencia); Controles: Participantes sin el desenlace.",
            "OR: Odds Ratio por máxima verosimilitud condicional con IC exacto al 95%; RR: Riesgo Relativo con corrección de Haldane-Anscombe.",
            "Fisher (p): Valor p bilateral de la Prueba Exacta de Fisher (regla de oro en n reducida); Chi² Yates (p): Chi-cuadrado con corrección de continuidad.",
            "En negrita se destacan las asociaciones y estimaciones con significancia estadística formal (p < 0.05)."
        ]

        title = f"Matriz Resumen de Factores de Riesgo vs. {target_label}"
        subtitle = "Análisis bivariante de variables cualitativas mediante epidemiología analítica de contingencia"

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={c: "l" if "Factor" in c else "c" for c in res_df.columns},
            spanners=spanners
        )
        return report

    # =========================================================================
    # Pilar 2: Cuantitativa vs Cualitativa (Comparación de Carga Corporal entre Grupos)
    # =========================================================================

    def compare_groups(
        self,
        quantitative: Optional[Union[str, Sequence[str]]] = None,
        group: Optional[str] = None,
        method: str = "nonparametric",
        group_col: Optional[str] = None,
        continuous_cols: Optional[Union[str, Sequence[str]]] = None,
        iqr_format: str = "range",
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Compara una variable cuantitativa (ej. concentración de mercurio o antropometría)
        o conjunto de variables continuas entre categorías definidas por una variable cualitativa (2 grupos o >2 grupos).
        Permite elegir entre enfoque no paramétrico (Mann-Whitney / Kruskal-Wallis + Hodges-Lehmann / Dunn)
        o paramétrico (t de Welch / ANOVA + Hedges' g / Tukey HSD).

        Parámetros
        ----------
        quantitative / continuous_cols : Union[str, Sequence[str]], opcional (por defecto 'Mercurio_ug_L')
            Variable cuantitativa continua dependiente o lista de variables continuas. Admite alias ('Mercurio', 'hg', etc.).
        group / group_col : str, opcional (por defecto 'Es_Expuesto')
            Variable cualitativa de agrupación (ej. 'Sexo', 'Es_Expuesto', 'Sector', 'Alim_Pescados').
        method : str, opcional (por defecto 'nonparametric')
            Enfoque metodológico: 'nonparametric' (mediana, Mann-Whitney/Kruskal) o 'parametric' (media, Welch t/ANOVA).
        iqr_format : str, opcional (por defecto 'range')
            Formato de dispersión para la mediana: 'range' -> [Q1 - Q3] o 'width' -> [RIQ].
        n_boot : int, opcional (por defecto 2000)
            Réplicas Bootstrap para intervalos de confianza en análisis no paramétrico.
        labels_map : Dict[str, str], opcional
            Mapeo de nombres legibles.
        filepath : str, opcional
            Ruta para guardar el reporte en HTML.

        Retorna
        -------
        BivariateTableReport
            Reporte con tabla descriptiva y de contraste de hipótesis.
        """
        if isinstance(self, pd.DataFrame):
            return BivariateTables(self).compare_groups(
                quantitative=quantitative, group=group, method=method,
                group_col=group_col, continuous_cols=continuous_cols,
                iqr_format=iqr_format, n_boot=n_boot, labels_map=labels_map, filepath=filepath
            )
        q_param = quantitative if quantitative is not None else continuous_cols
        if q_param is None:
            q_param = "Mercurio_ug_L"
        g_param = group or group_col or "Es_Expuesto"
        g_col = self._resolve_column(g_param)
        g_label = self.get_label(g_col, labels_map)
        is_param = method.lower().startswith("param")

        if isinstance(q_param, (list, tuple)) and len(q_param) > 1:
            records = []
            for col in q_param:
                resolved_col = self._resolve_column(col)
                if resolved_col not in self.df.columns:
                    continue
                single_rep = self.compare_groups(
                    quantitative=resolved_col, group=g_param, method=method,
                    iqr_format=iqr_format, n_boot=n_boot, labels_map=labels_map
                )
                df_single = single_rep.to_dataframe()
                if df_single.empty:
                    continue
                v_label = self.get_label(resolved_col, labels_map)

                if len(df_single) == 2 and "Dif. Medias [IC 95%]" in df_single.columns:
                    r1, r0 = df_single.iloc[0], df_single.iloc[1]
                    records.append({
                        "Variable Cuantitativa": f"**{v_label}**",
                        f"{r1['Categoría']} Media (DE)": r1["Media (DE)"],
                        f"{r0['Categoría']} Media (DE)": r0["Media (DE)"],
                        "Dif. Medias [IC 95%]": r1["Dif. Medias [IC 95%]"],
                        "Hedges' g": r1["Hedges' g"],
                        "Estadístico": f"t = {r1['t de Welch']}",
                        "Valor p": r1["Valor p"]
                    })
                elif len(df_single) == 2 and "Hodges-Lehmann [IC 95%]" in df_single.columns:
                    r1, r0 = df_single.iloc[0], df_single.iloc[1]
                    records.append({
                        "Variable Cuantitativa": f"**{v_label}**",
                        f"{r1['Categoría']} Mediana [RIQ]": r1["Mediana [RIQ]"],
                        f"{r0['Categoría']} Mediana [RIQ]": r0["Mediana [RIQ]"],
                        "Hodges-Lehmann [IC 95%]": r1["Hodges-Lehmann [IC 95%]"],
                        "r_rb [IC 95%]": r1["Efecto r_rb [IC 95%]"],
                        "Estadístico": f"U = {r1['Mann-Whitney U']}",
                        "Valor p": r1["Valor p"]
                    })
                else:
                    first_row = df_single.iloc[0]
                    rec = {"Variable Cuantitativa": f"**{v_label}**"}
                    for _, row in df_single.iterrows():
                        cat = str(row["Categoría"]).replace("*", "")
                        val_col = "Media (DE)" if "Media (DE)" in row else "Mediana [RIQ]"
                        rec[f"{cat} {val_col}"] = row[val_col]
                    if "ANOVA F" in first_row:
                        rec["Prueba"] = "ANOVA"
                        rec["Efecto (η²)"] = first_row["Eta² (η²)"]
                        rec["Estadístico"] = f"F = {first_row['ANOVA F']}"
                        rec["Valor p"] = first_row["Valor p"]
                    else:
                        rec["Prueba"] = "Kruskal-Wallis"
                        rec["Efecto (ε²)"] = first_row["Epsilon² (ε²)"]
                        rec["Estadístico"] = f"H = {first_row['Kruskal-Wallis H']}"
                        rec["Valor p"] = first_row["Valor p"]
                    records.append(rec)

            res_df = pd.DataFrame(records)
            title = f"Comparación de Variables Cuantitativas según {g_label}"
            subtitle = f"Enfoque {'no paramétrico' if not is_param else 'paramétrico'} para múltiples biomarcadores o covariables"
            return BivariateTableReport(
                df=res_df,
                title=title,
                subtitle=subtitle,
                filepath=filepath,
                column_alignments={c: "l" if "Variable" in c else "c" for c in res_df.columns}
            )

        q_target = q_param[0] if isinstance(q_param, (list, tuple)) else q_param
        q_col = self._resolve_column(q_target)
        df_sub = self.df.dropna(subset=[q_col, g_col]).copy()
        q_vals = pd.to_numeric(df_sub[q_col], errors="coerce")
        df_sub = df_sub[q_vals.notna()].copy()
        df_sub[q_col] = q_vals[q_vals.notna()]


        q_label = self.get_label(q_col, labels_map)
        g_label = self.get_label(g_col, labels_map)

        unique_groups = df_sub[g_col].unique()
        k = len(unique_groups)

        if k < 2:
            return BivariateTableReport(
                df=pd.DataFrame(),
                title=f"Comparación de {q_label} según {g_label}",
                subtitle="El factor de agrupación debe contener al menos 2 categorías distintas.",
                filepath=filepath
            )

        posthoc_df = None

        if k == 2:
            g_sorted = sorted(list(unique_groups), key=lambda x: str(x))
            g1_name, g0_name = g_sorted[1], g_sorted[0]
            if str(g0_name).lower() in ("si", "sí", "1", "true") and str(g1_name).lower() in ("no", "0", "false"):
                g1_name, g0_name = g0_name, g1_name

            v1 = df_sub[df_sub[g_col] == g1_name][q_col].to_numpy()
            v0 = df_sub[df_sub[g_col] == g0_name][q_col].to_numpy()

            if method.lower().startswith("param"):
                res_t = independent_t_test(v1, v0, equal_var=False)
                t_stat = res_t["t_stat"]
                p_val = res_t["p_val"]
                diff_m = res_t["diff_means"]
                ci_d = res_t["diff_means_ci"]
                g_hedges = res_t["hedges_g"]
                lev_p = res_t["levene_p"]

                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else ("<0.0001" if pd.notna(p_val) else "N/D")
                if pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"

                records = [
                    {
                        "Categoría": f"**{g1_name}**",
                        "n": len(v1),
                        "Media (DE)": f"{res_t['mean1']:.2f} ({res_t['std1']:.2f})",
                        "Mín - Máx": f"{v1.min():.2f} - {v1.max():.2f}" if len(v1) > 0 else "N/D",
                        "Dif. Medias [IC 95%]": f"{diff_m:.2f} [{ci_d[0]:.2f}, {ci_d[1]:.2f}]" if pd.notna(diff_m) else "N/D",
                        "Hedges' g": f"{g_hedges:+.2f}" if pd.notna(g_hedges) else "N/D",
                        "t de Welch": f"{t_stat:.2f}" if pd.notna(t_stat) else "N/D",
                        "Valor p": p_str
                    },
                    {
                        "Categoría": f"**{g0_name}**",
                        "n": len(v0),
                        "Media (DE)": f"{res_t['mean0']:.2f} ({res_t['std0']:.2f})",
                        "Mín - Máx": f"{v0.min():.2f} - {v0.max():.2f}" if len(v0) > 0 else "N/D",
                        "Dif. Medias [IC 95%]": "—",
                        "Hedges' g": "—",
                        "t de Welch": "—",
                        "Valor p": "—"
                    }
                ]
                res_df = pd.DataFrame(records)

                spanners = [
                    {"label": f"Factor: {g_label}", "columns": ["Categoría", "n"]},
                    {"label": f"Parámetros de {q_label}", "columns": ["Media (DE)", "Mín - Máx"]},
                    {"label": "Tamaño del Efecto e Inferencia Paramétrica", "columns": ["Dif. Medias [IC 95%]", "Hedges' g", "t de Welch", "Valor p"]}
                ]
                notes = [
                    f"Media (DE): Media aritmética y desviación estándar paramétrica.",
                    f"Hedges' g: Tamaño del efecto estandarizado corregido por sesgo de muestras pequeñas (n={len(df_sub)}).",
                    f"Prueba de Levene de homocedasticidad: p = {lev_p:.4f}." if pd.notna(lev_p) else "",
                    "Prueba t de Welch (no asume varianzas iguales)."
                ]
                notes = [n for n in notes if n]

                title = f"Comparación Paramétrica de {q_label} según {g_label}"
                subtitle = f"Prueba t de Welch y tamaño del efecto de Hedges ($g$) para 2 grupos independientes"

            else:
                res_mw = mann_whitney_test(v1, v0, n_boot=n_boot)
                u_stat = res_mw["u_stat"]
                p_val = res_mw["p_val"]
                hl = res_mw["hl_shift"]
                hl_ci = res_mw["hl_shift_ci"]
                r_rb = res_mw["r_rb"]
                r_ci = res_mw["r_rb_ci"]

                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else ("<0.0001" if pd.notna(p_val) else "N/D")
                if pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"

                q1_25, q1_75 = res_mw["iqr1"]
                q0_25, q0_75 = res_mw["iqr0"]

                if iqr_format == "range":
                    med1_str = f"{res_mw['median1']:.2f} [{q1_25:.2f} - {q1_75:.2f}]"
                    med0_str = f"{res_mw['median0']:.2f} [{q0_25:.2f} - {q0_75:.2f}]"
                else:
                    med1_str = f"{res_mw['median1']:.2f} [{q1_75 - q1_25:.2f}]"
                    med0_str = f"{res_mw['median0']:.2f} [{q0_75 - q0_25:.2f}]"

                hl_str = f"{hl:.2f} [{hl_ci[0]:.2f}, {hl_ci[1]:.2f}]" if pd.notna(hl) else "N/D"
                r_str = f"{r_rb:+.2f} [{r_ci[0]:+.2f}, {r_ci[1]:+.2f}]" if pd.notna(r_rb) and pd.notna(r_ci[0]) else (f"{r_rb:+.2f}" if pd.notna(r_rb) else "N/D")

                records = [
                    {
                        "Categoría": f"**{g1_name}**",
                        "n": len(v1),
                        "Mediana [RIQ]": med1_str,
                        "Mín - Máx": f"{v1.min():.2f} - {v1.max():.2f}" if len(v1) > 0 else "N/D",
                        "Hodges-Lehmann [IC 95%]": hl_str,
                        "Efecto r_rb [IC 95%]": r_str,
                        "Mann-Whitney U": f"{u_stat:.1f}" if pd.notna(u_stat) else "N/D",
                        "Valor p": p_str
                    },
                    {
                        "Categoría": f"**{g0_name}**",
                        "n": len(v0),
                        "Mediana [RIQ]": med0_str,
                        "Mín - Máx": f"{v0.min():.2f} - {v0.max():.2f}" if len(v0) > 0 else "N/D",
                        "Hodges-Lehmann [IC 95%]": "—",
                        "Efecto r_rb [IC 95%]": "—",
                        "Mann-Whitney U": "—",
                        "Valor p": "—"
                    }
                ]
                res_df = pd.DataFrame(records)

                spanners = [
                    {"label": f"Factor: {g_label}", "columns": ["Categoría", "n"]},
                    {"label": f"Distribución de {q_label}", "columns": ["Mediana [RIQ]", "Mín - Máx"]},
                    {"label": "Tamaño del Efecto e Inferencia no Paramétrica", "columns": ["Hodges-Lehmann [IC 95%]", "Efecto r_rb [IC 95%]", "Mann-Whitney U", "Valor p"]}
                ]
                notes = [
                    "Mediana [RIQ]: Mediana y rango intercuartílico en escala natural.",
                    "Hodges-Lehmann: Estimador de desplazamiento de localización de pseudomediana pareada con IC al 95%.",
                    "r_rb: Correlación biserial por rangos (r_rb > 0 indica niveles superiores en el primer grupo).",
                    "Mann-Whitney U: Prueba no paramétrica de suma de rangos de Wilcoxon-Mann-Whitney."
                ]
                title = f"Comparación no Paramétrica de {q_label} según {g_label}"
                subtitle = f"Prueba U de Mann-Whitney, estimador de Hodges-Lehmann y correlación biserial por rangos"

        else:
            g_sorted = sorted(list(unique_groups), key=lambda x: str(x))
            group_arrays = [df_sub[df_sub[g_col] == g_val][q_col].to_numpy() for g_val in g_sorted]

            if method.lower().startswith("param"):
                res_anova = anova_oneway_test(group_arrays, group_names=[str(g) for g in g_sorted])
                f_stat = res_anova["f_stat"]
                p_val = res_anova["p_val"]
                eta_sq = res_anova["eta_sq"]
                omega_sq = res_anova["omega_sq"]
                lev_p = res_anova["levene_p"]
                posthoc_df = res_anova["posthoc"]

                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else ("<0.0001" if pd.notna(p_val) else "N/D")
                if pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"

                records = []
                first = True
                for g_info in res_anova["group_stats"]:
                    arr_g = df_sub[df_sub[g_col] == g_info["group"]][q_col].to_numpy()
                    records.append({
                        "Categoría": f"**{g_info['group']}**",
                        "n": g_info["n"],
                        "Media (DE)": f"{g_info['mean']:.2f} ({g_info['std']:.2f})",
                        "Mín - Máx": f"{arr_g.min():.2f} - {arr_g.max():.2f}" if len(arr_g) > 0 else "N/D",
                        "ANOVA F": f"{f_stat:.2f}" if first and pd.notna(f_stat) else ("—" if not first else "N/D"),
                        "Eta² (η²)": f"{eta_sq:.3f}" if first and pd.notna(eta_sq) else ("—" if not first else "N/D"),
                        "Omega² (ω²)": f"{omega_sq:.3f}" if first and pd.notna(omega_sq) else ("—" if not first else "N/D"),
                        "Valor p": p_str if first else "—"
                    })
                    first = False
                res_df = pd.DataFrame(records)

                spanners = [
                    {"label": f"Factor: {g_label}", "columns": ["Categoría", "n"]},
                    {"label": f"Parámetros de {q_label}", "columns": ["Media (DE)", "Mín - Máx"]},
                    {"label": "Inferencia ANOVA y Tamaño del Efecto", "columns": ["ANOVA F", "Eta² (η²)", "Omega² (ω²)", "Valor p"]}
                ]
                notes = [
                    "ANOVA: Análisis de varianza de una vía de Fisher-Snedecor.",
                    "Eta² (η²): Proporción de varianza explicada por el factor; Omega² (ω²): Estimador insesgado de la varianza explicada.",
                    f"Prueba de Levene de homocedasticidad: p = {lev_p:.4f}." if pd.notna(lev_p) else "",
                    "Comparaciones múltiples post-hoc calculadas mediante la prueba de Tukey HSD (disponible en posthoc_df)."
                ]
                notes = [n for n in notes if n]

                title = f"Comparación Paramétrica de {q_label} según {g_label} (k={k})"
                subtitle = "Análisis de varianza (ANOVA) de 1 vía y tamaño del efecto global"

            else:
                res_kw = kruskal_wallis_test(group_arrays, group_names=[str(g) for g in g_sorted])
                h_stat = res_kw["h_stat"]
                p_val = res_kw["p_val"]
                eps_sq = res_kw["epsilon_sq"]
                posthoc_df = dunn_posthoc_test(group_arrays, group_names=[str(g) for g in g_sorted], p_adjust="holm")

                p_str = f"{p_val:.4f}" if pd.notna(p_val) and p_val >= 0.0001 else ("<0.0001" if pd.notna(p_val) else "N/D")
                if pd.notna(p_val) and p_val < 0.05:
                    p_str = f"**{p_str}**"

                records = []
                first = True
                for g_info in res_kw["group_stats"]:
                    arr_g = df_sub[df_sub[g_col] == g_info["group"]][q_col].to_numpy()
                    q25, q75 = g_info["q25"], g_info["q75"]
                    if iqr_format == "range":
                        med_str = f"{g_info['median']:.2f} [{q25:.2f} - {q75:.2f}]"
                    else:
                        med_str = f"{g_info['median']:.2f} [{q75 - q25:.2f}]"

                    records.append({
                        "Categoría": f"**{g_info['group']}**",
                        "n": g_info["n"],
                        "Mediana [RIQ]": med_str,
                        "Mín - Máx": f"{arr_g.min():.2f} - {arr_g.max():.2f}" if len(arr_g) > 0 else "N/D",
                        "Kruskal-Wallis H": f"{h_stat:.2f}" if first and pd.notna(h_stat) else ("—" if not first else "N/D"),
                        "Epsilon² (ε²)": f"{eps_sq:.3f}" if first and pd.notna(eps_sq) else ("—" if not first else "N/D"),
                        "Valor p": p_str if first else "—"
                    })
                    first = False
                res_df = pd.DataFrame(records)

                spanners = [
                    {"label": f"Factor: {g_label}", "columns": ["Categoría", "n"]},
                    {"label": f"Distribución de {q_label}", "columns": ["Mediana [RIQ]", "Mín - Máx"]},
                    {"label": "Inferencia no Paramétrica y Tamaño de Efecto", "columns": ["Kruskal-Wallis H", "Epsilon² (ε²)", "Valor p"]}
                ]
                notes = [
                    "Kruskal-Wallis H: Prueba no paramétrica de comparación de k grupos independientes sobre rangos.",
                    "Epsilon² (ε²): Coeficiente de tamaño de efecto para Kruskal-Wallis acotado en [0, 1].",
                    "Comparaciones múltiples por pares evaluadas mediante la prueba post-hoc de Dunn con corrección de Holm (disponible en posthoc_df)."
                ]
                title = f"Comparación no Paramétrica de {q_label} según {g_label} (k={k})"
                subtitle = "Prueba de Kruskal-Wallis y tamaño de efecto Epsilon-Cuadrado"

        col_alignments = {c: "c" for c in res_df.columns}
        col_alignments["Categoría"] = "l"

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=col_alignments,
            spanners=spanners,
            posthoc_df=posthoc_df
        )
        return report

    # =========================================================================
    # Pilar 3: Cuantitativa vs Cuantitativa (Gradientes Continuos y Co-exposición)
    # =========================================================================

    def correlation_analysis(
        self,
        x: Optional[Union[str, Sequence[str]]] = None,
        y: Optional[str] = None,
        target: Optional[str] = None,
        target_col: Optional[str] = None,
        continuous_cols: Optional[Union[str, Sequence[str]]] = None,
        method: str = "nonparametric",
        n_boot: int = 2000,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Evalúa la correlación y dependencia funcional entre dos variables cuantitativas continuas
        o una lista de variables continuas frente a un biomarcador diana.
        Enfoque no paramétrico: Spearman rho_s (con IC 95% Bootstrap) y Kendall tau_b.
        Enfoque paramétrico: Pearson r (con IC 95% Fisher z), R² y parámetros de regresión lineal simple.
        """
        if isinstance(self, pd.DataFrame):
            return BivariateTables(self).correlation_analysis(
                x=x, y=y, target=target, target_col=target_col,
                continuous_cols=continuous_cols, method=method,
                n_boot=n_boot, labels_map=labels_map, filepath=filepath
            )
        col_y = y or target or target_col or "Mercurio_ug_L"
        col_x = x if x is not None else continuous_cols
        if col_x is None:
            raise ValueError("Debe especificar la variable cuantitativa o lista de variables continuas a correlacionar.")

        y_col = self._resolve_column(col_y)
        y_label = self.get_label(y_col, labels_map)
        is_param = method.lower().startswith("param")

        if isinstance(col_x, (list, tuple)) and len(col_x) > 1:
            records = []
            for item in col_x:
                x_resolved = self._resolve_column(item)
                if x_resolved not in self.df.columns or x_resolved == y_col:
                    continue
                single_rep = self.correlation_analysis(
                    x=x_resolved, y=y_col, method=method, n_boot=n_boot, labels_map=labels_map
                )
                df_s = single_rep.to_dataframe()
                if not df_s.empty:
                    records.append(df_s.iloc[0].to_dict())

            res_df = pd.DataFrame(records)
            title = f"Matriz de Correlación vs. {y_label}"
            subtitle = f"Evaluación de gradientes continuos mediante enfoque {'no paramétrico' if not is_param else 'paramétrico'}"
            return BivariateTableReport(
                df=res_df,
                title=title,
                subtitle=subtitle,
                filepath=filepath,
                column_alignments={c: "l" if "Variable" in c else "c" for c in res_df.columns}
            )

        single_x = col_x[0] if isinstance(col_x, (list, tuple)) else col_x
        x_col = self._resolve_column(single_x)

        df_sub = self.df[[x_col, y_col]].dropna().apply(pd.to_numeric, errors="coerce").dropna()
        n_valid = len(df_sub)

        x_label = self.get_label(x_col, labels_map)

        if n_valid < 3:
            return BivariateTableReport(
                df=pd.DataFrame(),
                title=f"Correlación entre {x_label} y {y_label}",
                subtitle="Datos insuficientes (n < 3).",
                filepath=filepath
            )

        vx = df_sub[x_col].to_numpy()
        vy = df_sub[y_col].to_numpy()

        if is_param:
            res_p = pearson_correlation(vx, vy)
            r_val = res_p["r"]
            p_val = res_p["p_val"]
            ci_l, ci_h = res_p["ci_low"], res_p["ci_high"]
            r2 = res_p["r_squared"]
            slope = res_p["slope"]
            intercept = res_p["intercept"]

            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            if p_val < 0.05:
                p_str = f"**{p_str}**"

            records = [{
                "Variable X": f"**{x_label}**",
                "Variable Y": f"**{y_label}**",
                "Metodo": "Pearson",
                "n Válido": n_valid,
                "Pearson (r)": f"{r_val:+.3f}",
                "IC 95% Fisher z": f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) else "N/D",
                "R² (Varianza Explicada)": f"{r2:.3f}",
                "Pendiente (Beta)": f"{slope:+.3f}",
                "Intercepto": f"{intercept:.2f}",
                "Ecuación Regresión": f"Y = {intercept:.2f} + {slope:+.3f}·X",
                "Valor p": p_str
            }]
            res_df = pd.DataFrame(records)

            spanners = [
                {"label": "Variables Evaluadas", "columns": ["Variable X", "Variable Y", "Metodo", "n Válido"]},
                {"label": "Asociación Lineal de Pearson", "columns": ["Pearson (r)", "IC 95% Fisher z", "R² (Varianza Explicada)"]},
                {"label": "Modelo Lineal e Inferencia", "columns": ["Pendiente (Beta)", "Intercepto", "Ecuación Regresión", "Valor p"]}
            ]
            notes = [
                "Pearson (r): Coeficiente de correlación producto-momento que mide la fuerza y sentido de la relación lineal.",
                "IC 95% Fisher z: Intervalo de confianza asintótico al 95% obtenido mediante transformación z de Fisher.",
                "R²: Coeficiente de determinación lineal (proporción de varianza explicada).",
                "Regresión lineal simple estimada por mínimos cuadrados ordinarios (MCO)."
            ]
            title = f"Correlación Lineal Paramétrica: {x_label} vs. {y_label}"
            subtitle = f"Coeficiente de Pearson ($r$), intervalo de Fisher $z$ y parámetros de regresión (n={n_valid})"

        else:
            res_sp = spearman_correlation(vx, vy, n_boot=n_boot)
            res_kt = kendall_correlation(vx, vy)

            rho = res_sp["rho"]
            p_sp = res_sp["p_val"]
            ci_l, ci_h = res_sp["ci_low"], res_sp["ci_high"]
            tau = res_kt["tau"]
            p_kt = res_kt["p_val"]

            p_sp_str = f"{p_sp:.4f}" if p_sp >= 0.0001 else "<0.0001"
            p_kt_str = f"{p_kt:.4f}" if p_kt >= 0.0001 else "<0.0001"
            if p_sp < 0.05:
                p_sp_str = f"**{p_sp_str}**"
            if p_kt < 0.05:
                p_kt_str = f"**{p_kt_str}**"

            records = [{
                "Variable X": f"**{x_label}**",
                "Variable Y": f"**{y_label}**",
                "Metodo": "Spearman / Kendall",
                "n Válido": n_valid,
                "Spearman (ρₛ)": f"{rho:+.3f}",
                "IC 95% Bootstrap": f"[{ci_l:+.2f}, {ci_h:+.2f}]" if pd.notna(ci_l) else "N/D",
                "Spearman p": p_sp_str,
                "Kendall (τ_b)": f"{tau:+.3f}" if pd.notna(tau) else "N/D",
                "Kendall p": p_kt_str
            }]
            res_df = pd.DataFrame(records)

            spanners = [
                {"label": "Variables Evaluadas", "columns": ["Variable X", "Variable Y", "Metodo", "n Válido"]},
                {"label": "Correlación de Rangos de Spearman", "columns": ["Spearman (ρₛ)", "IC 95% Bootstrap", "Spearman p"]},
                {"label": "Correlación de Kendall", "columns": ["Kendall (τ_b)", "Kendall p"]}
            ]
            notes = [
                "Spearman (ρₛ): Coeficiente de correlación no paramétrica de Spearman para dependencia monótona.",
                "IC 95% Bootstrap: Intervalo de confianza obtenido mediante remuestreo bootstrap no paramétrico (2,000 réplicas).",
                "Kendall (τ_b): Coeficiente tau-b de Kendall, especialmente robusto para muestras pequeñas (n=20) y presencia de empates.",
                "En negrita se destacan las correlaciones con significancia estadística formal (p < 0.05)."
            ]
            title = f"Correlación no Paramétrica: {x_label} vs. {y_label}"
            subtitle = f"Dependencia monótona evaluada mediante coeficientes de Spearman ($\rho$) y Kendall ($\tau$) (n={n_valid})"

        col_alignments = {c: "c" for c in res_df.columns}
        col_alignments["Variable X"] = "l"
        col_alignments["Variable Y"] = "l"

        report = BivariateTableReport(
            df=res_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments=col_alignments,
            spanners=spanners
        )
        return report


    # =========================================================================
    # Transición Hacia el Análisis Multivariable (Screening y Selección)
    # =========================================================================

    def multivariate_screening(
        self,
        target: Optional[str] = None,
        candidate_features: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        candidates: Optional[Sequence[str]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        method: str = "nonparametric",
        screening_p_threshold: float = 0.20,
        fdr_alpha: Optional[float] = None,
        labels_map: Optional[Dict[str, str]] = None,
        filepath: Optional[str] = None
    ) -> BivariateTableReport:
        """
        Batería integral de tamizaje bivariante (*screening*) para preseleccionar factores de riesgo
        antes de ingresar a modelos multivariables (regresión lineal múltiple o regresión logística).
        """
        if isinstance(self, pd.DataFrame):
            return BivariateTables(self).multivariate_screening(
                target=target, candidate_features=candidate_features,
                target_col=target_col, candidates=candidates, feature_cols=feature_cols,
                method=method, screening_p_threshold=screening_p_threshold,
                fdr_alpha=fdr_alpha, labels_map=labels_map, filepath=filepath
            )
        t_col = target or target_col or "Mercurio_ug_L"
        target_col = self._resolve_column(t_col)
        c_feats = candidate_features if candidate_features is not None else (candidates if candidates is not None else feature_cols)
        p_threshold = fdr_alpha if fdr_alpha is not None else screening_p_threshold

        s_tar = self.df[target_col]
        is_tar_num = pd.to_numeric(s_tar, errors="coerce").notna().sum() > (0.5 * len(s_tar.dropna()))

        if c_feats is None:
            cand_features = [
                "Edad", "Sexo", "Peso_kg", "Altura_cm", "IMC", "Sector", "Institucion", "Es_Expuesto", "Score_Riesgo",
                "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria", "Exposicion_Cualquier_Lugar_Riesgo",
                "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
                "Exposicion_Industrias_fabrica_metales", "Exposicion_Industrias_fabrica_productos_quimicos",
                "Salud_Agua_pozo_profundo", "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada",
                "Salud_Bombillos", "Salud_Fuma", "Salud_Techo", "Salud_Joyeria",
                "Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo",
                "Alim_Pescados", "Alim_Carnes", "Alim_Lacteos", "Alim_Vegetales"
            ]
            cand_features = [c for c in cand_features if c in self.df.columns and c != target_col]
        else:
            cand_features = [self._resolve_column(c) for c in self._normalize_columns(c_feats) if self._resolve_column(c) in self.df.columns and self._resolve_column(c) != target_col]

        raw_records = []
        p_values = []
        is_param = method.lower().startswith("param")

        for feat in cand_features:
            s_f = self.df[feat].dropna()
            if len(s_f) == 0:
                continue

            is_f_num = pd.to_numeric(s_f, errors="coerce").notna().sum() > (0.8 * len(s_f))
            u_f = s_f.unique()
            k_f = len(u_f)

            rel_type = ""
            test_name = ""
            metric_name = ""
            effect_val = np.nan
            p_val = np.nan

            df_pair = self.df[[target_col, feat]].dropna()
            n_eff = len(df_pair)
            if n_eff < 3:
                continue

            if is_tar_num:
                y_arr = pd.to_numeric(df_pair[target_col], errors="coerce").to_numpy()

                if is_f_num and k_f > 5:
                    rel_type = "Cuant vs Cuant"
                    x_arr = pd.to_numeric(df_pair[feat], errors="coerce").to_numpy()
                    if is_param:
                        res_p = pearson_correlation(x_arr, y_arr)
                        test_name = "Pearson"
                        metric_name = "r"
                        effect_val = res_p["r"]
                        p_val = res_p["p_val"]
                    else:
                        res_s = spearman_correlation(x_arr, y_arr, n_boot=500)
                        test_name = "Spearman"
                        metric_name = "ρₛ"
                        effect_val = res_s["rho"]
                        p_val = res_s["p_val"]

                elif k_f == 2:
                    rel_type = "Cuant vs Cual (2g)"
                    u_sorted = sorted(list(u_f), key=lambda x: str(x))
                    g1_val, g0_val = u_sorted[1], u_sorted[0]
                    v1 = df_pair[df_pair[feat] == g1_val][target_col].to_numpy()
                    v0 = df_pair[df_pair[feat] == g0_val][target_col].to_numpy()

                    if is_param:
                        res_t = independent_t_test(v1, v0)
                        test_name = "Welch t"
                        metric_name = "Hedges' g"
                        effect_val = res_t["hedges_g"]
                        p_val = res_t["p_val"]
                    else:
                        res_mw = mann_whitney_test(v1, v0, n_boot=500)
                        test_name = "Mann-Whitney"
                        metric_name = "r_rb"
                        effect_val = res_mw["r_rb"]
                        p_val = res_mw["p_val"]

                else:
                    rel_type = "Cuant vs Cual (>2g)"
                    group_arrays = [df_pair[df_pair[feat] == g_val][target_col].to_numpy() for g_val in u_f]
                    if is_param:
                        res_a = anova_oneway_test(group_arrays)
                        test_name = "ANOVA"
                        metric_name = "Eta²"
                        effect_val = res_a["eta_sq"]
                        p_val = res_a["p_val"]
                    else:
                        res_k = kruskal_wallis_test(group_arrays)
                        test_name = "Kruskal-Wallis"
                        metric_name = "Epsilon²"
                        effect_val = res_k["epsilon_sq"]
                        p_val = res_k["p_val"]

            else:
                tar_arr = df_pair[target_col]
                if is_f_num:
                    rel_type = "Cual vs Cuant"
                    u_tar = list(tar_arr.unique())
                    if len(u_tar) == 2:
                        v1 = df_pair[df_pair[target_col] == u_tar[1]][feat].to_numpy()
                        v0 = df_pair[df_pair[target_col] == u_tar[0]][feat].to_numpy()
                        if is_param:
                            res_t = independent_t_test(v1, v0)
                            test_name = "Welch t"
                            metric_name = "Hedges' g"
                            effect_val = res_t["hedges_g"]
                            p_val = res_t["p_val"]
                        else:
                            res_mw = mann_whitney_test(v1, v0, n_boot=500)
                            test_name = "Mann-Whitney"
                            metric_name = "r_rb"
                            effect_val = res_mw["r_rb"]
                            p_val = res_mw["p_val"]
                else:
                    rel_type = "Cual vs Cual"
                    res_q = qualitative_association_test(df_pair[feat], df_pair[target_col])
                    if res_q["is_2x2"]:
                        test_name = "Fisher Exacto" if not is_param else "Chi² Yates"
                        metric_name = "Odds Ratio"
                        effect_val = res_q["odds_ratio"]
                        p_val = res_q["fisher_p"] if not is_param else res_q["chi2_yates_p"]
                    else:
                        test_name = "Chi² Pearson"
                        metric_name = "V de Cramér"
                        effect_val = res_q["cramers_v"]
                        p_val = res_q["chi2_p"]

            raw_records.append({
                "feat_raw": feat,
                "rel_type": rel_type,
                "test_name": test_name,
                "metric_name": metric_name,
                "effect_val": effect_val,
                "p_val": p_val,
                "n_eff": n_eff
            })
            p_values.append(p_val if pd.notna(p_val) else 1.0)

        adj_p = adjust_pvalues(p_values, method="fdr_bh")

        final_rows = []
        for rec, padj in zip(raw_records, adj_p):
            p_raw = rec["p_val"]
            eff = rec["effect_val"]
            feat = rec["feat_raw"]
            var_label = self.get_label(feat, labels_map)

            eff_abs = abs(eff) if pd.notna(eff) else 0.0
            if rec["metric_name"] == "Odds Ratio" and pd.notna(eff):
                eff_abs = max(eff, 1.0 / eff) if eff > 0 else 0.0

            if (pd.notna(p_raw) and p_raw < 0.05) or (padj < 0.10) or (eff_abs >= 0.40 and rec["metric_name"] != "Odds Ratio") or (rec["metric_name"] == "Odds Ratio" and eff_abs >= 3.0):
                prio = "Prioridad Alta"
                criterio = "Efecto sustancial y/o significancia formal"
            elif pd.notna(p_raw) and p_raw < p_threshold:
                prio = "Prioridad Intermedia"
                criterio = f"Candidato a modelo multivariable (p < {p_threshold:.2f})"
            else:
                prio = "Baja Prioridad"
                criterio = "Efecto débil o sin evidencia de asociación"

            p_raw_str = f"{p_raw:.4f}" if pd.notna(p_raw) and p_raw >= 0.0001 else ("<0.0001" if pd.notna(p_raw) else "N/D")
            p_fdr_str = f"{padj:.4f}" if pd.notna(padj) and padj >= 0.0001 else ("<0.0001" if pd.notna(padj) else "N/D")

            if pd.notna(p_raw) and p_raw < 0.05:
                p_raw_str = f"**{p_raw_str}**"
            if pd.notna(padj) and padj < 0.10:
                p_fdr_str = f"**{p_fdr_str}**"

            eff_str = f"{eff:+.2f}" if pd.notna(eff) and rec["metric_name"] != "Odds Ratio" else (f"{eff:.2f}" if pd.notna(eff) else "N/D")

            final_rows.append({
                "Variable Predictora": f"**{var_label}**",
                "Tipo de Relación": rec["rel_type"],
                "Prueba": rec["test_name"],
                "Métrica": rec["metric_name"],
                "Tamaño Efecto": eff_str,
                "p (Crudo)": p_raw_str,
                "FDR p-valor (BH)": p_fdr_str,
                "Prioridad Multivariable": f"**{prio}**" if prio == "Prioridad Alta" else prio,
                "Elegibilidad Multivariable": f"**{prio}**" if prio == "Prioridad Alta" else prio,
                "Criterio": criterio,
                "_raw_p": p_raw if pd.notna(p_raw) else 1.0,
                "_prio_ord": 0 if prio == "Prioridad Alta" else (1 if prio == "Prioridad Intermedia" else 2),
                "_feat": feat
            })

        if final_rows:
            final_rows.sort(key=lambda r: (r["_prio_ord"], r["_raw_p"]))

        # Asignar Rank y calcular Colinealidad (rho max con predictores de mayor rango)
        for i, row in enumerate(final_rows, start=1):
            row["Rank"] = i
            feat_i = row["_feat"]
            if i == 1:
                row["Colinealidad (rho max)"] = "Ref / —"
            else:
                higher_feats = [r["_feat"] for r in final_rows[:i-1]]
                max_c = 0.0
                for hf in higher_feats:
                    s1 = pd.to_numeric(self.df[feat_i], errors="coerce")
                    s2 = pd.to_numeric(self.df[hf], errors="coerce")
                    vmask = s1.notna() & s2.notna()
                    if vmask.sum() >= 4 and s1[vmask].nunique() > 1 and s2[vmask].nunique() > 1:
                        try:
                            if not is_param:
                                cv, _ = stats.spearmanr(s1[vmask], s2[vmask])
                            else:
                                cv, _ = stats.pearsonr(s1[vmask], s2[vmask])
                            if pd.notna(cv):
                                max_c = max(max_c, abs(float(cv)))
                        except Exception:
                            pass

                row["Colinealidad (rho max)"] = f"{max_c:.2f}" if max_c > 0 else "0.00"

        res_df = pd.DataFrame(final_rows)

        top_feats = [r["_feat"] for r in final_rows if r["_prio_ord"] <= 1][:8]
        collin_df = pd.DataFrame()
        if len(top_feats) >= 2:
            collin_df = collinearity_matrix(self.df, top_feats, method="spearman" if not is_param else "pearson")

        clean_df = res_df.drop(columns=["_raw_p", "_prio_ord", "_feat"])

        target_label = self.get_label(target_col, labels_map)
        title = f"Matriz Maestra de Tamizaje Bivariante (*Screening*) vs. {target_label}"
        subtitle = f"Batería epidemiológica completa {'no paramétrica' if not is_param else 'paramétrica'} con control FDR (Benjamini-Hochberg)"

        spanners = [
            {"label": "Jerarquía y Predictor", "columns": ["Rank", "Variable Predictora", "Tipo de Relación"]},
            {"label": "Prueba y Magnitud", "columns": ["Prueba", "Métrica", "Tamaño Efecto"]},
            {"label": "Significancia Estadística", "columns": ["p (Crudo)", "FDR p-valor (BH)"]},
            {"label": "Decisión para Modelado Multivariable", "columns": ["Prioridad Multivariable", "Colinealidad (rho max)", "Criterio"]}
        ]

        notes = [
            f"Umbral de inclusión para análisis multivariable: p (Crudo) < {p_threshold:.2f} (criterio estándar de Hosmer-Lemeshow).",
            "FDR p-valor (BH): Valor p ajustado mediante el procedimiento de Benjamini-Hochberg para controlar la tasa de falsos descubrimientos ante pruebas múltiples.",
            "Colinealidad (rho max): Correlación máxima absoluta con predictores precedentes de mayor prioridad para identificar redundancia.",
            "Regla de Parsimonia para n=20: En la muestra analítica disponible, el modelo multivariable final (regresión lineal o logística) no debe incorporar más de 2 a 3 predictores simultáneos para preservar los grados de libertad.",
            "En hojas adicionales ('Colinealidad_Top') se compendia la matriz de correlación entre los predictores preseleccionados."
        ]

        extra_sheets = {}
        if not collin_df.empty:
            extra_sheets["Colinealidad_Top"] = collin_df

        report = BivariateTableReport(
            df=clean_df,
            title=title,
            subtitle=subtitle,
            notes=notes,
            filepath=filepath,
            column_alignments={c: "l" if "Variable" in c or "Criterio" in c else "c" for c in clean_df.columns},
            spanners=spanners,
            extra_sheets=extra_sheets
        )
        return report

    # Aliases
    coexposure_matrix = metal_correlations


# =============================================================================

# Funciones de Acceso Directo a Nivel de Módulo
# =============================================================================

def qualitative_association(df: pd.DataFrame, *args: Any, **kwargs: Any) -> BivariateTableReport:
    """Función de acceso directo para análisis de asociación cualitativa (Pilar 1)."""
    return BivariateTables(df).qualitative_association(*args, **kwargs)


def qualitative_summary(df: pd.DataFrame, *args: Any, **kwargs: Any) -> BivariateTableReport:
    """Función de acceso directo para matriz resumen de factores cualitativos (Pilar 1)."""
    return BivariateTables(df).qualitative_summary(*args, **kwargs)


def compare_groups(df: pd.DataFrame, *args: Any, **kwargs: Any) -> BivariateTableReport:
    """Función de acceso directo para contraste de grupos paramétrico o no paramétrico (Pilar 2)."""
    return BivariateTables(df).compare_groups(*args, **kwargs)


def correlation_analysis(df: pd.DataFrame, *args: Any, **kwargs: Any) -> BivariateTableReport:
    """Función de acceso directo para análisis de correlación continuo paramétrico o no paramétrico (Pilar 3)."""
    return BivariateTables(df).correlation_analysis(*args, **kwargs)


def multivariate_screening(df: pd.DataFrame, *args: Any, **kwargs: Any) -> BivariateTableReport:
    """Función de acceso directo para tamizaje multivariable con control FDR (Transición Multivariable)."""
    return BivariateTables(df).multivariate_screening(*args, **kwargs)


__all__ = [
    "BivariateTableReport",
    "BivariateTables",
    "qualitative_association",
    "qualitative_summary",
    "compare_groups",
    "correlation_analysis",
    "multivariate_screening",
]

