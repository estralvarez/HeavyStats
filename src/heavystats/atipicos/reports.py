"""
Reportes editoriales en HTML, CSV y JSON para diagnóstico de valores atípicos y quimiometría.
Consistentes con la arquitectura de HeavyStats y las normativas analíticas de la UC / FACYT.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats

from heavystats.html_utils import (
    render_html_table,
    wrap_html_container,
    format_html_str,
    BaseReport,
)


class UnivariateOutlierReport(BaseReport):
    """
    Reporte editorial univariante de consistencia analítica y detección de valores atípicos.
    Proporciona exportación a HTML interactivo, CSV y JSON estructurado.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series, List[float]],
        sample_ids: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
        target_name: str = "Mercurio_ug_L",
        who_threshold: Optional[float] = 5.0,
        epa_threshold: Optional[float] = 5.8,
        cdc_threshold: Optional[float] = None,
        alpha: float = 0.05,
        title: str = "Diagnóstico Univariante de Valores Atípicos y Consistencia Analítica",
        subtitle: Optional[str] = None,
        filepath_html: Optional[str] = None,
        filepath_csv: Optional[str] = None,
        filepath_json: Optional[str] = None,
        filepath_tex: Optional[str] = None,
    ):
        if hasattr(y, "y"):
            diag = y
            self.y = np.asarray(diag.y, dtype=float)
            self.sample_ids = getattr(diag, "sample_ids", np.arange(1, len(self.y) + 1))
            self.target_name = getattr(diag, "target_col", target_name)
            self.who_threshold = getattr(diag, "who_threshold", who_threshold)
            self.epa_threshold = getattr(diag, "epa_threshold", epa_threshold)
            self.cdc_threshold = getattr(diag, "cdc_threshold", cdc_threshold)
        else:
            self.y = np.asarray(y, dtype=float)
            self.sample_ids = np.asarray(sample_ids) if sample_ids is not None else np.arange(1, len(self.y) + 1)
            self.target_name = target_name
            self.who_threshold = who_threshold
            self.epa_threshold = epa_threshold
            self.cdc_threshold = cdc_threshold

        self.alpha = float(alpha)


        self.n_obs = len(self.y)
        if self.n_obs > 0:
            self.mean = float(np.mean(self.y))
            self.std = float(np.std(self.y, ddof=1)) if self.n_obs > 1 else 0.0
            self.median = float(np.median(self.y))
            self.q25, self.q75 = np.percentile(self.y, [25, 75])
            self.iqr = float(self.q75 - self.q25)
            self.min_val = float(np.min(self.y))
            self.max_val = float(np.max(self.y))

            # IC 95%
            se_mean = self.std / np.sqrt(self.n_obs) if self.n_obs > 1 else 0.0
            t_ci = stats.t.ppf(1 - alpha / 2, df=max(1, self.n_obs - 1))
            self.ci_lower = self.mean - t_ci * se_mean
            self.ci_upper = self.mean + t_ci * se_mean

            # Vallas de Tukey
            self.tukey_lower = self.q25 - 1.5 * self.iqr
            self.tukey_upper = self.q75 + 1.5 * self.iqr

            # Test de Grubbs exacto
            if self.std > 0 and self.n_obs >= 3:
                self.g_max = float((self.max_val - self.mean) / self.std)
                self.g_min = float((self.mean - self.min_val) / self.std)
                t_crit = stats.t.ppf(1 - alpha / (2 * self.n_obs), df=self.n_obs - 2)
                self.g_crit = float(((self.n_obs - 1) / np.sqrt(self.n_obs)) * np.sqrt(t_crit**2 / (self.n_obs - 2 + t_crit**2)))
            else:
                self.g_max = 0.0
                self.g_min = 0.0
                self.g_crit = 0.0

            self.excede_who = int(np.sum(self.y > self.who_threshold)) if self.who_threshold else 0
            self.excede_epa = int(np.sum(self.y > self.epa_threshold)) if self.epa_threshold else 0
            self.excede_cdc = int(np.sum(self.y > self.cdc_threshold)) if self.cdc_threshold else 0
        else:
            self.mean = self.std = self.median = self.q25 = self.q75 = self.iqr = 0.0
            self.min_val = self.max_val = self.ci_lower = self.ci_upper = self.tukey_lower = self.tukey_upper = 0.0
            self.g_max = self.g_min = self.g_crit = 0.0
            self.excede_who = self.excede_epa = self.excede_cdc = 0

        self.title = title
        self.subtitle = subtitle or f"Evaluación de consistencia analítica y detección de atípicos para {self.target_name} (N = {self.n_obs})"

        if filepath_html:
            self.to_html(filepath_html)
        if filepath_csv:
            self.to_csv(filepath_csv)
        if filepath_json:
            self.to_json(filepath_json)

    def to_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "Categoría": "Estadísticos Paramétricos",
                "Parámetro": "Media ± Desviación Estándar",
                "Método": "Paramétrico (Distribución Normal)",
                "Valor": f"{self.mean:.2f} ± {self.std:.2f}",
                "Unidad": "µg/L",
                "Criterio / Límite": f"IC 95%: [{self.ci_lower:.2f}, {self.ci_upper:.2f}]",
            },
            {
                "Categoría": "Estadísticos No Paramétricos",
                "Parámetro": "Mediana (Rango Intercuartílico)",
                "Método": "No Paramétrico (Robusto)",
                "Valor": f"{self.median:.2f} ({self.iqr:.2f})",
                "Unidad": "µg/L",
                "Criterio / Límite": f"Q1 = {self.q25:.2f}, Q3 = {self.q75:.2f}",
            },
            {
                "Categoría": "Estadísticos No Paramétricos",
                "Parámetro": "Rango Muestral (Mín - Máx)",
                "Método": "No Paramétrico (Extremos)",
                "Valor": f"{self.min_val:.2f} - {self.max_val:.2f}",
                "Unidad": "µg/L",
                "Criterio / Límite": "Observaciones empíricas mín/máx",
            },
            {
                "Categoría": "Detección de Atípicos",
                "Parámetro": "Vallas de Tukey (1.5 × IQR)",
                "Método": "No Paramétrico (Vallas de Tukey)",
                "Valor": f"[{self.tukey_lower:.2f}, {self.tukey_upper:.2f}]",
                "Unidad": "µg/L",
                "Criterio / Límite": f"Valla Sup. = {self.tukey_upper:.2f} (Máx: {self.max_val:.2f})",
            },
            {
                "Categoría": "Detección de Atípicos",
                "Parámetro": "Test de Grubbs Máximo (G_calc)",
                "Método": "Paramétrico (Test de Grubbs bilateral)",
                "Valor": f"G = {self.g_max:.3f}",
                "Unidad": "Adimensional",
                "Criterio / Límite": f"G_crit (alpha={self.alpha:.2f}) = {self.g_crit:.3f} ({'Atípico detectado' if self.g_max >= self.g_crit else 'Conforme (Sin atípicos)'})",
            },
            {
                "Categoría": "Detección de Atípicos",
                "Parámetro": "Test de Grubbs Mínimo (G_calc)",
                "Método": "Paramétrico (Test de Grubbs bilateral)",
                "Valor": f"G = {self.g_min:.3f}",
                "Unidad": "Adimensional",
                "Criterio / Límite": f"G_crit (alpha={self.alpha:.2f}) = {self.g_crit:.3f} ({'Atípico detectado' if self.g_min >= self.g_crit else 'Conforme (Sin atípicos)'})",
            },
        ]
        if self.who_threshold:
            data.append({
                "Categoría": "Guías Toxicológicas",
                "Parámetro": "Umbral OMS Población No Expuesta",
                "Método": "Normativa Internacional (OMS)",
                "Valor": f"{self.who_threshold:.2f}",
                "Unidad": "µg/L",
                "Criterio / Límite": f"Superan: {self.excede_who} / {self.n_obs} ({self.excede_who/self.n_obs*100:.1f}%)" if self.n_obs else "---",
            })
        if self.epa_threshold:
            data.append({
                "Categoría": "Guías Toxicológicas",
                "Parámetro": "Dosis de Referencia Guía US EPA",
                "Método": "Normativa Internacional (US EPA)",
                "Valor": f"{self.epa_threshold:.2f}",
                "Unidad": "µg/L",
                "Criterio / Límite": f"Superan: {self.excede_epa} / {self.n_obs} ({self.excede_epa/self.n_obs*100:.1f}%)" if self.n_obs else "---",
            })
        if self.cdc_threshold:
            data.append({
                "Categoría": "Guías Toxicológicas",
                "Parámetro": "Nivel de Referencia CDC",
                "Método": "Normativa Internacional (CDC)",
                "Valor": f"{self.cdc_threshold:.2f}",
                "Unidad": "µg/dL",
                "Criterio / Límite": f"Superan: {self.excede_cdc} / {self.n_obs} ({self.excede_cdc/self.n_obs*100:.1f}%)" if self.n_obs else "---",
            })
        return pd.DataFrame(data)

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 20px;">
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #0284c7;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Media ± Desv. Estándar</div>
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px;">{self.mean:.2f} ± {self.std:.2f} µg/L</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">IC 95%: [{self.ci_lower:.2f}, {self.ci_upper:.2f}]</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #3b82f6;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Mediana (IQR)</div>
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px;">{self.median:.2f} ({self.iqr:.2f}) µg/L</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Q1: {self.q25:.2f} | Q3: {self.q75:.2f}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #8b5cf6;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Valla Superior Tukey</div>
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px;">{self.tukey_upper:.2f} µg/L</div>
            <div style="font-size: 11px; color: {'#16a34a' if self.max_val <= self.tukey_upper else '#dc2626'}; margin-top: 2px;">Máx: {self.max_val:.2f} ({'Conforme' if self.max_val <= self.tukey_upper else 'Atípico'})</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #10b981;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Test de Grubbs</div>
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px;">G = {self.g_max:.3f}</div>
            <div style="font-size: 11px; color: {'#16a34a' if self.g_max < self.g_crit else '#dc2626'}; margin-top: 2px;">G_crit: {self.g_crit:.3f} ({'Sin atípicos' if self.g_max < self.g_crit else 'Atípico detectado'})</div>
          </div>
        </div>
        """
        spanners = [
            {"label": "Clasificación y Metodología", "columns": ["Categoría", "Parámetro", "Método"]},
            {"label": "Resultado Numérico", "columns": ["Valor", "Unidad"]},
            {"label": "Criterio de Decisión / Umbral", "columns": ["Criterio / Límite"]},
        ]
        table_html = render_html_table(
            df=self.to_dataframe(),
            column_alignments={
                "Categoría": "l",
                "Parámetro": "l",
                "Método": "l",
                "Valor": "r",
                "Unidad": "c",
                "Criterio / Límite": "l",
            },
            spanners=spanners,
            full_page=False,
        )
        inner = kpi_html + table_html
        wrapped = wrap_html_container(
            inner_html=inner,
            title=self.title,
            subtitle=self.subtitle,
            notes=[
                "Determinaciones instrumentales cuantitativas por Espectrometría de Absorción Atómica con Horno de Grafito (GFAAS).",
                f"Vallas de Tukey evaluadas a 1.5 × IQR. Test de Grubbs evaluado a dos colas con significancia crítica α = {self.alpha}."
            ],
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, wrapped)
        return wrapped


class OutlierDiagnosticReport(BaseReport):
    """
    Reporte quimiométrico multivariante de apalancamiento, residuos y valores atípicos.
    Exporta a HTML interactivo de calidad editorial, CSV y JSON estructurado.
    """

    def __init__(
        self,
        df_diagnostics: Union[pd.DataFrame, Any],
        target_name: str = "Mercurio_ug_L",
        T2_limit: float = 0.0,
        leverage_limit: float = 0.0,
        cooks_threshold: float = 0.0,
        title: str = "Diagnóstico Quimiométrico Multivariante de Valores Atípicos e Influencia",
        subtitle: Optional[str] = None,
        filepath_html: Optional[str] = None,
        filepath_csv: Optional[str] = None,
        filepath_json: Optional[str] = None,
        filepath_tex: Optional[str] = None,
    ):
        if hasattr(df_diagnostics, "to_dataframe"):
            diag = df_diagnostics
            self._df = diag.to_dataframe()
            self.target_name = getattr(diag, "target_col", target_name)
            self.T2_limit = float(getattr(diag, "T2_limit", T2_limit))
            self.leverage_limit = float(getattr(diag, "leverage_limit", leverage_limit))
            self.cooks_threshold = float(getattr(diag, "cooks_threshold", cooks_threshold))
        else:
            self._df = df_diagnostics.copy()
            self.target_name = target_name
            self.T2_limit = float(T2_limit)
            self.leverage_limit = float(leverage_limit)
            self.cooks_threshold = float(cooks_threshold)
        self.n_obs = len(self._df)
        self.title = title
        self.subtitle = subtitle or f"Evaluación de influencia en espacio latente PLS-1 para {self.target_name} (N = {self.n_obs})"

        if filepath_html:
            self.to_html(filepath_html)
        if filepath_csv:
            self.to_csv(filepath_csv)
        if filepath_json:
            self.to_json(filepath_json)

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame de diagnóstico con columnas originales y precisión estandarizada."""
        return self._df.copy()

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera una tabla HTML editorial de alta calidad con encabezados formalizados,
        agrupadores (spanners), redondeo analítico riguroso y distintivos (badges) que
        resaltan de forma inequívoca las observaciones atípicas e influyentes.
        """
        df = self._df.copy()

        # KPIs superiores
        n_flagged = len(df[df["Diagnostico"] != "Homogéneo"]) if "Diagnostico" in df.columns else 0
        n_high_lev = len(df[df["Leverage"] > self.leverage_limit]) if "Leverage" in df.columns and self.leverage_limit > 0 else 0
        n_cook = len(df[df["Cooks_D"] > self.cooks_threshold]) if "Cooks_D" in df.columns and self.cooks_threshold > 0 else 0
        n_t2 = len(df[df["Hotelling_T2"] > self.T2_limit]) if "Hotelling_T2" in df.columns and self.T2_limit > 0 else 0

        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 20px;">
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; border-left: 4px solid #0284c7;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Apalancamiento Crítico (h*)</div>
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 3px;">2p/n = {self.leverage_limit:.3f}</div>
            <div style="font-size: 11px; color: {'#16a34a' if n_high_lev == 0 else '#b45309'}; margin-top: 2px;">{n_high_lev} muestra(s) superan límite</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; border-left: 4px solid #d97706;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Umbral Distancia Cook (D*)</div>
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 3px;">4/n = {self.cooks_threshold:.3f}</div>
            <div style="font-size: 11px; color: {'#16a34a' if n_cook == 0 else '#b91c1c'}; margin-top: 2px;">{n_cook} observación(es) influyentes</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; border-left: 4px solid #8b5cf6;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Límite Hotelling T² (F_95%)</div>
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 3px;">T²_lim = {self.T2_limit:.2f}</div>
            <div style="font-size: 11px; color: {'#16a34a' if n_t2 == 0 else '#b91c1c'}; margin-top: 2px;">{n_t2} fuera de elipse 95%</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; border-left: 4px solid {'#10b981' if n_flagged == 0 else '#ef4444'};">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Diagnóstico Integral</div>
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 3px;">{self.n_obs - n_flagged} / {self.n_obs} Conformes</div>
            <div style="font-size: 11px; color: {'#16a34a' if n_flagged == 0 else '#b91c1c'}; margin-top: 2px;">{n_flagged} caso(s) diferenciado(s)</div>
          </div>
        </div>
        """

        # Construcción manual de la tabla para lograr estilo editorial y badges de colores
        table_html = []
        table_html.append("<table class='hs-pub-table' style='width: 100%; font-size: 13px;'>")
        table_html.append("  <thead>")
        
        # Fila de Spanners Superiores
        table_html.append("    <tr style='background: #f8fafc;'>")
        table_html.append("      <th colspan='2' class='hs-center-col' style='border-bottom: 2px solid #0f172a; font-weight: 700; color: #0f172a;'>Sujeto y Biomarcador</th>")
        table_html.append("      <th colspan='3' class='hs-center-col' style='border-bottom: 2px solid #0f172a; font-weight: 700; color: #0f172a;'>Detección Univariante</th>")
        table_html.append("      <th colspan='2' class='hs-center-col' style='border-bottom: 2px solid #0f172a; font-weight: 700; color: #0f172a;'>Espacio Latente (PLS-1)</th>")
        table_html.append("      <th colspan='5' class='hs-center-col' style='border-bottom: 2px solid #0f172a; font-weight: 700; color: #0f172a;'>Influencia Multivariante y Residuos</th>")
        table_html.append("      <th class='hs-center-col' style='border-bottom: 2px solid #0f172a; font-weight: 700; color: #0f172a;'>Clasificación</th>")
        table_html.append("    </tr>")

        # Fila de Encabezados de Columna
        table_html.append("    <tr>")
        table_html.append("      <th class='hs-center-col'>ID Muestra</th>")
        table_html.append("      <th class='hs-num-col'>Conc. [µg/L]</th>")
        table_html.append("      <th class='hs-num-col'>Z-Score</th>")
        table_html.append("      <th class='hs-num-col'>Z-Modif.</th>")
        table_html.append("      <th class='hs-center-col'>Valla Tukey</th>")
        table_html.append("      <th class='hs-num-col'>Puntuación t₁</th>")
        table_html.append("      <th class='hs-num-col'>Puntuación t₂</th>")
        table_html.append("      <th class='hs-num-col'>Hotelling T²</th>")
        table_html.append("      <th class='hs-num-col'>Residuo Q</th>")
        table_html.append("      <th class='hs-num-col'>Leverage (hᵢᵢ)</th>")
        table_html.append("      <th class='hs-num-col'>Dist. Cook (Dᵢ)</th>")
        table_html.append("      <th class='hs-num-col'>Res. Estudent.</th>")
        table_html.append("      <th class='hs-center-col'>Diagnóstico Quimiométrico</th>")
        table_html.append("    </tr>")
        table_html.append("  </thead>")
        table_html.append("  <tbody>")

        for _, row in df.iterrows():
            sid = str(row.get("Muestra", ""))
            y_conc = float(row.get("Concentracion", 0.0))
            z_val = float(row.get("Z_Score", 0.0))
            mod_z = float(row.get("Modified_Z", 0.0))
            tukey = str(row.get("Tukey_Outlier", "Normal"))
            t1 = float(row.get("T1_Score", 0.0))
            t2 = float(row.get("T2_Score", 0.0))
            t2_hot = float(row.get("Hotelling_T2", 0.0))
            q_res = float(row.get("Q_Residual", 0.0))
            lev = float(row.get("Leverage", 0.0))
            ck = float(row.get("Cooks_D", 0.0))
            r_stud = float(row.get("Residual_Estudentizado", 0.0))
            diag_txt = str(row.get("Diagnostico", "Homogéneo"))

            # Determinar estilo de fila según severidad de atipicidad
            row_style = ""
            if diag_txt == "Punto Palanca Extremo":
                row_style = " style='background-color: #fef2f2 !important;'"
            elif diag_txt in ("Alto Apalancamiento", "Observación Influyente"):
                row_style = " style='background-color: #fffbeb !important;'"
            elif diag_txt == "Atípico Multivariante T²":
                row_style = " style='background-color: #fdf2f8 !important;'"

            # Formatear celdas con resaltado si exceden umbral
            # Hotelling T2
            t2_style = "font-weight: 700; color: #b91c1c;" if (self.T2_limit > 0 and t2_hot > self.T2_limit) else ""
            t2_cell = f"<span style='{t2_style}'>{t2_hot:.2f}{'*' if t2_style else ''}</span>"

            # Leverage
            lev_style = "font-weight: 700; color: #b45309;" if (self.leverage_limit > 0 and lev > self.leverage_limit) else ""
            lev_cell = f"<span style='{lev_style}'>{lev:.3f}{'*' if lev_style else ''}</span>"

            # Cook's D
            ck_style = "font-weight: 700; color: #b91c1c;" if (self.cooks_threshold > 0 and ck > self.cooks_threshold) else ""
            ck_cell = f"<span style='{ck_style}'>{ck:.3f}{'*' if ck_style else ''}</span>"

            # Badge Tukey
            if "Atípico" in tukey:
                tukey_badge = "<span class='hs-badge-danger'>Atípico</span>"
            else:
                tukey_badge = "<span class='hs-badge-ok'>Normal</span>"

            # Badge Diagnóstico
            if diag_txt == "Punto Palanca Extremo":
                diag_badge = "<span class='hs-badge-danger' style='font-size: 11px;'>Punto Palanca Extremo</span>"
            elif diag_txt == "Alto Apalancamiento":
                diag_badge = "<span class='hs-badge-warn' style='font-size: 11px;'>Alto Apalancamiento</span>"
            elif diag_txt == "Observación Influyente":
                diag_badge = "<span class='hs-badge-warn' style='background-color: #ffedd5 !important; color: #c2410c !important; border-color: #fed7aa; font-size: 11px;'>Observación Influyente</span>"
            elif diag_txt == "Atípico Multivariante T²":
                diag_badge = "<span class='hs-badge-danger' style='background-color: #fce7f3 !important; color: #9d174d !important; border-color: #fbcfe8; font-size: 11px;'>Atípico T²</span>"
            else:
                diag_badge = "<span class='hs-badge-ok' style='font-size: 11px;'>Homogéneo</span>"

            table_html.append(f"    <tr{row_style}>")
            table_html.append(f"      <td class='hs-center-col' style='font-weight: 600;'>{sid}</td>")
            table_html.append(f"      <td class='hs-num-col'>{y_conc:.2f}</td>")
            table_html.append(f"      <td class='hs-num-col'>{z_val:+.2f}</td>")
            table_html.append(f"      <td class='hs-num-col'>{mod_z:+.2f}</td>")
            table_html.append(f"      <td class='hs-center-col'>{tukey_badge}</td>")
            table_html.append(f"      <td class='hs-num-col'>{t1:+.2f}</td>")
            table_html.append(f"      <td class='hs-num-col'>{t2:+.2f}</td>")
            table_html.append(f"      <td class='hs-num-col'>{t2_cell}</td>")
            table_html.append(f"      <td class='hs-num-col'>{q_res:.2f}</td>")
            table_html.append(f"      <td class='hs-num-col'>{lev_cell}</td>")
            table_html.append(f"      <td class='hs-num-col'>{ck_cell}</td>")
            table_html.append(f"      <td class='hs-num-col'>{r_stud:+.2f}</td>")
            table_html.append(f"      <td class='hs-center-col'>{diag_badge}</td>")
            table_html.append("    </tr>")

        table_html.append("  </tbody>")
        table_html.append("</table>")

        inner = kpi_html + "\n".join(table_html)
        wrapped = wrap_html_container(
            inner_html=inner,
            title=self.title,
            subtitle=self.subtitle,
            notes=[
                f"Hotelling T² evaluado con distribución F de Snedecor al 95% de confianza (Límite crítico = {self.T2_limit:.2f}).",
                f"Límite de apalancamiento crítico: h* = 2p/n = {self.leverage_limit:.3f}. Valores con asterisco (*) superan el umbral.",
                f"Umbral de distancia de Cook: D* = 4/n = {self.cooks_threshold:.3f}. Valores con asterisco (*) indican influencia sobre el modelo.",
                "Los sujetos clasificados como no homogéneos representan variabilidad sociodemográfica genuina y deben retenerse en el análisis."
            ],
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, wrapped)
        return wrapped


class StratifiedContrastReport(BaseReport):
    """
    Reporte de contraste bivariante estratificado por niveles de concentración de biomarcadores.
    Exporta a HTML interactivo, CSV y JSON estructurado.
    """

    def __init__(
        self,
        df_merged: pd.DataFrame,
        target_col: str = "Mercurio_ug_L",
        strata_col: str = "Estrato_Hg",
        title: str = "Contraste Bivariante de Determinantes por Estratos de Concentración",
        subtitle: Optional[str] = None,
        filepath_html: Optional[str] = None,
        filepath_csv: Optional[str] = None,
        filepath_json: Optional[str] = None,
        filepath_tex: Optional[str] = None,
    ):
        self.df_merged = df_merged.copy()
        self.target_col = target_col
        self.strata_col = strata_col
        self.title = title
        self.subtitle = subtitle or f"Contraste de determinantes por estrato de {target_col} (N = {len(df_merged)})"

        self._build_tables()

        if filepath_html:
            self.to_html(filepath_html)
        if filepath_csv:
            self.to_csv(filepath_csv)
        if filepath_json:
            self.to_json(filepath_json)

    def _build_tables(self):
        def _to_clean_numeric(s):
            if not pd.api.types.is_numeric_dtype(s):
                s_clean = s.astype(str).str.strip().str.upper()
                s_mapped = s_clean.map({"SI": 1.0, "SÍ": 1.0, "TRUE": 1.0, "1": 1.0, "NO": 0.0, "FALSE": 0.0, "0": 0.0})
                return s_mapped.fillna(0.0).values.astype(float)
            return pd.to_numeric(s, errors="coerce").fillna(0.0).values.astype(float)

        target_vals = _to_clean_numeric(self.df_merged[self.target_col])

        est_sup = self.df_merged[self.df_merged[self.strata_col].str.contains("Superior|Alto|Max", case=False, na=False)]
        est_cen = self.df_merged[self.df_merged[self.strata_col].str.contains("Central|Medio|Prom", case=False, na=False)]
        est_inf = self.df_merged[self.df_merged[self.strata_col].str.contains("Inferior|Bajo|Min", case=False, na=False)]

        self.n_sup = len(est_sup) if len(est_sup) > 0 else 1
        self.n_cen = len(est_cen) if len(est_cen) > 0 else 1
        self.n_inf = len(est_inf) if len(est_inf) > 0 else 1

        factors = [
            ("Pescado Fresco (Alim_Pescados)", "Alim_Pescados"),
            ("Atún Enlatado (Alim_Atun)", "Alim_Atun"),
            ("Sardinas Enlatadas (Alim_Sardinas)", "Alim_Sardinas"),
            ("Proximidad a Talleres Mecánicos", "Exposicion_Talleres_Mecanicos"),
            ("Uso de Agua de Pozo / Subterránea", "Salud_Agua_Pozo"),
            ("Consumo de Agua Filtrada en Hogar", "Salud_Agua_Filtro"),
            ("Transporte Público Frecuente", "Salud_Transporte_Publico"),
            ("Presencia de Fumadores en el Hogar", "Salud_Fuma_Hogar"),
        ]

        contraste_rows = []
        corr_rows = []

        for label, col in factors:
            if col in self.df_merged.columns:
                s = self.df_merged[col]
                s_num = _to_clean_numeric(s)

                v_inf = _to_clean_numeric(est_inf[col]) if len(est_inf) > 0 else np.array([0.0])
                v_cen = _to_clean_numeric(est_cen[col]) if len(est_cen) > 0 else np.array([0.0])
                v_sup = _to_clean_numeric(est_sup[col]) if len(est_sup) > 0 else np.array([0.0])

                if col.startswith("Alim_"):
                    pct_inf = f"{np.mean(v_inf):.2f}"
                    pct_cen = f"{np.mean(v_cen):.2f}"
                    pct_sup = f"{np.mean(v_sup):.2f}"
                else:
                    pct_inf = f"{np.mean(v_inf) * 100:.1f}%"
                    pct_cen = f"{np.mean(v_cen) * 100:.1f}%"
                    pct_sup = f"{np.mean(v_sup) * 100:.1f}%"

                if len(s_num) > 3 and np.std(s_num) > 0 and np.std(target_vals) > 0:
                    rho, p_val = stats.spearmanr(s_num, target_vals)
                    rho_str = f"{rho:+.3f} (p={p_val:.3f})"
                else:
                    rho, p_val, rho_str = 0.0, 1.0, "---"

                contraste_rows.append({
                    "Variable / Factor": label,
                    "Inferior": pct_inf,
                    "Central": pct_cen,
                    "Superior": pct_sup,
                    "Spearman_rs": rho_str
                })
                corr_rows.append({
                    "Variable": label,
                    "Spearman_rho": float(rho),
                    "p_valor": float(p_val)
                })

        self.df_contraste = pd.DataFrame(contraste_rows)
        self.df_corr = pd.DataFrame(corr_rows).sort_values(by="p_valor").reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return self.df_contraste.copy()

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        t1 = render_html_table(self.df_contraste, full_page=False)
        t2 = render_html_table(self.df_corr, full_page=False)
        inner = f"<h3>1. Contraste por Estratos</h3>\n{t1}\n<h3>2. Correlaciones</h3>\n{t2}"
        wrapped = wrap_html_container(inner_html=inner, title=self.title, subtitle=self.subtitle, full_page=full_page)
        if filepath:
            self._write_file(filepath, wrapped)
        return wrapped
