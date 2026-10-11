"""
Generadores de reportes editoriales y exportación para el análisis multivariante:
- MultivariateTableReport: Reporte consolidado en HTML, CSV y JSON (FDR, LASSO, PLS-R).
- PlsVipReport: Reporte especializado en puntajes VIP de PLS-R con dominios y clasificación.
"""

from typing import Dict, List, Optional, Any, Union
import os
from pathlib import Path
import pandas as pd
import numpy as np

from heavystats.html_utils import (
    BaseReport,
    render_html_table,
    wrap_html_container,
    format_html_str,
)
# Re-exportar reportes de atípicos desde el módulo canónico heavystats.atipicos
from heavystats.atipicos import (
    UnivariateOutlierReport,
    OutlierDiagnosticReport,
    StratifiedContrastReport,
)


class MultivariateTableReport(BaseReport):
    """
    Reporte consolidado del análisis multivariante.
    Integra los resultados del Tamizaje Bivariante No Paramétrico, la Regresión LASSO
    y la Regresión por Mínimos Cuadrados Parciales (PLS-R) en una sola estructura analítica
    exportable a HTML (interactivo editorial), CSV y JSON estructurado.
    """

    def __init__(
        self,
        consolidated_df: pd.DataFrame,
        screening_df: Optional[pd.DataFrame] = None,
        lasso_summary_df: Optional[pd.DataFrame] = None,
        pls_summary_df: Optional[pd.DataFrame] = None,
        target_name: str = "Mercurio_ug_L",
        log_transformed: bool = True,
        title: str = "Tamizaje Bivariante y Modelado Multivariante Regularizado (LASSO / PLS-R)",
        subtitle: Optional[str] = None,
        notes: Optional[List[str]] = None,
        filepath: Optional[str] = None,
    ):
        self._df = consolidated_df.copy()
        self.screening_df = screening_df.copy() if screening_df is not None else None
        self.lasso_summary_df = lasso_summary_df.copy() if lasso_summary_df is not None else None
        self.pls_summary_df = pls_summary_df.copy() if pls_summary_df is not None else None
        self.target_name = target_name
        self.log_transformed = log_transformed
        self.title = title
        self.subtitle = subtitle or f"Evaluación de covariables de exposición frente a {target_name}"
        self.notes = notes if notes is not None else [
            f"Variable dependiente biológica modelada: {'ln(' + target_name + ')' if log_transformed else target_name}.",
            "Cribado bivariante ejecutado mediante contrastes no paramétricos de distribución libre con control de tasa de falso descubrimiento (q-FDR Benjamini-Hochberg).",
            "Regresión LASSO optimizada mediante validación cruzada con penalización L1.",
            "Regresión PLS-R basada en componentes latentes ortogonales; factores con VIP ≥ 1.0 se declaran determinantes críticos.",
            "Dirección del efecto: Aporte (↑, incremento de la concentración) o Atenuación (↓, efecto protector o de reducción)."
        ]

        if filepath:
            self.to_html(filepath)

    @property
    def df(self) -> pd.DataFrame:
        """Devuelve el DataFrame consolidado limpio."""
        return self._df.copy()

    def to_dataframe(self) -> pd.DataFrame:
        return self.df

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Genera el código HTML con calidad de publicación del reporte."""
        cols = ["Etiqueta", "Estadístico_Str", "p_valor", "q_fdr", "Beta_Estandarizado", "Efecto_Lasso", "VIP", "Coef_PLS", "Prioridad"]
        display_df = self._df[[c for c in cols if c in self._df.columns]].copy()

        # Renombrar columnas para la visualización
        display_df = display_df.rename(columns={
            "Etiqueta": "Determinante",
            "Estadístico_Str": "Estadístico Bivariado",
            "p_valor": "p (Crudo)",
            "q_fdr": "q (FDR)",
            "Beta_Estandarizado": "β LASSO",
            "Efecto_Lasso": "Efecto LASSO",
            "VIP": "VIP PLS-R",
            "Coef_PLS": "Coef. PLS",
            "Prioridad": "Prioridad"
        })

        spanners = [
            {"label": "Factores de Riesgo", "colspan": 1},
            {"label": "Tamizaje Bivariante", "colspan": 3},
            {"label": "Regresión LASSO", "colspan": 2},
            {"label": "Regresión PLS-R", "colspan": 2},
            {"label": "Clasificación", "colspan": 1},
        ]

        alignments = {
            "Determinante": "l",
            "Estadístico Bivariado": "c",
            "p (Crudo)": "r",
            "q (FDR)": "r",
            "β LASSO": "r",
            "Efecto LASSO": "c",
            "VIP PLS-R": "r",
            "Coef. PLS": "r",
            "Prioridad": "c",
        }

        html_code = render_html_table(
            df=display_df,
            title=self.title,
            subtitle=self.subtitle,
            notes=self.notes,
            column_alignments=alignments,
            spanners=spanners,
            full_page=full_page
        )

        if filepath:
            self._write_file(filepath, html_code)

        return html_code


DEFAULT_SURVEY_LABELS: Dict[str, str] = {
    "Alim_Pescados": "Consumo de Pescado / Atún",
    "Alim_Cereales": "Consumo de Cereales",
    "Alim_Leguminosas": "Consumo de Leguminosas",
    "Alim_Tuberculos": "Consumo de Tubérculos",
    "Alim_Carnes": "Consumo de Carnes",
    "Alim_Bebidas": "Consumo de Bebidas",
    "Alim_Huevos": "Consumo de Huevos",
    "Alim_Lacteos": "Consumo de Lácteos",
    "Alim_Frutas": "Consumo de Frutas",
    "Alim_Vegetales": "Consumo de Vegetales",
    "Alim_Azucar": "Consumo de Azúcar",
    "Alim_Grasas": "Consumo de Grasas",
    "Alim_Chocolate": "Consumo de Chocolate",
    "Salud_Fuma": "Exposición a Humo (Tabaco)",
    "Salud_Actividad": "Actividad Física Regular",
    "Salud_Bombillos": "Manipulación Bombillos Fluor.",
    "Salud_Joyeria": "Uso de Joyería / Bisutería",
    "Salud_Transporte_caminar": "Movilidad Peatonal (Caminar)",
    "Salud_Transporte_publico": "Transporte Público",
    "Salud_Transporte_vehiculo": "Vehículo Particular",
    "Salud_Agua_pozo_profundo": "Agua de Pozo Profundo",
    "Salud_Agua_filtrada": "Agua Filtrada",
    "Salud_Agua_mineral_embotellada": "Agua Mineral Embotellada",
    "Exposicion_Talleres_latoneria": "Taller de Latonería",
    "Exposicion_Talleres_mecanico": "Taller Mecánico",
    "Exposicion_Lugares_canales": "Cercanía a Canales",
    "Exposicion_Lugares_estacion_gasolina": "Estación de Gasolina",
    "Exposicion_Lugares_rectificadora_motores": "Rectificadora de Motores",
    "Exposicion_Lugares_rios": "Cercanía a Ríos",
    "Exposicion_Industrias_fabrica_metales": "Fábrica Metalmecánica",
    "Exposicion_Industrias_fabrica_productos_quimicos": "Industria Química",
    "Edad": "Edad (Años)",
    "Sexo": "Sexo",
    "IMC": "Índice de Masa Corporal (IMC)"
}


class PlsVipReport(BaseReport):
    """
    Reporte editorial de Importancia de Variables en la Proyección (VIP) y Coeficientes PLS-1.
    Diseñado para presentación interactiva en Jupyter Notebooks, exportación a HTML, CSV y JSON,
    con etiquetas descriptivas en español científico y clasificación de relevancia.
    """

    def __init__(
        self,
        df_vip: Optional[Union[pd.DataFrame, Any]] = None,
        pls_model: Optional[Any] = None,
        nipals_summary_df: Optional[pd.DataFrame] = None,
        r2_global: float = 0.0,
        rmse: float = 0.0,
        labels_map: Optional[Dict[str, str]] = None,
        target_name: Optional[str] = None,
        title: str = "Ranking de Importancia en la Proyección (VIP) y Parámetros PLS-1",
        subtitle: Optional[str] = None,
        filepath_html: Optional[str] = None,
        filepath_csv: Optional[str] = None,
        filepath_json: Optional[str] = None,
        filepath_tex: Optional[str] = None,
        **kwargs: Any,
    ):
        self.target_name = target_name or "Mercurio_ug_L"

        if pls_model is not None and hasattr(pls_model, "summary_df"):
            base_df = pls_model.summary_df.copy()
            if hasattr(pls_model, "r2_"):
                r2_global = float(pls_model.r2_)
        elif df_vip is not None and isinstance(df_vip, pd.DataFrame):
            base_df = df_vip.copy()
        elif nipals_summary_df is not None:
            base_df = nipals_summary_df.copy()
        else:
            base_df = pd.DataFrame(columns=["Variable", "VIP", "Beta_PLS", "Coef_PLS"])

        self._df = base_df.copy()

        # Coeficiente Beta PLS
        if "Beta_PLS" not in self._df.columns:
            if "Coef_PLS" in self._df.columns:
                self._df["Beta_PLS"] = self._df["Coef_PLS"]
            elif "Beta" in self._df.columns:
                self._df["Beta_PLS"] = self._df["Beta"]
            else:
                self._df["Beta_PLS"] = 0.0

        # Mapeo de etiquetas
        mapper = DEFAULT_SURVEY_LABELS.copy()
        if labels_map:
            mapper.update(labels_map)

        if "Etiqueta" not in self._df.columns and "Variable" in self._df.columns:
            self._df["Etiqueta"] = self._df["Variable"].map(mapper).fillna(self._df["Variable"])

        # Asignación de dominios
        if "Dominio" not in self._df.columns and "Variable" in self._df.columns:
            def _get_dom(v: str) -> str:
                s = str(v).lower()
                if s.startswith("alim_") or "dieta" in s or "pescado" in s or "atun" in s:
                    return "Hábitos Dietarios"
                elif s.startswith("salud_transporte") or "movilidad" in s:
                    return "Movilidad Urbana"
                elif s.startswith("salud_agua") or "agua" in s:
                    return "Fuentes Hídricas"
                elif s.startswith("salud_") or "fuma" in s:
                    return "Hogar y Estilo de Vida"
                elif s.startswith("exposicion_") or "taller" in s or "industria" in s:
                    return "Exposición Ambiental / Ocupacional"
                elif s in ("edad", "sexo", "imc"):
                    return "Sociodemográfico / Biometría"
                return "Otros Factores"
            self._df["Dominio"] = self._df["Variable"].apply(_get_dom)

        self.r2_global = float(r2_global)
        self.rmse = float(rmse)
        self.title = title
        self.subtitle = subtitle or f"Modelo PLS-1 evaluado (R² = {self.r2_global:.3f})"

        self.n_criticas = int((self._df["VIP"] >= 1.0).sum()) if "VIP" in self._df.columns else 0
        self.n_moderadas = int(((self._df["VIP"] >= 0.8) & (self._df["VIP"] < 1.0)).sum()) if "VIP" in self._df.columns else 0
        self.n_marginales = int((self._df["VIP"] < 0.8).sum()) if "VIP" in self._df.columns else 0

        top_row = self._df.iloc[0] if len(self._df) > 0 else None
        self.top_var = str(top_row.get("Etiqueta", top_row.get("Variable", "---"))) if top_row is not None else "---"
        self.top_vip = float(top_row["VIP"]) if top_row is not None and "VIP" in top_row else 0.0

        if filepath_html:
            self.to_html(filepath_html)
        if filepath_csv:
            self.to_csv(filepath_csv)
        if filepath_json:
            self.to_json(filepath_json)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def to_dataframe(self) -> pd.DataFrame:
        return self.df

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False, top_n: Optional[int] = None) -> str:
        """
        Genera reporte interactivo HTML con tarjetas KPI, callouts y formato tipográfico editorial.
        """
        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px;">
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #0284c7;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Bondad de Ajuste R²</div>
            <div style="font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px;">{self.r2_global * 100:.2f}%</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Varianza explicada PLS-1</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #dc2626;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Determinantes Críticos</div>
            <div style="font-size: 20px; font-weight: 700; color: #dc2626; margin-top: 4px;">{self.n_criticas} variables</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">VIP ≥ 1.0 (Impacto prioritario)</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #d97706;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Influencia Moderada</div>
            <div style="font-size: 20px; font-weight: 700; color: #d97706; margin-top: 4px;">{self.n_moderadas} variables</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">0.8 ≤ VIP &lt; 1.0</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #16a34a;">
            <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Factor Dominante</div>
            <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{self.top_var}">{self.top_var}</div>
            <div style="font-size: 11px; color: #16a34a; margin-top: 2px; font-weight: 600;">VIP Máximo = {self.top_vip:.3f}</div>
          </div>
        </div>
        """

        df_display = self._df.head(top_n).copy() if top_n is not None else self._df.copy()

        # Generar badges de clasificación
        def make_badge(v: float) -> str:
            if v >= 1.0:
                return "<span style='background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;'>Crítica (VIP ≥ 1.0)</span>"
            elif v >= 0.8:
                return "<span style='background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600;'>Moderada</span>"
            else:
                return "<span style='background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:12px; font-size:11px;'>Marginal</span>"

        def make_effect(b: float) -> str:
            if b > 0:
                return f"<span style='color:#b91c1c; font-weight:600;'>+{b:.3f} (Aporte ↑)</span>"
            else:
                return f"<span style='color:#15803d; font-weight:600;'>{b:.3f} (Atenuación ↓)</span>"

        df_table = pd.DataFrame()
        df_table["Determinante"] = df_display["Etiqueta"]
        df_table["Dominio"] = df_display["Dominio"]
        df_table["Puntaje VIP"] = df_display["VIP"].apply(lambda x: f"{float(x):.3f}")
        df_table["Efecto (β PLS)"] = df_display["Beta_PLS"].apply(make_effect)
        df_table["Clasificación"] = df_display["VIP"].apply(make_badge)

        table_html = render_html_table(
            df=df_table,
            column_alignments={"Determinante": "l", "Dominio": "l", "Puntaje VIP": "r", "Efecto (β PLS)": "c", "Clasificación": "c"},
            full_page=False
        )

        inner = kpi_html + table_html
        wrapped = wrap_html_container(
            inner_html=inner,
            title=self.title,
            subtitle=self.subtitle,
            notes=[
                "Predictores directos de la encuesta estandarizados por auto-escalado (Z-score).",
                f"Determinantes con VIP ≥ 1.0 representan factores de influencia primaria sobre la carga biológica.",
                f"El factor de mayor aporte es {self.top_var} con VIP = {self.top_vip:.3f}."
            ],
            full_page=full_page
        )

        if filepath:
            self._write_file(filepath, wrapped)

        return wrapped
