"""
Generadores de reportes editoriales y exportación para el análisis multivariante:
- MultivariateTableReport: Reporte interactivo en HTML (calidad de publicación) y generador de tablas en LaTeX (booktabs).
"""

from typing import Dict, List, Optional, Any, Union
import os
import pandas as pd
import numpy as np

from heavystats.html_utils import (
    BaseReport,
    render_html_table,
    wrap_html_container,
    format_html_str,
)


class MultivariateTableReport(BaseReport):
    """
    Reporte consolidado del análisis multivariante.
    
    Integra los resultados del Tamizaje Bivariante No Paramétrico, la Regresión LASSO
    y la Regresión por Mínimos Cuadrados Parciales (PLS-R) en una sola estructura analítica
    exportable a LaTeX (normativa FACYT / booktabs), HTML de publicación y Excel.
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
        self.subtitle = subtitle or f"Evaluación de determinantes biológicos sobre la concentración sérica de {target_name}"
        self.notes = notes if notes is not None else [
            f"Variable dependiente biológica modelada: {'ln(' + target_name + ')' if log_transformed else target_name}.",
            "Cribado bivariante ejecutado mediante contrastes no paramétricos de distribución libre (Mann-Whitney U, Kruskal-Wallis, Spearman) con control de tasa de falso descubrimiento (q-FDR Benjamini-Hochberg).",
            "Regresión LASSO optimizada mediante validación cruzada (k-fold CV) con penalización L1.",
            "Regresión PLS-R basada en componentes latentes ortogonales; factores con VIP ≥ 1.0 se declaran determinantes críticos de la carga xenobiótica.",
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

    def to_latex(
        self,
        filepath: Optional[str] = None,
        caption: Optional[str] = None,
        label: str = "tab:multivariante_factores",
        spanish_comma: bool = True
    ) -> str:
        """
        Genera el código LaTeX de la tabla con el paquete booktabs siguiendo rigurosamente
        el estándar editorial de FACYT (sin líneas verticales, encabezados spanner y tipografía matemática).
        """
        cap = caption or (
            f"Tamizaje bivariante no paramétrico y parámetros de los modelos multivariantes "
            f"(LASSO y PLS-R) para la concentración sérica de \\ce{{Hg_T}}."
        )

        def fmt_num(v: Any, dec: int = 3, sign: bool = False) -> str:
            if v is None or pd.isna(v) or v == "---":
                return "---"
            try:
                val = float(v)
                s = f"{val:+.{dec}f}" if sign else f"{val:.{dec}f}"
                if spanish_comma:
                    s = s.replace(".", "{,}")
                return s
            except Exception:
                return str(v)

        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("    \\centering")
        lines.append("    \\small")
        lines.append(f"    \\caption{{{cap}}}")
        lines.append(f"    \\label{{{label}}}")
        lines.append("    \\begin{tabular}{lcccccc}")
        lines.append("        \\toprule")
        lines.append("        & \\multicolumn{2}{c}{\\textbf{Cribado Bivariante}} & \\multicolumn{2}{c}{\\textbf{Regresión LASSO}} & \\multicolumn{2}{c}{\\textbf{Regresión PLS-R}} \\\\")
        lines.append("        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
        lines.append("        \\textbf{Determinante / Factor} & \\textbf{Estadístico} & \\textbf{Valor $p$} & $\\boldsymbol{\\beta}$ \\textbf{Estandarizado} & \\textbf{Efecto} & \\textbf{VIP} & \\textbf{Coef. PLS} \\\\")
        lines.append("        \\midrule")

        for _, row in self._df.iterrows():
            factor_name = str(row.get("Etiqueta", row.get("Variable", "")))
            stat_val = row.get("Estadístico")
            stat_name = row.get("Estadístico_Nombre", "")

            # Formatear el estadístico de forma matemática y tipográfica
            raw_stat = str(row.get("Estadístico_Str", "---"))
            if "ρ" in raw_stat or "rho" in raw_stat:
                val_num = fmt_num(stat_val, dec=3, sign=True)
                stat_tex = f"$\\rho_s = {val_num}$"
            elif raw_stat.startswith("U = "):
                val_num = fmt_num(stat_val, dec=1, sign=False)
                stat_tex = f"$U = {val_num}$"
            elif raw_stat.startswith("H = "):
                val_num = fmt_num(stat_val, dec=2, sign=False)
                stat_tex = f"$H = {val_num}$"
            else:
                stat_tex = raw_stat

            p_val = fmt_num(row.get("p_valor"), dec=3)
            
            beta_val = row.get("Beta_Estandarizado")
            if beta_val is not None and abs(float(beta_val)) < 1e-5:
                beta_lasso = "0{,}000" if spanish_comma else "0.000"
            else:
                beta_lasso = fmt_num(beta_val, dec=3, sign=True)

            efecto_lasso = str(row.get("Efecto_Lasso", "---"))
            efecto_lasso = efecto_lasso.replace(" (↑)", "").replace(" (↓)", "").replace(" (β = 0)", "")

            vip_val = fmt_num(row.get("VIP"), dec=2)
            pls_coef = fmt_num(row.get("Coef_PLS"), dec=3, sign=True)

            line_str = f"        {factor_name:<36} & {stat_tex:<14} & {p_val:<8} & {beta_lasso:<10} & {efecto_lasso:<12} & {vip_val:<8} & {pls_coef} \\\\"
            lines.append(line_str)

        lines.append("        \\bottomrule")
        lines.append("    \\end{tabular}")
        lines.append("    \\vspace{1.5mm}")
        
        target_clean = self.target_name.lower()
        if "mercurio" in target_clean or "hg" in target_clean:
            chem_target = "\\ce{Hg_T}"
        elif "plomo" in target_clean or "pb" in target_clean:
            chem_target = "\\ce{Pb}"
        elif "cadmio" in target_clean or "cd" in target_clean:
            chem_target = "\\ce{Cd}"
        else:
            chem_target = f"\\text{{{self.target_name}}}"

        target_tex = f"\\ln({chem_target})" if self.log_transformed else chem_target
        lines.append(f"    \\flushleft{{\\footnotesize \\textit{{Nota:}} Variable dependiente modelada: ${target_tex}$. "
                     "$\\beta$ de LASSO optimizado mediante validación cruzada ($k$-fold CV). "
                     "Factores con $\\text{VIP} \\ge 1{,}0$ se consideran determinantes altamente influyentes en el espacio latente de PLS-R. "
                     "Efecto: \\textit{Aporte} ($\\beta > 0$, incremento biológico) o \\textit{Atenuación} ($\\beta < 0$, reducción o factor protector).}")
        lines.append("\\end{table}")

        latex_code = "\n".join(lines)

        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(latex_code)

        return latex_code

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
            "Prioridad": "c"
        }

        html_code = render_html_table(
            df=display_df,
            title=self.title,
            subtitle=self.subtitle,
            notes=self.notes,
            spanners=spanners,
            column_alignments=alignments,
            full_page=full_page
        )

        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_code)

        return html_code

    def _repr_html_(self) -> str:
        """Renderizado HTML interactivo para Jupyter notebooks."""
        return self.to_html(full_page=False)

