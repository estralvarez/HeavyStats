import sys
import pathlib
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from heavystats.cleaning import load_data, standardize_boolean_columns

def calculate_numerical_smd(g1, g2):
    mean1, mean2 = g1.mean(), g2.mean()
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_sd = np.sqrt((var1 + var2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (mean1 - mean2) / pooled_sd

def calculate_binary_smd(p1, p2):
    variance = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    if variance == 0:
        return 0.0
    return (p1 - p2) / np.sqrt(variance)

from heavystats.html_utils import wrap_html_container, format_html_str, BaseReport


class ComparationReport(BaseReport):
    """
    Reporte de comparación de seleccionados vs no seleccionados.
    Permite exportar a texto plano, HTML de calidad de publicación (estilo Booktabs),
    y exportación a DataFrames y Excel.
    """
    def __init__(
        self,
        num_results: List[Dict[str, Any]],
        cat_results: List[Dict[str, Any]],
        risk_results: List[Dict[str, Any]],
        bias_vars: List[Tuple[str, float]],
        variability_warnings: List[str],
        n_selected: int,
        n_non_selected: int,
        n_total: int
    ):
        self.num_results = num_results
        self.cat_results = cat_results
        self.risk_results = risk_results
        self.bias_vars = bias_vars
        self.variability_warnings = variability_warnings
        self.n_selected = n_selected
        self.n_non_selected = n_non_selected
        self.n_total = n_total

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Devuelve un diccionario con los 4 DataFrames correspondientes a cada sección del análisis."""
        # 1. Numéricas
        df_num = pd.DataFrame([
            {
                "Variable": r["variable"],
                f"Seleccionados (n={self.n_selected}) Media": r["mean_sel"],
                f"Seleccionados (n={self.n_selected}) DE": r["sd_sel"],
                f"No Seleccionados (n={self.n_non_selected}) Media": r["mean_non"],
                f"No Seleccionados (n={self.n_non_selected}) DE": r["sd_non"],
                "SMD": r["smd"],
                "p_value": r["p_value"]
            }
            for r in self.num_results
        ])

        # 2. Categóricas
        cat_rows = []
        for r in self.cat_results:
            for lvl in r["levels"]:
                cat_rows.append({
                    "Variable": r["variable"],
                    "Categoría": lvl["level"],
                    f"Seleccionados (n={self.n_selected}) N": lvl["c_sel"],
                    f"Seleccionados (n={self.n_selected}) %": round(lvl["p_sel"] * 100, 2),
                    f"No Seleccionados (n={self.n_non_selected}) N": lvl["c_non"],
                    f"No Seleccionados (n={self.n_non_selected}) %": round(lvl["p_non"] * 100, 2),
                    "SMD": lvl["smd"],
                    "p_value_global": r["p_value"]
                })
        df_cat = pd.DataFrame(cat_rows)

        # 3. Variabilidad
        df_risk = pd.DataFrame([
            {
                "Variable de Riesgo": r["variable"],
                "Cantidad SI": r["si_cnt"],
                "Cantidad NO": r["no_cnt"],
                "Tiene Variabilidad": "Sí" if r["has_variability"] else "No",
                "Estado": r["status"]
            }
            for r in self.risk_results
        ])

        # 4. Sesgo
        df_bias = pd.DataFrame([
            {
                "Variable": var,
                "SMD Máximo": round(val, 3),
                "Nivel de Desequilibrio": "Importante (> 0.25)" if val > 0.25 else "Leve (> 0.10)"
            }
            for var, val in sorted(self.bias_vars, key=lambda x: x[1], reverse=True)
        ])

        return {
            "numericas": df_num,
            "categoricas": df_cat,
            "variabilidad": df_risk,
            "sesgo": df_bias
        }

    def to_excel(self, filepath: str, **kwargs: Any) -> None:
        """Exporta todas las tablas del reporte a un libro de Excel (.xlsx) con pestañas individuales."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        dfs = self.to_dataframes()
        with pd.ExcelWriter(filepath, **kwargs) as writer:
            dfs["numericas"].to_excel(writer, sheet_name="Numericas", index=False)
            dfs["categoricas"].to_excel(writer, sheet_name="Categoricas", index=False)
            dfs["variabilidad"].to_excel(writer, sheet_name="Variabilidad", index=False)
            dfs["sesgo"].to_excel(writer, sheet_name="Sesgo_Seleccion", index=False)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Genera el reporte en formato texto plano estructurado."""
        lines = [
            "=== REPORTE DE COMPARACIÓN: SELECCIONADOS vs NO SELECCIONADOS ===",
            f"Total Población: N={self.n_total} | Seleccionados: n={self.n_selected} | No Seleccionados: n={self.n_non_selected}",
            "-" * 85,
            "1. VARIABLES NUMÉRICAS:",
            f"{'Variable'.ljust(15)} | {f'Muestra (n={self.n_selected}) Media(DE)'.ljust(26)} | {f'No Muestra (n={self.n_non_selected}) Media(DE)'.ljust(26)} | {'SMD'.rjust(8)} | {'p-value'.rjust(8)}",
            "-" * 85
        ]
        for res in self.num_results:
            sel_str = f"{res['mean_sel']:.2f} ({res['sd_sel']:.2f})"
            non_str = f"{res['mean_non']:.2f} ({res['sd_non']:.2f})"
            lines.append(
                f"{res['variable'].ljust(15)} | {sel_str.ljust(26)} | {non_str.ljust(26)} | {res['smd']:+8.3f} | {res['p_value']:8.4f}"
            )
        lines.append("-" * 85)
        lines.append("2. VARIABLES CATEGÓRICAS:")
        lines.append(
            f"{'Variable'.ljust(15)} | {'Categoría'.ljust(20)} | {f'Muestra (n={self.n_selected}) N (%)'.ljust(22)} | {f'No Muestra (n={self.n_non_selected}) N (%)'.ljust(22)} | {'SMD'.rjust(8)} | {'p-value (Chi2 / Fisher)'.rjust(18)}"
        )
        lines.append("-" * 85)
        for res in self.cat_results:
            first = True
            p_str = f"{res['p_value']:.4f}{res['warn_flag']}" if pd.notna(res['p_value']) else ""
            for lvl in res["levels"]:
                var_label = res['variable'] if first else ""
                sel_str = f"{lvl['c_sel']} ({lvl['p_sel']*100:.1f}%)"
                non_str = f"{lvl['c_non']} ({lvl['p_non']*100:.1f}%)"
                lines.append(
                    f"{var_label.ljust(15)} | {lvl['level'].ljust(20)} | {sel_str.ljust(22)} | {non_str.ljust(22)} | {lvl['smd']:+8.3f} | {p_str.rjust(18) if first else ''.rjust(18)}"
                )
                first = False
        lines.append("-" * 85)
        lines.append("\nNota sobre el p-value:")
        lines.append("- Se calcula mediante la Prueba Exacta de Fisher para tablas de 2x2 con frecuencias bajas (conteo < 5).")
        lines.append("- Se calcula mediante la Prueba de Chi-cuadrado (Chi2) de Pearson en los demás casos.")
        lines.append("- La advertencia 'celdas < 5' señala que alguna frecuencia esperada en la prueba de Chi2 es inferior a 5.\n")
        lines.append("-" * 85)
        lines.append(f"3. VARIABILIDAD DE PREDICTORES EN LA MUESTRA (n={self.n_selected}):")
        lines.append(
            f"{'Predictor'.ljust(15)} | {'SI (N)'.rjust(6)} | {'NO (N)'.rjust(6)} | {'Variabilidad?'.ljust(14)} | {'Estado'}"
        )
        lines.append("-" * 85)
        for res in self.risk_results:
            clean_st = str(res['status']).replace('🟢 ', '').replace('⚠️ ', '').replace('🔴 ', '').replace('**', '')
            lines.append(
                f"{res['variable'].ljust(15)} | {str(res['si_cnt']).rjust(6)} | {str(res['no_cnt']).rjust(6)} | {'Sí'.ljust(14) if res['has_variability'] else 'No'.ljust(14)} | {clean_st}"
            )
        lines.append("-" * 85)
        lines.append("4. DIAGNÓSTICO DE SESGO DE SELECCIÓN:")
        if not self.bias_vars:
            lines.append("Sin evidencia de sesgo de selección (|SMD| <= 0.10).")
        else:
            for var, val in sorted(self.bias_vars, key=lambda x: x[1], reverse=True):
                nivel = "Importante (> 0.25)" if val > 0.25 else "Leve (> 0.10)"
                lines.append(f"- {var}: SMD Máximo = {val:.3f} ({nivel} desequilibrio)")
        if self.variability_warnings:
            lines.append("\nAlertas de variabilidad:")
            for warning in self.variability_warnings:
                lines.append(f"- {warning.replace('`', '')}")
                
        report_str = "\n".join(lines)
        if filepath:
            self._write_file(filepath, report_str)
        return report_str

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera un reporte HTML completo con calidad de publicación (estilo Booktabs)
        y tema claro forzado con colores explícitos para perfecta legibilidad en fondos oscuros o claros.
        """
        lines = []

        # 1. Numéricas
        lines.append("  <div class='hs-section-title'>1. Comparación de Variables Numéricas (Edad y Score de Riesgo)</div>")
        lines.append("  <table class='hs-pub-table'>")
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append("        <th class='hs-left-col'>Variable</th>")
        lines.append(f"        <th class='hs-center-col'>Seleccionados (n={self.n_selected})<br><span style='font-size:12px; font-weight:normal; color:#64748b !important;'>Media (DE)</span></th>")
        lines.append(f"        <th class='hs-center-col'>No Seleccionados (n={self.n_non_selected})<br><span style='font-size:12px; font-weight:normal; color:#64748b !important;'>Media (DE)</span></th>")
        lines.append("        <th class='hs-center-col'>SMD</th>")
        lines.append("        <th class='hs-center-col'>p-value (t-test)</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")
        for res in self.num_results:
            lines.append("      <tr>")
            lines.append(f"        <td class='hs-left-col'><code>{res['variable']}</code></td>")
            lines.append(f"        <td class='hs-center-col'>{res['mean_sel']:.2f} ({res['sd_sel']:.2f})</td>")
            lines.append(f"        <td class='hs-center-col'>{res['mean_non']:.2f} ({res['sd_non']:.2f})</td>")
            lines.append(f"        <td class='hs-center-col'><strong>{res['smd']:+.3f}</strong></td>")
            lines.append(f"        <td class='hs-center-col'>{res['p_value']:.4f}</td>")
            lines.append("      </tr>")
        lines.append("    </tbody>")
        lines.append("  </table>")

        # 2. Categóricas
        lines.append("  <div class='hs-section-title'>2. Comparación de Variables Categóricas (Sociodemográficas y Riesgos)</div>")
        lines.append("  <table class='hs-pub-table'>")
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append("        <th class='hs-left-col' style='width: 170px;'>Variable</th>")
        lines.append("        <th class='hs-left-col' style='width: 140px;'>Categoría</th>")
        lines.append(f"        <th class='hs-center-col'>Seleccionados (n={self.n_selected})<br><span style='font-size:12px; font-weight:normal; color:#64748b !important;'>N (%)</span></th>")
        lines.append(f"        <th class='hs-center-col'>No Seleccionados (n={self.n_non_selected})<br><span style='font-size:12px; font-weight:normal; color:#64748b !important;'>N (%)</span></th>")
        lines.append("        <th class='hs-center-col'>SMD</th>")
        lines.append("        <th class='hs-center-col'>p-value (&chi;² / Fisher)</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")
        for res_idx, res in enumerate(self.cat_results):
            first = True
            p_val = res.get('p_value')
            html_flag = res.get('html_flag', '')
            p_display = f"<strong>{p_val:.4f}</strong>{html_flag}" if pd.notna(p_val) else ""
            for lvl in res["levels"]:
                var_label = f"<code>{res['variable']}</code>" if first else ""
                td_style = " style='border-top: 1px solid #cbd5e1 !important;'" if (first and res_idx > 0) else ""
                lines.append("      <tr>")
                lines.append(f"        <td class='hs-left-col'{td_style}>{var_label}</td>")
                lines.append(f"        <td class='hs-left-col'{td_style}>{lvl['level']}</td>")
                lines.append(f"        <td class='hs-center-col'{td_style}>{lvl['c_sel']} ({lvl['p_sel']*100:.1f}%)</td>")
                lines.append(f"        <td class='hs-center-col'{td_style}>{lvl['c_non']} ({lvl['p_non']*100:.1f}%)</td>")
                lines.append(f"        <td class='hs-center-col'{td_style}><strong>{lvl['smd']:+.3f}</strong></td>")
                lines.append(f"        <td class='hs-center-col'{td_style}>{p_display if first else ''}</td>")
                lines.append("      </tr>")
                first = False
        lines.append("    </tbody>")
        lines.append("  </table>")
        lines.append("  <div class='hs-pub-notes'>Nota: Se calcula mediante Prueba Exacta de Fisher para tablas 2x2 con celdas &lt; 5, y Chi-cuadrado (&chi;²) en los demás casos.</div>")

        # 3. Variabilidad
        lines.append(f"  <div class='hs-section-title'>3. Exploración de Variabilidad de Riesgo en el Grupo Seleccionado (n={self.n_selected})</div>")
        lines.append("  <div style='font-size:13px; color:#475569 !important; margin-bottom:8px;'>Para realizar regresiones estadísticas o contrastes diagnósticos de manera robusta, los predictores (variables de riesgo de metales) no deben ser constantes en la muestra analítica.</div>")
        lines.append("  <table class='hs-pub-table'>")
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append("        <th class='hs-left-col' style='width: 170px;'>Variable de Riesgo</th>")
        lines.append("        <th class='hs-center-col' style='width: 120px;'>Cantidad de 'SI'</th>")
        lines.append("        <th class='hs-center-col' style='width: 120px;'>Cantidad de 'NO'</th>")
        lines.append("        <th class='hs-center-col' style='width: 150px;'>¿Tiene Variabilidad?</th>")
        lines.append("        <th class='hs-left-col'>Estado / Alerta en Muestra</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")
        for res in self.risk_results:
            si_c = res['si_cnt']
            no_c = res['no_cnt']
            has_v = res['has_variability']

            if not has_v:
                badge_status = f"<span class='hs-badge-danger'>🔴 Sin variabilidad</span> <span style='font-size:12px; color:#b91c1c !important; margin-left:6px;'>(Constante: todos los registros son '{'SI' if si_c > 0 else 'NO'}'). No se puede utilizar en regresiones.</span>"
            elif si_c < 5 or no_c < 5:
                min_obs = min(si_c, no_c)
                badge_status = f"<span class='hs-badge-warn'>⚠️ Baja variabilidad</span> <span style='font-size:12px; color:#475569 !important; margin-left:6px;'>(sólo {min_obs} observaciones en una categoría). Puede causar problemas de convergencia en modelos logísticos.</span>"
            else:
                badge_status = "<span class='hs-badge-ok'>🟢 Variabilidad adecuada</span>"

            has_var_badge = "<span class='hs-badge-ok'>Sí</span>" if has_v else "<span class='hs-badge-danger'>No</span>"
            lines.append("      <tr>")
            lines.append(f"        <td class='hs-left-col'><code>{res['variable']}</code></td>")
            lines.append(f"        <td class='hs-center-col'><strong>{si_c}</strong></td>")
            lines.append(f"        <td class='hs-center-col'><strong>{no_c}</strong></td>")
            lines.append(f"        <td class='hs-center-col'>{has_var_badge}</td>")
            lines.append(f"        <td class='hs-left-col'>{badge_status}</td>")
            lines.append("      </tr>")
        lines.append("    </tbody>")
        lines.append("  </table>")

        # 4. Diagnóstico de Sesgo
        lines.append("  <div class='hs-section-title'>4. Diagnóstico de Sesgo de Selección</div>")
        lines.append("  <div style='font-size:13px; color:#475569 !important; margin-bottom:8px;'>El sesgo de selección se evalúa examinando si hay desequilibrios significativos entre el grupo seleccionado y el no seleccionado:<br>&bull; <code>|SMD| &gt; 0.10</code> indica desequilibrio leve.<br>&bull; <code>|SMD| &gt; 0.25</code> indica desequilibrio importante (posible sesgo).</div>")
        
        if not self.bias_vars:
            lines.append("  <div style='margin-top:10px;'><span class='hs-badge-ok'>🟢 No se detectó evidencia de sesgo de selección</span> <span style='font-size:13px; color:#475569 !important; margin-left:6px;'>El grupo seleccionado y el no seleccionado son comparables en todas las covariables registradas (todas las |SMD| &le; 0.10).</span></div>")
        else:
            lines.append("  <table class='hs-pub-table'>")
            lines.append("    <thead>")
            lines.append("      <tr>")
            lines.append("        <th class='hs-left-col' style='width: 200px;'>Variable</th>")
            lines.append("        <th class='hs-center-col' style='width: 140px;'>SMD Máximo</th>")
            lines.append("        <th class='hs-left-col'>Nivel de Desequilibrio</th>")
            lines.append("      </tr>")
            lines.append("    </thead>")
            lines.append("    <tbody>")
            for var, val in sorted(self.bias_vars, key=lambda x: x[1], reverse=True):
                if val > 0.25:
                    nivel_badge = "<span class='hs-badge-danger'>Importante (&gt; 0.25)</span>"
                else:
                    nivel_badge = "<span class='hs-badge-warn'>Leve (&gt; 0.10)</span>"
                lines.append("      <tr>")
                lines.append(f"        <td class='hs-left-col'><code>{var}</code></td>")
                lines.append(f"        <td class='hs-center-col'><strong>{val:.3f}</strong></td>")
                lines.append(f"        <td class='hs-left-col'>{nivel_badge}</td>")
                lines.append("      </tr>")
            lines.append("    </tbody>")
            lines.append("  </table>")

        inner_html = "\n".join(lines)
        html_code = wrap_html_container(
            inner_html=inner_html,
            title="Reporte de Comparación: Seleccionados vs No Seleccionados",
            subtitle=f"Población Total: <strong>N={self.n_total}</strong> | Seleccionados (Muestra): <strong>n={self.n_selected}</strong> | No Seleccionados: <strong>n={self.n_non_selected}</strong>",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<ComparationReport selected={self.n_selected} non_selected={self.n_non_selected}>"


def compare_groups(
    data: pd.DataFrame,
    sample_col: str = "Muestra_Codificada"
) -> ComparationReport:
    """
    Compara las variables del grupo seleccionado vs el no seleccionado,
    y analiza la variabilidad de predictores en el grupo seleccionado.
    
    Retorna un ComparationReport.
    """
    import scipy.stats as stats
    
    if sample_col not in data.columns:
        raise ValueError(f"La columna identificadora '{sample_col}' no existe en el DataFrame.")
        
    df_clean = standardize_boolean_columns(data)
    sample_mask = df_clean[sample_col].notna()
    n_selected = int(sample_mask.sum())
    n_non_selected = int((~sample_mask).sum())
    n_total = len(df_clean)
    
    # 1. Variables Numéricas
    num_results = []
    num_vars = ["Edad", "Score_Riesgo"]
    for col in num_vars:
        if col in df_clean.columns:
            g_sel = df_clean.loc[sample_mask, col].dropna()
            g_non = df_clean.loc[~sample_mask, col].dropna()
            
            m_sel, sd_sel = float(g_sel.mean()), float(g_sel.std())
            m_non, sd_non = float(g_non.mean()), float(g_non.std())
            
            smd = float(calculate_numerical_smd(g_sel, g_non))
            _, p_val = stats.ttest_ind(g_sel, g_non, equal_var=False)
            
            num_results.append({
                "variable": col,
                "mean_sel": m_sel, "sd_sel": sd_sel,
                "mean_non": m_non, "sd_non": sd_non,
                "smd": smd, "p_value": float(p_val)
            })
            
    # 2. Variables Categóricas
    cat_results = []
    cat_vars = ["Sexo", "Sector", "Es_Expuesto", "Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
    for col in cat_vars:
        if col in df_clean.columns:
            counts_sel = df_clean.loc[sample_mask, col].value_counts()
            counts_non = df_clean.loc[~sample_mask, col].value_counts()
            
            # Orden natural consistente para categorías
            raw_cats = set(counts_sel.index) | set(counts_non.index)
            if raw_cats == {"SI", "NO"}:
                categories = ["NO", "SI"]
            elif raw_cats == {"Femenino", "Masculino"}:
                categories = ["Femenino", "Masculino"]
            else:
                categories = sorted(list(raw_cats))
            
            contingency = pd.crosstab(df_clean[col], sample_mask)
            p_val_global = np.nan
            warn_flag = ""
            html_flag = ""
            if contingency.shape[0] >= 2:
                try:
                    if contingency.shape == (2, 2) and contingency.min().min() < 5:
                        _, p_val_global = stats.fisher_exact(contingency)
                        warn_flag = " (Fisher)"
                        html_flag = " <span style='font-size:11px; color:#64748b !important;'>(Fisher)</span>"
                    else:
                        _, p_val_global, _, expected = stats.chi2_contingency(contingency)
                        if (expected < 5).any():
                            warn_flag = " ($\\chi^2$, ⚠️ celdas < 5)"
                            html_flag = " <span style='font-size:11px; color:#b45309 !important;'>(&chi;², ⚠️ celdas &lt; 5)</span>"
                        else:
                            warn_flag = " ($\\chi^2$)"
                            html_flag = " <span style='font-size:11px; color:#64748b !important;'>(&chi;²)</span>"
                except Exception:
                    pass
                    
            levels = []
            for cat in categories:
                c_sel = int(counts_sel.get(cat, 0))
                c_non = int(counts_non.get(cat, 0))
                
                p_sel = c_sel / n_selected if n_selected > 0 else 0.0
                p_non = c_non / n_non_selected if n_non_selected > 0 else 0.0
                
                smd = float(calculate_binary_smd(p_sel, p_non))
                levels.append({
                    "level": str(cat),
                    "c_sel": c_sel, "p_sel": p_sel,
                    "c_non": c_non, "p_non": p_non,
                    "smd": smd
                })
                
            cat_results.append({
                "variable": col,
                "p_value": float(p_val_global) if pd.notna(p_val_global) else np.nan,
                "warn_flag": warn_flag,
                "html_flag": html_flag,
                "levels": levels
            })
            
    # 3. Exploración de Variabilidad
    risk_results = []
    risk_cols = ["Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
    low_variability_warnings = []
    
    for col in risk_cols:
        if col in df_clean.columns:
            counts = df_clean.loc[sample_mask, col].value_counts()
            si_cnt = int(counts.get("SI", 0))
            no_cnt = int(counts.get("NO", 0))
            
            has_variability = (si_cnt > 0) and (no_cnt > 0)
            
            if not has_variability:
                status = f"🔴 SIN VARIABILIDAD (Constante). Todos son '{'SI' if si_cnt > 0 else 'NO'}'."
                low_variability_warnings.append(f"`{col}` no tiene variabilidad (todos los registros son '{'SI' if si_cnt > 0 else 'NO'}').")
            elif si_cnt < 5 or no_cnt < 5:
                min_c = min(si_cnt, no_cnt)
                status = f"⚠️ Baja variabilidad (sólo {min_c} observaciones en una categoría). Puede causar problemas de convergencia en modelos logísticos."
                low_variability_warnings.append(f"`{col}` tiene baja variabilidad (sólo {min_c} observaciones en la categoría minoritaria).")
            else:
                status = "🟢 Variabilidad adecuada."
                
            risk_results.append({
                "variable": col,
                "si_cnt": si_cnt,
                "no_cnt": no_cnt,
                "has_variability": has_variability,
                "status": status
            })
            
    # 4. Diagnóstico de Sesgo de Selección
    bias_vars = []
    for res in num_results:
        smd = abs(res["smd"])
        if smd > 0.1:
            bias_vars.append((res["variable"], smd))
    for res in cat_results:
        max_smd = max(abs(lvl["smd"]) for lvl in res["levels"]) if res["levels"] else 0.0
        if max_smd > 0.1:
            bias_vars.append((res["variable"], max_smd))
            
    return ComparationReport(
        num_results=num_results,
        cat_results=cat_results,
        risk_results=risk_results,
        bias_vars=bias_vars,
        variability_warnings=low_variability_warnings,
        n_selected=n_selected,
        n_non_selected=n_non_selected,
        n_total=n_total
    )

