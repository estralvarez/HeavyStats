import sys
import pathlib
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
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

class ComparationReport:
    """
    Reporte de comparación de seleccionados vs no seleccionados.
    Permite exportar a texto, Markdown y renderizarse automáticamente en Jupyter.
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

    def _write_file(self, filepath: Optional[str], content: str) -> None:
        if filepath:
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        """Genera el reporte comparativo en formato de tablas Markdown."""
        report_lines = []
        report_lines.append("# Reporte de Comparación: Seleccionados vs No Seleccionados")
        report_lines.append(f"**Población Total:** N={self.n_total} | **Seleccionados (Muestra):** n={self.n_selected} | **No Seleccionados:** n={self.n_non_selected}\n")
        
        # 1. Continuas
        report_lines.append("## 1. Comparación de Variables Numéricas (Edad y Score de Riesgo)\n")
        report_lines.append("| Variable | Seleccionados (n=20) <br> Media (DE) | No Seleccionados (n=28) <br> Media (DE) | SMD | p-value (t-test) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for res in self.num_results:
            report_lines.append(
                f"| **{res['variable']}** | {res['mean_sel']:.2f} ({res['sd_sel']:.2f}) | {res['mean_non']:.2f} ({res['sd_non']:.2f}) | {res['smd']:+.3f} | {res['p_value']:.4f} |"
            )
        report_lines.append("\n")
        
        # 2. Categóricas
        report_lines.append("## 2. Comparación de Variables Categóricas (Sociodemográficas y Riesgos)\n")
        report_lines.append("| Variable | Categoría | Seleccionados (n=20) <br> N (%) | No Seleccionados (n=28) <br> N (%) | SMD | p-value ($\\chi^2$ / Fisher) |")
        report_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for res in self.cat_results:
            first = True
            p_str = f"{res['p_value']:.4f}{res['warn_flag']}" if pd.notna(res['p_value']) else ""
            for lvl in res["levels"]:
                var_label = f"**{res['variable']}**" if first else ""
                report_lines.append(
                    f"| {var_label} | {lvl['level']} | {lvl['c_sel']} ({lvl['p_sel']*100:.1f}%) | {lvl['c_non']} ({lvl['p_non']*100:.1f}%) | {lvl['smd']:+.3f} | {p_str if first else ''} |"
                )
                first = False
            report_lines.append("| | | | | | |")
        if report_lines[-1] == "| | | | | | |":
            report_lines.pop()
        report_lines.append("\n> **Nota sobre el p-value:**")
        report_lines.append("> * Se calcula mediante la **Prueba Exacta de Fisher** para tablas de $2 \\times 2$ con frecuencias bajas (conteo < 5).")
        report_lines.append("> * Se calcula mediante la **Prueba de Chi-cuadrado ($\\chi^2$)** de Pearson en los demás casos.")
        report_lines.append("> * La advertencia `⚠️ celdas < 5` señala que alguna frecuencia esperada en la prueba de $\\chi^2$ es inferior a 5, por lo que la aproximación asintótica puede ser imprecisa.")
        report_lines.append("\n")
        
        # 3. Variabilidad
        report_lines.append("## 3. Exploración de Variabilidad de Riesgo en el Grupo Seleccionado (n=20)\n")
        report_lines.append("Para realizar regresiones estadísticas de manera robusta en etapas posteriores, los predictores (variables de riesgo de metales) no deben ser constantes en la muestra analítica.\n")
        report_lines.append("| Variable de Riesgo | Cantidad de 'SI' | Cantidad de 'NO' | ¿Tiene Variabilidad? | Estado / Alerta en Muestra |")
        report_lines.append("| :--- | :---: | :---: | :---: | :--- |")
        for res in self.risk_results:
            report_lines.append(
                f"| `{res['variable']}` | {res['si_cnt']} | {res['no_cnt']} | {'Sí' if res['has_variability'] else 'No'} | {res['status']} |"
            )
        report_lines.append("\n")
        
        # 4. Diagnóstico
        report_lines.append("## 4. Diagnóstico de Sesgo de Selección\n")
        report_lines.append("El sesgo de selección se evalúa examinando si hay desequilibrios significativos entre el grupo seleccionado y el no seleccionado. Generalmente:")
        report_lines.append("* $|SMD| > 0.1$ indica desequilibrio leve.")
        report_lines.append("* $|SMD| > 0.25$ indica desequilibrio importante (posible sesgo).\n")
        
        if not self.bias_vars:
            report_lines.append("🟢 **No se detectó evidencia de sesgo de selección.** El grupo seleccionado y el no seleccionado son comparables en todas las covariables registradas (todas las $|SMD| \\le 0.1$).")
        else:
            report_lines.append("⚠️ **Posible sesgo de selección detectado en las siguientes variables:**\n")
            report_lines.append("| Variable | SMD Máximo | Nivel de Desequilibrio |")
            report_lines.append("| :--- | :---: | :--- |")
            for var, val in sorted(self.bias_vars, key=lambda x: x[1], reverse=True):
                nivel = "Importante ($>0.25$)" if val > 0.25 else "Leve ($>0.10$)"
                report_lines.append(f"| `{var}` | {val:.3f} | {nivel} |")
            report_lines.append("\n")
            report_lines.append("### Recomendaciones Metodológicas:")
            report_lines.append("1. **Reportar estos desequilibrios:** Es mandatorio documentar que los dos grupos difieren en estas variables.")
            report_lines.append("2. **Ajuste en análisis posteriores:** Al modelar la asociación de los metales, se debe controlar por las covariables que muestran desequilibrio (ej. en regresiones multivariadas) para reducir el sesgo de confusión.")
            
        if self.variability_warnings:
            report_lines.append("\n### Alertas de Variabilidad para Modelos de Regresión:")
            for warning in self.variability_warnings:
                report_lines.append(f"* {warning}")
                
        report_str = "\n".join(report_lines)
        self._write_file(filepath, report_str)
        return report_str

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Genera el reporte en formato texto plano estructurado."""
        lines = [
            "=== REPORTE DE COMPARACIÓN: SELECCIONADOS vs NO SELECCIONADOS ===",
            f"Total Población: N={self.n_total} | Seleccionados: n={self.n_selected} | No Seleccionados: n={self.n_non_selected}",
            "-" * 85,
            "1. VARIABLES NUMÉRICAS:",
            f"{'Variable'.ljust(15)} | {'Muestra (n=20) Media(DE)'.ljust(26)} | {'No Muestra (n=28) Media(DE)'.ljust(26)} | {'SMD'.rjust(8)} | {'p-value'.rjust(8)}",
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
            f"{'Variable'.ljust(15)} | {'Categoría'.ljust(20)} | {'Muestra N (%)'.ljust(18)} | {'No Muestra N (%)'.ljust(18)} | {'SMD'.rjust(8)} | {'p-value ($\\chi^2$ / Fisher)'.rjust(8)}"
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
                    f"{var_label.ljust(15)} | {lvl['level'].ljust(20)} | {sel_str.ljust(18)} | {non_str.ljust(18)} | {lvl['smd']:+8.3f} | {p_str.rjust(8) if first else ''.rjust(8)}"
                )
                first = False
        lines.append("-" * 85)
        lines.append("\nNota sobre el p-value:")
        lines.append("- Se calcula mediante la Prueba Exacta de Fisher para tablas de 2x2 con frecuencias bajas (conteo < 5).")
        lines.append("- Se calcula mediante la Prueba de Chi-cuadrado ($\\chi^2$) de Pearson en los demás casos.")
        lines.append("- La advertencia '⚠️ celdas < 5' señala que alguna frecuencia esperada en la prueba de $\\chi^2$ es inferior a 5.\n")
        lines.append("-" * 85)
        lines.append("3. VARIABILIDAD DE PREDICTORES EN LA MUESTRA:")
        lines.append(
            f"{'Predictor'.ljust(15)} | {'SI (N)'.rjust(6)} | {'NO (N)'.rjust(6)} | {'Variabilidad?'.ljust(14)} | {'Estado'}"
        )
        lines.append("-" * 85)
        for res in self.risk_results:
            lines.append(
                f"{res['variable'].ljust(15)} | {str(res['si_cnt']).rjust(6)} | {str(res['no_cnt']).rjust(6)} | {'Sí'.ljust(14) if res['has_variability'] else 'No'.ljust(14)} | {res['status'].replace('🟢 ', '').replace('⚠️ ', '').replace('🔴 ', '')}"
            )
        lines.append("-" * 85)
        lines.append("4. DIAGNÓSTICO DE SESGO DE SELECCIÓN:")
        if not self.bias_vars:
            lines.append("Sin evidencia de sesgo de selección.")
        else:
            for var, val in sorted(self.bias_vars, key=lambda x: x[1], reverse=True):
                nivel = "Importante" if val > 0.25 else "Leve"
                lines.append(f"- {var}: SMD Máximo = {val:.3f} ({nivel} desequilibrio)")
        if self.variability_warnings:
            lines.append("\nAlertas de variabilidad:")
            for warning in self.variability_warnings:
                lines.append(f"- {warning}")
                
        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def _repr_markdown_(self) -> str:
        """Renderizado automático en Markdown para Jupyter Notebook."""
        return self.to_markdown()

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
    if sample_col not in data.columns:
        raise ValueError(f"La columna identificadora '{sample_col}' no existe en el DataFrame.")
        
    sample_mask = data[sample_col].notna()
    n_selected = sample_mask.sum()
    n_non_selected = (~sample_mask).sum()
    n_total = len(data)
    
    # 1. Variables Numéricas
    num_results = []
    num_vars = ["Edad", "Score_Riesgo"]
    for col in num_vars:
        if col in data.columns:
            g_sel = data.loc[sample_mask, col].dropna()
            g_non = data.loc[~sample_mask, col].dropna()
            
            m_sel, sd_sel = g_sel.mean(), g_sel.std()
            m_non, sd_non = g_non.mean(), g_non.std()
            
            smd = calculate_numerical_smd(g_sel, g_non)
            _, p_val = stats.ttest_ind(g_sel, g_non, equal_var=False)
            
            num_results.append({
                "variable": col,
                "mean_sel": m_sel, "sd_sel": sd_sel,
                "mean_non": m_non, "sd_non": sd_non,
                "smd": smd, "p_value": p_val
            })
            
    # 2. Variables Categóricas
    cat_results = []
    cat_vars = ["Sexo", "Sector", "Es_Expuesto", "Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
    for col in cat_vars:
        if col in data.columns:
            counts_sel = data.loc[sample_mask, col].value_counts()
            counts_non = data.loc[~sample_mask, col].value_counts()
            categories = sorted(list(set(counts_sel.index) | set(counts_non.index)))
            
            contingency = pd.crosstab(data[col], sample_mask)
            p_val_global = np.nan
            warn_flag = ""
            if contingency.shape[0] >= 2:
                try:
                    if contingency.shape == (2, 2) and contingency.min().min() < 5:
                        _, p_val_global = stats.fisher_exact(contingency)
                        warn_flag = " (Fisher)"
                    else:
                        _, p_val_global, _, expected = stats.chi2_contingency(contingency)
                        if (expected < 5).any():
                            warn_flag = " ($\\chi^2$, ⚠️ celdas < 5)"
                except Exception:
                    pass
                    
            levels = []
            for cat in categories:
                c_sel = counts_sel.get(cat, 0)
                c_non = counts_non.get(cat, 0)
                
                p_sel = c_sel / n_selected
                p_non = c_non / n_non_selected
                
                smd = calculate_binary_smd(p_sel, p_non)
                levels.append({
                    "level": cat,
                    "c_sel": c_sel, "p_sel": p_sel,
                    "c_non": c_non, "p_non": p_non,
                    "smd": smd
                })
                
            cat_results.append({
                "variable": col,
                "p_value": p_val_global,
                "warn_flag": warn_flag,
                "levels": levels
            })
            
    # 3. Exploración de Variabilidad
    risk_results = []
    risk_cols = ["Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
    low_variability_warnings = []
    
    for col in risk_cols:
        if col in data.columns:
            counts = data.loc[sample_mask, col].value_counts()
            si_cnt = counts.get("SI", 0)
            no_cnt = counts.get("NO", 0)
            
            has_variability = (si_cnt > 0) and (no_cnt > 0)
            
            if not has_variability:
                status = "🔴 **SIN VARIABILIDAD (Constante)**. No se puede utilizar en regresiones."
                low_variability_warnings.append(f"`{col}` no tiene variabilidad (todos son '{'SI' if si_cnt > 0 else 'NO'}').")
            elif si_cnt < 5 or no_cnt < 5:
                status = f"⚠️ **Baja variabilidad** (sólo {min(si_cnt, no_cnt)} observaciones en una categoría). Puede causar problemas de convergencia en modelos logísticos."
                low_variability_warnings.append(f"`{col}` tiene desbalance severo (sólo {min(si_cnt, no_cnt)} observaciones de la minoría).")
            else:
                status = "🟢 **Variabilidad adecuada**."
                
            risk_results.append({
                "variable": col,
                "si_cnt": si_cnt,
                "no_cnt": no_cnt,
                "has_variability": has_variability,
                "status": status
            })
            
    # 4. Diagnóstico de Sesgo de Selección
    bias_vars = []
    # Evaluar numéricas
    for res in num_results:
        smd = abs(res["smd"])
        if smd > 0.1:
            bias_vars.append((res["variable"], smd))
    # Evaluar categóricas
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

