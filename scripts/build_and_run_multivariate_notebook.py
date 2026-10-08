"""
Genera y ejecuta de forma interactiva el cuaderno '01_multivariate_analysis.ipynb'
en 'd:/vscode/results/HeavyStats/notebooks/multivariante/'.
Captura todos los outputs (HTML, texto, tablas) para que el cuaderno quede listo
para visualizar y explorar en VS Code / Jupyter / Positron.
"""

import os
import io
import json
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from IPython.core.interactiveshell import InteractiveShell

# Directorio de salida
nb_dir = Path(r"d:\vscode\results\HeavyStats\notebooks\multivariante")
nb_dir.mkdir(parents=True, exist_ok=True)
nb_path = nb_dir / "01_multivariate_analysis.ipynb"

# Inicializar IPython shell
shell = InteractiveShell.instance()

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_and_run_code(code_str):
    code_lines = [l + "\n" for l in code_str.strip().split("\n")]
    if code_lines:
        code_lines[-1] = code_lines[-1].rstrip("\n")

    f_out = io.StringIO()
    f_err = io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_err):
        exec_result = shell.run_cell(code_str)

    stdout_text = f_out.getvalue()
    stderr_text = f_err.getvalue()

    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [l + "\n" for l in stdout_text.splitlines()]
        })

    if stderr_text and not ("Warning" in stderr_text and len(stderr_text) < 150):
        outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [l + "\n" for l in stderr_text.splitlines()]
        })

    # Si la última expresión devolvió un objeto (display/repr)
    if exec_result.result is not None:
        obj = exec_result.result
        data_dict = {}
        # Verificar si tiene _repr_html_
        if hasattr(obj, "_repr_html_"):
            try:
                html_repr = obj._repr_html_()
                if html_repr:
                    data_dict["text/html"] = [html_repr]
            except Exception:
                pass
        
        # Plain text
        data_dict["text/plain"] = [repr(obj)]

        outputs.append({
            "data": data_dict,
            "execution_count": len([c for c in nb["cells"] if c["cell_type"] == "code"]) + 1,
            "metadata": {},
            "output_type": "execute_result"
        })

    nb["cells"].append({
        "cell_type": "code",
        "execution_count": len([c for c in nb["cells"] if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": code_lines
    })


print("Construyendo celdas del cuaderno...")

# -------------------------------------------------------------
# Celda 1: Markdown Título
# -------------------------------------------------------------
add_markdown("""# Módulo de Análisis Multivariante (`heavystats.multivariante`)
## Pipeline en Dos Etapas para Biomonitoreo Toxicológico y Epidemiología

Este cuaderno demuestra el flujo bioestadístico y computacional de la librería **HeavyStats** para dar respuesta al **Objetivo Específico 4** de la investigación:

> **Objetivo 4**: Evaluar el impacto individual y conjunto de los factores de riesgo sobre las concentraciones biológicas de metales pesados ($\\ce{Hg_T}$ en plasma/sangre).

### Desafíos Metodológicos Resueltos:
1. **Desbalance Dimensional ($p > n$):** Con $n = 20$ muestras analíticas y $p > 35$ factores de exposición, las regresiones clásicas colapsan por pérdida de grados de libertad e hiper-sobreajuste (*overfitting*).
2. **Asimetría No Paramétrica:** La variable biológica presenta asimetría positiva con valores concentrados cerca del límite de detección ($LOD$).
3. **Solución en Dos Etapas:**
   - **Etapa 1: Tamizaje Bivariante No Paramétrico (*Screening*)** con contrastes libres de supuestos (Mann-Whitney $U$, Kruskal-Wallis $H$, Spearman $\\rho_s$) y control de la Tasa de Falso Descubrimiento (**Benjamini-Hochberg FDR**).
   - **Etapa 2: Modelado Multivariante Regularizado**, combinando **Regresión LASSO** (selección parsimoniosa de factores netos) y **Regresión PLS-R** (proyección en componentes latentes y cálculo de **VIP $\\ge 1{,}0$**).""")

# -------------------------------------------------------------
# Celda 2: Carga de librerías y configuración
# -------------------------------------------------------------
add_and_run_code("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import heavystats as hs
from heavystats.multivariante import (
    MultivariatePipeline,
    MultivariateScreening,
    LassoModeler,
    PlsModeler,
    MultivariateTableReport
)

print(f"Versión de HeavyStats: {hs.__version__}")
print(f"Módulos multivariantes activos: {[cls for cls in dir(hs.multivariante) if not cls.startswith('_')]}")""")

# -------------------------------------------------------------
# Celda 3: Carga de datos
# -------------------------------------------------------------
add_markdown("""---
## 1. Carga y Estructuración de los Datos Analíticos de Mercurio ($n = 20$)
Cargamos la submuestra cuantitativa con determinación analítica de $\\ce{Hg_T}$ en sangre total.""")

add_and_run_code("""# Localizar el archivo de datos del proyecto
data_candidates = [
    Path(r"d:\\vscode\\results\\resultados_mercurio\\datos\\procesados\\datos_mercurio_n20.csv"),
    Path.cwd().parent / "datos" / "procesados" / "datos_mercurio_n20.csv",
    Path.cwd() / "datos" / "datos_mercurio_n20.csv",
]

data_path = None
for p in data_candidates:
    if p.exists():
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("No se encontró el archivo de datos analíticos.")

df_hg = pd.read_csv(data_path)
print(f"Dataset analítico cargado exitosamente: {df_hg.shape[0]} observaciones y {df_hg.shape[1]} variables.")
print(f"Resumen de la variable respuesta (Mercurio_ug_L):")
print(df_hg['Mercurio_ug_L'].describe())""")

# -------------------------------------------------------------
# Celda 4: Visualización inicial de covariables
# -------------------------------------------------------------
add_markdown("""Observamos las primeras filas de covariables candidatas (dietarias, ambientales y sociodemográficas):""")

add_and_run_code("""cols_muestra = ['Muestra_Codificada', 'Edad', 'Sexo', 'Mercurio_ug_L', 'Alim_Pescados', 'Salud_Transporte_vehiculo', 'Alim_Cereales', 'Es_Expuesto']
df_hg[cols_muestra].head(10)""")

# -------------------------------------------------------------
# Celda 5: Etapa 1 Screening Bivariante
# -------------------------------------------------------------
add_markdown("""---
## 2. Etapa 1: Tamizaje Bivariante No Paramétrico (*Screening*)

La clase `MultivariateScreening` evalúa sistemáticamente cada covariable independiente:
- **Variables dicotómicas (2 grupos):** Prueba $U$ de Mann-Whitney, estimador de Hodges-Lehmann y correlación biserial por rangos ($r_{\\text{rb}}$ con intervalo de confianza *bootstrap* al 95%).
- **Variables politómicas ($\\ge 3$ grupos):** Prueba de Kruskal-Wallis ($H$) y tamaño de efecto Épsilon Cuadrado ($\\epsilon^2$).
- **Variables continuas / ordinales:** Coeficiente de correlación de Spearman ($\\rho_s$) y Kendall ($\\tau_b$).
- **Control de Multiplicidad:** Corrección Benjamini-Hochberg FDR para derivar los $q$-valores.
- **Clasificación Multicriterio:** Clasificación en prioridades **Alta**, **Intermedia** y **Baja / Descarte**.""")

add_and_run_code("""# Inicializar y ejecutar el tamizaje
screening = MultivariateScreening(
    data=df_hg,
    target_col="Mercurio_ug_L",
    alpha=0.05,
    fdr_threshold=0.10,
    n_boot=1000,
    random_state=42
)

df_screening = screening.run()

# Visualizar el top 12 de factores ordenados por significancia
cols_view = ['Variable', 'Etiqueta', 'Tipo', 'Estadístico_Str', 'p_valor', 'q_fdr', 'Direccion', 'Prioridad']
df_screening[cols_view].head(12)""")

# -------------------------------------------------------------
# Celda 6: Resumen de prioridades
# -------------------------------------------------------------
add_markdown("""Analizamos la distribución de covariables según su nivel de prioridad para el modelado multivariante:""")

add_and_run_code("""print("Distribución de covariables por nivel de prioridad:")
print(df_screening['Prioridad'].value_counts())

factores_priorizados = screening.get_prioritized_features(min_priority="Intermedia")
print(f"\\nFactores seleccionados para la Etapa 2 ({len(factores_priorizados)} covariables):")
for f in factores_priorizados:
    row = df_screening[df_screening['Variable'] == f].iloc[0]
    print(f"  • {row['Etiqueta']:<35} ({row['Variable']}): {row['Estadístico_Str']}, p={row['p_valor']:.3f}, {row['Direccion']}")""")

# -------------------------------------------------------------
# Celda 7: Etapa 2 Modelos Multivariantes
# -------------------------------------------------------------
add_markdown("""---
## 3. Etapa 2: Modelado Multivariante Regularizado (LASSO y PLS-R)

### 3.1. Regresión LASSO ($L_1$ Penalizado)
Minimiza el error cuadrático incorporando una penalización sobre la norma $L_1$ de los coeficientes:
$$\\min_{\\beta} \\left\\{ \\frac{1}{2n} \\sum_{i=1}^n \\left( y_i - \\beta_0 - \\sum_{j=1}^p \\beta_j x_{ij} \\right)^2 + \\lambda \\sum_{j=1}^p |\\beta_j| \\right\\}$$

El hiperparámetro de penalización $\\lambda$ se calibra mediante validación cruzada ($k$-fold CV con $k=5$).""")

add_and_run_code("""from heavystats.multivariante.models import prepare_features_matrix

# Preparar matriz de predictores numéricos estandarizados
X_mat, feat_names = prepare_features_matrix(df_hg, factores_priorizados)
y_log = np.log(df_hg['Mercurio_ug_L'].values)

# Ajustar LassoModeler
lasso = LassoModeler(cv_folds=5, random_state=42)
lasso.fit(X_mat, y_log)

print(f"Parámetro de regularización óptimo (lambda/alpha): {lasso.alpha_:.4f}")
print(f"Coeficiente de determinación R² del modelo LASSO: {lasso.r2_:.3f}")
print(f"\\nResumen de coeficientes estandarizados beta:")
print(lasso.summary_df.to_string(index=False))""")

# -------------------------------------------------------------
# Celda 8: Regresión PLS-R y VIP
# -------------------------------------------------------------
add_markdown("""### 3.2. Regresión por Mínimos Cuadrados Parciales (PLS-R) y Estadístico VIP
Modela la covarianza entre la matriz $X$ y el vector $y$ proyectándolos hacia $A = 2$ componentes latentes ortogonales.
La importancia de cada predictor se cuantifica mediante el estadístico **VIP** (Variable Importance in Projection):
$$\\text{VIP}_j = \\sqrt{ \\frac{p \\sum_{a=1}^A \\text{SS}_a (w_{aj} / \\|w_a\\|)^2}{\\sum_{a=1}^A \\text{SS}_a} }$$
**Regla toxicológica:** Variables con $\\text{VIP} \\ge 1{,}0$ son determinantes críticos de la carga xenobiótica.""")

add_and_run_code("""# Ajustar PlsModeler con 2 componentes latentes
pls = PlsModeler(n_components=2, scale=True)
pls.fit(X_mat, y_log)

print(f"Coeficiente de determinación R² del modelo PLS-R: {pls.r2_:.3f}")
print(f"\\nResumen de importancia de variables en la proyección (VIP):")
print(pls.summary_df.to_string(index=False))""")

# -------------------------------------------------------------
# Celda 9: Pipeline Integral Unificado
# -------------------------------------------------------------
add_markdown("""---
## 4. Ejecución del Pipeline Integral Unificado (`MultivariatePipeline`)

En la práctica, todas las etapas anteriores se orquestan en una única llamada a la clase `MultivariatePipeline`:""")

add_and_run_code("""pipeline = MultivariatePipeline(
    data=df_hg,
    target_col="Mercurio_ug_L",
    log_transform_target=True,
    alpha=0.05,
    fdr_threshold=0.10,
    random_state=42
)

# 1. Ejecutar tamizaje
pipeline.run_screening()

# 2. Ajustar modelos multivariantes sobre factores priorizados
fit_res = pipeline.fit_models(
    min_priority="Intermedia",
    lasso_cv_folds=5,
    pls_components=2
)

# 3. Obtener reporte consolidado
report = pipeline.get_report()
print("Pipeline ejecutado exitosamente. Tabla consolidada generada.")
report.df""")

# -------------------------------------------------------------
# Celda 10: Renderizado HTML interactivo
# -------------------------------------------------------------
add_markdown("""Visualizamos la tabla con calidad de publicación mediante su renderizado interactivo HTML:""")

add_and_run_code("""# Renderizado interactivo con estilos de publicación de HeavyStats
report""")

# -------------------------------------------------------------
# Celda 11: Exportación Editorial
# -------------------------------------------------------------
add_markdown("""---
## 5. Exportación Automática de Resultados (LaTeX, HTML, Excel)

Generamos los entregables editoriales listos para incluir directamente en el manuscrito de la tesis en LaTeX y para consulta interactiva.""")

add_and_run_code("""salidas_dir = Path(r"d:\\vscode\\results\\HeavyStats\\notebooks\\multivariante\\salidas")
salidas_dir.mkdir(parents=True, exist_ok=True)

# 1. Exportar tabla en código LaTeX con booktabs
tex_file = salidas_dir / "tabla_multivariante_hg.tex"
latex_code = pipeline.export_latex(
    str(tex_file),
    caption="Tamizaje bivariante no paramétrico y parámetros de los modelos multivariantes (LASSO y PLS-R) para la concentración basal de mercurio total en sangre (\\\\ce{Hg_T}).",
    label="tab:multivariante_mercurio"
)

# 2. Exportar reporte HTML autónomo
html_file = salidas_dir / "reporte_multivariante_hg.html"
pipeline.export_html(str(html_file))

# 3. Exportar libro de Excel con hojas separadas
excel_file = salidas_dir / "resultados_multivariante_hg.xlsx"
with pd.ExcelWriter(excel_file) as writer:
    pipeline.screening_df.to_excel(writer, sheet_name="Cribado_Bivariante", index=False)
    pipeline.consolidated_df.to_excel(writer, sheet_name="Modelos_Multivariantes", index=False)

print(f"Archivos exportados exitosamente en: {salidas_dir}")
print(f"  • LaTeX: {tex_file.name} ({tex_file.stat().st_size} bytes)")
print(f"  • HTML:  {html_file.name} ({html_file.stat().st_size} bytes)")
print(f"  • Excel: {excel_file.name} ({excel_file.stat().st_size} bytes)")""")

# -------------------------------------------------------------
# Celda 12: Visualización del código LaTeX generado
# -------------------------------------------------------------
add_markdown("""A continuación se muestra el código LaTeX generado automáticamente, listo para copiar o compilar con `\\input{}`:""");

add_and_run_code("""print(latex_code)""")

# Guardar notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Cuaderno generado y ejecutado exitosamente en:\n{nb_path}")
