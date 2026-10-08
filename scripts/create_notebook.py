"""
Script para generar el cuaderno interactivo de Jupyter para el análisis multivariante.
"""

import os
import json

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

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()]
    })


# Celda Markdown: Título y Contexto
add_md("""# Análisis Multivariante: Determinantes de la Concentración Basal de Mercurio (Hg)

**Trabajo Especial de Grado - FACYT / Departamento de Química (Universidad de Carabobo)**  
**Autor:** Br. Jose Daniel Estrada Alvarez  
**Tutor:** Dr. Jose Ramon Jimenez Ochoa  

---

## Fundamentación y Justificación Metodológica (Objetivo Específico 4)
En la evaluación de biomarcadores toxicológicos con desbalance dimensional ($p > n$, donde $n = 20$ y $p > 35$ covariables), la regresión ordinaria OLS colapsa por multicolinealidad y pérdida de grados de libertad. Además, las concentraciones séricas de $\\ce{Hg_T}$ presentan asimetría positiva no paramétrica (Shapiro-Wilk $W = 0{,}8857, p = 0{,}0225$).

Por ello, se implementa una arquitectura secuenciada en **dos etapas**:
1. **Etapa 1 (Tamizaje Bivariante No Paramétrico / Screening):** Contrastes libres de supuestos (Mann-Whitney $U$, Kruskal-Wallis $H$, correlación de Spearman $\\rho_s$) con control de multiplicidad de hipótesis mediante la Tasa de Falso Descubrimiento (**Benjamini-Hochberg FDR**, $q < 0{,}10$).
2. **Etapa 2 (Modelado Multivariante Regularizado):**
   - **Regresión LASSO ($L_1$ penalizado):** Selección parsimoniosa de covariables donde los coeficientes redundantes colapsan a exactamente cero ($\\beta = 0$).
   - **Regresión por Mínimos Cuadrados Parciales (PLS-R):** Manejo de colinealidad mediante componentes latentes ortogonales y cuantificación de la Importancia de la Variable en la Proyección (**VIP $\\ge 1{,}0$**).
""")

# Celda Código: Carga de paquetes
add_code("""import os
from pathlib import Path
import pandas as pd
import numpy as np

import heavystats as hs
from heavystats.multivariante import MultivariatePipeline

print(f"HeavyStats versión: {hs.__version__}")
""")

# Celda Markdown: Paso 1 Carga de datos
add_md("""---
## 1. Carga de Datos de la Submuestra Analítica de Mercurio ($n = 20$)
Cargamos el conjunto de datos analíticos depurado (`datos_mercurio_n20.csv`).
""")

# Celda Código: Carga de datos
add_code("""base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
data_path = base_dir / "datos" / "procesados" / "datos_mercurio_n20.csv"
df = pd.read_csv(data_path)
print(f"Dimensiones del dataset analítico: {df.shape}")
df[['Muestra_Codificada', 'Edad', 'Sexo', 'Mercurio_ug_L', 'Alim_Pescados', 'Es_Expuesto']].head()
""")

# Celda Markdown: Paso 2 Inicialización del Pipeline
add_md("""---
## 2. Configuración e Inicialización de `MultivariatePipeline`
Inicializamos el pipeline configurando:
- Variable dependiente: `Mercurio_ug_L`.
- Transformación logarítmica: `log_transform_target=True` ($\\ln(\\ce{Hg_T})$ para estabilizar la varianza residual).
- Umbrales de inferencia: $\\alpha = 0{,}05$ y tasa FDR $q = 0{,}10$.
""")

# Celda Código: Inicialización
add_code("""pipeline = MultivariatePipeline(
    data=df,
    target_col="Mercurio_ug_L",
    log_transform_target=True,
    alpha=0.05,
    fdr_threshold=0.10,
    random_state=42
)
print("MultivariatePipeline configurado e inicializado exitosamente.")
""")

# Celda Markdown: Paso 3 Tamizaje Bivariante
add_md("""---
## 3. Etapa 1: Tamizaje Bivariante No Paramétrico (*Screening*)
Se evalúan todas las covariables candidatas contra los niveles de $\\ce{Hg_T}$, categorizando su sentido de impacto:
- **Aporte (↑):** Incrementa la concentración biológica del metal.
- **Atenuación (↓):** Efecto protector o de reducción biológica.
- **Contrastante:** Diferencias significativas entre $\\ge 3$ categorías.
""")

# Celda Código: Ejecutar Screening
add_code("""screening_df = pipeline.run_screening()
screening_df[['Variable', 'Etiqueta', 'Tipo', 'Estadístico_Str', 'p_valor', 'q_fdr', 'Direccion', 'Prioridad']].head(15)
""")

# Celda Markdown: Paso 4 Modelado Multivariante
add_md("""---
## 4. Etapa 2: Modelado Multivariante Regularizado (LASSO y PLS-R)
Ajustamos los modelos multivariantes sobre los factores de riesgo priorizados:
- **Regresión LASSO:** Validación cruzada en 5 pliegues ($k$-fold CV).
- **Regresión PLS-R:** Extracción de componentes latentes ortogonales y estadístico VIP.
""")

# Celda Código: Ajustar modelos
add_code("""fit_results = pipeline.fit_models(
    min_priority="Intermedia",
    lasso_cv_folds=5,
    pls_components=2
)
report = fit_results['report']
report
""")

# Celda Markdown: Paso 5 Exportación
add_md("""---
## 5. Exportación Editorial de Resultados (LaTeX, HTML, Excel)
Exportamos la tabla con formato `booktabs` lista para inclusión en el manuscrito de la tesis en LaTeX y el reporte HTML interactivo.
""")

# Celda Código: Exportar
add_code("""salidas_tablas = base_dir / "salidas" / "multivariante" / "tablas"
salidas_tablas.mkdir(parents=True, exist_ok=True)

# 1. Exportar tabla LaTeX lista para el manuscrito
tex_path = salidas_tablas / "tabla_multivariante_mercurio.tex"
pipeline.export_latex(
    str(tex_path),
    caption="Tamizaje bivariante no paramétrico y parámetros de los modelos multivariantes (LASSO y PLS-R) para la concentración basal de mercurio total en sangre (\\ce{Hg_T}).",
    label="tab:multivariante_mercurio"
)

# 2. Exportar reporte HTML interactivo
html_path = salidas_tablas / "reporte_multivariante_mercurio.html"
pipeline.export_html(str(html_path))

# 3. Exportar libro de Excel con hojas de tamizaje y modelos
excel_path = salidas_tablas / "resultados_multivariante_mercurio.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    screening_df.to_excel(writer, sheet_name="Cribado_Bivariante", index=False)
    fit_results['consolidated'].to_excel(writer, sheet_name="Modelos_Multivariantes", index=False)

print(f"¡Resultados multivariantes exportados exitosamente en: {salidas_tablas}!")
""")

nb_out = r"d:\vscode\results\resultados_mercurio\multivariante\01_analisis_multivariante.ipynb"
os.makedirs(os.path.dirname(nb_out), exist_ok=True)
with open(nb_out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook creado con éxito en: {nb_out}")

