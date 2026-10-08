"""
Plantillas para los cuadernos de Análisis Bivariante (Estadística y Gráficos).
Diseñado con calidad de publicación editorial a 300 DPI y alineado con heavystats.bivariate.
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_bivariante_tablas(cfg):
    """
    Función deprecada: el análisis bivariante se enfoca exclusivamente en gráficos.
    """
    return None


def generar_cuaderno_bivariante_graficos(cfg):
    """
    Construye bivariante/02_graficos_bivariantes.ipynb para el metal seleccionado,
    alineado con la estructura analítica y las funciones del paquete heavystats.bivariate.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    archivo_datos = cfg["archivo_datos"]
    archivo_proc = cfg["archivo_proc"]
    col_conc = cfg["col_concentracion"]
    col_dieta = cfg["col_dieta_prioritaria"]
    nombre_dieta = cfg["nombre_dieta"]

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""# Análisis Bivariante: Visualización Epidemiológica y Toxicológica de Asociaciones ({nombre} - {simbolo})
### Paquete Bioestadístico HeavyStats (`heavystats.bivariate`)

Motor gráfico editorial a 300 DPI estructurado según los principios del análisis bivariante bioestadístico:
1. **Variable Cuantitativa vs. Variable Cualitativa**:
   - Contrastes no paramétricos (Mann–Whitney $U$, Kruskal–Wallis $H$, diferencias de medianas $\\Delta\\text{{Med}}$ y tamaños de efecto $r_{{rb}}$ / $\\epsilon^2$).
   - Contrastes paramétricos (prueba $t$ de Welch, ANOVA unidireccional, diferencia de medias y Hedges' $g$ / $\\eta^2$).
   - Anotación directa en corchetes (*brackets*), tamaños muestrales ($n=\\dots$) en ejes y estilización de publicaciones científicas.
2. **Hábitos Dietarios y Validación del Algoritmo de Riesgo**:
   - Gradientes ordinales dosis–respuesta (prueba de Jonckheere–Terpstra o Pearson).
   - Mosaicos facetados dietarios en cuadrícula (`layout="grid"`), perfiles polares (radar / spider charts) y regresión del score de riesgo.
3. **Variable Cuantitativa vs. Variable Cuantitativa**:
   - Dispersión continua, correlación de Spearman $\\rho_s$ con intervalos de confianza al 95% calculados por **Bootstrap (2,000 réplicas)**, Kendall $\\tau_b$ y regresión lineal OLS / Pearson $r$ ($R^2$).
   - Matrices de co-exposición bivariante: Heatmap triangular inferior estilizado y matriz de dispersión compacta con distribuciones KDE univariantes.
4. **Mosaico Bivariante Integrado y Funciones a Nivel de Módulo**:
   - Cuadrícula multipanel consolidada (`plot_grid`) y funciones de acceso directo funcional (`compare_groups_plot`, etc.).
5. **Análisis de Similitud Dietaria y Contrastes de Subcohorte de Alta Carga Corporal**:
   - Ranking de correlación / distancia frente a la concentración plasmática (`similarity_ranking_plot`).
   - Contraste modal vs resto de la muestra (`subcohort_contrast_plot`) con opciones paramétricas y no paramétricas.
   - Tamizaje sistemático automatizado de contrastes significativos (`scan_subcohort_contrasts`).
   - Consenso de factores comunes de riesgo (`shared_factors_plot`) y mosaico editorial de subcohorte (`subcohort_mosaic_plot`).
6. **Radiografía Epidemiológica de la Encuesta y Matriz de Síntesis Normalizada**:
   - Mapas de calor temáticos por bloque con columnas normalizadas $[0.0 - 1.0]$ (`survey_block_heatmap`).
   - Suite completa automatizada de los 7 bloques temáticos a 300 DPI (`plot_survey_radiography`).
   - Matriz de síntesis ejecutiva de exposición multidominio clasificada por carga biológica (`plot_survey_synthesis`)."""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 1. Carga de Datos y Configuración del Motor Gráfico Bivariante ({nombre})

Inicializamos `BivariatePlots` para renderizar figuras analíticas coordinadas a 300 DPI con estética editorial predefinida."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Asegurar carga del módulo de desarrollo de HeavyStats
heavystats_src = Path('d:/vscode/results/HeavyStats/src')
if heavystats_src.exists() and str(heavystats_src) not in sys.path:
    sys.path.insert(0, str(heavystats_src))

import heavystats as hs
from heavystats.bivariate import (
    BivariatePlots,
    compare_groups_plot,
    correlation_analysis_plot,
    coexposure_matrix_plot,
    diet_radar_plot,
    diet_boxplots_plot,
    similarity_ranking_plot,
    subcohort_contrast_plot,
    shared_factors_plot,
    subcohort_mosaic_plot,
    survey_block_heatmap_plot,
    survey_radiography_plot,
    survey_synthesis_plot,
    scan_subcohort_contrasts,
)

# Definición de rutas base y directorios de salida
base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_datos = base_dir / "datos" / "procesados" / "{archivo_proc}"

if not ruta_datos.exists():
    df_raw = hs.load_data(base_dir / "datos" / "{archivo_datos}")
    df_proc = hs.create_composite_indicators(
        hs.encode_dietary_frequencies(
            hs.desaggregate_multiple_responses(
                hs.standardize_boolean_columns(df_raw)
            )
        )
    )
    df_analytical = hs.select_metal(df_proc, concentration_col="{key}")
else:
    df_analytical = pd.read_csv(ruta_datos, encoding="utf-8")

# Instanciación del motor bivariante con paleta editorial predefinida
bp = BivariatePlots(
    df=df_analytical,
    palette="crest",
    style="ticks",
    context="notebook"
)

salidas_graficos = base_dir / "salidas" / "bivariante" / "graficos"
salidas_graficos.mkdir(parents=True, exist_ok=True)

print(f"Motor BivariatePlots inicializado exitosamente (N={{len(df_analytical)}} observaciones, {{len(df_analytical.columns)}} variables).")
print(f"Directorio de figuras configurado en: {{salidas_graficos}}")"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 2. Variable Cuantitativa vs. Variable Cualitativa (Carga Corporal)

Evaluación de la concentración sérica del biomarcador frente a factores sociodemográficos, clínicos y ambientales.
- **Factores Dicotómicos (`plot_binary`)**: Diagrama de cajas con puntos individuales (*stripplot*), medianas [RIQ], prueba de Mann–Whitney $U$, correlación biserial de rangos ($r_{rb}$) y diferencia de medianas. Diseñado estrictamente para contraste de 2 grupos (sin línea de límite toxicológico).
- **Factores Politómicos (`plot_categorical`)**: Comparación entre >2 grupos con prueba de Kruskal–Wallis $H$ y tamaño del efecto $\\epsilon^2$.
- **Contraste Paramétrico (`compare_groups(method="parametric")`)**: Medias $\\pm$ DE con prueba $t$ de Welch o ANOVA unidireccional y Hedges' $g$ / $\\eta^2$."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.1 Factores Dicotómicos: Concentración de {nombre} según Sexo (Femenino vs Masculino)
fig_sexo, ax_sexo = bp.plot_binary(
    metal="{col_conc}",
    group_col="Sexo",
    show_points=True,
    title="Concentración Sérica de {nombre} según Sexo",
    filepath=str(salidas_graficos / "boxplot_{key}_sexo.png")
)
fig_sexo.savefig(salidas_graficos / "boxplot_{key}_sexo.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.2 Factores Dicotómicos: {nombre} según Condición de Exposición Ocupacional/Ambiental
fig_exp, ax_exp = bp.plot_binary(
    metal="{col_conc}",
    group_col="Es_Expuesto",
    show_points=True,
    title="Niveles de {nombre} según Condición de Expuesto",
    filepath=str(salidas_graficos / "boxplot_{key}_es_expuesto.png")
)
fig_exp.savefig(salidas_graficos / "boxplot_{key}_es_expuesto.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.3 Factores Politómicos: Concentración de {nombre} según Sector Geográfico (>2 grupos)
fig_sec, ax_sec = bp.plot_categorical(
    metal="{col_conc}",
    group_col="Sector",
    show_points=True,
    title="Distribución de {nombre} por Sector de Residencia (Kruskal–Wallis)",
    filepath=str(salidas_graficos / "boxplot_{key}_sector.png")
)
fig_sec.savefig(salidas_graficos / "boxplot_{key}_sector.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.4 Contraste Paramétrico: Prueba t de Welch con Media ± DE y Tamaño de Efecto Hedges' g
fig_param, ax_param = bp.compare_groups(
    quantitative="{col_conc}",
    group="Sexo",
    method="parametric",
    show_points=True,
    title="Contraste Paramétrico: {nombre} por Sexo (Welch t / Hedges' g)",
    filepath=str(salidas_graficos / "boxplot_{key}_sexo_parametrico.png")
)
fig_param.savefig(salidas_graficos / "boxplot_{key}_sexo_parametrico.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.5 Factores de Exposición por Variables Dummy de Proximidad Ambiental
fig_dummies, ax_dummies = bp.plot_dummies(
    metal="{col_conc}",
    dimension="all",
    show_points=True,
    title="Carga de {nombre} según Fuentes Ambientales de Exposición",
    filepath=str(salidas_graficos / "boxplot_{key}_factores_ambientales.png")
)
fig_dummies.savefig(salidas_graficos / "boxplot_{key}_factores_ambientales.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 1.6 Interacción de Factores: Estratificación Cruzada [Sexo - Condición de Expuesto]
fig_inter, ax_inter = bp.compare_groups(
    quantitative="{col_conc}",
    group=["Sexo", "Es_Expuesto"],
    show_points=True,
    title="Interacción Sexo × Exposición frente a {nombre} en Sangre",
    filepath=str(salidas_graficos / "boxplot_{key}_interaccion_sexo_expuesto.png")
)
fig_inter.savefig(salidas_graficos / "boxplot_{key}_interaccion_sexo_expuesto.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. Hábitos Dietarios y Validación del Algoritmo de Riesgo

- **Gradiente Dosis–Respuesta**: Análisis de tendencia ordinal (no paramétrico con Jonckheere–Terpstra o paramétrico con Pearson).
- **Mosaicos Dietarios ("Todo en Uno")**: Formato facetado en cuadrícula (`layout="grid"`), con contrastes estadísticos seleccionables.
- **Perfil Multivariante Polar (Radar / Spider Chart)**: Comparación de patrones alimentarios mediante `ax.set_thetagrids()` estratificada por mediana o terciles de concentración.
- **Validación del Score de Riesgo**: Regresión empírica frente al biomarcador biológico con bandas de confianza al 95%."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.1 Tendencia Ordinal: Frecuencia de Consumo de {nombre_dieta} vs {nombre}
fig_dieta, ax_dieta = bp.plot_dietary(
    ordinal_col="{col_dieta}",
    metal="{col_conc}",
    method="nonparametric",  # o "parametric"
    show_points=True,
    filepath=str(salidas_graficos / "tendencia_ordinal_{key}_{col_dieta}.png")
)
fig_dieta.savefig(salidas_graficos / "tendencia_ordinal_{key}_{col_dieta}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.2 Mosaico Dietario Multivariable Facetado: Todos los Grupos de Alimentos vs {nombre}
fig_diet_grid, axes_diet_grid = bp.plot_diet_boxplots(
    metal="{col_conc}",
    method="nonparametric",  # o "parametric"
    layout="grid",
    ncols=4,
    filepath=str(salidas_graficos / "mosaico_dietario_grid_{key}.png")
)
fig_diet_grid.savefig(salidas_graficos / "mosaico_dietario_grid_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.4 Perfil Dietario en Coordenadas Polares (Radar Chart) Estratificado por Mediana
fig_radar, axes_radar = bp.plot_diet_radar(
    metal="{col_conc}",
    stratify_by="median",
    split_panels=True,
    figsize=(15.0, 7.5),
    filepath=str(salidas_graficos / "radar_dietario_{key}_split.png")
)
fig_radar.savefig(salidas_graficos / "radar_dietario_{key}_split.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.5 Validación Empírica del Algoritmo de Riesgo frente a Concentraciones Biológicas
fig_risk, axes_risk = bp.plot_risk_algorithm(
    metals=["{col_conc}"],
    figsize=(6.5, 4.5),
    filepath=str(salidas_graficos / "validacion_algoritmo_riesgo_{key}.png")
)
fig_risk.savefig(salidas_graficos / "validacion_algoritmo_riesgo_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 4. Variable Cuantitativa vs. Variable Cuantitativa (Gradientes Continuos y Co-Exposición)

- **Correlación Continua Bivariante**: Ajuste de regresión lineal con banda de confianza al 95%, coeficiente de Spearman $\\rho_s$ con IC 95% Bootstrap (2,000 remuestreos) y Kendall $\\tau_b$.
- **Matriz de Co-Exposición Bivariante**:
  - `kind="heatmap"`: Matriz triangular inferior con paleta divergente, coeficientes e intervalos Bootstrap.
  - `kind="scatter"`: Matriz compacta de dispersión bivariante con densidades KDE univariantes en la diagonal y resúmenes estadísticos en el triángulo superior."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 3.1 Dispersión y Correlación No Paramétrica: Edad vs Concentración de {nombre}
fig_corr_sp, ax_corr_sp = bp.correlation_analysis(
    x="Edad",
    y="{col_conc}",
    method="nonparametric",
    n_boot=2000,
    title="Correlación No Paramétrica: Edad vs {nombre} (Bootstrap IC 95%)",
    filepath=str(salidas_graficos / "dispersion_{key}_edad_spearman.png")
)
fig_corr_sp.savefig(salidas_graficos / "dispersion_{key}_edad_spearman.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 3.2 Regresión Paramétrica: Score de Riesgo vs {nombre} (Pearson r, Ecuación OLS y R²)
fig_corr_pr, ax_corr_pr = bp.correlation_analysis(
    x="Score_Riesgo",
    y="{col_conc}",
    method="parametric",
    title="Regresión Lineal OLS: Score de Riesgo vs {nombre} Sérico",
    filepath=str(salidas_graficos / "regresion_{key}_score_riesgo_pearson.png")
)
fig_corr_pr.savefig(salidas_graficos / "regresion_{key}_score_riesgo_pearson.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 3.3 Matriz de Co-Exposición Cuantitativa: Heatmap Triangular Editorial
cols_cuant = [c for c in ["{col_conc}", "Edad", "Score_Riesgo", "IMC", "Peso_kg", "Altura_cm"] if c in df_analytical.columns]

fig_coexp_heat, ax_coexp_heat = bp.plot_coexposure_matrix(
    variables=cols_cuant,
    kind="heatmap",
    method="nonparametric",
    n_boot=2000,
    title=r"Matriz de Co-Exposición Bivariante (Spearman $\\rho_s$ con IC 95% Bootstrap)",
    filepath=str(salidas_graficos / "matriz_coexposicion_{key}_heatmap.png")
)
fig_coexp_heat.savefig(salidas_graficos / "matriz_coexposicion_{key}_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 3.4 Matriz de Dispersión y Co-Exposición Compacta (Scatter + KDE + Estadísticos)
core_cuant = [c for c in ["{col_conc}", "Edad", "Score_Riesgo", "IMC"] if c in df_analytical.columns]

fig_coexp_scat, axes_coexp_scat = bp.plot_coexposure_matrix(
    variables=core_cuant,
    kind="scatter",
    method="nonparametric",
    n_boot=2000,
    title="Matriz Bivariante Compacta: Dispersión, Densidad y Correlación",
    filepath=str(salidas_graficos / "matriz_coexposicion_{key}_dispersion_compacta.png")
)
fig_coexp_scat.savefig(salidas_graficos / "matriz_coexposicion_{key}_dispersion_compacta.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 5. Mosaico Bivariante Integrado y Funciones a Nivel de Módulo

- **Cuadrícula Exploratoria Multideterminante (`plot_grid`)**: Agrupa en un único lienzo factores binarios, politómicos, continuos y ordinales.
- **Funciones de Conveniencia a Nivel de Módulo**: Acceso funcional directo sin necesidad de instanciar `BivariatePlots` (ideal para scripts analíticos y pipelines automatizados)."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 4.1 Mosaico Bivariante Integrado: Factores Sociodemográficos, Clínicos y Ambientales
fig_grid, axes_grid = bp.plot_grid(
    metal="{col_conc}",
    binary_cols=["Sexo", "Es_Expuesto"],
    categorical_cols=["Institucion", "Sector"],
    continuous_cols=["Edad", "Score_Riesgo"],
    ordinal_cols=["{col_dieta}"],
    ncols=2,
    filepath=str(salidas_graficos / "mosaico_editorial_bivariante_{key}.png")
)
fig_grid.savefig(salidas_graficos / "mosaico_editorial_bivariante_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 4.2 Uso de Funciones Funcionales Directas del Módulo (API Funcional sin Instanciación)
fig_fn_comp, ax_fn_comp = compare_groups_plot(
    df=df_analytical,
    quantitative="{col_conc}",
    group="Es_Expuesto",
    title="Llamada Funcional Directa: compare_groups_plot()"
)
plt.show()

fig_fn_corr, ax_fn_corr = correlation_analysis_plot(
    df=df_analytical,
    x="Score_Riesgo",
    y="{col_conc}",
    title="Llamada Funcional Directa: correlation_analysis_plot()"
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 6. Análisis de Similitud Dietaria y Contrastes de Subcohorte de Alta Carga Corporal ({nombre})

- **Ranking de Similitud (`similarity_ranking_plot`)**: Ordenación de frecuencias dietarias por coeficiente de correlación con {nombre}.
- **Contraste de Subcohorte (`subcohort_contrast_plot`)**: Contraste de la categoría modal de mayor consumo vs. el resto de la cohorte con pruebas no paramétricas (Mann–Whitney $U$) o paramétricas ($t$ de Welch).
- **Tamizaje Sistemático de Contrastes (`scan_subcohort_contrasts`)**: Evaluación automatizada en toda la batería de alimentos para identificar asociaciones significativas ($p < 0.05$).
- **Consenso de Factores Compartidos (`shared_factors_plot`)**: Minería de covariables de exposición compartidas en el estrato de mayor carga.
- **Mosaico Integrado de Subcohorte (`subcohort_mosaic_plot`)**: Panel multipanel consolidado (A, B y C)."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 5.1 Ranking de Similitud Dietaria frente a la Concentración Sérica de {nombre}
fig_sim, ax_sim, df_sim = bp.similarity_ranking_plot(
    target_metal="{col_conc}",
    method="nonparametric",
    title="Ranking de Asociación Dietaria vs Concentración de {nombre}",
    filepath=str(salidas_graficos / "ranking_similitud_{key}.png")
)
fig_sim.savefig(salidas_graficos / "ranking_similitud_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 5.2 Contraste de Subcohorte: Categoría Modal de Consumo de {nombre_dieta} vs Resto de la Muestra
fig_contrast, ax_contrast, res_contrast = bp.subcohort_contrast_plot(
    target_metal="{col_conc}",
    criterion_col="{col_dieta}",
    criterion_val=None,  # Selecciona automáticamente la categoría modal de mayor frecuencia
    method="nonparametric",  # Admite "nonparametric" o "parametric"
    annotate_ids=False,
    title="Contraste de Carga Corporal: {nombre} ({nombre_dieta} Modal vs Resto)",
    filepath=str(salidas_graficos / "contraste_subcohorte_{key}_{col_dieta}.png")
)
fig_contrast.savefig(salidas_graficos / "contraste_subcohorte_{key}_{col_dieta}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 5.3 Tamizaje Sistemático de Contrastes Dietarios Significativos (p < 0.05)
df_scan = bp.scan_subcohort_contrasts(
    target_metal="{col_conc}",
    method="nonparametric",
    alpha=0.05,
    plot_significant=False,
)
print(f"Tamizaje de contrastes completado ({{len(df_scan)}} variables evaluadas):")
display(df_scan.head(10))"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 5.4 Minería de Factores de Riesgo Compartidos en el Estrato Crítico de {nombre}
fig_shared, ax_shared, df_shared = bp.shared_factors_plot(
    target_metal="{col_conc}",
    filepath=str(salidas_graficos / "factores_compartidos_{key}.png")
)
fig_shared.savefig(salidas_graficos / "factores_compartidos_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 5.5 Mosaico Editorial Integrado de Subcohorte (Ranking + Contraste + Factores)
fig_sub_mosaic, axes_sub_mosaic, res_sub_mosaic = bp.subcohort_mosaic_plot(
    target_metal="{col_conc}",
    criterion_col="{col_dieta}",
    filepath=str(salidas_graficos / "mosaico_subcohorte_editorial_{key}.png")
)
fig_sub_mosaic.savefig(salidas_graficos / "mosaico_subcohorte_editorial_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 7. Radiografía Epidemiológica de la Encuesta y Matriz de Síntesis Normalizada ({nombre})

- **Mapas de Calor Normalizados (`survey_block_heatmap`)**: Columnas escaladas en $[0.0 - 1.0]$, columna totalizadora separada y corte visual por percentiles ($\\leq P_{{25}}$, $P_{{25}} - P_{{75}}$, $\\geq P_{{75}}$) según la concentración de {nombre}.
- **Suite Completa de 7 Bloques (`plot_survey_radiography`)**: Generación masiva automatizada de los 7 paneles temáticos a 300 DPI (PNG y PDF).
- **Matriz de Síntesis Ejecutiva (`plot_survey_synthesis`)**: Visualización consolidada de exposición total normalizada clasificada en orden ascendente de biomarcador plasmático."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 6.1 Radiografía Temática: Bloque 4 de Hábitos Dietarios
res_b4 = bp.plot_survey_radiography(
    target_metal="{col_conc}",
    blocks=["bloque4_dieta"],
    output_dir=salidas_graficos,
    show=True,
    dpi=300
)"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 6.2 Generación Masiva Automatizada de la Suite de 7 Bloques de la Encuesta a 300 DPI
suite_radiografia = bp.plot_survey_radiography(
    target_metal="{col_conc}",
    output_dir=salidas_graficos,
    show=False,
    dpi=300
)
print("Suite completa de radiografía generada exitosamente:")
for bloq_k in suite_radiografia.keys():
    print(f"  -> {{bloq_k}} exportado a {{salidas_graficos}} (PNG y PDF)")"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 6.3 Matriz Ejecutiva de Síntesis Global: Exposición Multidominio clasificada por {nombre}
fig_sintesis, ax_sintesis = bp.plot_survey_synthesis(
    target_metal="{col_conc}",
    output_dir=salidas_graficos,
    show=True,
    dpi=300
)
plt.show()

print(f"\\n¡Todas las figuras y radiografías bivariantes de {nombre} fueron generadas y exportadas con éxito en:\\n  -> {{salidas_graficos.resolve()}}")"""
        }
    ]

    return crear_cuaderno(
        ruta="bivariante/02_graficos_bivariantes",
        titulo=f"Análisis Bivariante: Visualización de Asociaciones para {nombre} ({simbolo})",
        descripcion=f"Visualizaciones bivariantes epidemiológicas: contrastes no paramétricos y paramétricos, gradientes dietarios ({nombre_dieta}), co-exposición y mosaicos multipanel.",
        celdas=celdas
    )
