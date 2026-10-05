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
   - Gradientes ordinales dietarios (prueba de Jonckheere–Terpstra), mosaicos de consumo y perfiles polares (radar / spider charts).
2. **Variable Cuantitativa vs. Variable Cuantitativa**:
   - Dispersión continua, correlación de Spearman $\\rho_s$ con intervalos de confianza al 95% calculados por **Bootstrap (2,000 réplicas)**, Kendall $\\tau_b$ y regresión lineal OLS / Pearson $r$ ($R^2$).
   - Matrices de co-exposición bivariante: Heatmap triangular inferior estilizado y matriz de dispersión compacta con distribuciones KDE univariantes.
3. **Mosaico Bivariante Integrado y Funciones a Nivel de Módulo**:
   - Cuadrícula multipanel consolidada (`plot_grid`) y funciones funcionales de acceso directo (`compare_groups_plot`, `correlation_analysis_plot`, `diet_radar_plot`, etc.)."""
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

- **Gradiente Dosis–Respuesta Monótono**: Prueba de Jonckheere–Terpstra ($JT$) para contrastar tendencias crecientes/decrecientes ordenadas por frecuencia de consumo.
- **Mosaicos Dietarios ("Todo en Uno")**: Formato facetado en cuadrícula (`layout="grid"`) o consolidado horizontal (`layout="consolidated"`), adoptando la paleta editorial predefinida.
- **Perfil Multivariante Polar (Radar / Spider Chart)**: Comparación de patrones alimentarios mediante `ax.set_thetagrids()` estratificada por mediana o terciles de concentración.
- **Validación del Score de Riesgo**: Regresión empírica frente al biomarcador biológico con bandas de confianza al 95%."""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.1 Tendencia Monótona Ordinal: Frecuencia de Consumo de {nombre_dieta} vs {nombre}
fig_dieta, ax_dieta = bp.plot_dietary(
    ordinal_col="{col_dieta}",
    metal="{col_conc}",
    show_points=True,
    title="Gradiente Ordinal de Ingesta de {nombre_dieta} vs {nombre} Sérico (Jonckheere–Terpstra)",
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
    layout="grid",
    ncols=4,
    filepath=str(salidas_graficos / "mosaico_dietario_grid_{key}.png")
)
fig_diet_grid.savefig(salidas_graficos / "mosaico_dietario_grid_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        },
        {
            "tipo": "code",
            "contenido": f"""# 2.3 Diagrama Dietario Consolidado Horizontal (Frecuente/Diario vs Ocasional/Nunca)
fig_diet_cons, ax_diet_cons = bp.plot_diet_boxplots(
    metal="{col_conc}",
    layout="consolidated",
    filepath=str(salidas_graficos / "diagrama_dietario_consolidado_{key}.png")
)
fig_diet_cons.savefig(salidas_graficos / "diagrama_dietario_consolidado_{key}.pdf", dpi=300, bbox_inches="tight")
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
plt.show()

print(f"\\n¡Figuras bivariantes de {nombre} guardadas exitosamente en:\\n  -> {{salidas_graficos.resolve()}}")"""
        }
    ]

    return crear_cuaderno(
        ruta="bivariante/02_graficos_bivariantes",
        titulo=f"Análisis Bivariante: Visualización de Asociaciones para {nombre} ({simbolo})",
        descripcion=f"Visualizaciones bivariantes epidemiológicas: contrastes no paramétricos y paramétricos, gradientes dietarios ({nombre_dieta}), co-exposición y mosaicos multipanel.",
        celdas=celdas
    )
