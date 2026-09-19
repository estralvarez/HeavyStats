"""
Plantillas para los cuadernos de Análisis Bivariante (Estadística y Gráficos).
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_bivariante_tablas(cfg):
    """
    Construye bivariante/01_analisis_estadistico.ipynb para el metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    archivo_datos = cfg["archivo_datos"]
    archivo_proc = cfg["archivo_proc"]
    col_conc = cfg["col_concentracion"]
    nombre_dieta = cfg["nombre_dieta"]

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""---
## 1. Carga de Datos y Configuración del Motor Bivariante ({nombre})

Cargamos la muestra analítica e instanciamos `BivariateTables` para ejecutar contrastes no paramétricos rigurosos adaptados a muestras pequeñas ($n=20$) y concentraciones asimétricas."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import heavystats as hs
from heavystats.bivariate import BivariateTables

base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_datos = base_dir / "datos" / "procesados" / "{archivo_proc}"

if not ruta_datos.exists():
    df_raw = hs.load_data(base_dir / "datos" / "{archivo_datos}")
    df_proc = hs.create_composite_indicators(hs.encode_dietary_frequencies(hs.desaggregate_multiple_responses(hs.standardize_boolean_columns(df_raw))))
    df_analytical = hs.select_metal(df_proc, concentration_col="{key}")
else:
    df_analytical = pd.read_csv(ruta_datos)

bt = BivariateTables(df_analytical)
salidas_tablas = base_dir / "salidas" / "bivariante" / "tablas"
salidas_tablas.mkdir(parents=True, exist_ok=True)

print(f"Motor BivariateTables listo para {nombre} ({simbolo}).")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. {nombre} vs Variables Cuantitativas y Antropométricas

Cálculo de correlaciones de Spearman ($\\rho_s$) con intervalos de confianza al 95% calculados mediante remuestreo **Bootstrap no paramétrico (2,000 réplicas)** frente a Edad, Peso, Talla, IMC y Score de Riesgo."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_cont = bt.continuous_summary(metal="{col_conc}")
display(report_cont)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. {nombre} vs Factores Categóricos y Binarios de Exposición

Contrastes de hipótesis no paramétricos:
- **Mann-Whitney U** y correlación biserial por rangos ($r_{{rb}}$) para factores dicotómicos (Sexo, Condición de expuesto, Proximidad a talleres).
- **Kruskal-Wallis H** y tamaño de efecto $\\epsilon^2$ con comparaciones post-hoc de Dunn para variables politómicas (Sectores, Instituciones)."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_bin = bt.binary_summary(metal="{col_conc}")
report_cat = bt.categorical_summary(metal="{col_conc}")

display(report_bin)
display(report_cat)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 4. {nombre} vs Hábitos Dietarios ({nombre_dieta})

Análisis de correlación ordinal y prueba de tendencia monótona de Jonckheere-Terpstra frente a la ingesta dietaria específica."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_diet = bt.dietary_summary(metal="{col_conc}")
display(report_diet)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 5. Validación Empírica del Algoritmo de Riesgo de Metales ({nombre})

Contraste entre las categorías de riesgo calculadas a priori y los niveles séricos reales medidos en sangre."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_risk = bt.risk_score_summary(metal="{col_conc}")
display(report_risk)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 6. Co-Exposición Inter-Metálica (Plomo ↔ Mercurio ↔ Cadmio)

Evaluación de la sincronía toxicológica entre biomarcadores séricos para caracterizar co-exposiciones ambientales conjuntas."""
        },
        {
            "tipo": "code",
            "contenido": """report_metals = bt.metal_correlations()
display(report_metals)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 7. Matriz Maestra de Screening Bivariante y Ranking de Variables

Integración de todos los predictores evaluados, ajuste por tasa de falso descubrimiento (**FDR Benjamini-Hochberg**, $q < 0.10$) y priorización multicriterio de factores de riesgo."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_master = bt.master_association_matrix(metal="{col_conc}")
report_ranking = bt.variable_ranking(metal="{col_conc}")

display(report_master)
display(report_ranking)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 8. Diagnóstico de Colinealidad y Preparación de Bloques para PCA / PLS

Identificación de redundancias y preparación de las matrices $X$ (exposición) e $Y$ (respuesta sérica) para el modelado multivariante parsimonioso."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_pca = bt.pca_candidates(metal="{col_conc}")
report_pls = bt.pls_candidates(metal="{col_conc}")

display(report_pca)
display(report_pls)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 9. Exportación de Resultados Bivariantes ({nombre})

Guardamos todos los reportes consolidados en `salidas/bivariante/tablas/`."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_master.to_excel(str(salidas_tablas / "matriz_maestra_{key}.xlsx"))
report_master.to_html(str(salidas_tablas / "matriz_maestra_{key}.html"), full_page=True)
report_ranking.to_excel(str(salidas_tablas / "ranking_variables_{key}.xlsx"))
report_diet.to_excel(str(salidas_tablas / "asociacion_dietaria_{key}.xlsx"))

print(f"¡Reportes bivariantes de {nombre} exportados exitosamente en: {{salidas_tablas}}!")"""
        }
    ]

    return crear_cuaderno(
        ruta="bivariante/01_analisis_estadistico",
        titulo=f"Análisis Bivariante: Tablas y Pruebas Estadísticas para {nombre} ({simbolo})",
        descripcion=f"Contraste de hipótesis no paramétricas (Spearman con Bootstrap, Mann-Whitney, Kruskal-Wallis, Dunn, FDR) y screening maestro para {nombre}.",
        celdas=celdas
    )


def generar_cuaderno_bivariante_graficos(cfg):
    """
    Construye bivariante/02_graficos_bivariantes.ipynb para el metal seleccionado.
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
            "contenido": f"""---
## 1. Carga de Datos y Configuración del Motor Gráfico Bivariante ({nombre})

Inicializamos `BivariatePlots` para renderizar figuras analíticas coordinadas a 300 DPI con anotación de límites de referencia toxicológica."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import heavystats as hs
from heavystats.bivariate import BivariatePlots, BivariateTables

base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_datos = base_dir / "datos" / "procesados" / "{archivo_proc}"

if not ruta_datos.exists():
    df_raw = hs.load_data(base_dir / "datos" / "{archivo_datos}")
    df_proc = hs.create_composite_indicators(hs.encode_dietary_frequencies(hs.desaggregate_multiple_responses(hs.standardize_boolean_columns(df_raw))))
    df_analytical = hs.select_metal(df_proc, concentration_col="{key}")
else:
    df_analytical = pd.read_csv(ruta_datos)

bp = BivariatePlots(df_analytical)
bt = BivariateTables(df_analytical)
salidas_graficos = base_dir / "salidas" / "bivariante" / "graficos"
salidas_graficos.mkdir(parents=True, exist_ok=True)

print(f"Motor BivariatePlots listo para figuras de {nombre} ({simbolo}).")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. {nombre} vs Variables Cuantitativas (Dispersión y Correlación)

Gráficos de dispersión continua con ajuste monotónico, stripplot y bandas de confianza para Edad, Score de Riesgo e IMC."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Dispersión {nombre} vs Edad
fig_edad, ax_edad = bp.plot_continuous(
    metal="{col_conc}",
    column="Edad",
    filepath=str(salidas_graficos / "dispersion_{key}_edad.png")
)
plt.show()

# Dispersión {nombre} vs Score de Riesgo
fig_score, ax_score = bp.plot_continuous(
    metal="{col_conc}",
    column="Score_Riesgo",
    filepath=str(salidas_graficos / "dispersion_{key}_score_riesgo.png")
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. {nombre} vs Factores Categóricos (Boxplots Estratificados)

Distribución de concentraciones de {nombre} estratificada por Institución Educativa y Sector residencial."""
        },
        {
            "tipo": "code",
            "contenido": f"""fig_inst, ax_inst = bp.plot_categorical(
    metal="{col_conc}",
    group_col="Institucion",
    filepath=str(salidas_graficos / "boxplot_{key}_institucion.png")
)
plt.show()

fig_sec, ax_sec = bp.plot_categorical(
    metal="{col_conc}",
    group_col="Sector",
    filepath=str(salidas_graficos / "boxplot_{key}_sector.png")
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 4. Tendencia Ordinal Dietaria: {nombre_dieta} vs {nombre} en Sangre

Visualización de la gradiente de acumulación sérica según la frecuencia de ingesta alimentaria (Escala 0: Nunca a 4: Diario)."""
        },
        {
            "tipo": "code",
            "contenido": f"""fig_dieta, ax_dieta = bp.plot_dietary(
    metal="{col_conc}",
    dietary_col="{col_dieta}",
    filepath=str(salidas_graficos / "tendencia_{key}_{col_dieta}.png")
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 5. Matriz de Co-Exposición Inter-Metálica

Matriz de correlación gráfica entre Plomo, Mercurio y Cadmio."""
        },
        {
            "tipo": "code",
            "contenido": """fig_matriz, ax_matriz = bp.plot_metal_matrix(
    filepath=str(salidas_graficos / "matriz_coexposicion_metales.png")
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 6. Visualización de Screening: Gráfico Volcano y Forest Plot de Efectos ({nombre})

- **Volcano Plot**: Magnitud de la asociación ($r_{{rb}} / \\rho_s$) frente a la significancia estadística ($-\\log_{{10}}(p)$) con umbrales de FDR.
- **Forest Plot**: Intervalos de confianza al 95% para los predictores priorizados en el ranking."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Obtener matriz de screening
report_master = bt.master_association_matrix(metal="{col_conc}")
df_screening = report_master.df

# Volcano Plot
fig_volcano, ax_volcano = bp.plot_volcano(
    metal="{col_conc}",
    screening_df=df_screening,
    filepath=str(salidas_graficos / "volcano_plot_{key}.png")
)
plt.show()

# Forest Plot de Efectos
fig_forest, ax_forest = bp.plot_forest_effects(
    metal="{col_conc}",
    screening_df=df_screening,
    filepath=str(salidas_graficos / "forest_plot_{key}.png")
)
plt.show()

print(f"¡Figuras bivariantes de {nombre} guardadas exitosamente en: {{salidas_graficos}}!")"""
        }
    ]

    return crear_cuaderno(
        ruta="bivariante/02_graficos_bivariantes",
        titulo=f"Análisis Bivariante: Visualización de Asociaciones para {nombre} ({simbolo})",
        descripcion=f"Boxplots estratificados, dispersión con remuestreo Bootstrap, tendencia dietaria ({nombre_dieta}), Volcano y Forest plots.",
        celdas=celdas
    )
