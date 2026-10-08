"""
Plantillas para los cuadernos de Análisis Univariante (Tablas y Gráficos).
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_univariante_tablas(cfg):
    """
    Construye univariante/01_tablas.ipynb para el metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    unidad = cfg["unidad"]
    archivo_datos = cfg["archivo_datos"]
    archivo_proc = cfg["archivo_proc"]
    param_metal = cfg["param_metal"]
    lod = cfg["lod"]
    entidad = cfg["entidad_referencia"]

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""---
## 1. Carga de Datos y Configuración del Motor Univariante ({nombre})

Cargamos los datasets limpios y preprocesados desde `datos/procesados/` e inicializamos la clase `UnivariateTables`."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import heavystats as hs
from heavystats.univariate import UnivariateTables

base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_poblacion = base_dir / "datos" / "procesados" / "datos_poblacion_procesados_N48.csv"
ruta_metal = base_dir / "datos" / "procesados" / "{archivo_proc}"

# Fallback si aún no se generaron los procesados
if not ruta_metal.exists():
    df_raw = hs.load_data(base_dir / "datos" / "{archivo_datos}")
    df_total = hs.create_composite_indicators(hs.encode_dietary_frequencies(hs.desaggregate_multiple_responses(hs.standardize_boolean_columns(df_raw))))
    df_analytical = hs.select_metal(df_total, concentration_col="{key}")
else:
    df_total = pd.read_csv(ruta_poblacion, encoding="utf-8") if ruta_poblacion.exists() else None
    df_analytical = pd.read_csv(ruta_metal, encoding="utf-8")

tables = UnivariateTables(df_analytical, df_total=df_total)
print(f"Motor UnivariateTables listo para {nombre} ({simbolo}).")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. Perfil Toxicológico de {nombre} en Sangre ({simbolo}, {unidad})

Evaluamos los biomarcadores séricos de {nombre} considerando:
- **Límite de Detección Analítico (LOD)**: {lod}.
- **Límite de Referencia Internacional {entidad}**.
- Parámetros descriptivos: Mediana [RIQ], Media (DE) y percentiles $p_5$ a $p_{{95}}$."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Resumen toxicológico unimetálico de {nombre}
report_metal = tables.metal_summary(columns="{param_metal}")
display(report_metal)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 3. Caracterización de Variables Categóricas y Factores de Exposición

Frecuencias absolutas y relativas $n\\ (\\%)$ para la muestra analítica ($n=20$) y comparación con la población general $N\\ (\\%)$ ($N=48$), organizadas en bloques:
- Sociodemográficas (Sexo, Sector, Institución).
- Exposición y Riesgo (Condición de expuesto, proximidad industrial).
- Fuentes de agua y hábitos."""
        },
        {
            "tipo": "code",
            "contenido": """# Resumen por bloques conceptuales
report_cat = tables.categorical_summary()
display(report_cat)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 4. Parámetros de Variables Numéricas Continuas y Normalidad

Cálculo de Media (DE) y Mediana [RIQ], evaluadas mediante Asimetría ($\\gamma_1$), Curtosis ($\\gamma_2$) y prueba de normalidad de Shapiro-Wilk ($W, p$)."""
        },
        {
            "tipo": "code",
            "contenido": """report_num = tables.numerical_summary()
display(report_num)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 5. Evaluación Formal de la Transformación Logarítmica $\\ln(\\text{{{simbolo}}})$

Diagnóstico formal para justificar si la transformación $\\ln(\\text{{{nombre}}})$ normaliza la distribución y estabiliza la varianza para modelos lineales."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_log = tables.log_transform_evaluation(columns="{param_metal}")
display(report_log)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 6. Clasificación Nutricional e IMC Pediátrico (CDC)

Determinación de percentiles de IMC ajustados por edad y sexo según las curvas pediátricas del CDC."""
        },
        {
            "tipo": "code",
            "contenido": """report_bmi = tables.bmi_summary()
display(report_bmi)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 7. Exportación de Tablas Científicas ({nombre})

Exportamos todas las tablas generadas a la carpeta `salidas/univariante/tablas/` en formato HTML Booktabs, Excel y CSV."""
        },
        {
            "tipo": "code",
            "contenido": f"""salidas_tablas = base_dir / "salidas" / "univariante" / "tablas"
salidas_tablas.mkdir(parents=True, exist_ok=True)

report_metal.to_html(str(salidas_tablas / "tabla_toxicologica_{key}.html"), full_page=True)
report_metal.to_excel(str(salidas_tablas / "tabla_toxicologica_{key}.xlsx"))

report_cat.to_excel(str(salidas_tablas / "resumen_categorico.xlsx"), sheet_name="Categoricas")
report_num.to_excel(str(salidas_tablas / "resumen_numerico.xlsx"), sheet_name="Numericas")
report_log.to_html(str(salidas_tablas / "evaluacion_logaritmica_{key}.html"), full_page=True)

print(f"¡Tablas univariantes de {nombre} exportadas con éxito en: {{salidas_tablas}}!")"""
        }
    ]

    return crear_cuaderno(
        ruta="univariante/01_tablas",
        titulo=f"Análisis Univariante: Tablas Descriptivas y Epidemiológicas de {nombre} ({simbolo})",
        descripcion=f"Perfil toxicológico de {nombre} en sangre, caracterización sociodemográfica, diagnóstico de normalidad y evaluación logarítmica.",
        celdas=celdas
    )


def generar_cuaderno_univariante_graficos(cfg):
    """
    Construye univariante/02_graficos.ipynb para el metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    unidad = cfg["unidad"]
    archivo_datos = cfg["archivo_datos"]
    archivo_proc = cfg["archivo_proc"]
    col_conc = cfg["col_concentracion"]
    limite = cfg["limite_permisible"]
    entidad = cfg["entidad_referencia"]
    key = cfg["key"]

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""---
## 1. Carga de Datos y Configuración del Motor Gráfico ({nombre})

Cargamos la muestra analítica de {nombre} y configuramos `UnivariatePlots` con paleta científica de alta legibilidad editorial."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import heavystats as hs
from heavystats.univariate import UnivariatePlots

base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_metal = base_dir / "datos" / "procesados" / "{archivo_proc}"

if not ruta_metal.exists():
    df_raw = hs.load_data(base_dir / "datos" / "{archivo_datos}")
    df_analytical = hs.select_metal(hs.standardize_boolean_columns(df_raw), concentration_col="{key}")
else:
    df_analytical = pd.read_csv(ruta_metal, encoding="utf-8")

uplots = UnivariatePlots(df_analytical, palette="crest")
salidas_graficos = base_dir / "salidas" / "univariante" / "graficos"
salidas_graficos.mkdir(parents=True, exist_ok=True)

print(f"Motor UnivariatePlots listo para generación de figuras de {nombre} a 300 DPI.")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. Pipeline de Distribución: Boxplot e Histograma de {nombre} con Límite {entidad} y Máscara de Percentiles

Visualización integrada en dos paneles coordinados que combina la dispersión individual (jitter points), cuartiles, mediana, histograma con KDE, umbral de referencia toxicológico ({limite} {unidad}) y **máscara de estratificación por percentiles** ($\\leq P_{{25}}$, $> P_{{25}} \\text{{ y }} < P_{{75}}$, $\\geq P_{{75}}$) con bandas sombreadas, líneas verticales y conteo de muestras con porcentajes."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Pipeline de Distribución con límite permisible, escala natural y máscara de percentiles
figs_metal = uplots.plot_distribucion(
    columns="{col_conc}",
    percentile_mask=True,
    show_limit=True,
    log_scale=False,
    color="#2b6cb0",
    save_dir=str(salidas_graficos),
    save_format=["png", "pdf"],
    shrink=0.8,
    bins=5,
    dpi=300
)
plt.show()

# Resumen analítico de los estratos percentilares de {nombre}
mascara_info = uplots.obtener_mascara_percentiles(column="{col_conc}")
print("Estratificación por percentiles (<=P25, P25-P75, >=P75):")
print(f"  P25 = {{mascara_info['p_lower']:.3f}}, P75 = {{mascara_info['p_upper']:.3f}}")
for estrato, count in mascara_info['counts'].items():
    pct = mascara_info['percentages'][estrato]
    print(f"  {{estrato}}: n={{count}} ({{pct:.1f}}%)")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. Histograma Estratificado con KDE por Factor de Exposición ({nombre})

Visualización de la distribución sérica de {nombre} estratificada por covariables clave (como Sexo), utilizando barras apiladas (`multiple="stack"`), separación visual (`shrink=0.8`) y curvas de densidad KDE superpuestas para contrastar subpoblaciones."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Histograma estratificado por Sexo con barras apiladas y estimación de densidad KDE
figs_hist_sexo = uplots.plot_histograma_estratificado(
    columns="{col_conc}",
    hue="Sexo",
    multiple="stack",
    shrink=0.8,
    kde=True,
    palette="Set1",
    save_dir=str(salidas_graficos),
    save_format=["png", "pdf"],
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 4. Gráfico Q-Q Normal de {nombre}

Evaluación visual del ajuste de la distribución observada frente a los cuantiles teóricos de la distribución Normal."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Gráficos Q-Q para {nombre}
figs_qq = uplots.plot_qq(
    columns="{col_conc}",
    save_dir=str(salidas_graficos),
    save_format=["png", "pdf"],
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 5. Distribución de Variables Categóricas

Diagramas de barras horizontales y verticales con frecuencias absolutas y relativas para las covariables sociodemográficas."""
        },
        {
            "tipo": "code",
            "contenido": """# Visualización de covariables categóricas
figs_cat = uplots.plot_categorico(
    columns=["Sexo", "Sector", "Institucion"],
    save_dir=str(salidas_graficos),
    save_format=["png", "pdf"],
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 6. Pipeline Multirrespuesta: Fuentes de Agua

Frecuencia y porcentaje de acceso a fuentes de agua para consumo en la población escolar y comunitaria."""
        },
        {
            "tipo": "code",
            "contenido": """# Gráfico de selección múltiple para fuentes de agua
if "Salud_Agua" in df_analytical.columns:
    fig_agua = uplots.plot_multirrespuesta(
        "Salud_Agua",
        title="Fuentes de Agua para Consumo Humano",
        save_dir=str(salidas_graficos),
        save_format=["png", "pdf"],
        dpi=300
    )
    plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 7. Pipeline de Factores de Exposición Agrupados

Factores ambientales y ocupacionales agrupados por dimensiones analíticas (talleres, industrias, lugares de riesgo)."""
        },
        {
            "tipo": "code",
            "contenido": """# Factores de exposición agrupados
fig_factores = uplots.plot_factores(
    dimensions={
        "Exposicion_Lugares": "Lugares de Riesgo",
        "Exposicion_Talleres": "Talleres y Servicios",
        "Exposicion_Industrias": "Industrias Químicas/Metales"
    },
    save_dir=str(salidas_graficos),
    save_format=["png", "pdf"],
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 8. Pipeline Mosaico de Determinantes de Riesgo

Mosaico multipanel que integra el consumo de pescado/mariscos, bombillos ahorradores y convivencia con fumadores."""
        },
        {
            "tipo": "code",
            "contenido": """# Mosaico de determinantes toxicológicos mediante mosaico y plot_categorico
fig_determinantes, axd = uplots.crear_mosaico(
    [["pescado", "pescado"],
     ["bombillos", "fumadores"]],
    figsize=(10.5, 7.2),
    height_ratios=(1.25, 0.95),
    bold=True
)

uplots.plot_categorico(
    "Alim_Pescados",
    ax=axd["pescado"],
    title="(a) Consumo de Pescado y Mariscos",
    color="#2b6cb0"
)
uplots.plot_categorico(
    "Salud_Bombillos",
    ax=axd["bombillos"],
    title="(b) Bombillos Ahorradores en el Hogar",
    order=["No", "Sí"],
    color="#319795"
)
uplots.plot_categorico(
    "Salud_Fuma",
    ax=axd["fumadores"],
    title="(c) Convivencia con Fumadores en el Hogar",
    order=["No", "Sí"],
    color="#4a5568"
)

fig_determinantes.tight_layout()
plt.show()

print(f"¡Figuras univariantes guardadas exitosamente en: {salidas_graficos}!")"""
        }
    ]

    return crear_cuaderno(
        ruta="univariante/02_graficos",
        titulo=f"Análisis Univariante: Gráficos de Distribución de {nombre} ({simbolo})",
        descripcion=f"Histogramas, boxplots integrados con límite permisible {entidad} y gráficos Q-Q a 300 DPI.",
        celdas=celdas
    )
