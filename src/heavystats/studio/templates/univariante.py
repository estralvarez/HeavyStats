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
    df_total = pd.read_csv(ruta_poblacion) if ruta_poblacion.exists() else None
    df_analytical = pd.read_csv(ruta_metal)

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
- Parámetros robustos: Media Geométrica (GM), Desviación Estándar Geométrica (GSD), Mediana [RIQ] y percentiles $p_5$ a $p_{{95}}$."""
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
    df_analytical = pd.read_csv(ruta_metal)

uplots = UnivariatePlots(df_analytical, palette="crest")
salidas_graficos = base_dir / "salidas" / "univariante" / "graficos"
salidas_graficos.mkdir(parents=True, exist_ok=True)

print(f"Motor UnivariatePlots listo para generación de figuras de {nombre} a 300 DPI.")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. Diagrama Combinado: Boxplot e Histograma de {nombre} con Límite {entidad}

Visualización que integra la dispersión individual (jitter points), cuartiles, mediana y la referencia de seguridad toxicológica ({limite} {unidad})."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Boxplot + Histograma con límite permisible anotado
figs_metal = uplots.plot_box_histograms(
    columns="{col_conc}",
    show_limit=True,
    permissible_limit={limite},
    save_dir=str(salidas_graficos),
    save_format="png",
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. Gráfico Q-Q Normal de {nombre}

Evaluación visual del ajuste de la distribución observada frente a los cuantiles teóricos de la distribución Normal."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Gráficos Q-Q para {nombre}
figs_qq = uplots.plot_qq(
    columns="{col_conc}",
    save_dir=str(salidas_graficos),
    save_format="png",
    dpi=300
)
plt.show()"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 4. Distribución de Variables Categóricas y Factores de Exposición

Gráficos de barras con frecuencias absolutas y porcentajes para las principales covariables estratificadas."""
        },
        {
            "tipo": "code",
            "contenido": """# Visualización de covariables categóricas
figs_cat = uplots.plot_categorical(
    columns=["Sexo", "Sector", "Institucion"],
    save_dir=str(salidas_graficos),
    save_format="png",
    dpi=300
)
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
