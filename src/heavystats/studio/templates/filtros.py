"""
Plantilla para el cuaderno de Limpieza, Preprocesamiento y Filtrado (filtros.ipynb).
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_filtros(cfg):
    """
    Construye filtros.ipynb adaptado con la configuración del metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    archivo = cfg["archivo_datos"]
    archivo_proc = cfg["archivo_proc"]
    col_conc = cfg["col_concentracion"]
    col_risk = cfg["col_riesgo"]

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""---
## 1. Carga de Datos y Pipeline de Preprocesamiento ({nombre})

Aplicamos la secuencia metodológica completa de estandarización:
1. Normalización de campos de texto binario (`standardize_boolean_columns`).
2. Desagregación de respuestas múltiples separadas por punto y coma (`desaggregate_multiple_responses`).
3. Codificación ordinal de frecuencias alimentarias (`encode_dietary_frequencies`).
4. Creación de indicadores sintéticos de riesgo ambiental (`create_composite_indicators`)."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import heavystats as hs

base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
ruta_datos = base_dir / "datos" / "{archivo}"
df_raw = hs.load_data(ruta_datos)

# Pipeline de limpieza y transformación
df_proc = hs.standardize_boolean_columns(df_raw)
df_proc = hs.desaggregate_multiple_responses(
    df_proc,
    columns=["Salud_Transporte", "Salud_Agua", "Exposicion_Talleres", "Exposicion_Lugares", "Exposicion_Industrias"]
)
df_proc = hs.encode_dietary_frequencies(df_proc)
df_proc = hs.create_composite_indicators(df_proc)

print(f"Población total preprocesada: N = {{df_proc.shape[0]}} observaciones y {{df_proc.shape[1]}} columnas.")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. Extracción de Muestra Analítica y Filtrado Específico de {nombre}

Para el análisis toxicológico y multivariado riguroso, se aíslan los biomarcadores unimetálicos para evitar multicolinealidad estructural:
- `get_analytical_sample`: Extrae los participantes que cuentan con mediciones de laboratorio en sangre ($n=20$).
- `select_metal`: Filtra el dataset conservando exclusivamente la concentración de **{nombre}** (`{col_conc}`) y su correspondiente clasificación de riesgo (`{col_risk}`)."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Muestra analítica con biomarcadores séricos (n=20)
df_analytical = hs.get_analytical_sample(df_proc)

# Dataset unimetálico específico de {nombre}
df_metal = hs.select_metal(df_proc, concentration_col="{key}")

print(f"Muestra analítica general (n=20): {{df_analytical.shape}}")
print(f"Dataset unimetálico de {nombre} (n=20): {{df_metal.shape}}")
display(df_metal[["Muestra_Codificada", "{col_conc}", "{col_risk}"]].head())"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 3. Diagnóstico de Sesgo de Selección y Balance de Muestras

Evaluamos los desequilibrios basales entre los pacientes seleccionados para analítica de sangre ($n=20$) y los no seleccionados ($N-n=28$) mediante:
- **Diferencia Media Estandarizada (SMD)**: $|\\text{SMD}| > 0.10$ leve, $|\\text{SMD}| > 0.25$ moderado/severo.
- **Prueba t de Welch** para variables numéricas continuas.
- **Prueba Exacta de Fisher / Chi-cuadrado** para variables cualitativas."""
        },
        {
            "tipo": "code",
            "contenido": """# Evaluación formal de sesgo de selección
report_balance = hs.compare_groups(df_proc)
display(report_balance)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 4. Exportación de Datasets Limpios y Reportes de Balance

Exportamos los datasets procesados a la subcarpeta `datos/procesados/` para que los cuadernos de análisis univariante y bivariante los consuman directamente."""
        },
        {
            "tipo": "code",
            "contenido": f"""datos_proc = base_dir / "datos" / "procesados"
datos_proc.mkdir(parents=True, exist_ok=True)

salidas_qc = base_dir / "salidas" / "control_calidad"
salidas_qc.mkdir(parents=True, exist_ok=True)

# Exportar datasets en formato CSV estándar
df_proc.to_csv(datos_proc / "datos_poblacion_procesados_N48.csv", index=False, encoding="utf-8")
df_metal.to_csv(datos_proc / "{archivo_proc}", index=False, encoding="utf-8")

# Exportar reporte de balance
report_balance.to_excel(str(salidas_qc / "reporte_balance_sesgo_{key}.xlsx"))
report_balance.to_html(str(salidas_qc / "reporte_balance_sesgo_{key}.html"), full_page=True)

print(f"¡Dataset procesado de {nombre} exportado en: {{datos_proc / '{archivo_proc}'}}!")"""
        }
    ]

    return crear_cuaderno(
        ruta="filtros",
        titulo=f"Limpieza, Preprocesamiento y Filtrado de {nombre} ({simbolo})",
        descripcion=f"Pipeline de estandarización binaria, codificación ordinal, selección unimetálica de {nombre} y evaluación de sesgo muestral.",
        celdas=celdas
    )
