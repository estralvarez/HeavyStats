"""
Plantilla para el cuaderno de Modelado Multivariante (PLS-1, VIP, Regularización).
Diseñado con calidad de publicación editorial a 300 DPI y alineado con heavystats.multivariante.
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_multivariante_pls(cfg):
    """
    Construye multivariante/01_analisis_multivariante_pls.ipynb para el metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    archivo_proc = cfg.get("archivo_proc", f"datos_{key}_n20.csv")
    col_conc = cfg["col_concentracion"]
    unidad = cfg["unidad"]
    tipo_analisis = cfg.get("tipo_analisis", "parametrico")

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""# Trabajo Especial de Grado — FACYT (Universidad de Carabobo)
## Departamento de Química — Cátedra de Instrumental y Toxicología
---
### Cuaderno 01: Modelado Multivariante Quimiométrico por Mínimos Cuadrados Parciales (PLS-1)
**Metal:** {nombre} ({simbolo})  
**Módulo:** `heavystats.multivariante` (accesible en español)

Este cuaderno implementa el análisis multivariante integrando:
1. **Preprocesamiento por Dominios** (`prepare_pls_dataframe`):
   - Hábitos dietarios estandarizados [0.0 - 1.0].
   - Hábitos del hogar, movilidad y fuentes hídricas binarizadas.
   - Indicadores consolidados agregados por dominio.
2. **Modelado Quimiométrico PLS-1** (`PlsModeler`):
   - Proyección sobre componentes latentes ortogonales.
   - Cálculo analítico riguroso del vector de Importancia de Variables en la Proyección (**VIP**).
3. **Tablas Editoriales** (`PlsVipReport`):
   - Exportación multiformato (.html, .csv, .json).
4. **Figuras de Diagnóstico a 300 DPI** (`MultivariatePlots`):
   - Ranking horizontal de VIP Scores con umbral crítico (VIP ≥ 1.0).
   - Biplot de pesos PLS (w1 vs w2) con proyección de respuesta.
   - Dispersión de valores observados vs predichos.
   - Cuadrantes epidemiológicos de impacto (Beta_PLS vs VIP)."""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 1. Configuración del Entorno y Carga de HeavyStats

Importamos la biblioteca `heavystats` y configuramos las rutas del proyecto."""
        },
        {
            "tipo": "code",
            "contenido": f"""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Asegurar carga de HeavyStats en desarrollo
heavystats_src = Path('../../src').resolve() if (Path('../../src').resolve() / "heavystats").exists() else Path('../src').resolve()
if heavystats_src.exists() and str(heavystats_src) not in sys.path:
    sys.path.insert(0, str(heavystats_src))

import heavystats as hs
from heavystats.multivariante import (
    prepare_pls_dataframe,
    PlsModeler,
    MultivariatePlots,
    PlsVipReport,
    DEFAULT_SURVEY_LABELS,
)

# Definición de rutas base y directorios de salida
base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
OUTPUT_DIR = base_dir / "salidas" / "multivariante"
TABLAS_DIR = OUTPUT_DIR / "tablas"
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
TABLAS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[OK] HeavyStats v{{hs.__version__}} cargado exitosamente.")
print(f"[OK] Directorio de tablas configurado en: {{TABLAS_DIR}}")
print(f"[OK] Directorio de figuras configurado en: {{GRAFICOS_DIR}}")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 2. Carga y Estructuración de la Matriz de Predictores ({nombre})

Cargamos la subcohorte analítica con determinaciones de {nombre} y estructuramos la matriz $X$ y el vector respuesta $y$."""
        },
        {
            "tipo": "code",
            "contenido": f"""# Búsqueda automática de la ruta de datos
data_candidates = [
    base_dir / "datos" / "procesados" / "{archivo_proc}",
    base_dir / "datos" / "procesados" / "datos_poblacion_procesados_N48.csv",
    base_dir / "datos" / "{cfg.get('archivo_datos', 'datos.csv')}",
]
data_path = next((p for p in data_candidates if p.exists()), None)
if not data_path:
    raise FileNotFoundError("No se localizó el archivo de datos analíticos.")

df_raw = hs.load_data(data_path)
print(f"Dataset cargado con éxito: {{df_raw.shape[0]}} registros y {{df_raw.shape[1]}} columnas.")

# Filtrar subcohorte con mediciones cuantitativas del metal
target_metal = "{col_conc}"
df_metal = df_raw.dropna(subset=[target_metal]).copy()
print(f"Subcohorte analítica con mediciones de {nombre}: N = {{len(df_metal)}}")

# Estructurar matriz numérica por dominios analíticos
df_pls = prepare_pls_dataframe(
    data=df_metal,
    target_metal=target_metal,
    id_col="Muestra_Codificada",
    include_domains_totals=True,
    include_individual_features=True,
    normalize_diet=True,
    log_target=False
)

feature_cols = [c for c in df_pls.columns if c not in ["Muestra_Codificada", target_metal]]
X = df_pls[feature_cols].copy()
y = df_pls[target_metal].values

print(f"Matriz X estructurada: {{X.shape[0]}} observaciones × {{X.shape[1]}} predictores.")
print(f"Respuesta y ({nombre}): Media = {{np.mean(y):.3f}} {unidad}, Mediana = {{np.median(y):.3f}} {unidad}.")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. Ajuste del Modelo Quimiométrico PLS-1 y Cálculo de VIP

Ajustamos la regresión por Mínimos Cuadrados Parciales con auto-escalado analítico."""
        },
        {
            "tipo": "code",
            "contenido": """pls = PlsModeler(n_components=2, scale=True)
pls.fit(X, y)

df_vip_summary = pls.summary_df
print(f"Modelo PLS ajustado exitosamente con {pls.n_components} componentes latentes.")
print(f"Coeficiente de Determinación R² = {pls.r2_ * 100:.2f}%")

# Vista preliminar de los factores con mayor influencia
display(df_vip_summary.head(10))"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 4. Generación y Exportación de Tablas de Relevancia (VIP)

Exportamos la tabla estructurada a HTML editorial (Booktabs), CSV y JSON."""
        },
        {
            "tipo": "code",
            "contenido": f"""report_vip = PlsVipReport(
    pls_model=pls,
    target_name=target_metal,
    filepath_html=str(TABLAS_DIR / "reporte_pls_vip_{key}.html"),
    filepath_csv=str(TABLAS_DIR / "tabla_pls_vip_{key}.csv"),
    filepath_json=str(TABLAS_DIR / "tabla_pls_vip_{key}.json")
)

# Guardar también el modelo serializado en JSON y CSV
pls.to_json(str(TABLAS_DIR / "modelo_pls_resumen.json"))
pls.to_csv(str(TABLAS_DIR / "modelo_pls_resumen.csv"))

print(f"[OK] Reportes y tablas exportados exitosamente en: {{TABLAS_DIR}}")
display(report_vip)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 5. Figuras Diagnósticas de Publicación a 300 DPI ({nombre})

Generamos los cuatro paneles gráficos principales para el análisis toxicológico y epidemiológico."""
        },
        {
            "tipo": "code",
            "contenido": f"""mp = MultivariatePlots(
    pls_model=pls,
    labels_map=DEFAULT_SURVEY_LABELS,
    target_label="{nombre} Plasmático [{unidad}]"
)

# FIGURA 1: Ranking VIP Horizontal
fig1, ax1 = mp.plot_vip_ranking(
    top_n=15,
    threshold=1.0,
    filepath=str(GRAFICOS_DIR / "figura_01_ranking_vip_{key}.png")
)
fig1.savefig(GRAFICOS_DIR / "figura_01_ranking_vip_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 2: Biplot de Pesos PLS (w1 vs w2)
fig2, ax2 = mp.plot_weights_biplot(
    w1_idx=0,
    w2_idx=1,
    top_labels=12,
    filepath=str(GRAFICOS_DIR / "figura_02_biplot_pesos_pls.png")
)
fig2.savefig(GRAFICOS_DIR / "figura_02_biplot_pesos_pls.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 3: Curva de Calibración: Observados vs Predichos
fig3, ax3 = mp.plot_calibration(
    y_obs=y,
    y_pred=pls.predict(X),
    filepath=str(GRAFICOS_DIR / "figura_03_observados_vs_predichos.png")
)
fig3.savefig(GRAFICOS_DIR / "figura_03_observados_vs_predichos.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 4: Cuadrantes Epidemiológicos de Impacto
fig4, ax4 = mp.plot_impact_quadrants(
    filepath=str(GRAFICOS_DIR / "figura_04_cuadrantes_impacto_vip.png")
)
fig4.savefig(GRAFICOS_DIR / "figura_04_cuadrantes_impacto_vip.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        }
    ]

    return crear_cuaderno(
        ruta="multivariante/01_analisis_multivariante_pls",
        titulo=f"Modelado Multivariante Quimiométrico PLS-1 para {nombre} ({simbolo})",
        descripcion=f"Regresión por Mínimos Cuadrados Parciales (PLS-1), deducción del vector VIP y selección de factores determinantes de {nombre}.",
        celdas=celdas
    )
