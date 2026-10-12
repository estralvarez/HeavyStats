"""
Plantilla para el cuaderno de Diagnóstico Quimiométrico y Valores Atípicos (Outliers).
Diseñado con calidad de publicación editorial a 300 DPI y alineado con heavystats.atipicos.
"""

from heavystats.studio.notebook_builder import crear_cuaderno


def generar_cuaderno_atipicos_diagnostico(cfg):
    """
    Construye multivariante/02_analisis_outliers_diagnostico.ipynb para el metal seleccionado.
    """
    nombre = cfg["nombre"]
    simbolo = cfg["simbolo"]
    key = cfg["key"]
    archivo_proc = cfg.get("archivo_proc", f"datos_{key}_n20.csv")
    col_conc = cfg["col_concentracion"]
    unidad = cfg["unidad"]
    limite = cfg.get("limite_permisible", 5.0)

    celdas = [
        {
            "tipo": "markdown",
            "contenido": f"""# Trabajo Especial de Grado — FACYT (Universidad de Carabobo)
## Departamento de Química — Cátedra de Instrumental y Toxicología
---
### Cuaderno 02: Diagnóstico Quimiométrico y Detección de Valores Atípicos (Outliers)
**Metal:** {nombre} ({simbolo})  
**Módulo:** `heavystats.atipicos` (accesible en español)

Este cuaderno ejecuta el protocolo completo de evaluación de atípicos e influencia:
1. **Detección Univariante**:
   - Vallas de Tukey (1.5 IQR y 3.0 IQR extremos).
   - Test de Grubbs exacto ($t$-Student bilateral).
   - Z-Scores y Modified Z-Scores (MAD robusto).
2. **Diagnóstico Quimiométrico Multivariante en Espacio Latente**:
   - Puntuaciones $T_1$ y $T_2$ con elipse de confianza.
   - Estadístico $T^2$ de Hotelling frente al umbral crítico $F$ de Snedecor al 95%.
   - Residuales $Q$ (SPE) frente al límite percentil 95.
   - Apalancamiento ($h_{{ii}}$) frente al umbral crítico $2k/n$.
   - Distancia de Cook ($D_i$) frente al límite de influencia $4/n$.
3. **Tabulación Analítica Editorial**:
   - Reporte univariante de consistencia (`ReporteAtipicosUnivariante`).
   - Reporte multivariante consolidado (`ReporteDiagnosticoAtipicos`).
4. **Figuras Científicas a 300 DPI**:
   - Vallas de Tukey univariantes.
   - Espacio latente de puntuaciones ($T_1$ vs $T_2$).
   - Espacio de influencia ($T^2$ vs $Q$).
   - Apalancamiento vs Distancia de Cook.
   - Dashboard integral multipanel 2x2."""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 1. Configuración del Entorno y Carga de HeavyStats

Importamos la biblioteca `heavystats` y los submódulos especializados de `atipicos`."""
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
from heavystats.atipicos import (
    DiagnosticoAtipicos,
    GraficosAtipicos,
    ReporteAtipicosUnivariante,
    ReporteDiagnosticoAtipicos,
    ReporteContrasteEstratos,
)

# Definición de rutas base y directorios de salida
base_dir = Path.cwd() if (Path.cwd() / "datos").exists() else Path.cwd().parent
OUTPUT_DIR = base_dir / "salidas" / "atipicos"
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
## 2. Inicialización del Motor de Diagnóstico Quimiométrico ({nombre})

Ejecutamos `DiagnosticoAtipicos` para cuantificar consistencia analítica univariante y multivariante."""
        },
        {
            "tipo": "code",
            "contenido": f"""data_candidates = [
    base_dir / "datos" / "procesados" / "{archivo_proc}",
    base_dir / "datos" / "procesados" / "datos_poblacion_procesados_N48.csv",
    base_dir / "datos" / "{cfg.get('archivo_datos', 'datos.csv')}",
]
data_path = next((p for p in data_candidates if p.exists()), None)
if not data_path:
    raise FileNotFoundError("No se localizó el archivo de datos analíticos.")

df_raw = hs.load_data(data_path)
target_metal = "{col_conc}"
df_metal = df_raw.dropna(subset=[target_metal]).copy()

diag = DiagnosticoAtipicos(
    df=df_metal,
    target_col=target_metal,
    id_col="Muestra_Codificada",
    n_components=2,
    alpha=0.05,
    who_threshold={limite}
)

kpis = diag.get_kpis()
print(f"Diagnóstico completado para N = {{kpis['n_obs']}} sujetos analíticos.")
print(f"Media ± DE: {{kpis['mean']:.3f}} ± {{kpis['std']:.3f}} {unidad} | Mediana (IQR): {{kpis['median']:.3f}} ({{kpis['iqr']:.3f}})")
print(f"Valla Superior de Tukey (1.5 IQR): {{kpis['tukey_fences'][1]:.3f}} {unidad}")
print(f"Test de Grubbs: G_calc = {{kpis['grubbs']['g_max']:.3f}}, G_crit = {{kpis['grubbs']['g_crit']:.3f}} (Atípico: {{kpis['grubbs']['reject_h0']}})")"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 3. Reporte Tabular Univariante de {nombre}

Generación y exportación a HTML editorial, CSV y JSON."""
        },
        {
            "tipo": "code",
            "contenido": f"""rep_uni = ReporteAtipicosUnivariante(
    diag,
    filepath_html=str(TABLAS_DIR / "diagnostico_univariante_{key}.html"),
    filepath_csv=str(TABLAS_DIR / "diagnostico_univariante_{key}.csv"),
    filepath_json=str(TABLAS_DIR / "diagnostico_univariante_{key}.json")
)
display(rep_uni)"""
        },
        {
            "tipo": "markdown",
            "contenido": """---
## 4. Reporte Quimiométrico Multivariante (Hotelling T², Apalancamiento y Cook)

Evaluación de observaciones extremas en el espacio multidimensional."""
        },
        {
            "tipo": "code",
            "contenido": f"""df_resumen = diag.to_dataframe()

rep_multi = ReporteDiagnosticoAtipicos(
    diag,
    filepath_html=str(TABLAS_DIR / "diagnostico_quimiometrico_{key}.html"),
    filepath_csv=str(TABLAS_DIR / "diagnostico_quimiometrico_{key}.csv"),
    filepath_json=str(TABLAS_DIR / "diagnostico_quimiometrico_{key}.json")
)

display(rep_multi)"""
        },
        {
            "tipo": "markdown",
            "contenido": f"""---
## 5. Figuras Diagnósticas a 300 DPI ({nombre})

Generamos las cinco figuras del panel quimiométrico."""
        },
        {
            "tipo": "code",
            "contenido": f"""op = GraficosAtipicos(diag)

# FIGURA 1: Distribución Univariante y Vallas de Tukey
fig1, _ = op.plot_univariate_fences(
    metal_label="{nombre} Plasmático [{unidad}]",
    filepath=str(GRAFICOS_DIR / "figura_01_univariante_tukey_{key}.png")
)
fig1.savefig(GRAFICOS_DIR / "figura_01_univariante_tukey_{key}.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 2: Espacio Latente de Puntuaciones (t1 vs t2)
fig2, _ = op.plot_latent_scores(
    filepath=str(GRAFICOS_DIR / "figura_02_espacio_latente_t1_t2.png")
)
fig2.savefig(GRAFICOS_DIR / "figura_02_espacio_latente_t1_t2.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 3: Diagrama de Influencia (T² vs Q)
fig3, _ = op.plot_influence_space(
    filepath=str(GRAFICOS_DIR / "figura_03_influencia_hotelling_q.png")
)
fig3.savefig(GRAFICOS_DIR / "figura_03_influencia_hotelling_q.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 4: Apalancamiento vs Distancia de Cook
fig4, _ = op.plot_leverage_cook(
    filepath=str(GRAFICOS_DIR / "figura_04_apalancamiento_cook.png")
)
fig4.savefig(GRAFICOS_DIR / "figura_04_apalancamiento_cook.pdf", dpi=300, bbox_inches="tight")
plt.show()

# FIGURA 5: Dashboard Integral Consolidado (2x2 a 300 DPI)
fig5, _ = op.plot_dashboard_2x2(
    metal_label="{nombre} Plasmático [{unidad}]",
    filepath=str(GRAFICOS_DIR / "figura_05_dashboard_diagnostico_2x2.png")
)
fig5.savefig(GRAFICOS_DIR / "figura_05_dashboard_diagnostico_2x2.pdf", dpi=300, bbox_inches="tight")
plt.show()"""
        }
    ]

    return crear_cuaderno(
        ruta="multivariante/02_analisis_outliers_diagnostico",
        titulo=f"Diagnóstico Quimiométrico y Valores Atípicos para {nombre} ({simbolo})",
        descripcion=f"Detección univariante (Tukey, Grubbs), diagnóstico multivariante (Hotelling T², Apalancamiento, Cook) y gráficos a 300 DPI.",
        celdas=celdas
    )
