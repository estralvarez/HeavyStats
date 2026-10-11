<div align="center">

<img src="assets/logo.png" alt="HeavyStats Logo" width="180" />

# <span style="color: red;">HeavyStats</span>

**Framework estadístico y quimiométrico para el biomonitoreo de metales pesados y epidemiología toxicológica.**

</div>

[![PyPI Version](https://img.shields.io/pypi/v/heavystats.svg?color=blue)](https://pypi.org/project/heavystats/)
[![Python Versions](https://img.shields.io/pypi/pyversions/heavystats.svg)](https://pypi.org/project/heavystats/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-blueviolet.svg)](https://peps.python.org/pep-0008/)
[![Typing: Typed](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](https://peps.python.org/pep-0561/)

---

*Paquete bioestadístico integral en Python diseñado para el control de calidad analítico, bioestadística univariante, análisis bivariante y contrastes de subcohorte, modelado quimiométrico multivariante (PLS-1 / VIP) y diagnóstico de valores atípicos en biomonitoreo de metales pesados (Plomo, Mercurio, Cadmio) en salud infantil y poblaciones vulnerables.*

---

## Características Principales

- **HeavyStats Studio & CLI**: Entorno de terminal interactivo (TUI basada en Textual) y orquestador automatizado para generar proyectos completos con sus 4 pilares analíticos y **7 cuadernos reproducibles** organizados por metal (Plomo, Mercurio, Cadmio).
- **Control de Calidad Automatizado**: Validación metodológica formal mediante 10 reglas sistemáticas (límites analíticos LOD/LOQ, rangos biológicos plausibles, consistencia antropométrica CDC/OMS y trazabilidad de duplicados).
- **Preprocesamiento Epidemiológico y por Dominios**: Limpieza de encuestas, decodificación estandarizada, desagregación de multirrespuestas, matrices dietarias normalizadas e ingeniería de indicadores compuestos de exposición.
- **Bioestadística Univariante con Calidad de Publicación**:
  - Resúmenes robustos (mediana, IQR, media ± DE, percentiles biológicos y contraste frente a umbrales CDC/OMS/EPA).
  - Tablas editoriales formateadas (estilo APA/médico) con visualización HTML forzada en modo claro, exportables a Excel, CSV y LaTeX (`booktabs`).
  - Figuras descriptivas a 300 DPI (histogramas con densidad KDE, boxplots con umbrales toxicológicos y gráficos Q-Q).
- **Bioestadística Bivariante y Contrastes de Subcohorte**:
  - Pruebas no paramétricas y paramétricas con anotación directa en corchetes (*brackets*): Mann-Whitney $U$, Kruskal-Wallis, Dunn *post-hoc*, $t$ de Welch y ANOVA de 1 vía.
  - Gradientes ordinales dosis-respuesta (Jonckheere-Terpstra y Pearson).
  - Correlaciones con intervalos de confianza al 95% calculados por **Bootstrap (2,000 réplicas)**: Spearman $\rho_s$, Kendall $\tau_b$, Pearson $r$ y matrices térmicas de co-exposición.
  - Análisis dietario y de subcohorte: perfiles polares (radar charts), rankings de correlación y distancia (`similarity_ranking_plot`), y tamizaje automatizado de contrastes significativos en sujetos de alta carga corporal (`scan_subcohort_contrasts`).
- **Modelado Quimiométrico Multivariante (PLS-1 & VIP)**:
  - Regresión por Mínimos Cuadrados Parciales (PLS-1) con proyección en componentes latentes y auto-escalado.
  - Deducción analítica del vector de **Importancia de Variables en la Proyección (VIP Scores)** con umbral de relevancia crítica ($\text{VIP} \ge 1.0$).
  - Biplots de pesos $w_1$ vs $w_2$, cuadrantes epidemiológicos de impacto ($\beta_{\text{PLS}}$ vs $\text{VIP}$) y curvas observados vs predichos a 300 DPI.
  - Regularización lineal (Lasso) y screening automatizado de predictores relevantes.
- **Diagnóstico Quimiométrico y Detección de Valores Atípicos**:
  - Detección univariante: Vallas de Tukey (1.5 IQR y 3.0 IQR), Test exacto de Grubbs bilateral y Z-scores modificados por MAD.
  - Diagnóstico multivariante en espacio latente: Puntuaciones con elipse de confianza al 95%, distancia $T^2$ de Hotelling frente al umbral crítico $F$ de Snedecor, residuales $Q$ (SPE), apalancamiento ($h_{ii}$) y distancia de Cook ($D_i$).
- **Compilación Editorial en PDF (`unir_pilar_pdf`)**: Consolidación automática de tablas y figuras por pilar analítico en informes PDF multipágina con portada institucional.

---

## Instalación

Instala la versión estable desde PyPI con `pip`:

```bash
pip install heavystats
```

O utilizando `uv`:

```bash
uv add heavystats
```

Para desarrollo local o modo editable:

```bash
git clone https://github.com/estralvarez/HeavyStats.git
cd HeavyStats
uv pip install -e .
```

---

## HeavyStats Studio (TUI & CLI)

HeavyStats incluye un estudio interactivo de terminal para inicializar, auditar y generar automáticamente proyectos completos de investigación estructurados en 4 pilares y **7 cuadernos reproducibles**.

### Ejecución Interactiva (TUI)

```bash
# Lanzar la interfaz gráfica de terminal
uv run heavystats

# Comando alternativo equivalente
uv run heavystats-studio
```

### Ejecución Directa por Línea de Comandos (CLI / Headless)

```bash
# Generar proyecto para un metal específico sin interfaz visual
uv run heavystats -m plomo --no-tui
uv run heavystats -m mercurio --no-tui
uv run heavystats -m cadmio --no-tui

# Consultar ayuda y opciones
uv run heavystats --help
```

### Iniciar desde Python

```python
import heavystats as hs

# Lanza la aplicación interactiva Textual
hs.launch_studio()
```

### Jerarquía Generada por HeavyStats Studio

Al seleccionar un metal (ej. Mercurio), Studio genera la estructura analítica normalizada:

```text
├── datos/
│   ├── datos_{metal}.csv
│   └── procesados/
├── salidas/
│   ├── control_calidad/
│   ├── univariante/
│   │   ├── tablas/
│   │   └── graficos/
│   ├── bivariante/
│   │   └── graficos/
│   ├── multivariante/
│   │   ├── tablas/
│   │   └── graficos/
│   └── atipicos/
│       ├── tablas/
│       └── graficos/
├── validacion.ipynb                                  # Auditoría formal (10 reglas de calidad)
├── filtros.ipynb                                      # Selección de muestra analítica
├── univariante/
│   ├── 01_tablas.ipynb                               # Resumen descriptivo y umbrales CDC/OMS
│   └── 02_graficos.ipynb                             # Distribuciones, boxplots y Q-Q a 300 DPI
├── bivariante/
│   └── 02_graficos_bivariantes.ipynb                 # Contrastes, correlaciones y subcohorte
└── multivariante/
    ├── 01_analisis_multivariante_pls.ipynb           # Modelado PLS-1 y ranking VIP
    └── 02_analisis_outliers_diagnostico.ipynb        # Diagnóstico quimiométrico de atípicos
```

---

## 🚀 Guía de Inicio Rápido (Quickstart)

### 1. Carga y Control de Calidad de Datos

```python
import heavystats as hs

# Cargar dataset (usa el dataset integrado si no se especifica ruta)
df = hs.load_data()

# Ejecutar auditoría metodológica (10 reglas de calidad)
reporte = hs.validate_data(df)

# Visualizar en Jupyter / VSCode / Positron
reporte.show()

# Exportar reporte
reporte.to_excel("salidas/control_calidad/reporte_calidad.xlsx")
reporte.to_text("salidas/control_calidad/reporte_calidad.txt")
```

---

### 2. Análisis Univariante (Tablas y Gráficos)

```python
from heavystats.univariante import UnivariateTables, UnivariatePlots

# Tablas descriptivas con percentiles y límites permisibles
tables = UnivariateTables(df)
tabla_hg = tables.metals_table()
tabla_hg.show()
tabla_hg.to_excel("salidas/univariante/tablas/tabla_metales.xlsx")

# Figuras de distribución a 300 DPI
plots = UnivariatePlots(df)
fig = plots.plot_distribution("Mercurio_ug_L")
```

---

### 3. Bioestadística Bivariante y Contrastes de Subcohorte

```python
from heavystats.bivariante import (
    mann_whitney_test,
    spearman_matrix,
    compare_groups_plot,
    similarity_ranking_plot,
    scan_subcohort_contrasts,
)

# Prueba de hipótesis no paramétrica entre grupos
res_mw = mann_whitney_test(
    df=df,
    group_col="Sector",
    val_col="Mercurio_ug_L",
    group_a="Zona_A",
    group_b="Zona_B"
)
print(res_mw)

# Matriz de co-exposición con correlación de Spearman
matriz = spearman_matrix(df, metal_cols=["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L"])
matriz.show()

# Gráfico comparativo editorial con corchetes y valores p
fig_comp = compare_groups_plot(
    data=df,
    x_col="Frecuencia_Pescado",
    y_col="Mercurio_ug_L",
    metal="Mercurio",
    non_parametric=True,
    annotate_brackets=True
)

# Tamizaje sistemático de factores determinantes en subcohorte de alta exposición
contrastes = scan_subcohort_contrasts(
    data=df,
    target_metal="Mercurio_ug_L",
    percentile_threshold=75,
    top_n=10
)
contrastes.show()
```

---

### 4. Modelado Quimiométrico Multivariante (PLS-1 & VIP)

```python
from heavystats.multivariante import (
    prepare_pls_dataframe,
    PlsModeler,
    MultivariatePlots,
    PlsVipReport,
)

# Estructurar matriz numérica por dominios estandarizados
df_pls = prepare_pls_dataframe(
    data=df.dropna(subset=["Mercurio_ug_L"]),
    target_metal="Mercurio_ug_L",
    id_col="Muestra_Codificada",
    include_domains_totals=True,
    normalize_diet=True
)

X = df_pls.drop(columns=["Muestra_Codificada", "Mercurio_ug_L"])
y = df_pls["Mercurio_ug_L"].values

# Ajustar regresión PLS-1 con 2 componentes latentes y auto-escalado
pls = PlsModeler(n_components=2, scale=True)
pls.fit(X, y)
print(f"R² = {pls.r2_ * 100:.2f}%")

# Reporte tabular con ranking VIP
reporte_vip = PlsVipReport(pls, vip_threshold=1.0)
reporte_vip.show()
reporte_vip.to_html("salidas/multivariante/tablas/vip_report.html")

# Gráficos multivariantes a 300 DPI (VIP ranking, Biplot, Cuadrantes de Impacto)
mv_plots = MultivariatePlots(pls)
fig_vip = mv_plots.plot_vip_scores(top_n=15, threshold=1.0)
fig_biplot = mv_plots.plot_biplot()
fig_quad = mv_plots.plot_impact_quadrants()
```

---

### 5. Diagnóstico Quimiométrico de Valores Atípicos e Influencia

```python
from heavystats.atipicos import (
    DiagnosticoAtipicos,
    GraficosAtipicos,
    ReporteAtipicosUnivariante,
    ReporteDiagnosticoAtipicos,
)

# Inicializar motor de diagnóstico univariante y multivariante
diag = DiagnosticoAtipicos(
    df=df.dropna(subset=["Mercurio_ug_L"]),
    target_col="Mercurio_ug_L",
    id_col="Muestra_Codificada",
    n_components=2,
    alpha=0.05,
    who_threshold=5.0
)

# Reportes editoriales
rep_uni = ReporteAtipicosUnivariante(diag)
rep_uni.show()

rep_multi = ReporteDiagnosticoAtipicos(diag)
rep_multi.show()

# Visualización científica a 300 DPI (Hotelling T², Q-residuals, Cook, Apalancamiento)
graficos_out = GraficosAtipicos(diag)
fig_latente = graficos_out.scores_latent_space()
fig_influencia = graficos_out.t2_vs_q_residual_plot()
fig_cook = graficos_out.leverage_vs_cook_plot()
fig_dashboard = graficos_out.multipanel_dashboard()
```

---

### 6. Consolidación Editorial en PDF

```python
from heavystats.reporte_pdf import unir_pilar_pdf
from pathlib import Path

# Compilar todas las tablas y figuras del pilar en un informe único
pdf_path = unir_pilar_pdf(
    pilar="multivariante",
    ruta_pdf=Path("salidas/multivariante/informe_multivariante_consolidado.pdf"),
    directorio_salidas=Path("salidas/multivariante"),
    titulo="PILAR MULTIVARIANTE: MODELADO PLS-1 Y RANKING VIP",
    subtitulo="Determinantes de Exposición a Mercurio en Sangre",
    autor="Br. Jose Daniel Estrada Alvarez",
    tutor="Dr. Jose Ramon Jimenez Ochoa"
)
print(f"Informe compilado: {pdf_path}")
```

---

## Módulos de la Biblioteca

HeavyStats implementa una arquitectura modular con nombres canónicos en español y alias equivalentes en inglés:

| Módulo Canónico | Alias Inglés | Descripción |
| :--- | :--- | :--- |
| `heavystats.studio` | — | Entorno TUI interactivo (Textual) y generador de proyectos bioestadísticos. |
| `heavystats.cli` | — | Interfaz de línea de comandos unificada (`heavystats`, `heavystats-studio`). |
| `heavystats.validation` | — | Auditoría y control de calidad formal bajo 10 reglas metodológicas sistemáticas. |
| `heavystats.cleaning` | — | Limpieza, tipificación de encuestas e ingeniería de indicadores compuestos. |
| `heavystats.univariante` | `heavystats.univariate` | Bioestadística univariante, tablas estilo APA, percentiles y gráficos de distribución. |
| `heavystats.bivariante` | `heavystats.bivariate` | Pruebas no paramétricas, correlación bootstrap, gráficos anotados y análisis de subcohorte. |
| `heavystats.multivariante` | `heavystats.multivariate` | Modelado PLS-1, cálculo analítico de VIP, regularización Lasso y biplots 300 DPI. |
| `heavystats.atipicos` | `heavystats.outliers` | Detección univariante (Tukey, Grubbs) y quimiométrica ($T^2$ de Hotelling, $Q$, Cook, apalancamiento). |
| `heavystats.comparation` | — | Comparación poblacional de riesgo y análisis de exposición por sectores. |
| `heavystats.reporte_pdf` | — | Motor de consolidación y compilación de informes PDF institucionales con portada. |
| `heavystats.html_utils` | — | Renderizado HTML editorial con soporte forzado para visualización clara en Jupyter/IDE. |

---

## Pruebas Automatizadas

La biblioteca cuenta con pruebas unitarias exhaustivas con `pytest`:

```bash
uv run pytest tests/ -v
```

---

## Cita y Referencias

Si utilizas `heavystats` en tus investigaciones científicas, tesis de grado o proyectos epidemiológicos, por favor cita la biblioteca:

```bibtex
@software{heavystats2026,
  author = {Estrada Alvarez, Jose Daniel},
  title = {HeavyStats: Framework bioestadístico y quimiométrico para el biomonitoreo de metales pesados en salud infantil},
  year = {2026},
  version = {0.5.0},
  url = {https://github.com/estralvarez/HeavyStats}
}
```

---

## Licencia

Este proyecto está bajo la Licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

