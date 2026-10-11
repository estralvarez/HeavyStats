"""
Módulo de Análisis Multivariante de HeavyStats (`heavystats.multivariante`).
Implementa el flujo en dos etapas:
1. Tamizaje bivariante no paramétrico con control FDR de Benjamini-Hochberg.
2. Modelado multivariante conjunto mediante Regresión LASSO y Mínimos Cuadrados Parciales (PLS-R) con cálculo de VIP.
3. Generación de gráficos científicos a 300 DPI mediante `MultivariatePlots`.
4. Compilación de informes consolidados de tablas y gráficos en PDF.
"""

from functools import partial
from heavystats.multivariante.screening import MultivariateScreening
from heavystats.multivariante.models import (
    LassoModeler,
    PlsModeler,
    prepare_features_matrix,
    prepare_pls_dataframe,
)
from heavystats.multivariante.plots import MultivariatePlots
from heavystats.multivariante.reports import (
    MultivariateTableReport,
    PlsVipReport,
    DEFAULT_SURVEY_LABELS,
    UnivariateOutlierReport,
    OutlierDiagnosticReport,
    StratifiedContrastReport,
)
from heavystats.multivariante.pipeline import MultivariatePipeline
from heavystats.reporte_pdf import unir_pilar_pdf, compilar_pilar_pdf

# Nombres y alias canónicos en español para máxima accesibilidad
PipelineMultivariante = MultivariatePipeline
CribadoMultivariante = MultivariateScreening
ModeladorLasso = LassoModeler
ModeladorPls = PlsModeler
GraficosMultivariantes = MultivariatePlots
preparar_matriz_predictores = prepare_features_matrix
preparar_dataframe_pls = prepare_pls_dataframe
ReporteTablaMultivariante = MultivariateTableReport
ReporteVipPls = PlsVipReport

# Compilador especializado de PDF para el pilar multivariante
unir_multivariante_pdf = partial(unir_pilar_pdf, "multivariante")
compilar_multivariante_pdf = unir_multivariante_pdf

__all__ = [
    # Clases en español e inglés
    "PipelineMultivariante",
    "MultivariatePipeline",
    "CribadoMultivariante",
    "MultivariateScreening",
    "ModeladorLasso",
    "LassoModeler",
    "ModeladorPls",
    "PlsModeler",
    "GraficosMultivariantes",
    "MultivariatePlots",
    "ReporteTablaMultivariante",
    "MultivariateTableReport",
    "ReporteVipPls",
    "PlsVipReport",
    # Funciones de preparación
    "preparar_matriz_predictores",
    "prepare_features_matrix",
    "preparar_dataframe_pls",
    "prepare_pls_dataframe",
    "DEFAULT_SURVEY_LABELS",
    # Reportes de diagnóstico
    "UnivariateOutlierReport",
    "OutlierDiagnosticReport",
    "StratifiedContrastReport",
    # Compilación en PDF
    "unir_pilar_pdf",
    "compilar_pilar_pdf",
    "unir_multivariante_pdf",
    "compilar_multivariante_pdf",
]
