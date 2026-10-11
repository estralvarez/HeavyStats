"""
Módulo de Detección de Valores Atípicos y Diagnóstico Quimiométrico (`heavystats.atipicos`).
Implementa vallas de Tukey, prueba de Grubbs, distancias Hotelling T², residuales Q (SPE),
apalancamiento (Leverage hii), distancia de Cook, espacio de influencia quimiométrica,
gráficos a 300 DPI y compilación de informes en PDF.
"""

from functools import partial
from heavystats.atipicos.diagnostics import OutlierDiagnostics
from heavystats.atipicos.plots import OutlierPlots
from heavystats.atipicos.reports import (
    UnivariateOutlierReport,
    OutlierDiagnosticReport,
    StratifiedContrastReport,
)
from heavystats.reporte_pdf import unir_pilar_pdf, compilar_pilar_pdf

# Nombres canónicos en español para máxima accesibilidad
DiagnosticoAtipicos = OutlierDiagnostics
GraficosAtipicos = OutlierPlots
ReporteAtipicosUnivariante = UnivariateOutlierReport
ReporteDiagnosticoAtipicos = OutlierDiagnosticReport
ReporteContrasteEstratos = StratifiedContrastReport

# Compilador especializado de PDF para el pilar de valores atípicos
unir_atipicos_pdf = partial(unir_pilar_pdf, "atipicos")
compilar_atipicos_pdf = unir_atipicos_pdf

__all__ = [
    # Clases en español e inglés
    "DiagnosticoAtipicos",
    "OutlierDiagnostics",
    "GraficosAtipicos",
    "OutlierPlots",
    "ReporteAtipicosUnivariante",
    "UnivariateOutlierReport",
    "ReporteDiagnosticoAtipicos",
    "OutlierDiagnosticReport",
    "ReporteContrasteEstratos",
    "StratifiedContrastReport",
    # Compilación en PDF
    "unir_pilar_pdf",
    "compilar_pilar_pdf",
    "unir_atipicos_pdf",
    "compilar_atipicos_pdf",
]
