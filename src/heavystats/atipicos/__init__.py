"""
Módulo de Detección de Valores Atípicos y Diagnóstico Quimiométrico (`heavystats.atipicos`).
Implementa vallas de Tukey, prueba de Grubbs, distancias Hotelling T², residuales Q (SPE),
apalancamiento (Leverage hii), distancia de Cook, espacio de influencia quimiométrica
y gráficos a 300 DPI.
"""

from heavystats.atipicos.diagnostics import OutlierDiagnostics
from heavystats.atipicos.plots import OutlierPlots
from heavystats.atipicos.reports import (
    UnivariateOutlierReport,
    OutlierDiagnosticReport,
    StratifiedContrastReport,
)

# Nombres canónicos en español para máxima accesibilidad
DiagnosticoAtipicos = OutlierDiagnostics
GraficosAtipicos = OutlierPlots
ReporteAtipicosUnivariante = UnivariateOutlierReport
ReporteDiagnosticoAtipicos = OutlierDiagnosticReport
ReporteContrasteEstratos = StratifiedContrastReport

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
]
