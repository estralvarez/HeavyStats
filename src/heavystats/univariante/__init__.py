"""
Módulo de Análisis y Visualización Univariante de HeavyStats (`heavystats.univariante`).
Implementa tablas estadísticas (paramétricas y no paramétricas), pruebas de normalidad,
gráficos univariantes a 300 DPI y compilación de informes en PDF.
"""

from functools import partial
from heavystats.univariante.tables import (
    UnivariateTables,
    UnivariateTableReport,
)
from heavystats.univariante.plots import (
    BasePlots,
    UnivariatePlots,
    desglosar_multirrespuesta,
    obtener_mascara_percentiles,
    get_percentile_mask,
    mascara_percentiles,
)
from heavystats.univariante.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    CDC_BMI_REFERENCE,
    get_label,
)
from heavystats.reporte_pdf import unir_pilar_pdf, compilar_pilar_pdf

# Nombres y alias canónicos en español para máxima accesibilidad
TablasUnivariantes = UnivariateTables
GraficosUnivariantes = UnivariatePlots
ReporteTablaUnivariante = UnivariateTableReport

# Constantes en español
MAPA_ETIQUETAS_POR_DEFECTO = DEFAULT_LABELS_MAP
PARAMETROS_PERSONALIZADOS_POR_DEFECTO = DEFAULT_CUSTOM_PARAMS
LIMITES_PERMISIBLES_POR_DEFECTO = DEFAULT_PERMISSIBLE_LIMITS
REFERENCIA_IMC_CDC = CDC_BMI_REFERENCE
obtener_etiqueta = get_label

# Función especializada de compilación del pilar univariante
unir_univariante_pdf = partial(unir_pilar_pdf, "univariante")
compilar_univariante_pdf = unir_univariante_pdf

__all__ = [
    # Clases principales en español e inglés
    "TablasUnivariantes",
    "UnivariateTables",
    "GraficosUnivariantes",
    "UnivariatePlots",
    "ReporteTablaUnivariante",
    "UnivariateTableReport",
    "BasePlots",
    # Funciones auxiliares
    "desglosar_multirrespuesta",
    "obtener_mascara_percentiles",
    "get_percentile_mask",
    "mascara_percentiles",
    # Constantes
    "MAPA_ETIQUETAS_POR_DEFECTO",
    "PARAMETROS_PERSONALIZADOS_POR_DEFECTO",
    "LIMITES_PERMISIBLES_POR_DEFECTO",
    "REFERENCIA_IMC_CDC",
    "obtener_etiqueta",
    "DEFAULT_LABELS_MAP",
    "DEFAULT_CUSTOM_PARAMS",
    "DEFAULT_PERMISSIBLE_LIMITS",
    "CDC_BMI_REFERENCE",
    "get_label",
    # Compilación en PDF
    "unir_pilar_pdf",
    "compilar_pilar_pdf",
    "unir_univariante_pdf",
    "compilar_univariante_pdf",
]
