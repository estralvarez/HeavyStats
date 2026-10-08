"""
Módulo de Análisis Multivariante de HeavyStats (`heavystats.multivariante`).
Implementa el flujo en dos etapas:
1. Tamizaje bivariante no paramétrico con control FDR de Benjamini-Hochberg.
2. Modelado multivariante conjunto mediante Regresión LASSO y Mínimos Cuadrados Parciales (PLS-R) con cálculo de VIP.
"""

from heavystats.multivariante.screening import MultivariateScreening
from heavystats.multivariante.models import (
    LassoModeler,
    PlsModeler,
    prepare_features_matrix,
)
from heavystats.multivariante.reports import MultivariateTableReport
from heavystats.multivariante.pipeline import MultivariatePipeline

__all__ = [
    "MultivariatePipeline",
    "MultivariateScreening",
    "LassoModeler",
    "PlsModeler",
    "prepare_features_matrix",
    "MultivariateTableReport",
]
