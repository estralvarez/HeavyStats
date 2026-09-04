"""
Constantes, mapas de etiquetas y configuraciones predeterminadas para análisis bivariantes.
"""

from typing import Dict, List, Any
from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)

# Variables primarias predefinidas (Análisis Primario a priori)
DEFAULT_PRIMARY_BINARY_VARS: List[str] = [
    "Sexo",
    "Sector",
    "Es_Expuesto",
    "Riesgo_Pb",
    "Riesgo_Hg",
    "Riesgo_Cd",
]

DEFAULT_PRIMARY_METALS: List[str] = [
    "Plomo_ug_dL",
    "Mercurio_ug_L",
    "Cadmio_ug_L",
]

# Grupos de variables exploratorias
DEFAULT_EXPLORATORY_GROUPS: Dict[str, List[str]] = {
    "Consumo Dietario (Frecuencias Ordinales)": [
        "Alim_Pescados",
        "Alim_Carnes",
        "Alim_Tuberculos",
        "Alim_Leguminosas",
        "Alim_Cereales",
        "Alim_Vegetales",
        "Alim_Frutas",
        "Alim_Lacteos",
        "Alim_Huevos",
        "Alim_Bebidas",
        "Alim_Azucar",
        "Alim_Grasas",
        "Alim_Chocolate",
    ],
    "Fuentes de Agua y Hábitos de Salud": [
        "Salud_Agua_filtrada",
        "Salud_Agua_mineral_embotellada",
        "Salud_Agua_pozo_profundo",
        "Salud_Transporte_caminar",
        "Salud_Transporte_publico",
        "Salud_Transporte_vehiculo",
        "Salud_Fuma",
        "Salud_Actividad",
        "Salud_Bombillos",
        "Salud_Techo",
        "Salud_Joyeria",
        "Salud_Suplementos_multivitaminico",
        "Salud_Suplementos_proteico",
    ],
    "Exposición Ocupacional y Ambiental en Entorno": [
        "Exposicion_Talleres_carpinteria",
        "Exposicion_Talleres_latoneria",
        "Exposicion_Talleres_mecanico",
        "Exposicion_Industrias_fabrica_metales",
        "Exposicion_Industrias_fabrica_productos_quimicos",
    ],
}

# Mapeo de niveles ordinales de consumo de alimentos
DIET_ORDINAL_MAP: Dict[str, int] = {
    "nunca": 0,
    "rara vez": 1,
    "a veces": 2,
    "frecuentemente": 3,
    "diario": 4,
}

DIET_ORDINAL_LABELS: Dict[int, str] = {
    0: "Nunca",
    1: "Rara vez",
    2: "A veces",
    3: "Frecuentemente",
    4: "Diario",
}

__all__ = [
    "DEFAULT_LABELS_MAP",
    "DEFAULT_CUSTOM_PARAMS",
    "DEFAULT_PERMISSIBLE_LIMITS",
    "DEFAULT_PRIMARY_BINARY_VARS",
    "DEFAULT_PRIMARY_METALS",
    "DEFAULT_EXPLORATORY_GROUPS",
    "DIET_ORDINAL_MAP",
    "DIET_ORDINAL_LABELS",
    "get_label",
]
