"""
Constantes, mapas de etiquetas, límites permisibles y configuraciones del modulo bivariante.
"""

from typing import Dict, List, Tuple, Any, Optional
from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)

# Metales principales analizados
DEFAULT_PRIMARY_METALS: List[str] = [
    "Plomo_ug_dL",
    "Mercurio_ug_L",
    "Cadmio_ug_L",
]

# Límites de referencia toxicológicos y normativos internacionales
CDC_LEAD_REFERENCE_VALUE: float = 3.5    # µg/dL (CDC 2021 BLRV para niños)
EPA_MERCURY_REFERENCE_VALUE: float = 5.0  # µg/L (EPA / OMS)
OMS_CADMIUM_REFERENCE_VALUE: float = 1.0  # µg/L (OMS / ATSDR)

DEFAULT_METAL_LIMITS: Dict[str, float] = {
    "Plomo_ug_dL": CDC_LEAD_REFERENCE_VALUE,
    "Mercurio_ug_L": EPA_MERCURY_REFERENCE_VALUE,
    "Cadmio_ug_L": OMS_CADMIUM_REFERENCE_VALUE,
    "Plomo": CDC_LEAD_REFERENCE_VALUE,
    "Mercurio": EPA_MERCURY_REFERENCE_VALUE,
    "Cadmio": OMS_CADMIUM_REFERENCE_VALUE,
    "pb": CDC_LEAD_REFERENCE_VALUE,
    "hg": EPA_MERCURY_REFERENCE_VALUE,
    "cd": OMS_CADMIUM_REFERENCE_VALUE,
}

# Límites de detección analíticos (LOD) por defecto
DEFAULT_METAL_LODS: Dict[str, float] = {
    "Plomo_ug_dL": 0.10,   # µg/dL
    "Mercurio_ug_L": 0.10,  # µg/L
    "Cadmio_ug_L": 0.05,   # µg/L
    "Plomo": 0.10,
    "Mercurio": 0.10,
    "Cadmio": 0.05,
    "pb": 0.10,
    "hg": 0.10,
    "cd": 0.05,
}

# Puntos de corte para clasificación toxicológica Bajo / Medio / Alto (Etapa 12)
# Formato: (corte_bajo_medio, corte_medio_alto)
# Plomo: Bajo (<1.0 µg/dL), Medio (1.0 a <3.5 µg/dL), Alto (≥3.5 µg/dL - CDC BLRV)
# Mercurio: Bajo (<1.0 µg/L), Medio (1.0 a <5.0 µg/L), Alto (≥5.0 µg/L - EPA/OMS)
# Cadmio: Bajo (<0.2 µg/L), Medio (0.2 a <1.0 µg/L), Alto (≥1.0 µg/L - OMS)
DEFAULT_METAL_CUTOFFS: Dict[str, Tuple[float, float]] = {
    "Plomo_ug_dL": (1.0, CDC_LEAD_REFERENCE_VALUE),
    "Mercurio_ug_L": (1.0, EPA_MERCURY_REFERENCE_VALUE),
    "Cadmio_ug_L": (0.2, OMS_CADMIUM_REFERENCE_VALUE),
    "Plomo": (1.0, CDC_LEAD_REFERENCE_VALUE),
    "Mercurio": (1.0, EPA_MERCURY_REFERENCE_VALUE),
    "Cadmio": (0.2, OMS_CADMIUM_REFERENCE_VALUE),
    "pb": (1.0, CDC_LEAD_REFERENCE_VALUE),
    "hg": (1.0, EPA_MERCURY_REFERENCE_VALUE),
    "cd": (0.2, OMS_CADMIUM_REFERENCE_VALUE),
}

# Pares de Co-Exposición Metal-Metal (Etapa 11)
DEFAULT_METAL_PAIRS: List[Tuple[str, str]] = [
    ("Plomo_ug_dL", "Mercurio_ug_L"),
    ("Plomo_ug_dL", "Cadmio_ug_L"),
    ("Mercurio_ug_L", "Cadmio_ug_L"),
]

# Codificación ordinal de frecuencias de consumo alimenticio (Etapa 7)
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

# Diccionario de grupos conceptuales de variables para análisis bivariante y tamizaje
DEFAULT_BIVARIATE_GROUPS: Dict[str, List[str]] = {
    "Variables Sociodemográficas y Antropométricas": [
        "Edad", "Sexo", "Sector", "Institucion", "Peso_kg", "Altura_cm", "IMC"
    ],
    "Factores de Exposición Ambiental y Talleres": [
        "Es_Expuesto",
        "Exposicion_Talleres_carpinteria",
        "Exposicion_Talleres_latoneria",
        "Exposicion_Talleres_mecanico",
        "Exposicion_Industrias_fabrica_metales",
        "Exposicion_Industrias_fabrica_productos_quimicos",
        "Exposicion_Lugares_canale",
        "Exposicion_Lugares_canales",
        "Exposicion_Lugares_estacion_gasolina",
        "Exposicion_Lugares_llenadora_gas_natural",
        "Exposicion_Lugares_rectificadora_motores",
        "Exposicion_Lugares_rios",
        "Exposicion_Cualquier_Taller",
        "Exposicion_Cualquier_Industria",
        "Exposicion_Cualquier_Lugar_Riesgo",
    ],
    "Hábitos de Salud y Fuentes de Agua": [
        "Salud_Agua_filtrada",
        "Salud_Agua_mineral_embotellada",
        "Salud_Agua_pozo_profundo",
        "Salud_Fuma",
        "Salud_Actividad",
        "Salud_Bombillos",
        "Salud_Techo",
        "Salud_Joyeria",
        "Salud_Transporte_caminar",
        "Salud_Transporte_publico",
        "Salud_Transporte_vehiculo",
        "Salud_Suplementos_multivitaminico",
        "Salud_Suplementos_proteico",
    ],
    "Frecuencia de Consumo Dietario (Ordinal 0-4)": [
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
    "Indicadores y Score del Algoritmo de Riesgo": [
        "Score_Riesgo",
        "Riesgo_Pb",
        "Riesgo_Hg",
        "Riesgo_Cd",
    ],
}

# Etiquetas extendidas bivariantes para presentación
EXTENDED_BIVARIATE_LABELS: Dict[str, str] = {
    "Exposicion_Cualquier_Taller": "Cualquier Taller en Entorno",
    "Exposicion_Cualquier_Industria": "Cualquier Industria en Entorno",
    "Exposicion_Cualquier_Lugar_Riesgo": "Cualquier Punto Crítico Ambiental",
    "Salud_Cualquier_Agua_Riesgo": "Fuente de Agua no Tratada / Pozo",
    "Score_Riesgo": "Score de Riesgo",
    "Nivel_Riesgo": "Nivel Categórico de Riesgo",
    "Riesgo_Pb": "Riesgo Plomo (Pb)",
    "Riesgo_Hg": "Riesgo Mercurio (Hg)",
    "Riesgo_Cd": "Riesgo Cadmio (Cd)",
    "IMC": "Índice de Masa Corporal (IMC)",
}

DEFAULT_LABELS_MAP.update(EXTENDED_BIVARIATE_LABELS)

__all__ = [
    "DEFAULT_PRIMARY_METALS",
    "CDC_LEAD_REFERENCE_VALUE",
    "EPA_MERCURY_REFERENCE_VALUE",
    "OMS_CADMIUM_REFERENCE_VALUE",
    "DEFAULT_METAL_LIMITS",
    "DEFAULT_METAL_LODS",
    "DEFAULT_METAL_CUTOFFS",
    "DEFAULT_METAL_PAIRS",
    "DIET_ORDINAL_MAP",
    "DIET_ORDINAL_LABELS",
    "DEFAULT_BIVARIATE_GROUPS",
    "DEFAULT_LABELS_MAP",
    "DEFAULT_CUSTOM_PARAMS",
    "DEFAULT_PERMISSIBLE_LIMITS",
    "get_label",
]
