"""
Constantes y configuraciones por defecto para análisis y gráficos univariantes.
"""
from typing import Dict, Optional

# Referencias del IMC Pediátrico de la CDC (2000/2022 Extendida) para edades de 5 a 10 años
# Valores de percentiles P5, P85 y P95 al punto medio del año de edad (ej. 5.5 años / 66 meses).
CDC_BMI_REFERENCE = {
    "masculino": {
        5: {"P5": 13.84, "P85": 16.84, "P95": 17.94},
        6: {"P5": 13.74, "P85": 17.01, "P95": 18.41},
        7: {"P5": 13.72, "P85": 17.40, "P95": 19.15},
        8: {"P5": 13.80, "P85": 17.96, "P95": 20.07},
        9: {"P5": 13.96, "P85": 18.63, "P95": 21.09},
        10: {"P5": 14.22, "P85": 19.39, "P95": 22.15},
    },
    "femenino": {
        5: {"P5": 13.52, "P85": 16.80, "P95": 18.26},
        6: {"P5": 13.43, "P85": 17.10, "P95": 18.84},
        7: {"P5": 13.43, "P85": 17.63, "P95": 19.68},
        8: {"P5": 13.54, "P85": 18.32, "P95": 20.70},
        9: {"P5": 13.74, "P85": 19.12, "P95": 21.82},
        10: {"P5": 14.04, "P85": 19.98, "P95": 22.98},
    }
}

# Parámetros de estilo rcParams por defecto (eliminación de spines superior y derecho y sin cuadrícula)
DEFAULT_CUSTOM_PARAMS: Dict[str, bool] = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": False,
}

# Límites permisibles / valores de referencia internacionales de metales en sangre (CDC / OMS / EPA)
DEFAULT_PERMISSIBLE_LIMITS: Dict[str, float] = {
    "Plomo_ug_dL": 5.0,      # µg/dL (CDC/OMS nivel de referencia en sangre)
    "Mercurio_ug_L": 5.0,    # µg/L (EPA/OMS valor de referencia)
    "Cadmio_ug_L": 1.0,      # µg/L (OMS valor de referencia poblacional)
    "Plomo": 5.0,
    "Mercurio": 5.0,
    "Cadmio": 1.0,
    "pb": 5.0,
    "hg": 5.0,
    "cd": 1.0,
}

# Diccionario por defecto para renombrar variables eliminando snake_case y formateando unidades
DEFAULT_LABELS_MAP: Dict[str, str] = {
    # Metales Pesados
    "Plomo_ug_dL": r"Concentración de Plomo en Sangre ($\mu$g/dL)",
    "Mercurio_ug_L": r"Concentración de Mercurio en Sangre ($\mu$g/L)",
    "Cadmio_ug_L": r"Concentración de Cadmio en Sangre ($\mu$g/L)",
    "Plomo": r"Plomo ($\mu$g/dL)",
    "Mercurio": r"Mercurio ($\mu$g/L)",
    "Cadmio": r"Cadmio ($\mu$g/L)",
    "pb": r"Plomo ($\mu$g/dL)",
    "hg": r"Mercurio ($\mu$g/L)",
    "cd": r"Cadmio ($\mu$g/L)",
    # Antropometría y Datos Clínicos
    "Edad": "Edad",
    "Peso_kg": "Peso (kg)",
    "Altura_cm": "Altura (cm)",
    "Score_Riesgo": "Score de Riesgo",
    "Muestra_Codificada": "ID Muestra",
    # Sociodemográficas
    "Sexo": "Sexo",
    "Sector": "Sector",
    "Institucion": "Institución",
    # Exposición
    "Es_Expuesto": "Es Expuesto",
    "Riesgo_Pb": "Riesgo Plomo (Pb)",
    "Riesgo_Hg": "Riesgo Mercurio (Hg)",
    "Riesgo_Cd": "Riesgo Cadmio (Cd)",
    "Exposicion_Talleres": "Exposición a Talleres",
    "Exposicion_Talleres_carpinteria": "Taller de Carpintería",
    "Exposicion_Talleres_latoneria": "Taller de Latonería",
    "Exposicion_Talleres_mecanico": "Taller Mecánico",
    "Exposicion_Industrias": "Exposición a Industrias",
    "Exposicion_Industrias_fabrica_metales": "Fábrica de Metales",
    "Exposicion_Industrias_fabrica_productos_quimicos": "Fábrica de Productos Químicos",
    "Exposicion_Lugares": "Exposición a Lugares",
    "Exposicion_Lugares_canale": "Canales",
    "Exposicion_Lugares_canales": "Canales",
    "Exposicion_Lugares_estacion_gasolina": "Estación de Gasolina",
    "Exposicion_Lugares_llenadora_gas_natural": "Llenadora de Gas Natural",
    "Exposicion_Lugares_rectificadora_motores": "Rectificadora de Motores",
    "Exposicion_Lugares_rios": "Ríos",
    # Hábitos / Salud
    "Salud_Transporte": "Medio de Transporte",
    "Salud_Transporte_caminar": "Caminar",
    "Salud_Transporte_publico": "Transporte Público",
    "Salud_Transporte_vehiculo": "Vehículo",
    "Salud_Agua": "Fuente de Agua",
    "Salud_Agua_filtrada": "Agua Filtrada",
    "Salud_Agua_mineral_embotellada": "Agua Mineral Embotellada",
    "Salud_Agua_pozo_profundo": "Agua de Pozo Profundo",
    "Salud_Fuma": "Hábito de Fumar",
    "Salud_Actividad": "Actividad Física",
    "Salud_Bombillos": "Bombillos Ahorradores",
    "Salud_Techo": "Tipo de Techo",
    "Salud_Joyeria": "Uso de Joyería",
    "Salud_Suplementos": "Suplementos Dietéticos",
    "Salud_Suplementos_multivitaminico": "Suplemento Multivitamínico",
    "Salud_Suplementos_proteico": "Suplemento Proteico",
    # Grupos de Alimentos
    "Alim_Cereales": "Cereales",
    "Alim_Leguminosas": "Leguminosas",
    "Alim_Tuberculos": "Tubérculos",
    "Alim_Carnes": "Carnes",
    "Alim_Pescados": "Pescados",
    "Alim_Bebidas": "Bebidas",
    "Alim_Huevos": "Huevos",
    "Alim_Lacteos": "Lácteos",
    "Alim_Frutas": "Frutas",
    "Alim_Vegetales": "Vegetales",
    "Alim_Azucar": "Azúcar",
    "Alim_Grasas": "Grasas",
    "Alim_Chocolate": "Chocolate",
}


def get_label(col: str, labels_map: Optional[Dict[str, str]] = None) -> str:
    """
    Obtiene la etiqueta legible para una columna, eliminando snake_case.

    Prioridad:
    1. Si se pasa un diccionario `labels_map` personalizado y la columna está en él.
    2. Si la columna está en `DEFAULT_LABELS_MAP`.
    3. Heurísticas automáticas para unidades comunes (_ug_dL, _ug_L, _kg, _cm).
    4. Reemplazo general de guiones bajos por espacios en formato Title / Natural.
    """
    if labels_map and col in labels_map:
        return labels_map[col]
    if col in DEFAULT_LABELS_MAP:
        return DEFAULT_LABELS_MAP[col]

    # Heurísticas de unidades comunes
    if col.endswith("_ug_dL"):
        metal = col[:-6].replace("_", " ").strip().title()
        return rf"Concentración de {metal} en Sangre ($\mu$g/dL)"
    if col.endswith("_ug_L"):
        metal = col[:-5].replace("_", " ").strip().title()
        return rf"Concentración de {metal} en Sangre ($\mu$g/L)"
    if col.endswith("_kg"):
        var = col[:-3].replace("_", " ").strip().title()
        return f"{var} (kg)"
    if col.endswith("_cm"):
        var = col[:-3].replace("_", " ").strip().title()
        return f"{var} (cm)"

    return col.replace("_", " ").strip().title()
