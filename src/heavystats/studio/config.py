"""
Configuración Metodológica y Toxicológica de Metales Pesados para HeavyStats Studio.
"""

from pathlib import Path

# Directorios principales del proyecto (relativos al directorio actual de ejecución)
DIR_BASE = Path.cwd()
DIR_DATOS = DIR_BASE / "datos"
DIR_DATOS_PROC = DIR_DATOS / "procesados"
DIR_SALIDAS = DIR_BASE / "salidas"
DIR_SALIDAS_QC = DIR_SALIDAS / "control_calidad"
DIR_SALIDAS_UNIVAR_TABLAS = DIR_SALIDAS / "univariante" / "tablas"
DIR_SALIDAS_UNIVAR_GRAFICOS = DIR_SALIDAS / "univariante" / "graficos"
DIR_SALIDAS_BIVAR_TABLAS = DIR_SALIDAS / "bivariante" / "tablas"
DIR_SALIDAS_BIVAR_GRAFICOS = DIR_SALIDAS / "bivariante" / "graficos"

METALES_CONFIG = {
    "plomo": {
        "key": "plomo",
        "nombre": "Plomo",
        "simbolo": "Pb",
        "unidad": "µg/dL",
        "col_concentracion": "Plomo_ug_dL",
        "col_riesgo": "Riesgo_Pb",
        "param_metal": "Plomo",
        "param_biv": "Plomo_ug_dL",
        "limite_permisible": 3.5,
        "entidad_referencia": "CDC (BLRV: 3.5 µg/dL)",
        "col_dieta_prioritaria": "Alim_Carnes",
        "nombre_dieta": "Consumo de Carnes / Alimentos",
        "archivo_datos": "datos_plomo.csv",
        "archivo_proc": "datos_plomo_n20.csv",
        "lod": "0.5 µg/dL",
        "color": "cyan",
    },
    "mercurio": {
        "key": "mercurio",
        "nombre": "Mercurio",
        "simbolo": "Hg",
        "unidad": "µg/L",
        "col_concentracion": "Mercurio_ug_L",
        "col_riesgo": "Riesgo_Hg",
        "param_metal": "Mercurio",
        "param_biv": "Mercurio_ug_L",
        "limite_permisible": 5.0,
        "entidad_referencia": "EPA / OMS (5.0 µg/L)",
        "col_dieta_prioritaria": "Alim_Pescados",
        "nombre_dieta": "Consumo de Pescados y Mariscos",
        "archivo_datos": "datos_mercurio.csv",
        "archivo_proc": "datos_mercurio_n20.csv",
        "lod": "0.1 µg/L",
        "color": "magenta",
    },
    "cadmio": {
        "key": "cadmio",
        "nombre": "Cadmio",
        "simbolo": "Cd",
        "unidad": "µg/L",
        "col_concentracion": "Cadmio_ug_L",
        "col_riesgo": "Riesgo_Cd",
        "param_metal": "Cadmio",
        "param_biv": "Cadmio_ug_L",
        "limite_permisible": 1.0,
        "entidad_referencia": "OMS / ATSDR (1.0 µg/L)",
        "col_dieta_prioritaria": "Alim_Vegetales",
        "nombre_dieta": "Consumo de Vegetales y Granos",
        "archivo_datos": "datos_cadmio.csv",
        "archivo_proc": "datos_cadmio_n20.csv",
        "lod": "0.2 µg/L",
        "color": "yellow",
    }
}

DEFAULT_METAL = "plomo"

MAPA_ALIAS = {
    "1": "plomo", "plomo": "plomo", "pb": "plomo", "lead": "plomo",
    "2": "mercurio", "mercurio": "mercurio", "hg": "mercurio", "mercury": "mercurio",
    "3": "cadmio", "cadmio": "cadmio", "cd": "cadmio", "cadmium": "cadmio"
}


def resolver_metal(alias_o_nombre):
    """
    Resuelve el identificador de metal a partir de un alias, número o símbolo.
    Retorna la configuración del metal (por defecto: Plomo).
    """
    if not alias_o_nombre:
        return METALES_CONFIG[DEFAULT_METAL]
    clave = str(alias_o_nombre).strip().lower()
    metal_key = MAPA_ALIAS.get(clave, DEFAULT_METAL)
    return METALES_CONFIG[metal_key]
