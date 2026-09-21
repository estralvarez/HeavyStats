"""
Módulo de gestión del sistema de archivos y datasets para HeavyStats Studio.
"""

import shutil
from pathlib import Path
import heavystats as hs
from heavystats.studio.config import (
    DIR_BASE,
    DIR_DATOS,
    DIR_DATOS_PROC,
    DIR_SALIDAS,
    DIR_SALIDAS_QC,
    DIR_SALIDAS_UNIVAR_TABLAS,
    DIR_SALIDAS_UNIVAR_GRAFICOS,
    DIR_SALIDAS_BIVAR_TABLAS,
    DIR_SALIDAS_BIVAR_GRAFICOS,
)


def crear_estructura_directorios():
    """
    Garantiza la existencia de la jerarquía de carpetas requeridas para el proyecto:
    - datos/ y datos/procesados/
    - salidas/ organizadas por etapa analítica (QC, Univariante, Bivariante)
    """
    directorios = [
        DIR_DATOS,
        DIR_DATOS_PROC,
        DIR_SALIDAS,
        DIR_SALIDAS_QC,
        DIR_SALIDAS_UNIVAR_TABLAS,
        DIR_SALIDAS_UNIVAR_GRAFICOS,
        DIR_SALIDAS_BIVAR_TABLAS,
        DIR_SALIDAS_BIVAR_GRAFICOS,
    ]
    for d in directorios:
        d.mkdir(parents=True, exist_ok=True)
    return [str(d) for d in directorios]


def auditar_dataset(cfg):
    """
    Inspecciona el archivo de datos asociado al metal en 'datos/'.
    Retorna un diccionario con el diagnóstico para consumo del CLI y de la TUI.
    """
    archivo_csv = DIR_DATOS / cfg["archivo_datos"]
    resultado = {
        "ruta": str(archivo_csv),
        "nombre_archivo": cfg["archivo_datos"],
        "existe": archivo_csv.exists(),
        "n_total": 0,
        "n_metal": 0,
        "columna": cfg["col_concentracion"],
        "metal": cfg["nombre"],
        "simbolo": cfg["simbolo"],
        "estado": "desconocido",
        "mensaje": "",
    }

    if not archivo_csv.exists():
        resultado["estado"] = "ausente"
        resultado["mensaje"] = f"Archivo '{cfg['archivo_datos']}' no encontrado en datos/."
        return resultado

    try:
        df = hs.load_data(archivo_csv)
        resultado["n_total"] = len(df)
        col_m = cfg["col_concentracion"]
        if col_m in df.columns:
            resultado["n_metal"] = int(df[col_m].notna().sum())

        if resultado["n_metal"] > 0:
            resultado["estado"] = "optimo"
            resultado["mensaje"] = f"{resultado['n_metal']} determinaciones analíticas listas para {cfg['nombre']}."
        else:
            resultado["estado"] = "incompleto"
            resultado["mensaje"] = f"La columna '{col_m}' no posee observaciones numéricas válidas."
    except Exception as e:
        resultado["estado"] = "error"
        resultado["mensaje"] = f"Error al inspeccionar dataset: {e}"

    return resultado


def garantizar_dataset_base(cfg):
    """
    Asegura la disponibilidad del archivo de datos sin sobreescribir ni eliminar archivos preexistentes.
    Si se selecciona Plomo y datos_plomo.csv aún no existía, pero datos_mercurio.csv
    alberga las determinaciones de Plomo, lo vincula de forma segura.
    """
    archivo_destino = DIR_DATOS / cfg["archivo_datos"]
    archivo_mercurio = DIR_DATOS / "datos_mercurio.csv"

    # Preservación de datos de Plomo si estaban en datos_mercurio.csv
    if cfg["key"] == "plomo" and not archivo_destino.exists() and archivo_mercurio.exists():
        try:
            df_temp = hs.load_data(archivo_mercurio)
            if "Plomo_ug_dL" in df_temp.columns and df_temp["Plomo_ug_dL"].notna().sum() > 0:
                shutil.copy2(archivo_mercurio, archivo_destino)
                print(f"[OK] Datos de Plomo preservados en: {archivo_destino}")
        except Exception as e:
            print(f"[AVISO] Error al verificar datos_mercurio.csv: {e}")

    # Fallback: generar desde HeavyStats si no existe archivo
    if not archivo_destino.exists():
        try:
            df_base = hs.load_data()
            df_base.to_csv(archivo_destino, sep=";", decimal=",", encoding="utf-8", index=False)
            print(f"[OK] Dataset generado a partir de HeavyStats en: {archivo_destino}")
        except Exception as e:
            print(f"[AVISO] No se pudo generar {archivo_destino}: {e}")

    return auditar_dataset(cfg)


def verificar_proyecto_existente(cfg):
    """
    Determina si la jerarquía de carpetas y los cuadernos analíticos para el metal
    especificado ya han sido generados previamente.
    Retorna True si todas las carpetas requeridas y los 6 cuadernos existen
    y corresponden al metal seleccionado.
    """
    if not cfg:
        return False

    directorios_requeridos = [
        DIR_DATOS,
        DIR_SALIDAS,
        DIR_SALIDAS_QC,
        DIR_SALIDAS_UNIVAR_TABLAS,
        DIR_SALIDAS_UNIVAR_GRAFICOS,
        DIR_SALIDAS_BIVAR_TABLAS,
        DIR_SALIDAS_BIVAR_GRAFICOS,
        DIR_BASE / "univariante",
        DIR_BASE / "bivariante",
    ]
    for d in directorios_requeridos:
        if not d.exists() or not d.is_dir():
            return False

    cuadernos_requeridos = [
        DIR_BASE / "validacion.ipynb",
        DIR_BASE / "filtros.ipynb",
        DIR_BASE / "univariante" / "01_tablas.ipynb",
        DIR_BASE / "univariante" / "02_graficos.ipynb",
        DIR_BASE / "bivariante" / "01_analisis_estadistico.ipynb",
        DIR_BASE / "bivariante" / "02_graficos_bivariantes.ipynb",
    ]
    for nb in cuadernos_requeridos:
        if not nb.exists():
            return False

    # Verificar que el cuaderno de validación corresponda al metal activo
    nb_val = DIR_BASE / "validacion.ipynb"
    try:
        with open(nb_val, "r", encoding="utf-8") as f:
            contenido = f.read()
            nombre = cfg.get("nombre", "").lower()
            simbolo = cfg.get("simbolo", "").lower()
            if nombre not in contenido.lower() and simbolo not in contenido.lower():
                return False
    except Exception:
        return False

    return True
