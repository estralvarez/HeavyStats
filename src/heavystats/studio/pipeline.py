"""
Orquestador del pipeline de generación de proyectos y cuadernos de HeavyStats Studio.
"""

from heavystats.studio.filesystem import (
    crear_estructura_directorios,
    garantizar_dataset_base,
    verificar_proyecto_existente,
    auditar_dataset,
)
from heavystats.studio.templates import (
    generar_cuaderno_validacion,
    generar_cuaderno_filtros,
    generar_cuaderno_univariante_tablas,
    generar_cuaderno_univariante_graficos,
    generar_cuaderno_bivariante_tablas,
    generar_cuaderno_bivariante_graficos,
)


def ejecutar_pipeline(cfg, progress_callback=None, force=False):
    """
    Ejecuta la secuencia completa de preparación y generación de cuadernos.
    Si las carpetas y cuadernos del metal ya están creadas y force=False,
    omite la re-ejecución innecesaria y retorna el estado existente.

    Parameters
    ----------
    cfg : dict
        Configuración del metal seleccionado (de METALES_CONFIG).
    progress_callback : callable, optional
        Función callback con firma (paso_actual: int, total_pasos: int, mensaje: str).
    force : bool, optional
        Si True, fuerza la re-creación aunque ya existan carpetas y cuadernos.

    Returns
    -------
    dict
        Resumen de ejecución con metales, cuadernos generados y diagnóstico de datos.
    """
    # Si las carpetas y cuadernos ya existen, omitir re-ejecución innecesaria
    if not force and verificar_proyecto_existente(cfg):
        if progress_callback:
            progress_callback(8, 8, f"Las carpetas de {cfg['nombre']} ya están creadas. Omitiendo re-ejecución.")
        return {
            "metal": cfg,
            "directorios": [],
            "dataset_audit": auditar_dataset(cfg),
            "cuadernos": [
                "validacion.ipynb",
                "filtros.ipynb",
                "univariante/01_tablas.ipynb",
                "univariante/02_graficos.ipynb",
                "bivariante/01_analisis_estadistico.ipynb",
                "bivariante/02_graficos_bivariantes.ipynb",
            ],
            "ya_existia": True,
        }
    pasos_totales = 8
    paso = 0

    def notificar(mensaje):
        nonlocal paso
        paso += 1
        if progress_callback:
            progress_callback(paso, pasos_totales, mensaje)
        else:
            print(f"[{paso}/{pasos_totales}] {mensaje}")

    # 1. Estructura de directorios
    notificar("Creando y asegurando jerarquía de directorios...")
    dirs_creados = crear_estructura_directorios()

    # 2. Aseguramiento de dataset
    notificar(f"Asegurando dataset para {cfg['nombre']} ({cfg['archivo_datos']})...")
    audit_data = garantizar_dataset_base(cfg)

    cuadernos_generados = []

    # 3. Cuaderno de validación
    notificar("Construyendo validacion.ipynb (10 reglas de calidad)...")
    nb_val = generar_cuaderno_validacion(cfg)
    cuadernos_generados.append(nb_val)

    # 4. Cuaderno de filtros
    notificar(f"Construyendo filtros.ipynb (muestra analítica de {cfg['nombre']})...")
    nb_fil = generar_cuaderno_filtros(cfg)
    cuadernos_generados.append(nb_fil)

    # 5. Univariante tablas
    notificar("Construyendo univariante/01_tablas.ipynb (perfil toxicológico)...")
    nb_u1 = generar_cuaderno_univariante_tablas(cfg)
    cuadernos_generados.append(nb_u1)

    # 6. Univariante gráficos
    notificar("Construyendo univariante/02_graficos.ipynb (boxplots, hist, Q-Q)...")
    nb_u2 = generar_cuaderno_univariante_graficos(cfg)
    cuadernos_generados.append(nb_u2)

    # 7. Bivariante tablas
    notificar(f"Construyendo bivariante/01_analisis_estadistico.ipynb ({cfg['nombre']})...")
    nb_b1 = generar_cuaderno_bivariante_tablas(cfg)
    cuadernos_generados.append(nb_b1)

    # 8. Bivariante gráficos
    notificar("Construyendo bivariante/02_graficos_bivariantes.ipynb (asociaciones)...")
    nb_b2 = generar_cuaderno_bivariante_graficos(cfg)
    cuadernos_generados.append(nb_b2)

    return {
        "metal": cfg,
        "directorios": dirs_creados,
        "dataset_audit": audit_data,
        "cuadernos": [str(c) for c in cuadernos_generados],
    }
