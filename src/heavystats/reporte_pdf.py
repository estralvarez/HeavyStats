"""
Módulo de Compilación de Informes Editoriales en PDF (`heavystats.reporte_pdf`).
Consolida tablas y gráficos de cada pilar analítico (Univariante, Bivariante, Multivariante, Atípicos)
en un único documento PDF de calidad de publicación institucional.
No requiere dependencias externas de compilación C (utiliza Matplotlib PdfPages y Pillow).
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from heavystats.html_utils import BaseReport


# Nombres canónicos de los 4 pilares analíticos de HeavyStats
PILARES_INFO = {
    "univariante": {
        "titulo": "PILAR I: ANÁLISIS UNIVARIANTE Y PERFIL SOCIODEMOGRÁFICO",
        "descripcion": "Estadística descriptiva, pruebas de normalidad, frecuencias dietarias e indicadores de riesgo.",
        "color": "#1e3a8a",
    },
    "bivariante": {
        "titulo": "PILAR II: CONTRASTES BIVARIANTES Y CORRELACIONES",
        "descripcion": "Pruebas de hipótesis no paramétricas (Mann-Whitney, Kruskal-Wallis) y matrices de correlación.",
        "color": "#0d9488",
    },
    "multivariante": {
        "titulo": "PILAR III: MODELADO MULTIVARIANTE Y VARIABLES LATENTES",
        "descripcion": "Cribado FDR Benjamini-Hochberg, regularización LASSO y regresión PLS-R con puntajes VIP.",
        "color": "#4338ca",
    },
    "atipicos": {
        "titulo": "PILAR IV: DIAGNÓSTICO QUIMIOMÉTRICO Y VALORES ATÍPICOS",
        "descripcion": "Vallas de Tukey, prueba de Grubbs, distancias Hotelling T², residuales Q, Leverage y Cook's D.",
        "color": "#b91c1c",
    },
    "outliers": {
        "titulo": "PILAR IV: DIAGNÓSTICO QUIMIOMÉTRICO Y VALORES ATÍPICOS",
        "descripcion": "Vallas de Tukey, prueba de Grubbs, distancias Hotelling T², residuales Q, Leverage y Cook's D.",
        "color": "#b91c1c",
    },
}


def _natural_sort_key(s: Union[str, Path]) -> List[Union[int, str]]:
    """Clave de ordenamiento natural para archivos (ej. figura_1 antes de figura_10)."""
    name = Path(s).name
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


def _crear_portada(
    pdf: PdfPages,
    pilar_key: str,
    titulo: Optional[str] = None,
    subtitulo: Optional[str] = None,
    autor: Optional[str] = None,
    tutor: Optional[str] = None,
    institucion: Optional[str] = None,
    fecha: Optional[str] = None,
) -> None:
    """Genera una portada institucional elegante y formal para el informe en PDF."""
    info = PILARES_INFO.get(pilar_key.lower(), PILARES_INFO["multivariante"])
    titulo_texto = titulo or info["titulo"]
    subtitulo_texto = subtitulo or "Monitoreo de los Niveles Basales de Mercurio en Sangre\nComunidades Educativas del Municipio San Diego, Estado Carabobo"
    autor_texto = autor or "Br. Jose Daniel Estrada Alvarez"
    tutor_texto = tutor or "Dr. Jose Ramon Jimenez Ochoa"
    inst_texto = institucion or "UNIVERSIDAD DE CARABOBO\nFACULTAD EXPERIMENTAL DE CIENCIAS Y TECNOLOGÍA\nDEPARTAMENTO DE QUÍMICA"

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    # Borde decorativo exterior
    rect = plt.Rectangle((0.05, 0.05), 0.90, 0.90, fill=False, edgecolor="#cbd5e1", linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect)
    rect_inner = plt.Rectangle((0.055, 0.055), 0.89, 0.89, fill=False, edgecolor="#e2e8f0", linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(rect_inner)

    # Encabezado institucional
    ax.text(0.5, 0.85, inst_texto, ha="center", va="center", fontsize=11, fontweight="bold", linespacing=1.6, color="#0f172a")

    # Línea divisoria superior
    ax.plot([0.2, 0.8], [0.76, 0.76], color=info["color"], linewidth=2.5, transform=ax.transAxes)

    # Título del Pilar
    ax.text(0.5, 0.65, "INFORME TÉCNICO Y ANALÍTICO", ha="center", va="center", fontsize=13, fontweight="semibold", color="#64748b", transform=ax.transAxes)
    ax.text(0.5, 0.58, titulo_texto, ha="center", va="center", fontsize=15, fontweight="bold", color=info["color"], wrap=True, transform=ax.transAxes)

    # Subtítulo y alcance del estudio
    ax.text(0.5, 0.47, subtitulo_texto, ha="center", va="center", fontsize=11, style="italic", color="#334155", linespacing=1.5, transform=ax.transAxes)
    ax.text(0.5, 0.38, info["descripcion"], ha="center", va="center", fontsize=9.5, color="#64748b", wrap=True, transform=ax.transAxes)

    # Línea divisoria inferior
    ax.plot([0.3, 0.7], [0.32, 0.32], color="#cbd5e1", linewidth=1.0, transform=ax.transAxes)

    # Metadatos de autoría y tutoría
    ax.text(0.5, 0.25, f"Autor: {autor_texto}\nTutor: {tutor_texto}", ha="center", va="center", fontsize=10.5, linespacing=1.6, color="#0f172a", fontweight="medium", transform=ax.transAxes)

    # Pie de portada
    pie_fecha = fecha or "HeavyStats Analytics Engine • Metodología GFAAS y Quimiometría UC"
    ax.text(0.5, 0.10, pie_fecha, ha="center", va="center", fontsize=8.5, color="#94a3b8", transform=ax.transAxes)

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _renderizar_tabla_pdf(
    pdf: PdfPages,
    df: pd.DataFrame,
    titulo: str = "Tabla Analítica",
    notas: Optional[str] = None,
    max_filas_por_pagina: int = 22
) -> None:
    """Renderiza un DataFrame en páginas de PDF con estilo editorial booktabs y paginación limpia."""
    # Sanitizar DataFrame
    df_clean = df.copy()
    # Descartar columnas técnicas internas
    df_clean = df_clean[[c for c in df_clean.columns if not str(c).startswith("_")]]

    if df_clean.empty:
        return

    # Formatear números a 3-4 decimales para visualización compacta
    for col in df_clean.columns:
        if pd.api.types.is_float_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

    n_total = len(df_clean)
    num_paginas = max(1, int(np.ceil(n_total / max_filas_por_pagina)))

    for p_idx in range(num_paginas):
        start_row = p_idx * max_filas_por_pagina
        end_row = min((p_idx + 1) * max_filas_por_pagina, n_total)
        df_chunk = df_clean.iloc[start_row:end_row]

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Título de la tabla y número de página si aplica
        pag_info = f" (Pág. {p_idx + 1}/{num_paginas})" if num_paginas > 1 else ""
        ax.text(0.08, 0.94, f"{titulo}{pag_info}", fontsize=12, fontweight="bold", color="#0f172a", transform=ax.transAxes)

        # Matriz de celdas
        col_labels = [str(c) for c in df_chunk.columns]
        cell_values = df_chunk.astype(str).values

        table = ax.table(
            cellText=cell_values,
            colLabels=col_labels,
            loc="upper center",
            bbox=[0.05, 0.12, 0.90, 0.78],
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)

        # Aplicar estilo editorial booktabs
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor("#1e293b")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                cell.set_linewidth(1.2)
                cell.set_edgecolor("#0f172a")
            else:
                # Celdas de datos con alternancia
                bg = "#f8fafc" if r % 2 == 0 else "#ffffff"
                cell.set_facecolor(bg)
                cell.get_text().set_color("#1e293b")

        # Notas al pie de la tabla
        if notas and (p_idx == num_paginas - 1):
            ax.text(0.08, 0.07, f"Nota: {notas}", fontsize=8, color="#64748b", style="italic", wrap=True, transform=ax.transAxes)

        pdf.savefig(fig, dpi=300)
        plt.close(fig)


def _renderizar_imagen_pdf(
    pdf: PdfPages,
    img_path: Union[str, Path],
    titulo: Optional[str] = None
) -> None:
    """Inserta una imagen (PNG, JPG) en una página dedicada del PDF conservando alta fidelidad."""
    img_path = Path(img_path)
    if not img_path.exists():
        return

    try:
        with Image.open(img_path) as img:
            w, h = img.size
            aspect = h / w

            # Crear figura adaptada a hoja carta (8.5 x 11 pulgadas)
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")

            # Título de la figura en la parte superior
            tit_text = titulo or img_path.stem.replace("_", " ").title()
            ax.text(0.08, 0.95, tit_text, fontsize=12, fontweight="bold", color="#0f172a", transform=ax.transAxes)

            # Insertar imagen centrada
            # Área disponible: x de 0.05 a 0.95 (ancho 0.90), y de 0.06 a 0.92 (alto 0.86)
            max_w_in = 7.5
            max_h_in = 9.2
            if aspect > (max_h_in / max_w_in):
                plot_h = max_h_in
                plot_w = plot_h / aspect
            else:
                plot_w = max_w_in
                plot_h = plot_w * aspect

            x_left = (8.5 - plot_w) / 2.0 / 8.5
            y_bottom = 0.06 + (max_h_in - plot_h) / 2.0 / 11.0
            w_norm = plot_w / 8.5
            h_norm = plot_h / 11.0

            img_ax = fig.add_axes([x_left, y_bottom, w_norm, h_norm])
            img_ax.imshow(img)
            img_ax.axis("off")

            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    except Exception as e:
        print(f"Advertencia: no se pudo renderizar la imagen {img_path}: {e}")


def unir_pilar_pdf(
    pilar: str,
    ruta_pdf: Union[str, Path],
    directorio_salidas: Optional[Union[str, Path]] = None,
    tablas: Optional[Sequence[Union[pd.DataFrame, BaseReport, str, Path]]] = None,
    graficos: Optional[Sequence[Union[str, Path, plt.Figure]]] = None,
    titulo: Optional[str] = None,
    subtitulo: Optional[str] = None,
    autor: Optional[str] = None,
    tutor: Optional[str] = None,
    institucion: Optional[str] = None,
    fecha: Optional[str] = None,
) -> Path:
    """
    Une todas las tablas e imágenes correspondientes a un pilar analítico en un único documento PDF.

    Parámetros
    ----------
    pilar : str
        Nombre del pilar analítico ('univariante', 'bivariante', 'multivariante', 'atipicos').
    ruta_pdf : Union[str, Path]
        Ruta del archivo PDF destino a generar.
    directorio_salidas : Optional[Union[str, Path]], opcional
        Directorio donde se encuentran guardadas las salidas (tablas en .json/.csv y figuras en .png).
        Si se especifica, se exploran y recopilan automáticamente todos los elementos.
    tablas : Optional[Sequence[Union[pd.DataFrame, BaseReport, str, Path]]], opcional
        Lista explícita de DataFrames, instancias de Reporte o rutas a archivos de tablas (.json, .csv).
    graficos : Optional[Sequence[Union[str, Path, plt.Figure]]], opcional
        Lista explícita de rutas a archivos de imagen (.png, .jpg) o figuras de Matplotlib.
    titulo : Optional[str], opcional
        Título principal de la portada.
    subtitulo : Optional[str], opcional
        Subtítulo descriptivo de la portada.
    autor : Optional[str], opcional
        Nombre del autor del informe.
    tutor : Optional[str], opcional
        Nombre del tutor de la investigación.
    institucion : Optional[str], opcional
        Nombre de la institución universitaria.
    fecha : Optional[str], opcional
        Texto de fecha o pie de página.

    Retorna
    -------
    Path
        Ruta canónica del archivo PDF generado y cerrado.
    """
    out_path = Path(ruta_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lista_tablas_dfs: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    lista_graficos_rutas: List[Union[str, Path, plt.Figure]] = []

    # 1. Exploración automática si se provee directorio_salidas
    if directorio_salidas is not None:
        dir_salidas = Path(directorio_salidas)
        if dir_salidas.exists() and dir_salidas.is_dir():
            # Buscar tablas (.json preferido, o .csv)
            archivos_tabla = []
            for ext in ["*.json", "*.csv"]:
                archivos_tabla.extend(list(dir_salidas.rglob(ext)))
            # Eliminar duplicados stem (preferir .json sobre .csv si ambos existen)
            vistos_stem = set()
            archivos_tabla_filtrados = []
            for at in sorted(archivos_tabla, key=_natural_sort_key):
                if at.stem not in vistos_stem:
                    archivos_tabla_filtrados.append(at)
                    vistos_stem.add(at.stem)

            for at in archivos_tabla_filtrados:
                try:
                    if at.suffix.lower() == ".json":
                        df_loaded = pd.read_json(at)
                    else:
                        df_loaded = pd.read_csv(at)
                    tit_tab = at.stem.replace("_", " ").title()
                    lista_tablas_dfs.append((tit_tab, df_loaded, None))
                except Exception as e:
                    pass

            # Buscar imágenes (.png, .jpg)
            archivos_img = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                archivos_img.extend(list(dir_salidas.rglob(ext)))
            for ai in sorted(archivos_img, key=_natural_sort_key):
                lista_graficos_rutas.append(ai)

    # 2. Agregar tablas pasadas explícitamente
    if tablas is not None:
        for idx, t in enumerate(tablas):
            if isinstance(t, BaseReport) or hasattr(t, "to_dataframe"):
                df_t = t.to_dataframe()
                tit_t = getattr(t, "title", f"Tabla Analítica {idx + 1}")
                lista_tablas_dfs.append((tit_t, df_t, None))
            elif isinstance(t, pd.DataFrame):
                lista_tablas_dfs.append((f"Tabla Analítica {idx + 1}", t.copy(), None))
            elif isinstance(t, (str, Path)):
                p = Path(t)
                if p.exists():
                    df_t = pd.read_json(p) if p.suffix.lower() == ".json" else pd.read_csv(p)
                    lista_tablas_dfs.append((p.stem.replace("_", " ").title(), df_t, None))

    # 3. Agregar gráficos pasados explícitamente
    if graficos is not None:
        for g in graficos:
            lista_graficos_rutas.append(g)

    # 4. Compilación del documento multipágina en PDF
    with PdfPages(out_path) as pdf:
        # Página 1: Portada
        _crear_portada(
            pdf=pdf,
            pilar_key=pilar,
            titulo=titulo,
            subtitulo=subtitulo,
            autor=autor,
            tutor=tutor,
            institucion=institucion,
            fecha=fecha,
        )

        # Páginas de Gráficos (conservando 300 DPI)
        for g in lista_graficos_rutas:
            if isinstance(g, plt.Figure):
                pdf.savefig(g, dpi=300, bbox_inches="tight")
            elif isinstance(g, (str, Path)):
                _renderizar_imagen_pdf(pdf=pdf, img_path=g)

        # Páginas de Tablas (estilo editorial booktabs)
        for tit_t, df_t, notas_t in lista_tablas_dfs:
            _renderizar_tabla_pdf(pdf=pdf, df=df_t, titulo=tit_t, notas=notas_t)

    print(f"[OK] Informe de pilar compilado exitosamente: {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path



# Alias idéntico en español para máxima accesibilidad y coherencia metodológica
compilar_pilar_pdf = unir_pilar_pdf
