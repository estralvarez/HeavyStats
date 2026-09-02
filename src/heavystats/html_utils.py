import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union


def get_publication_css() -> str:
    """
    Retorna el bloque CSS unificado para todos los reportes y tablas de HeavyStats.
    Diseñado con tema claro forzado (!important) y contraste óptimo para legibilidad
    impecable tanto en temas oscuros (JupyterLab, Positron, VSCode) como claros.
    """
    return """
  .hs-pub-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    padding: 24px;
    overflow-x: auto;
    margin: 16px 0;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    text-align: left;
    line-height: 1.5;
  }
  .hs-pub-container * { box-sizing: border-box; }
  .hs-pub-container p, .hs-pub-container div, .hs-pub-container span, .hs-pub-container td, .hs-pub-container th, .hs-pub-container li {
    color: #334155 !important;
  }
  .hs-pub-container h1, .hs-pub-container h2, .hs-pub-container h3, .hs-pub-container h4, .hs-pub-container h5 {
    color: #0f172a !important;
  }
  .hs-pub-container strong {
    color: #0f172a !important;
    font-weight: 600;
  }
  .hs-pub-container em {
    color: #475569 !important;
  }
  .hs-pub-container code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    border: 1px solid #cbd5e1 !important;
    display: inline-block;
  }
  .hs-pub-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 10px 0;
    font-size: 14px;
    text-align: left;
    border-top: 2px solid #0f172a;
    border-bottom: 2px solid #0f172a;
    background-color: #ffffff !important;
  }
  .hs-pub-table th {
    font-weight: 600;
    color: #0f172a !important;
    padding: 11px 9px;
    border-bottom: 1px solid #0f172a;
    background-color: #ffffff !important;
  }
  .hs-pub-table td {
    padding: 9px 8px;
    color: #334155 !important;
    border: none;
    background-color: #ffffff !important;
    vertical-align: middle;
  }
  .hs-pub-table tbody tr:hover td {
    background-color: #f8fafc !important;
  }
  .hs-pub-table tbody tr:not(:last-child) td {
    border-bottom: 1px solid #f1f5f9;
  }
  .hs-pub-group-row td {
    font-weight: 700;
    color: #0f172a !important;
    background-color: #f8fafc !important;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    padding-top: 12px;
    padding-bottom: 6px;
  }
  .hs-num-col { text-align: right; }
  .hs-center-col { text-align: center; }
  .hs-left-col { text-align: left; }
  .hs-pub-caption {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
    text-align: left;
    color: #0f172a !important;
  }
  .hs-pub-subtitle {
    font-size: 13px;
    color: #64748b !important;
    margin-bottom: 14px;
    text-align: left;
  }
  .hs-section-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a !important;
    margin: 24px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }
  .hs-badge-ok, .hs-badge-success {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
    background-color: #dcfce7 !important;
    color: #15803d !important;
    border: 1px solid #bbf7d0;
  }
  .hs-badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
    background-color: #fef3c7 !important;
    color: #b45309 !important;
    border: 1px solid #fde68a;
  }
  .hs-badge-danger {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
    background-color: #fee2e2 !important;
    color: #b91c1c !important;
    border: 1px solid #fecaca;
  }
  .hs-badge-cat {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background-color: #f3e8ff !important;
    color: #7e22ce !important;
    border: 1px solid #e9d5ff;
  }
  .hs-badge-num {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background-color: #e0f2fe !important;
    color: #0369a1 !important;
    border: 1px solid #bae6fd;
  }
  .hs-badge-other {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0;
  }
  .hs-var-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin: 2px 3px;
    padding: 2px 7px;
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    font-size: 12px;
    color: #1e293b !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  .hs-null-badge {
    color: #b45309 !important;
    background-color: #fef3c7 !important;
    border: 1px solid #fde68a;
    padding: 0 4px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
  }
  .hs-count-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
    background-color: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0;
  }
  .hs-dtype-code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background-color: #f1f5f9 !important;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    color: #0f172a !important;
    border: 1px solid #e2e8f0;
  }
  .hs-summary-card-success {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: #ecfdf5 !important;
    border: 1px solid #a7f3d0;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 12px;
    color: #065f46 !important;
    font-size: 13px;
    font-weight: 600;
  }
  .hs-summary-card-danger {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: #fef2f2 !important;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 12px;
    color: #991b1b !important;
    font-size: 13px;
    font-weight: 600;
  }
  .hs-pub-notes {
    font-size: 12px;
    color: #64748b !important;
    margin-top: 8px;
    font-style: italic;
    text-align: left;
  }
"""


def format_html_str(text: Any) -> str:
    r"""
    Formatea cadenas de texto para renderizado HTML de calidad de publicación:
    - Escapa caracteres HTML (<, >) para evitar tags rotos.
    - Convierte fórmulas y símbolos matemáticos ($\mu$, \mu, \chi^2, >=, <=) a entidades HTML.
    - Convierte **negrita** a <strong> y *cursiva* a <em>.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # Escapar caracteres reservados
    s = s.replace("<", "&lt;").replace(">", "&gt;")

    # Símbolos matemáticos y unidades a entidades HTML
    s = s.replace(r"$\mu$", "&mu;")
    s = s.replace("$\\mu$", "&mu;")
    s = s.replace(r"\mu", "&mu;")
    s = s.replace(r"$\chi^2$", "&chi;²")
    s = s.replace("$\\chi^2$", "&chi;²")
    s = s.replace(r"\chi^2", "&chi;²")
    s = s.replace("&gt;=LOD", "&ge; LOD")
    s = s.replace("&gt;=", "&ge;")
    s = s.replace("&lt;=", "&le;")
    s = s.replace("$", "")

    # Marcado a etiquetas HTML
    s = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.*?)\*", r"<em>\1</em>", s)
    s = re.sub(r"`(.*?)`", r"<code>\1</code>", s)
    return s


def wrap_html_container(
    inner_html: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    notes: Optional[List[str]] = None,
    full_page: bool = False,
    page_title: Optional[str] = None
) -> str:
    """
    Envuelve el cuerpo de un reporte en el contenedor estilizado .hs-pub-container
    y opcionalmente en una estructura HTML5 completa.
    """
    lines = []
    if full_page:
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='es'>")
        lines.append("<head>")
        lines.append("  <meta charset='UTF-8'>")
        lines.append(f"  <title>{page_title or title or 'Reporte'}</title>")

    lines.append("<style>")
    lines.append(get_publication_css())
    lines.append("</style>")

    if full_page:
        lines.append("</head>")
        lines.append("<body>")

    lines.append("<div class='hs-pub-container'>")

    if title:
        lines.append(f"  <div class='hs-pub-caption'>{format_html_str(title)}</div>")
    if subtitle:
        lines.append(f"  <div class='hs-pub-subtitle'>{format_html_str(subtitle)}</div>")

    lines.append(inner_html)

    if notes:
        notes_html = "<br>".join(f"Nota: {format_html_str(n)}" for n in notes)
        lines.append(f"  <div class='hs-pub-notes'>{notes_html}</div>")

    lines.append("</div>")

    if full_page:
        lines.append("</body>")
        lines.append("</html>")

    return "\n".join(lines)


def render_html_table(
    df: pd.DataFrame,
    title: str = "",
    subtitle: Optional[str] = None,
    notes: Optional[List[str]] = None,
    column_alignments: Optional[Dict[str, str]] = None,
    spanners: Optional[List[Dict[str, Any]]] = None,
    group_col: Optional[str] = None,
    classified_df: Optional[pd.DataFrame] = None,
    full_page: bool = False,
    divider_borders: bool = False
) -> str:
    """
    Genera el código HTML responsivo de calidad de publicación con tema claro forzado.
    """
    alignments = column_alignments or {}
    spanner_list = spanners or []
    note_list = notes or []

    cols = [c for c in df.columns if c != group_col]
    ncols = len(cols)

    table_lines = []
    table_lines.append("  <table class='hs-pub-table'>")
    table_lines.append("    <thead>")

    # Spanners de columnas
    if spanner_list:
        table_lines.append("      <tr>")
        for sp in spanner_list:
            sp_label = format_html_str(sp.get("label", ""))
            sp_cols = sp.get("columns", [])
            sp_width = len(sp_cols)
            if sp_width > 1:
                table_lines.append(f"        <th colspan='{sp_width}' class='hs-center-col' style='border-bottom: 1px solid #0f172a;'>{sp_label}</th>")
            elif sp_width == 1:
                table_lines.append("        <th style='border-bottom: none;'></th>")
        table_lines.append("      </tr>")

    # Encabezados de columnas
    table_lines.append("      <tr>")
    for c in cols:
        align = alignments.get(c, "l")
        cls = "hs-num-col" if align == "r" else ("hs-center-col" if align == "c" else "hs-left-col")
        table_lines.append(f"        <th class='{cls}'>{format_html_str(c)}</th>")
    table_lines.append("      </tr>")
    table_lines.append("    </thead>")

    # Cuerpo de datos
    table_lines.append("    <tbody>")
    for _, row in df.iterrows():
        if group_col and row.get(group_col) == "__GROUP_HEADER__":
            raw_group = str(row[cols[0]]).replace("**", "").strip()
            table_lines.append(f"      <tr class='hs-pub-group-row'><td colspan='{ncols}'>{format_html_str(raw_group)}</td></tr>")
            continue

        td_style = ""
        if divider_borders:
            is_new_var = bool(str(row[cols[0]]).strip())
            if is_new_var:
                td_style = " style='border-top: 1px solid #cbd5e1 !important;'"

        table_lines.append("      <tr>")
        for c in cols:
            align = alignments.get(c, "l")
            cls = "hs-num-col" if align == "r" else ("hs-center-col" if align == "c" else "hs-left-col")
            val_str = format_html_str(row[c])
            table_lines.append(f"        <td class='{cls}'{td_style}>{val_str}</td>")
        table_lines.append("      </tr>")
    table_lines.append("    </tbody>")
    table_lines.append("  </table>")

    # Tabla secundaria si incluye detalle de pacientes (ej. bmi_summary)
    if classified_df is not None and not classified_df.empty:
        df_pat = classified_df
        total_pat = len(df_pat)
        pat_cols = [c for c in ["Muestra_Codificada", "Edad", "Sexo", "Peso_kg", "Altura_cm", "Calculated_IMC", "Clasificacion_IMC"] if c in df_pat.columns]

        table_lines.append("  <div class='hs-section-title'>Detalle de IMC por Paciente</div>")
        table_lines.append("  <table class='hs-pub-table'>")
        table_lines.append("    <thead>")
        table_lines.append("      <tr>")
        for pc in pat_cols:
            header_title = {
                "Muestra_Codificada": "ID Paciente",
                "Edad": "Edad",
                "Sexo": "Sexo",
                "Peso_kg": "Peso (kg)",
                "Altura_cm": "Altura (cm)",
                "Calculated_IMC": "IMC (kg/m²)",
                "Clasificacion_IMC": "Clasificación"
            }.get(pc, pc)
            p_cls = "hs-num-col" if pc in ("Edad", "Peso_kg", "Altura_cm", "Calculated_IMC") else "hs-left-col"
            table_lines.append(f"        <th class='{p_cls}'>{header_title}</th>")
        table_lines.append("      </tr>")
        table_lines.append("    </thead>")
        table_lines.append("    <tbody>")
        for _, prow in df_pat.head(20).iterrows():
            table_lines.append("      <tr>")
            for pc in pat_cols:
                pval = prow[pc]
                if pc in ("Peso_kg", "Altura_cm") and pd.notna(pval):
                    pstr = f"{float(pval):.1f}"
                elif pc == "Calculated_IMC" and pd.notna(pval):
                    pstr = f"{float(pval):.2f}"
                else:
                    pstr = str(pval) if pd.notna(pval) else ""
                p_cls = "hs-num-col" if pc in ("Edad", "Peso_kg", "Altura_cm", "Calculated_IMC") else "hs-left-col"
                table_lines.append(f"        <td class='{p_cls}'>{format_html_str(pstr)}</td>")
            table_lines.append("      </tr>")
        table_lines.append("    </tbody>")
        table_lines.append("  </table>")
        if total_pat > 20:
            table_lines.append(f"  <div class='hs-pub-notes'>Mostrando primeras 20 observaciones de {total_pat} pacientes evaluados. Utilice report.get_classified_df() para acceder al DataFrame completo.</div>")

    inner_content = "\n".join(table_lines)
    return wrap_html_container(
        inner_html=inner_content,
        title=title,
        subtitle=subtitle,
        notes=note_list,
        full_page=full_page
    )


class BaseReport:
    """
    Clase base reutilizable para todos los reportes de HeavyStats.
    Proporciona exportación a archivos (HTML, CSV, Excel), renderizado HTML para Jupyter
    y métodos estándar de representación.
    """
    def _write_file(self, filepath: Optional[str], content: str) -> None:
        """Escribe contenido de texto en el archivo especificado creando directorios si no existen."""
        if filepath:
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Genera el código HTML con calidad de publicación del reporte."""
        raise NotImplementedError("Subclase debe implementar to_html()")

    def _repr_html_(self) -> str:
        """Renderizado interactivo automático en HTML para celdas de Jupyter Notebook."""
        return self.to_html()
