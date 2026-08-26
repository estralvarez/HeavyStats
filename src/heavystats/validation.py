import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import json
import os

class ValidationReport:
    def __init__(self, checks: List[Dict[str, Any]]):
        self.checks = checks
        self.all_passed = all(check.get('passed', False) for check in checks)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Genera un reporte legible en texto plano."""
        status_header = "[PASÓ]" if self.all_passed else "[FALLÓ]"
        lines = [
            f"=== REPORTE DE VALIDACIÓN {status_header} ===",
            f"Estado general: {'Aprobado' if self.all_passed else 'Requiere atención'}",
            "-" * 50
        ]
        
        for check in self.checks:
            mark = "[OK]" if check["passed"] else "[X] "
            lines.append(f"{mark} {check['criterio']}: {check['message']}")
            
        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        """Genera un reporte en formato Markdown con tabla de resultados."""
        badge = "🟢 **VALIDACIÓN EXITOSA**" if self.all_passed else "🔴 **VALIDACIÓN CON FALLOS**"
        
        lines = [
            "# Reporte de Calidad de Datos",
            f"**Estado General:** {badge}\n",
            "| Estado | Criterio | Detalle / Mensaje |",
            "| :---: | :--- | :--- |"
        ]
        
        for check in self.checks:
            icon = "✅" if check["passed"] else "❌"
            criterio = check["criterio"]
            # Escapar pipes en el mensaje para evitar romper la tabla Markdown
            mensaje = str(check["message"]).replace("|", "\\|")
            lines.append(f"| {icon} | **{criterio}** | {mensaje} |")
            
        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def _write_file(self, filepath: Optional[str], content: str) -> None:
        if filepath:
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def to_json(self, filepath: Optional[str] = None) -> str:
        serialized_checks = [
            {
                "criterio": str(check["criterio"]),
                "passed": bool(check["passed"]),
                "message": str(check["message"])
            }
            for check in self.checks
        ]
        data = {
            "all_passed": bool(self.all_passed),
            "checks": serialized_checks
        }
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        self._write_file(filepath, json_str)
        return json_str

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados en un DataFrame."""
        return pd.DataFrame(self.checks)

    def _repr_markdown_(self) -> str:
        """Renderizado automático en Markdown para celdas de Jupyter Notebook."""
        return self.to_markdown()

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<ValidationReport passed={self.all_passed} total_checks={len(self.checks)}>"


def validate_data(
    data: pd.DataFrame, 
    expected_shape: Tuple[int, int] = (48, 39),
    expected_sample_size: int = 20,
    sample_col: str = "Muestra_Codificada",
    critical_cols: Optional[List[str]] = None
) -> ValidationReport:
    """Evalúa un DataFrame contra reglas de calidad de datos."""
    
    checks = []
    
    # Parámetros por defecto para columnas críticas
    if critical_cols is None:
        critical_cols = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L", "Peso_kg", "Altura_cm"]

    # 1. Validar la forma (shape) - Se evalúa siempre
    shape_passed = data.shape == expected_shape
    checks.append({
        "criterio": "Dimensiones",
        "passed": shape_passed,
        "message": f"{data.shape[0]} filas y {data.shape[1]} columnas." if shape_passed else f"Forma actual {data.shape}, esperada {expected_shape}."
    })
    
    # 2. Validar columna de muestra y tamaño de la muestra
    sample_col_exists = sample_col in data.columns
    if not sample_col_exists:
        checks.append({
            "criterio": "Población y Muestra",
            "passed": False,
            "message": f"Columna identificadora '{sample_col}' no existe."
        })
    else:
        sample_mask = data[sample_col].notna()
        muestra_observada = sample_mask.sum()
        sample_passed = (muestra_observada == expected_sample_size)
        checks.append({
            "criterio": "Población y Muestra",
            "passed": sample_passed,
            "message": f"Población N={len(data)} | Muestra n={muestra_observada}." if sample_passed else f"Muestra de {muestra_observada} (se esperaban {expected_sample_size})."
        })
    
    # 3. Validar presencia de columnas críticas
    columnas_faltantes = [col for col in critical_cols if col not in data.columns]
    cols_exist = len(columnas_faltantes) == 0
    checks.append({
        "criterio": "Columnas Críticas",
        "passed": cols_exist,
        "message": "Todas las columnas requeridas están presentes." if cols_exist else f"Faltan las columnas: {', '.join(columnas_faltantes)}."
    })
    
    # 4. Validar registros duplicados
    duplicate_count = int(data.duplicated().sum())
    checks.append({
        "criterio": "Valores Duplicados",
        "passed": duplicate_count == 0,
        "message": "Sin registros duplicados en el dataset." if duplicate_count == 0 else f"Se detectaron {duplicate_count} filas completamente duplicadas."
    })
        
    # Validaciones sobre la muestra (SÓLO si existen las columnas críticas y la columna de muestra)
    if cols_exist and sample_col_exists:
        datos_muestra = data.loc[sample_mask, critical_cols]
        
        # 5. Completitud (Nulos en la muestra)
        nulos_por_columna = datos_muestra.isna().sum()
        if nulos_por_columna.any():
            cols_con_nulos = nulos_por_columna[nulos_por_columna > 0].index.tolist()
            checks.append({
                "criterio": "Completitud",
                "passed": False,
                "message": f"Nulos detectados en la muestra para: {', '.join(cols_con_nulos)}."
            })
        else:
            checks.append({
                "criterio": "Completitud",
                "passed": True,
                "message": "Sin valores nulos en los datos de la muestra."
            })
            
        # 6. Tipos de Datos (Valida tipo numérico para variables críticas)
        invalid_cols_info = []
        for col in critical_cols:
            if not pd.api.types.is_numeric_dtype(datos_muestra[col]):
                coerced = pd.to_numeric(datos_muestra[col], errors='coerce')
                bad_mask_col = coerced.isna() & datos_muestra[col].notna()
                if bad_mask_col.any():
                    invalid_idx = bad_mask_col.idxmax()
                    invalid_val = datos_muestra.loc[invalid_idx, col]
                    invalid_cols_info.append(f"'{col}' (ej. fila {invalid_idx}: '{invalid_val}')")
                else:
                    invalid_cols_info.append(f"'{col}'")

        if invalid_cols_info:
            checks.append({
                "criterio": "Tipos de Datos",
                "passed": False,
                "message": f"Valores no numéricos en: {', '.join(invalid_cols_info)}."
            })
        else:
            checks.append({
                "criterio": "Tipos de Datos",
                "passed": True,
                "message": "Todas las columnas críticas tienen un formato numérico válido."
            })

        # 7. Valores Imposibles (Límites Físicos y Lógicos)
        impossible_checks = []
        if not data["Edad"].between(0, 18).all():
            impossible_checks.append("Edad fuera del rango esperado (0-18 años)")
        if not datos_muestra["Peso_kg"].between(5, 150).all():
            impossible_checks.append("Peso fuera del rango esperado (5-150 kg)")
        if not datos_muestra["Altura_cm"].between(50, 220).all():
            impossible_checks.append("Altura fuera del rango esperado (50-220 cm)")
            
        for col in ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L"]:
            if col in data.columns:
                if (datos_muestra[col] < 0).any():
                    impossible_checks.append(f"{col} contiene valores negativos")
                    
        passed_impossible = len(impossible_checks) == 0
        checks.append({
            "criterio": "Valores Imposibles",
            "passed": passed_impossible,
            "message": "Todos los límites lógicos y físicos se cumplen." if passed_impossible else f"Inconsistencias detectadas: {', '.join(impossible_checks)}."
        })

        # 8. Codificación de Categorías (Consistencia de etiquetas)
        invalid_categories = []
        if "Sexo" in data.columns:
            sex_values = set(data["Sexo"].dropna().unique())
            if not sex_values.issubset({"femenino", "masculino"}):
                invalid_categories.append(f"Sexo: {sex_values}")
                
        for col in ["Es_Expuesto", "Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]:
            if col in data.columns:
                col_values = set(data[col].dropna().unique())
                if not col_values.issubset({"SI", "NO"}):
                    invalid_categories.append(f"{col}: {col_values}")
                    
        passed_categories = len(invalid_categories) == 0
        checks.append({
            "criterio": "Codificación Categorías",
            "passed": passed_categories,
            "message": "Codificación de variables categóricas consistente." if passed_categories else f"Inconsistencias de codificación detectadas en: {', '.join(invalid_categories)}."
        })

        # 9. Límites de Detección (LOD)
        lod_issues = []
        lods = {
            "Plomo_ug_dL": 0.1,
            "Mercurio_ug_L": 0.1,
            "Cadmio_ug_L": 0.05
        }
        for col, lod_val in lods.items():
            if col in data.columns:
                below_lod_count = (datos_muestra[col] < lod_val).sum()
                if below_lod_count > 0:
                    lod_issues.append(f"{col} tiene {below_lod_count} registros bajo el LOD ({lod_val})")
                    
        passed_lod = len(lod_issues) == 0
        checks.append({
            "criterio": "Límites de Detección",
            "passed": passed_lod,
            "message": "Todas las concentraciones superan los límites de detección típicos." if passed_lod else f"Valores bajo el LOD detectados: {', '.join(lod_issues)}."
        })

    # Ya no imprimimos/mostramos aquí. Retornamos el objeto limpiamente.
    return ValidationReport(checks)

if __name__ == "__main__":
    import pathlib
    from heavystats.cleaning import load_default_data
    BASE_DIR = pathlib.Path(__file__).parent
    df = load_default_data()
    
    report = validate_data(df)
    
    # Exportar a JSON para el portal de documentación de Astro
    json_path = BASE_DIR / "../../../../QABLOG/HeavyDocs/src/data/validation_report.json"
    try:
        report.to_json(json_path)
        print(f"Reporte de validación exportado exitosamente a JSON en: {json_path}")
    except Exception as e:
        print(f"No se pudo exportar el reporte a JSON: {e}")
    
    try:
        from IPython.display import display
        from IPython import get_ipython
        if get_ipython() is not None:
            display(report)
        else:
            print(report)
    except (ImportError, NameError):
        print(report)
