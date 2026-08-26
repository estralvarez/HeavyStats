import pathlib
import json
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


def load_default_data() -> pd.DataFrame:
    """Carga el DataFrame por defecto del archivo CSV de manera robusta,
    corrigiendo tipos de datos y caracteres mal codificados."""
    BASE_DIR = pathlib.Path(__file__).parent
    CSV_PATH = BASE_DIR / "../data/muestra_metales_pesados_23_07_2026.csv"
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se pudo encontrar el archivo CSV en la ruta: {CSV_PATH.resolve()}")
    
    # Cargar usando codificación UTF-8 y forzando Muestra_Codificada a tipo entero nullable (Int64)
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8", dtype={"Muestra_Codificada": "Int64"})
    
    # Diccionario de reemplazo de cadenas mis-encodadas (que contienen \ufffd)
    replacements = {
        "Mari\ufffdo": "Mariño",
        "Veh\ufffdculo": "Vehículo",
        "multivitam\ufffdnico": "multivitamínico",
        "Estaci\ufffdn": "Estación",
        "R\ufffdos": "Ríos",
        "qu\ufffdmicos": "químicos",
        "Latoner\ufffda": "Latonería",
        "Carpinter\ufffda": "Carpintería",
        "p\ufffdblico": "público",
        "prote\ufffdco": "proteico",
        "F\uffdbrica": "Fábrica",
        "mec\ufffdnico": "mecánico",
    }
    
    # Aplicar la limpieza en todas las columnas de tipo string
    for col in df.columns:
        if df[col].dtype == "object":
            for bad, good in replacements.items():
                df[col] = df[col].str.replace(bad, good, regex=False)
                
    return df


_df = None

def __getattr__(name: str):
    if name == "df":
        global _df
        if _df is None:
            _df = load_default_data()
        return _df
    raise AttributeError(f"module {__name__} has no attribute {name}")


class VariableTypeReport(dict):
    """
    Reporte de tipos de variables en formato horizontal (ancho).
    Hereda de dict y permite exportar a Texto, Markdown, HTML y JSON.
    """
    def __init__(self, data: pd.DataFrame, grouped_dict: Dict[str, List[str]]):
        super().__init__(grouped_dict)
        self.data = data
        self.shape = data.shape
        self._nulls = data.isna().sum().to_dict()

    def _write_file(self, filepath: Optional[str], content: str) -> None:
        if filepath:
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Emite el reporte en texto plano estructurado a lo ancho."""
        lines = [
            "=" * 90,
            f"REPORTE DE TIPOS DE VARIABLES | Dimensiones: {self.shape[0]} filas x {self.shape[1]} columnas",
            "=" * 90,
            f"{'Tipo de Dato'.ljust(15)} | {'Cant.'.rjust(5)} | Variables (Nulos detectados)",
            "-" * 90
        ]

        for dtype, cols in self.items():
            formatted_cols = []
            for col in cols:
                n_nulls = self._nulls.get(col, 0)
                formatted_cols.append(f"{col} [{n_nulls} nulos]" if n_nulls > 0 else col)
            
            vars_str = ", ".join(formatted_cols)
            lines.append(f"{str(dtype).ljust(15)} | {str(len(cols)).rjust(5)} | {vars_str}")

        lines.append("=" * 90)
        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        """Emite el reporte en Markdown con tabla extendida a lo ancho."""
        lines = [
            "## Distribución de Variables por Tipo",
            f"**Dimensiones del Dataset:** `{self.shape[0]}` filas × `{self.shape[1]}` columnas\n",
            "| Tipo de Dato | N° Columnas | Variables Asignadas |",
            "| :--- | :---: | :--- |"
        ]

        for dtype, cols in self.items():
            formatted_cols = []
            for col in cols:
                n_nulls = self._nulls.get(col, 0)
                if n_nulls > 0:
                    formatted_cols.append(f"`{col}` *(⚠️ {n_nulls} nulos)*")
                else:
                    formatted_cols.append(f"`{col}`")
            
            vars_cell = ", ".join(formatted_cols).replace("|", "\\|")
            lines.append(f"| **`{dtype}`** | {len(cols)} | {vars_cell} |")

        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Exporta la estructura a JSON."""
        report_data = {
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "types": {}
        }
        for dtype, cols in self.items():
            report_data["types"][str(dtype)] = {
                "count": len(cols),
                "columns": [
                    {
                        "name": str(col),
                        "nulls": int(self._nulls.get(col, 0))
                    }
                    for col in cols
                ]
            }
        json_str = json.dumps(report_data, indent=4, ensure_ascii=False)
        self._write_file(filepath, json_str)
        return json_str

    def _repr_markdown_(self) -> str:
        """Renderizado automático Markdown en Jupyter Notebooks."""
        return self.to_markdown()
    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<VariableTypeReport types={list(self.keys())} total_cols={self.shape[1]}>"


def columns_type(data: pd.DataFrame) -> VariableTypeReport:
    """Devuelve el reporte agrupado por tipo de columna."""
    grouped = {}
    for col in data.columns:
        dtype_str = str(data[col].dtype)
        grouped.setdefault(dtype_str, []).append(col)
    
    sorted_grouped = dict(sorted(grouped.items()))
    return VariableTypeReport(data, sorted_grouped)

def categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame conteniendo únicamente las variables categóricas."""
    return df.select_dtypes(include=["object", "category"])

def numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame conteniendo únicamente las variables numéricas."""
    return df.select_dtypes(include=[np.number])

class VariablesTableReport:
    """Reporte de tabla de variables clasificadas en Categóricas o Numéricas."""
    def __init__(self, df: pd.DataFrame, rows: List[Dict[str, Any]]):
        self.df = df
        self.rows = rows

    def _write_file(self, filepath: Optional[str], content: str) -> None:
        if filepath:
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve la tabla como un pandas DataFrame."""
        return pd.DataFrame(self.rows)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Devuelve la tabla en texto plano."""
        report_str = self.to_dataframe().to_string(index=False)
        self._write_file(filepath, report_str)
        return report_str

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        """Devuelve la clasificación formateada en una tabla Markdown horizontal."""
        grouped = {}
        for row in self.rows:
            grouped.setdefault(row["Tipo"], []).append(row["Variable"])
            
        lines = [
            "## Clasificación de Variables",
            f"**Total de Variables:** `{len(self.rows)}` columnas\n",
            "| Clasificación | N° Columnas | Variables Asignadas |",
            "| :--- | :---: | :--- |"
        ]
        
        for tipo in sorted(grouped.keys()):
            cols = grouped[tipo]
            formatted_cols = [f"`{col}`" for col in cols]
            vars_cell = ", ".join(formatted_cols)
            lines.append(f"| **{tipo}** | {len(cols)} | {vars_cell} |")
            
        report_str = "\n".join(lines)
        self._write_file(filepath, report_str)
        return report_str

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Exporta la estructura a JSON."""
        data = {"variables": self.rows}
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        self._write_file(filepath, json_str)
        return json_str

    def _repr_markdown_(self) -> str:
        """Renderizado automático en Markdown para Jupyter Notebook."""
        return self.to_markdown()

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"<VariablesTableReport variables={len(self.rows)}>"


def variables_table(df: pd.DataFrame) -> VariablesTableReport:
    """Construye una tabla (reporte) clasificando todas las variables del dataset en categóricas o numéricas."""
    cat_df = categorical_columns(df)
    num_df = numerical_columns(df)
    
    rows = []
    for col in df.columns:
        if col in cat_df.columns:
            tipo = "Categórica"
        elif col in num_df.columns:
            tipo = "Numérica"
        else:
            tipo = "Otro"
        rows.append({
            "Variable": col,
            "Tipo": tipo,
            "Tipo_Dato": str(df[col].dtype)
        })
    return VariablesTableReport(df, rows)

def get_analytical_sample(
    data: pd.DataFrame, 
    sample_col: str = "Muestra_Codificada"
) -> pd.DataFrame:
    """Extrae la muestra analítica a partir del dataset completo."""
    if sample_col not in data.columns:
        raise ValueError(f"La columna identificadora '{sample_col}' no existe en el DataFrame.")
    return data[data[sample_col].notna()].copy()


def select_metal(
    data: pd.DataFrame, 
    concentration_col: str,
    dropna: bool = True
) -> pd.DataFrame:
    """Filtra el dataset para conservar únicamente la concentración seleccionada y su riesgo."""
    mapping = {
        "plomo": "Plomo_ug_dL", "pb": "Plomo_ug_dL", "plomo_ug_dl": "Plomo_ug_dL",
        "mercurio": "Mercurio_ug_L", "hg": "Mercurio_ug_L", "mercurio_ug_l": "Mercurio_ug_L",
        "cadmio": "Cadmio_ug_L", "cd": "Cadmio_ug_L", "cadmio_ug_l": "Cadmio_ug_L"
    }
    col_normalized = concentration_col.strip().lower()
    target_col = mapping.get(col_normalized)

    if target_col is None:
        for c in data.columns:
            if c.lower() == col_normalized:
                target_col = c
                break
                
    if target_col is None or target_col not in data.columns:
        raise ValueError(
            f"Concentración '{concentration_col}' no reconocida. Opciones: 'Plomo_ug_dL' (Pb), 'Mercurio_ug_L' (Hg), 'Cadmio_ug_L' (Cd)."
        )

    risk_mapping = {
        "Plomo_ug_dL": "Riesgo_Pb",
        "Mercurio_ug_L": "Riesgo_Hg",
        "Cadmio_ug_L": "Riesgo_Cd"
    }
    
    all_concentrations = ["Plomo_ug_dL", "Mercurio_ug_L", "Cadmio_ug_L"]
    all_risks = ["Riesgo_Pb", "Riesgo_Hg", "Riesgo_Cd"]
    
    cols_to_exclude = [c for c in all_concentrations if c != target_col and c in data.columns]
    target_risk = risk_mapping.get(target_col)
    risks_to_exclude = [r for r in all_risks if r != target_risk and r in data.columns]
    
    filtered_data = data.drop(columns=cols_to_exclude + risks_to_exclude)
    
    if dropna:
        filtered_data = filtered_data.dropna(subset=[target_col])

    return filtered_data.reset_index(drop=True)