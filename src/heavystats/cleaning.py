import pathlib
import os
import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Tuple, List, Dict, Optional, Any


def load_data(csv_path: Optional[Any] = None) -> pd.DataFrame:
    """Carga el DataFrame de un archivo CSV de manera robusta,
    corrigiendo tipos de datos y caracteres mal codificados.
    
    Si no se proporciona csv_path, se usará el archivo por defecto."""
    BASE_DIR = pathlib.Path(__file__).parent
    
    if csv_path is None:
        CSV_PATH = BASE_DIR / "data/data_example.csv"
    else:
        CSV_PATH = pathlib.Path(csv_path)
        
    # Crear la carpeta contenedora si no existe
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se pudo encontrar el archivo CSV en la ruta: {CSV_PATH.resolve()}")
    
    # Cargar usando codificación UTF-8 (utf-8-sig elimina automáticamente el BOM) con fallback a latin1
    try:
        df = pd.read_csv(CSV_PATH, sep=";", decimal=',', encoding="utf-8-sig", dtype={"Muestra_Codificada": "Int64"})
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, sep=";", decimal=',', encoding="latin1", dtype={"Muestra_Codificada": "Int64"})
    
    # Sanitizar nombres de columnas por si persiste algún BOM invisible (\ufeff) o visible (ï»¿)
    df.columns = [str(c).lstrip("\ufeff").lstrip("ï»¿").strip() for c in df.columns]
    
    # Limpieza preventiva para archivos que hayan sido guardados previamente con el carácter de reemplazo (\ufffd)
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
    
    # Aplicar la limpieza en todas las columnas de tipo string si contienen \ufffd
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            if df[col].astype(str).str.contains("\ufffd").any():
                for bad, good in replacements.items():
                    df[col] = df[col].str.replace(bad, good, regex=False)
                
    return df


_df = None

def __getattr__(name: str):
    if name == "df":
        global _df
        if _df is None:
            _df = load_data()
        return _df
    raise AttributeError(f"module {__name__} has no attribute {name}")


from heavystats.html_utils import wrap_html_container, format_html_str, BaseReport


class VariableTypeReport(dict, BaseReport):
    """
    Reporte de tipos de variables estructurado.
    Hereda de dict y permite renderizado HTML interactivo estilo publicación (Booktabs),
    texto plano y exportación a DataFrame/CSV/Excel.
    """
    def __init__(self, data: pd.DataFrame, grouped_dict: Dict[str, List[str]]):
        super().__init__(grouped_dict)
        self.data = data
        self.shape = data.shape
        self._nulls = data.isna().sum().to_dict()

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el reporte a un DataFrame estructurado por variable."""
        rows = []
        for dtype, cols in self.items():
            for col in cols:
                n_nulls = self._nulls.get(col, 0)
                rows.append({
                    "Variable": col,
                    "Tipo de Dato": str(dtype),
                    "Nulos": n_nulls,
                    "Porcentaje Nulos (%)": round((n_nulls / self.shape[0] * 100), 2) if self.shape[0] > 0 else 0.0
                })
        return pd.DataFrame(rows)

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Exporta los datos estructurados a un archivo CSV."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        kwargs.setdefault("encoding", "utf-8")
        self.to_dataframe().to_csv(filepath, index=kwargs.get("index", False), **kwargs)

    def to_excel(self, filepath: str, sheet_name: str = "Tipos_Variables", **kwargs: Any) -> None:
        """Exporta los datos estructurados a un archivo Excel (.xlsx)."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.to_dataframe().to_excel(filepath, sheet_name=sheet_name, index=kwargs.get("index", False), **kwargs)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Emite el reporte en texto plano estructurado."""
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
        if filepath:
            self._write_file(filepath, report_str)
        return report_str

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera una tabla HTML responsiva con calidad de publicación (estilo Booktabs)
        y tema claro forzado para perfecta legibilidad en fondos oscuros o claros.
        """
        lines = []
        lines.append("  <table class='hs-pub-table'>")
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append("        <th class='hs-left-col' style='width: 160px;'>Tipo de Dato</th>")
        lines.append("        <th class='hs-center-col' style='width: 110px;'>N° Columnas</th>")
        lines.append("        <th class='hs-left-col'>Variables Asignadas</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")

        for dtype, cols in self.items():
            var_pills = []
            for col in cols:
                n_nulls = self._nulls.get(col, 0)
                safe_col = format_html_str(col)
                if n_nulls > 0:
                    var_pills.append(f"<span class='hs-var-pill'>{safe_col} <span class='hs-null-badge'>⚠️ {n_nulls} nulos</span></span>")
                else:
                    var_pills.append(f"<span class='hs-var-pill'>{safe_col}</span>")
            
            pills_html = "".join(var_pills)
            safe_dtype = format_html_str(dtype)
            lines.append("      <tr>")
            lines.append(f"        <td class='hs-left-col'><span class='hs-dtype-code'>{safe_dtype}</span></td>")
            lines.append(f"        <td class='hs-center-col'><span class='hs-count-badge'>{len(cols)}</span></td>")
            lines.append(f"        <td class='hs-left-col'>{pills_html}</td>")
            lines.append("      </tr>")

        lines.append("    </tbody>")
        lines.append("  </table>")

        inner_html = "\n".join(lines)
        html_code = wrap_html_container(
            inner_html=inner_html,
            title="Distribución de Variables por Tipo de Dato",
            subtitle=f"Dimensiones del Dataset: **{self.shape[0]}** filas × **{self.shape[1]}** columnas",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

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
    return df.select_dtypes(include=["object", "category", "str"])

def numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame conteniendo únicamente las variables numéricas."""
    return df.select_dtypes(include=[np.number])

class VariablesTableReport(BaseReport):
    """
    Reporte de tabla de variables clasificadas en Categóricas o Numéricas.
    Permite renderizado HTML interactivo estilo publicación (Booktabs),
    texto plano y exportación a DataFrame/CSV/Excel.
    """
    def __init__(self, df: pd.DataFrame, rows: List[Dict[str, Any]]):
        self.df = df
        self.rows = rows

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve la tabla como un pandas DataFrame."""
        return pd.DataFrame(self.rows)

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Exporta la clasificación a un archivo CSV."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        kwargs.setdefault("encoding", "utf-8")
        self.to_dataframe().to_csv(filepath, index=kwargs.get("index", False), **kwargs)

    def to_excel(self, filepath: str, sheet_name: str = "Clasificacion_Variables", **kwargs: Any) -> None:
        """Exporta la clasificación a un libro de Excel (.xlsx)."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.to_dataframe().to_excel(filepath, sheet_name=sheet_name, index=kwargs.get("index", False), **kwargs)

    def to_text(self, filepath: Optional[str] = None) -> str:
        """Devuelve la tabla en texto plano."""
        report_str = self.to_dataframe().to_string(index=False)
        if filepath:
            self._write_file(filepath, report_str)
        return report_str

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """
        Genera una tabla HTML responsiva con calidad de publicación (estilo Booktabs)
        y tema claro forzado para perfecta legibilidad en fondos oscuros o claros.
        """
        grouped = {}
        for row in self.rows:
            grouped.setdefault(row["Tipo"], []).append(row["Variable"])

        lines = []
        lines.append("  <table class='hs-pub-table'>")
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append("        <th class='hs-left-col' style='width: 160px;'>Clasificación</th>")
        lines.append("        <th class='hs-center-col' style='width: 110px;'>N° Columnas</th>")
        lines.append("        <th class='hs-left-col'>Variables Asignadas</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")

        for tipo in sorted(grouped.keys()):
            cols = grouped[tipo]
            if str(tipo).lower().startswith("cat"):
                badge_html = f"<span class='hs-badge-cat'>{tipo}</span>"
            elif str(tipo).lower().startswith("num"):
                badge_html = f"<span class='hs-badge-num'>{tipo}</span>"
            else:
                badge_html = f"<span class='hs-badge-other'>{tipo}</span>"

            var_pills = [f"<span class='hs-var-pill'>{format_html_str(c)}</span>" for c in cols]
            pills_html = "".join(var_pills)

            lines.append("      <tr>")
            lines.append(f"        <td class='hs-left-col'>{badge_html}</td>")
            lines.append(f"        <td class='hs-center-col'><span class='hs-count-badge'>{len(cols)}</span></td>")
            lines.append(f"        <td class='hs-left-col'>{pills_html}</td>")
            lines.append("      </tr>")

        lines.append("    </tbody>")
        lines.append("  </table>")

        inner_html = "\n".join(lines)
        html_code = wrap_html_container(
            inner_html=inner_html,
            title="Clasificación de Variables",
            subtitle=f"Total de Variables: **{len(self.rows)}** columnas",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

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


def standardize_boolean_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Estandariza columnas de texto binario a valores limpios 'SI' o 'NO',
    preservando los valores nulos (NaN)."""
    df_clean = df.copy()
    if columns is None:
        columns = df_clean.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        
    mapping = {
        "sí": "SI", "si": "SI", "sï": "SI", "sI": "SI",
        "no": "NO"
    }
    
    for col in columns:
        if col in df_clean.columns:
            # Aplicar sólo si el tipo de columna es de tipo string/objeto
            if pd.api.types.is_string_dtype(df_clean[col]) or pd.api.types.is_categorical_dtype(df_clean[col]):
                def clean_val(val):
                    if pd.isna(val):
                        return val
                    val_str = str(val).strip().lower()
                    if val_str in mapping:
                        return mapping[val_str]
                    return val
                df_clean[col] = df_clean[col].apply(clean_val)
    return df_clean


def _clean_choice_suffix(col: str, choice: str) -> str:
    """Simplifica y normaliza la opción para construir un nombre de columna limpio y corto."""
    def normalize(text: str) -> str:
        text = text.lower()
        nfkd_form = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        text = re.sub(r'[^a-z0-9_]', '_', text)
        text = re.sub(r'_+', '_', text)
        return text.strip('_')
        
    col_norm = normalize(col)
    choice_norm = normalize(choice)
    
    col_words = col_norm.split('_')
    choice_words = choice_norm.split('_')
    
    def clean_word(w):
        if w.endswith('es'):
            return w[:-2]
        if w.endswith('s'):
            return w[:-1]
        return w
        
    col_stems = {clean_word(w) for w in col_words}
    
    # Spanish stop words
    STOP_WORDS = {'de', 'del', 'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'en', 'con', 'a', 'para', 'por'}
    
    filtered_words = []
    for w in choice_words:
        if clean_word(w) in col_stems or w in col_words:
            continue
        if w in STOP_WORDS:
            continue
        filtered_words.append(w)
        
    if not filtered_words:
        filtered_words = [w for w in choice_words if w not in STOP_WORDS]
        if not filtered_words:
            filtered_words = choice_words
            
    return "_".join(filtered_words)


def desaggregate_multiple_responses(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    separator: str = ";"
) -> pd.DataFrame:
    """Desagrega columnas de respuestas múltiples separadas por un delimitador
    en variables binarias individuales (valores 0 o 1, y NaN para valores nulos)."""
    if columns is None:
        columns = ["Salud_Transporte", "Salud_Agua", "Exposicion_Talleres", "Exposicion_Lugares", "Exposicion_Industrias", "Salud_Suplementos"]
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        # Obtener todas las opciones únicas
        all_choices = set()
        for val in df_clean[col].dropna():
            choices = [c.strip() for c in str(val).split(separator) if c.strip()]
            all_choices.update(choices)
            
        # Filtrar opciones negativas típicas
        negative_phrases = {
            "ninguna de las anteriores", "ninguno de los anteriores", 
            "ninguna de las anteriores.", "ninguno de los anteriores.",
            "ninguno", "ninguna", "ningún", "ningun", "nada"
        }
        valid_choices = [c for c in all_choices if c.lower() not in negative_phrases]
        
        # Crear columnas binarias
        for choice in sorted(valid_choices):
            suffix = _clean_choice_suffix(col, choice)
            new_col_name = f"{col}_{suffix}"
            df_clean[new_col_name] = df_clean[col].apply(
                lambda val: 1 if pd.notna(val) and choice in [c.strip() for c in str(val).split(separator)]
                else (np.nan if pd.isna(val) else 0)
            )
            df_clean[new_col_name] = df_clean[new_col_name].astype("Int64")
            
    return df_clean


def encode_dietary_frequencies(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    mapping: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """Codifica frecuencias de consumo de alimentos a variables ordinales
    de tipo entero (Int64), preservando los valores nulos (NaN)."""
    df_clean = df.copy()
    if columns is None:
        columns = [col for col in df_clean.columns if col.startswith("Alim_")]
        
    if mapping is None:
        mapping = {
            "nunca": 0,
            "rara vez": 1,
            "a veces": 2,
            "frecuentemente": 3,
            "diario": 4
        }
        
    for col in columns:
        if col in df_clean.columns:
            def map_freq(val):
                if pd.isna(val):
                    return val
                val_str = str(val).strip().lower()
                return mapping.get(val_str, val)
                
            df_clean[col] = df_clean[col].apply(map_freq)
            try:
                df_clean[col] = df_clean[col].astype("Int64")
            except Exception:
                pass
                
    return df_clean


def create_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea indicadores booleanos compuestos para agrupar fuentes ambientales o de hábitos
    con frecuencias bajas individuales (ej. Cualquier Taller, Cualquier Industria, Cualquier Lugar de Riesgo)
    y calcula el Índice de Masa Corporal (IMC) si no está presente.
    """
    df_clean = df.copy()

    # 1. Cualquier Taller
    taller_cols = [c for c in df_clean.columns if c.startswith("Exposicion_Talleres_")]
    if taller_cols:
        df_clean["Exposicion_Cualquier_Taller"] = df_clean[taller_cols].fillna(0).max(axis=1).astype("Int64")

    # 2. Cualquier Industria
    ind_cols = [c for c in df_clean.columns if c.startswith("Exposicion_Industrias_")]
    if ind_cols:
        df_clean["Exposicion_Cualquier_Industria"] = df_clean[ind_cols].fillna(0).max(axis=1).astype("Int64")

    # 3. Cualquier Lugar de Riesgo
    lugar_cols = [c for c in df_clean.columns if c.startswith("Exposicion_Lugares_")]
    if lugar_cols:
        df_clean["Exposicion_Cualquier_Lugar_Riesgo"] = df_clean[lugar_cols].fillna(0).max(axis=1).astype("Int64")

    # 4. Agua de Riesgo
    if "Salud_Agua_pozo_profundo" in df_clean.columns:
        df_clean["Salud_Cualquier_Agua_Riesgo"] = df_clean["Salud_Agua_pozo_profundo"].fillna(0).astype("Int64")

    # 5. Cálculo del IMC si están Peso_kg y Altura_cm
    if "Peso_kg" in df_clean.columns and "Altura_cm" in df_clean.columns and "IMC" not in df_clean.columns:
        altura_m = df_clean["Altura_cm"] / 100.0
        df_clean["IMC"] = (df_clean["Peso_kg"] / (altura_m ** 2)).round(2)

    return df_clean
