"""
Módulo base de estilos, utilidades y renderizado gráfico bivariante.
Define configuraciones editoriales (#0f172a), resolución de variables, límites toxicológicos
y utilidades comunes para el principio DRY.
"""

import os
import re
import warnings
import textwrap
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heavystats.univariate.plots.base import BasePlots
from heavystats.univariate.constants import (
    DEFAULT_LABELS_MAP,
    DEFAULT_CUSTOM_PARAMS,
    DEFAULT_PERMISSIBLE_LIMITS,
    get_label,
)
from heavystats.bivariate.constants import (
    DEFAULT_PRIMARY_METALS,
    DEFAULT_METAL_LIMITS,
    DEFAULT_METAL_PAIRS,
    CDC_LEAD_REFERENCE_VALUE,
    EPA_MERCURY_REFERENCE_VALUE,
    OMS_CADMIUM_REFERENCE_VALUE,
    DIET_ORDINAL_MAP,
    DIET_ORDINAL_LABELS,
)

# Etiquetas toxicológicas y de unidades estándar para publicaciones biomédicas
DEFAULT_BIOMEDICAL_METAL_LABELS: Dict[str, str] = {
    "Plomo_ug_dL": "Plomo en sangre (µg/dL)",
    "Mercurio_ug_L": "Mercurio en sangre (µg/L)",
    "Cadmio_ug_L": "Cadmio en sangre (µg/L)",
    "Plomo": "Plomo en sangre (µg/dL)",
    "Mercurio": "Mercurio en sangre (µg/L)",
    "Cadmio": "Cadmio en sangre (µg/L)",
    "pb": "Plomo en sangre (µg/dL)",
    "hg": "Mercurio en sangre (µg/L)",
    "cd": "Cadmio en sangre (µg/L)",
}

# Fuentes y especificación técnica rigurosa de límites de referencia toxicológicos
DEFAULT_REFERENCE_LABELS: Dict[str, str] = {
    "Plomo_ug_dL": "Ref. CDC BLRV (3.5 µg/dL)",
    "Mercurio_ug_L": "Ref. EPA / OMS (5.0 µg/L)",
    "Cadmio_ug_L": "Ref. OMS (1.0 µg/L)",
    "Plomo": "Ref. CDC BLRV (3.5 µg/dL)",
    "Mercurio": "Ref. EPA / OMS (5.0 µg/L)",
    "Cadmio": "Ref. OMS (1.0 µg/L)",
    "pb": "Ref. CDC BLRV (3.5 µg/dL)",
    "hg": "Ref. EPA / OMS (5.0 µg/L)",
    "cd": "Ref. OMS (1.0 µg/L)",
}


class BivariateBasePlots(BasePlots):
    """
    Clase base con la infraestructura común de configuración gráfica,
    resolución semántica de variables, etiquetado y utilidades DRY.
    Hereda del motor BasePlots para soportar canvas, mosaicos y estilizado editorial unificado.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        palette: Union[str, Sequence[str]] = "crest",
        style: str = "ticks",
        context: str = "notebook",
        labels_map: Optional[Dict[str, str]] = None,
        rc: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            df=df,
            palette=palette if isinstance(palette, str) else "crest",
            style=style,
            context=context,
            labels_map=labels_map,
            rc=rc,
        )
        if isinstance(palette, (list, tuple)):
            self.palette = palette

        if self.df is not None:
            # Consolidación definitiva de Exposicion_Lugares_canale en Exposicion_Lugares_canales
            if "Exposicion_Lugares_canale" in self.df.columns:
                if "Exposicion_Lugares_canales" in self.df.columns:
                    self.df["Exposicion_Lugares_canales"] = (
                        self.df[["Exposicion_Lugares_canales", "Exposicion_Lugares_canale"]]
                        .fillna(0)
                        .max(axis=1)
                        .astype("Int64")
                    )
                    self.df = self.df.drop(columns=["Exposicion_Lugares_canale"])
                else:
                    self.df = self.df.rename(columns={"Exposicion_Lugares_canale": "Exposicion_Lugares_canales"})

    def get_label(self, col: str) -> str:
        """Obtiene la etiqueta limpia de una variable."""
        return get_label(col, self.labels_map)

    def _resolve_limit(self, col: str, permissible_limit: Optional[float] = None) -> Optional[float]:
        """Resuelve el límite permisible de referencia toxicológica."""
        if permissible_limit is not None:
            return permissible_limit
        return DEFAULT_PERMISSIBLE_LIMITS.get(col)

    def _resolve_column(self, col: str) -> str:
        """Resuelve el nombre exacto de una columna en el DataFrame admitiendo alias comunes y formato enriquecido."""
        if self.df is None:
            return col
        clean_c = str(col).replace("**", "").replace("<strong>", "").replace("</strong>", "").strip()
        if clean_c in self.df.columns:
            return clean_c

        clean_underscore = clean_c.replace(" ", "_")
        if clean_underscore in self.df.columns:
            return clean_underscore

        # Mapeo directo para errata de canale
        if clean_underscore.lower() in ("exposicion_lugares_canale", "exposicion_lugares_canales"):
            if "Exposicion_Lugares_canales" in self.df.columns:
                return "Exposicion_Lugares_canales"

        alias_map = {
            "plomo": "Plomo_ug_dL",
            "pb": "Plomo_ug_dL",
            "lead": "Plomo_ug_dL",
            "plomo_ug_dl": "Plomo_ug_dL",
            "mercurio": "Mercurio_ug_L",
            "hg": "Mercurio_ug_L",
            "mercury": "Mercurio_ug_L",
            "mercurio_ug_l": "Mercurio_ug_L",
            "cadmio": "Cadmio_ug_L",
            "cd": "Cadmio_ug_L",
            "cadmium": "Cadmio_ug_L",
            "cadmio_ug_l": "Cadmio_ug_L",
            "edad": "Edad",
            "age": "Edad",
            "peso": "Peso_kg",
            "peso_kg": "Peso_kg",
            "altura": "Altura_cm",
            "talla": "Altura_cm",
            "altura_cm": "Altura_cm",
            "score": "Score_Riesgo",
            "score_riesgo": "Score_Riesgo",
            "riesgo": "Score_Riesgo",
            "sexo": "Sexo",
            "sector": "Sector",
            "institucion": "Institucion",
            "es_expuesto": "Es_Expuesto",
            "es expuesto": "Es_Expuesto",
            "exposicion_lugares_canale": "Exposicion_Lugares_canales",
            "exposicion_lugares_canales": "Exposicion_Lugares_canales",
        }
        lower_c = clean_underscore.lower()
        if lower_c in alias_map and alias_map[lower_c] in self.df.columns:
            return alias_map[lower_c]

        for c in self.df.columns:
            if c.lower() == lower_c or c.lower().replace("_", " ") == clean_c.lower():
                return c

        return clean_underscore

    @staticmethod
    def _wrap_text(text: str, width: int = 13) -> str:
        """Envuelve texto largo con saltos de línea para mantener rotación 0°."""
        if not text:
            return ""
        lines = str(text).split("\n")
        wrapped = [textwrap.fill(line, width=width, break_long_words=False) for line in lines]
        return "\n".join(wrapped)

    def _format_category_label(self, group_col: str, cat_val: Any, count: int) -> str:
        """Formatea el nombre de la categoría incluyendo el tamaño muestral (n=...) con salto de línea."""
        val_str = str(cat_val).strip()
        val_lower = val_str.lower()

        # Mapeos semánticos contextuales limpios
        if group_col in ("Es_Expuesto", "es_expuesto") or group_col.startswith("Exposicion_") or group_col.startswith("Salud_"):
            if val_lower in ("0", "0.0", "false", "no", "no expuesto", "no_expuesto"):
                clean_name = "No expuesto" if "expuesto" in group_col.lower() else "No"
            elif val_lower in ("1", "1.0", "true", "si", "sí", "expuesto"):
                clean_name = "Expuesto" if "expuesto" in group_col.lower() else "Sí"
            else:
                clean_name = val_str.replace("_", " ").title()
        elif group_col.lower() == "sexo":
            if val_lower in ("f", "fem", "femenino"):
                clean_name = "Femenino"
            elif val_lower in ("m", "masc", "masculino"):
                clean_name = "Masculino"
            else:
                clean_name = val_str.title()
        else:
            clean_name = val_str.replace("_", " ").title()

        wrapped_name = self._wrap_text(clean_name, width=13)
        return f"{wrapped_name}\n(n={count})"

    @staticmethod
    def _clean_spines_and_ticks(ax: plt.Axes, rotation: int = 0, labelsize: float = 9.0) -> None:
        """Aplica la directriz editorial de spines limpios y ticks en negrita (#0f172a)."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
            if rotation != 0:
                tick.set_rotation(rotation)
                tick.set_ha("right")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.tick_params(axis="both", labelsize=labelsize)

    @staticmethod
    def _save_figure(
        fig: plt.Figure,
        base_name_or_filepath: Optional[str] = None,
        save_dir: Optional[str] = None,
        formats: Union[str, Sequence[str]] = "png",
        dpi: int = 300,
        close: bool = False,
        filepath: Optional[str] = None,
        save_format: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Exporta de forma robusta la figura en alta resolución (PNG/PDF) aplicando tight_layout.
        Compatible con ambas convenciones:
        - Bivariante directa: `_save_figure(fig, filepath=...)` o `_save_figure(fig, "ruta/archivo.png")`
        - Univariante modular: `_save_figure(fig, base_name, save_dir=..., formats=...)`
        """
        try:
            fig.tight_layout()
        except Exception:
            pass

        target_file = filepath or base_name_or_filepath
        active_formats = save_format if save_format is not None else formats

        # Si se especifica save_dir, se utiliza la convención BasePlots (guardado en múltiples formatos)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            name = base_name_or_filepath or "figura"
            fmt_list = [active_formats] if isinstance(active_formats, str) else list(active_formats)
            for fmt in fmt_list:
                clean_fmt = fmt.lstrip(".").lower()
                out_path = os.path.join(save_dir, f"{name}.{clean_fmt}")
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        elif target_file:
            dir_path = os.path.dirname(target_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            fig.savefig(target_file, dpi=dpi, bbox_inches="tight")

        if close:
            plt.close(fig)

    @staticmethod
    def _format_p_val(p_val: float, threshold: float = 0.001, precision: int = 3) -> str:
        """Formatea p-valores de forma consistente para publicación científica."""
        if pd.isna(p_val):
            return "p = N/D"
        if p_val < threshold:
            return f"p < {threshold}"
        return f"p = {p_val:.{precision}f}"
