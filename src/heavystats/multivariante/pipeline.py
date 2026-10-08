"""
Pipeline orquestador para el análisis multivariante en HeavyStats.
Integra el tamizaje bivariante no paramétrico con los modelos multivariantes (LASSO y PLS-R)
y la generación automática de reportes de calidad de publicación.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import numpy as np
import pandas as pd

from heavystats.multivariante.screening import MultivariateScreening
from heavystats.multivariante.models import LassoModeler, PlsModeler, prepare_features_matrix
from heavystats.multivariante.reports import MultivariateTableReport
from heavystats.bivariate.constants import DEFAULT_LABELS_MAP, get_label


class MultivariatePipeline:
    """
    Orquestador integral del análisis multivariante.
    
    Flujo de trabajo:
    1. Screening bivariado no paramétrico factor por factor con corrección Benjamini-Hochberg FDR.
    2. Filtrado y priorización de factores clave de riesgo o exposición.
    3. Modelado conjunto con Regresión LASSO (selección parsimoniosa) y PLS-R (proyección latente y VIP).
    4. Generación y exportación de reportes consolidados en LaTeX (booktabs) y HTML interactivo.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        candidate_cols: Optional[List[str]] = None,
        log_transform_target: bool = True,
        alpha: float = 0.05,
        fdr_threshold: float = 0.10,
        labels_map: Optional[Dict[str, str]] = None,
        random_state: int = 42,
    ):
        self.data = data.copy()
        self.target_col = target_col
        self.candidate_cols = candidate_cols
        self.log_transform_target = bool(log_transform_target)
        self.alpha = float(alpha)
        self.fdr_threshold = float(fdr_threshold)
        self.labels_map = labels_map or DEFAULT_LABELS_MAP
        self.random_state = int(random_state)

        if target_col not in self.data.columns:
            raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

        # Componentes
        self.screening = MultivariateScreening(
            data=self.data,
            target_col=self.target_col,
            candidate_cols=self.candidate_cols,
            labels_map=self.labels_map,
            alpha=self.alpha,
            fdr_threshold=self.fdr_threshold,
            random_state=self.random_state,
        )

        self.screening_df: Optional[pd.DataFrame] = None
        self.lasso_model: Optional[LassoModeler] = None
        self.pls_model: Optional[PlsModeler] = None
        self.consolidated_df: Optional[pd.DataFrame] = None
        self.report_: Optional[MultivariateTableReport] = None

    def run_screening(self) -> pd.DataFrame:
        """
        Ejecuta la Etapa 1: Tamizaje bivariante no paramétrico.
        """
        self.screening_df = self.screening.run()
        return self.screening_df.copy()

    def fit_models(
        self,
        min_priority: str = "Intermedia",
        custom_features: Optional[List[str]] = None,
        lasso_cv_folds: int = 5,
        pls_components: int = 2,
        lasso_criterion: str = "cv",
    ) -> Dict[str, Any]:
        """
        Ejecuta la Etapa 2: Ajuste de modelos multivariantes (LASSO y PLS-R).
        
        Parámetros
        ----------
        min_priority : str
            Nivel mínimo de prioridad para incluir variables ('Alta', 'Intermedia' o 'Todas').
        custom_features : Optional[List[str]]
            Lista explícita de variables si se desea omitir la selección automática.
        lasso_cv_folds : int
            Número de pliegues para validación cruzada en LASSO.
        pls_components : int
            Número de componentes latentes ortogonales para PLS-R.
        lasso_criterion : str
            Criterio de regularización ('cv', 'aic', 'bic').
        """
        if self.screening_df is None:
            self.run_screening()

        if custom_features is not None:
            selected_features = [f for f in custom_features if f in self.data.columns]
        else:
            selected_features = self.screening.get_prioritized_features(min_priority=min_priority)

        if not selected_features:
            raise ValueError("No se encontraron variables candidatas que cumplan el criterio de selección.")

        # Preparar matriz numérica X
        X_df, feature_cols = prepare_features_matrix(self.data, selected_features)

        # Preparar variable respuesta y
        y_raw = pd.to_numeric(self.data[self.target_col], errors="coerce").values
        if self.log_transform_target:
            # Asegurar valores estrictamente positivos para logaritmo
            min_val = np.nanmin(y_raw[y_raw > 0]) if np.any(y_raw > 0) else 1e-3
            y_clean = np.where(y_raw <= 0, min_val / 2.0, y_raw)
            y_target = np.log(y_clean)
        else:
            y_target = y_raw

        # 1. Ajustar LASSO
        self.lasso_model = LassoModeler(
            cv_folds=lasso_cv_folds,
            criterion=lasso_criterion,
            random_state=self.random_state
        )
        self.lasso_model.fit(X_df, y_target)

        # 2. Ajustar PLS-R
        self.pls_model = PlsModeler(
            n_components=pls_components,
            scale=True
        )
        self.pls_model.fit(X_df, y_target)

        # 3. Consolidar resultados
        lasso_res = self.lasso_model.summary_df.set_index("Variable")
        pls_res = self.pls_model.summary_df.set_index("Variable")

        # Vincular con la tabla de tamizaje
        records = []
        scr_indexed = self.screening_df.set_index("Variable") if self.screening_df is not None else pd.DataFrame()

        for feat in feature_cols:
            label = get_label(feat, self.labels_map)
            rec: Dict[str, Any] = {
                "Variable": feat,
                "Etiqueta": label,
            }

            if feat in scr_indexed.index:
                row_scr = scr_indexed.loc[feat]
                rec["Grupo"] = row_scr.get("Grupo", "General")
                rec["Tipo"] = row_scr.get("Tipo", "")
                rec["Estadístico"] = row_scr.get("Estadístico", np.nan)
                rec["Estadístico_Str"] = row_scr.get("Estadístico_Str", "---")
                rec["p_valor"] = row_scr.get("p_valor", np.nan)
                rec["q_fdr"] = row_scr.get("q_fdr", np.nan)
                rec["Direccion_Biv"] = row_scr.get("Direccion", "---")
                rec["Prioridad"] = row_scr.get("Prioridad", "---")
            else:
                rec["Grupo"] = "General"
                rec["Tipo"] = "Codificada"
                rec["Estadístico_Str"] = "---"
                rec["p_valor"] = np.nan
                rec["q_fdr"] = np.nan
                rec["Direccion_Biv"] = "---"
                rec["Prioridad"] = "Incluida"

            if feat in lasso_res.index:
                rec["Beta_Estandarizado"] = lasso_res.loc[feat, "Beta_Estandarizado"]
                rec["Efecto_Lasso"] = lasso_res.loc[feat, "Efecto_Lasso"]
            else:
                rec["Beta_Estandarizado"] = 0.0
                rec["Efecto_Lasso"] = "Descartado (β = 0)"

            if feat in pls_res.index:
                rec["VIP"] = pls_res.loc[feat, "VIP"]
                rec["Coef_PLS"] = pls_res.loc[feat, "Coef_PLS"]
                rec["Influencia_PLS"] = pls_res.loc[feat, "Influencia_PLS"]
            else:
                rec["VIP"] = np.nan
                rec["Coef_PLS"] = np.nan
                rec["Influencia_PLS"] = "---"

            records.append(rec)

        cons_df = pd.DataFrame(records)
        # Ordenar por VIP o Beta
        cons_df = cons_df.sort_values(by="VIP", ascending=False).reset_index(drop=True)
        self.consolidated_df = cons_df

        self.report_ = MultivariateTableReport(
            consolidated_df=self.consolidated_df,
            screening_df=self.screening_df,
            lasso_summary_df=self.lasso_model.summary_df,
            pls_summary_df=self.pls_model.summary_df,
            target_name=self.target_col,
            log_transformed=self.log_transform_target,
        )

        return {
            "consolidated": self.consolidated_df,
            "lasso": self.lasso_model,
            "pls": self.pls_model,
            "report": self.report_
        }

    def get_report(self) -> MultivariateTableReport:
        """Retorna la instancia del reporte editorial consolidado."""
        if self.report_ is None:
            self.fit_models()
        assert self.report_ is not None
        return self.report_

    def export_latex(
        self, 
        filepath: str, 
        caption: Optional[str] = None, 
        label: str = "tab:multivariante_factores"
    ) -> str:
        """Exporta la tabla a formato LaTeX con el paquete booktabs."""
        rep = self.get_report()
        return rep.to_latex(filepath=filepath, caption=caption, label=label)

    def export_html(self, filepath: str) -> None:
        """Exporta el reporte interactivo completo en formato HTML."""
        rep = self.get_report()
        rep.to_html(filepath)

    def export_excel(self, filepath: str) -> None:
        """Exporta las tablas a un libro de Excel con múltiples hojas."""
        rep = self.get_report()
        rep.to_excel(filepath)
