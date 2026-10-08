"""
Motor de tamizaje bivariante no paramétrico para análisis multivariante.
Implementa el cribado factor por factor, determinación de dirección (Aporte/Atenuación),
corrección de multiplicidad (Benjamini-Hochberg FDR) y asignación multicriterio de prioridades.
"""

from typing import Dict, List, Optional, Any, Union, Sequence, Tuple
import numpy as np
import pandas as pd

from heavystats.bivariate.tests import (
    mann_whitney_test,
    kruskal_wallis_test,
    spearman_correlation,
    kendall_correlation,
    adjust_pvalues,
)
from heavystats.bivariate.constants import (
    DEFAULT_BIVARIATE_GROUPS,
    DEFAULT_LABELS_MAP,
    get_label,
)


class MultivariateScreening:
    """
    Motor de tamizaje bivariante no paramétrico (Screening).
    
    Evalúa sistemáticamente cada covariable candidata frente a la variable respuesta biológica
    mediante contrastes de distribución libre, determina la dirección del efecto (Aporte/Atenuación),
    calcula valores q de FDR y clasifica los factores por prioridad para el modelado multivariante.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        candidate_cols: Optional[List[str]] = None,
        group_dict: Optional[Dict[str, List[str]]] = None,
        labels_map: Optional[Dict[str, str]] = None,
        alpha: float = 0.05,
        fdr_threshold: float = 0.10,
        n_boot: int = 1000,
        random_state: int = 42,
    ):
        self.data = data.copy()
        self.target_col = target_col
        self.alpha = float(alpha)
        self.fdr_threshold = float(fdr_threshold)
        self.n_boot = int(n_boot)
        self.random_state = int(random_state)
        self.labels_map = labels_map or DEFAULT_LABELS_MAP
        self.group_dict = group_dict or DEFAULT_BIVARIATE_GROUPS

        if target_col not in self.data.columns:
            raise ValueError(f"La columna objetivo '{target_col}' no se encuentra en el DataFrame.")

        # Filtrar candidatos
        if candidate_cols is not None:
            self.candidate_cols = [c for c in candidate_cols if c in self.data.columns and c != target_col]
        else:
            # Detección automática descartando metadatos o texto sin estructurar
            exclude = {
                target_col, "Muestra_Codificada", "ID", "id",
                "Exposicion_Talleres", "Exposicion_Industrias", "Exposicion_Lugares",
                "Salud_Transporte", "Salud_Agua", "Salud_Suplementos"
            }
            self.candidate_cols = [c for c in self.data.columns if c not in exclude]

        self._results_df: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el tamizaje sobre todas las covariables candidatas.
        
        Retorna
        -------
        pd.DataFrame
            Tabla con métricas de screening, tamaños de efecto, p-valores, q-FDR y prioridades.
        """
        y = pd.to_numeric(self.data[self.target_col], errors="coerce")
        valid_y_mask = y.notna()

        records: List[Dict[str, Any]] = []

        for col in self.candidate_cols:
            s = self.data[col]
            valid_mask = valid_y_mask & s.notna()
            s_clean = s[valid_mask]
            y_clean = y[valid_mask]

            if len(s_clean) < 3:
                continue

            # Buscar categoría conceptual
            category_group = "Otros Factores"
            for grp_name, grp_cols in self.group_dict.items():
                if col in grp_cols:
                    category_group = grp_name
                    break

            unique_vals = s_clean.unique()
            n_unique = len(unique_vals)

            label = get_label(col, self.labels_map)

            # Caso 1: Variable Dicotómica (2 categorías)
            is_bool_like = (
                n_unique == 2 or
                set(s_clean.astype(str).str.upper()).issubset({"NO", "SI", "0", "1", "FALSE", "TRUE"})
            )

            if is_bool_like and n_unique <= 2:
                # Normalizar grupo expuesto vs no expuesto
                str_vals = s_clean.astype(str).str.upper()
                if set(str_vals).issubset({"NO", "SI", "0", "1", "FALSE", "TRUE"}):
                    mask_exposed = str_vals.isin({"SI", "1", "TRUE"})
                else:
                    # Segundo valor como referencia expuesta
                    mask_exposed = (s_clean == unique_vals[1])

                g1 = y_clean[mask_exposed].values
                g0 = y_clean[~mask_exposed].values

                if len(g1) >= 2 and len(g0) >= 2:
                    mw = mann_whitney_test(
                        g1, g0, 
                        n_boot=self.n_boot, 
                        confidence=1.0 - self.alpha, 
                        random_state=self.random_state
                    )
                    u_stat = mw["u_stat"]
                    p_val = mw["p_val"]
                    r_rb = mw["r_rb"]
                    ci_l, ci_h = mw["r_rb_ci"]
                    hl = mw["hl_shift"]

                    direction = (
                        "Aporte (↑)" if r_rb > 0.05 else ("Atenuación (↓)" if r_rb < -0.05 else "Neutro (≈)")
                    )

                    records.append({
                        "Variable": col,
                        "Etiqueta": label,
                        "Grupo": category_group,
                        "Tipo": "Binaria",
                        "Prueba": "Mann-Whitney U",
                        "Estadístico": u_stat,
                        "Estadístico_Str": f"U = {u_stat:.1f}",
                        "Efecto": r_rb,
                        "Efecto_Nombre": "r_rb",
                        "Efecto_Str": f"{r_rb:+.2f}",
                        "IC95_Efecto": f"[{ci_l:+.2f}, {ci_h:+.2f}]",
                        "Hodges_Lehmann": hl,
                        "Direccion": direction,
                        "p_valor": p_val,
                        "n_valid": len(y_clean),
                    })

            # Caso 2: Variable Politómica Cualitativa (>= 3 categorías discretas)
            elif (s_clean.dtype == object or str(s_clean.dtype) == "category") and n_unique >= 3:
                group_arrays = [y_clean[s_clean == uv].values for uv in unique_vals if len(y_clean[s_clean == uv]) >= 2]
                if len(group_arrays) >= 2:
                    kw = kruskal_wallis_test(group_arrays)
                    h_stat = kw["h_stat"]
                    p_val = kw["p_val"]
                    eps_sq = kw["epsilon_sq"]

                    records.append({
                        "Variable": col,
                        "Etiqueta": label,
                        "Grupo": category_group,
                        "Tipo": "Politómica",
                        "Prueba": "Kruskal-Wallis",
                        "Estadístico": h_stat,
                        "Estadístico_Str": f"H = {h_stat:.2f}",
                        "Efecto": eps_sq,
                        "Efecto_Nombre": "eps^2",
                        "Efecto_Str": f"{eps_sq:.2f}",
                        "IC95_Efecto": "---",
                        "Hodges_Lehmann": np.nan,
                        "Direccion": "Contrastante",
                        "p_valor": p_val,
                        "n_valid": len(y_clean),
                    })

            # Caso 3: Continua u Ordinal Numérica
            else:
                s_num = pd.to_numeric(s_clean, errors="coerce")
                valid_num = s_num.notna() & y_clean.notna()
                if valid_num.sum() >= 3:
                    sp = spearman_correlation(
                        s_num[valid_num], y_clean[valid_num], 
                        n_boot=self.n_boot, 
                        confidence=1.0 - self.alpha, 
                        random_state=self.random_state
                    )
                    kt = kendall_correlation(s_num[valid_num], y_clean[valid_num])

                    rho = sp["rho"]
                    p_val = sp["p_val"]
                    ci_l = sp["ci_low"]
                    ci_h = sp["ci_high"]

                    direction = (
                        "Aporte (↑)" if rho > 0.05 else ("Atenuación (↓)" if rho < -0.05 else "Neutro (≈)")
                    )

                    records.append({
                        "Variable": col,
                        "Etiqueta": label,
                        "Grupo": category_group,
                        "Tipo": "Ordinal / Continua",
                        "Prueba": "Spearman",
                        "Estadístico": rho,
                        "Estadístico_Str": f"ρ = {rho:+.3f}",
                        "Efecto": rho,
                        "Efecto_Nombre": "rho",
                        "Efecto_Str": f"{rho:+.2f}",
                        "IC95_Efecto": f"[{ci_l:+.2f}, {ci_h:+.2f}]",
                        "Hodges_Lehmann": np.nan,
                        "Direccion": direction,
                        "p_valor": p_val,
                        "n_valid": int(valid_num.sum()),
                    })

        res_df = pd.DataFrame(records)

        if res_df.empty:
            self._results_df = res_df
            return res_df

        # Ajuste de multiplicidad Benjamini-Hochberg (FDR)
        p_vals = res_df["p_valor"].values
        res_df["q_fdr"] = adjust_pvalues(p_vals, method="fdr_bh")

        # Clasificación de Prioridad Multicriterio
        priorities: List[str] = []
        for _, row in res_df.iterrows():
            p = row["p_valor"]
            q = row["q_fdr"]
            eff = abs(row["Efecto"]) if pd.notna(row["Efecto"]) else 0.0

            # Criterio Alta Prioridad:
            # Significancia estadística fuerte o tamaño de efecto sustancial
            if (p < self.alpha or q < self.fdr_threshold) and eff >= 0.40:
                priorities.append("Alta")
            elif p < 0.05:
                priorities.append("Alta")
            # Criterio Intermedia:
            # Marginalmente relevante (p < 0.20) o efecto moderado (eff >= 0.30)
            elif p < 0.20 or eff >= 0.30:
                priorities.append("Intermedia")
            else:
                priorities.append("Baja / Descarte")

        res_df["Prioridad"] = priorities

        # Ordenar por p-valor
        res_df = res_df.sort_values(by=["p_valor", "Prioridad"], ascending=[True, True]).reset_index(drop=True)
        self._results_df = res_df
        return res_df

    @property
    def results_df(self) -> pd.DataFrame:
        if self._results_df is None:
            return self.run()
        return self._results_df.copy()

    def get_prioritized_features(self, min_priority: str = "Intermedia") -> List[str]:
        """
        Retorna la lista de nombres de variables que cumplen con el nivel mínimo de prioridad.
        """
        df = self.results_df
        if min_priority == "Alta":
            return df[df["Prioridad"] == "Alta"]["Variable"].tolist()
        elif min_priority == "Intermedia":
            return df[df["Prioridad"].isin(["Alta", "Intermedia"])]["Variable"].tolist()
        else:
            return df["Variable"].tolist()
