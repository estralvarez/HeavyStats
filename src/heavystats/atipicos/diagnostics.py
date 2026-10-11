"""
Módulo de diagnóstico quimiométrico y valores atípicos (outliers) de HeavyStats.
Implementa metodologías estadísticas formales para:
1. Detección univariante: Estadísticos robustos, vallas de Tukey (1.5 IQR y 3.0 IQR),
   Z-scores, Modified Z-score (MAD) y test de Grubbs exacto (t-Student a dos colas).
2. Diagnóstico multivariante en espacio latente (PLS-1 / PCA):
   Puntuaciones T, residuales Q (SPE), estadístico T² de Hotelling (distribución F de Snedecor),
   Apalancamiento (Leverage hii) y Distancia de Cook (Di).
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from heavystats.html_utils import BaseReport, render_html_table


class OutlierDiagnostics(BaseReport):
    """
    Motor analítico y quimiométrico para la detección de observaciones atípicas e influyentes.
    
    Permite evaluar tanto la consistencia analítica univariante de determinaciones instrumentales
    (p. ej. mercurio, plomo o cadmio por GFAAS) como la influencia multidimensional de los sujetos
    en el espacio latente de predictores de exposición ambiental y dietaria.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        id_col: Optional[str] = "Muestra_Codificada",
        n_components: int = 2,
        alpha: float = 0.05,
        who_threshold: Optional[float] = None,
        epa_threshold: Optional[float] = None,
        cdc_threshold: Optional[float] = None,
    ):
        """
        Inicializa el motor de diagnóstico de atípicos.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con las observaciones analíticas completas.
        target_col : str
            Columna de concentración del biomarcador biológico (variable respuesta).
        feature_cols : List[str], optional
            Columnas numéricas de exposición para el análisis multivariante. Si es None,
            se detectan automáticamente todas las numéricas excluyendo target_col e id_col.
        id_col : str, optional
            Columna identificadora de las muestras (por defecto 'Muestra_Codificada').
        n_components : int, default=2
            Número de componentes latentes para el modelo quimiométrico PLS-1.
        alpha : float, default=0.05
            Nivel de significancia para los umbrales críticos teóricos (F y t).
        who_threshold, epa_threshold, cdc_threshold : float, optional
            Umbrales toxicológicos de referencia internacional.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.id_col = id_col if id_col in df.columns else None
        self.n_components = int(n_components)
        self.alpha = float(alpha)

        # Configurar límites toxicológicos por defecto según el metal
        target_lower = target_col.lower()
        if who_threshold is not None:
            self.who_threshold = who_threshold
        elif "mercurio" in target_lower or "hg" in target_lower:
            self.who_threshold = 5.0
        elif "cadmio" in target_lower or "cd" in target_lower:
            self.who_threshold = 1.0
        else:
            self.who_threshold = 5.0

        if epa_threshold is not None:
            self.epa_threshold = epa_threshold
        elif "mercurio" in target_lower or "hg" in target_lower:
            self.epa_threshold = 5.8
        else:
            self.epa_threshold = None

        if cdc_threshold is not None:
            self.cdc_threshold = cdc_threshold
        elif "plomo" in target_lower or "pb" in target_lower:
            self.cdc_threshold = 3.5
        else:
            self.cdc_threshold = None

        # 1. Extracción y validación del vector univariante y
        self.y_raw = pd.to_numeric(self.df[target_col], errors="coerce")
        valid_mask = self.y_raw.notna()
        self.sub_df = self.df[valid_mask].copy()
        self.y = self.y_raw[valid_mask].values.astype(float)
        self.n_obs = len(self.y)

        if self.n_obs < 4:
            raise ValueError(f"Se requieren al menos 4 observaciones válidas para diagnóstico (obtenidas: {self.n_obs}).")

        # IDs
        if self.id_col:
            self.ids = self.sub_df[self.id_col].astype(str).tolist()
        else:
            self.ids = [f"M-{i+1:02d}" for i in range(self.n_obs)]

        # 2. Resolución de predictores multivariantes X
        if feature_cols:
            self.feature_cols = [c for c in feature_cols if c in self.sub_df.columns]
        else:
            ignore = {self.target_col}
            if self.id_col:
                ignore.add(self.id_col)
            num_cols = self.sub_df.select_dtypes(include=[np.number]).columns
            self.feature_cols = [c for c in num_cols if c not in ignore]

        # Ejecución del diagnóstico
        self._compute_univariate()
        if len(self.feature_cols) >= 2:
            self._compute_multivariate()
        else:
            self._multivariate_computed = False

    def _compute_univariate(self) -> None:
        """Cálculo de métricas univariantes, vallas de Tukey y test de Grubbs."""
        self.mean = float(np.mean(self.y))
        self.std = float(np.std(self.y, ddof=1))
        self.cv_percent = float((self.std / self.mean) * 100.0) if self.mean != 0 else 0.0
        self.median = float(np.median(self.y))
        self.q25, self.q75 = np.percentile(self.y, [25, 75])
        self.iqr = float(self.q75 - self.q25)
        self.min_val = float(np.min(self.y))
        self.max_val = float(np.max(self.y))

        # Intervalo de confianza 95% para la media
        se_mean = self.std / np.sqrt(self.n_obs)
        t_ci = stats.t.ppf(1.0 - self.alpha / 2.0, df=self.n_obs - 1)
        self.ci_lower = self.mean - t_ci * se_mean
        self.ci_upper = self.mean + t_ci * se_mean

        # Vallas de Tukey (1.5 IQR y 3.0 IQR)
        self.tukey_lower = float(self.q25 - 1.5 * self.iqr)
        self.tukey_upper = float(self.q75 + 1.5 * self.iqr)
        self.tukey_lower_extreme = float(self.q25 - 3.0 * self.iqr)
        self.tukey_upper_extreme = float(self.q75 + 3.0 * self.iqr)

        # Test de Grubbs (detección formal de atípico extremo singular)
        self.g_max = float((self.max_val - self.mean) / self.std) if self.std > 0 else 0.0
        self.g_min = float((self.mean - self.min_val) / self.std) if self.std > 0 else 0.0
        t_crit = stats.t.ppf(1.0 - self.alpha / (2.0 * self.n_obs), df=self.n_obs - 2)
        self.g_crit = float(((self.n_obs - 1) / np.sqrt(self.n_obs)) * np.sqrt(t_crit**2 / (self.n_obs - 2 + t_crit**2)))

        # Z-Scores y Modified Z-Scores
        self.z_scores = (self.y - self.mean) / self.std if self.std > 0 else np.zeros_like(self.y)
        med_abs_dev = float(np.median(np.abs(self.y - self.median)))
        if med_abs_dev > 0:
            self.modified_z_scores = 0.6745 * (self.y - self.median) / med_abs_dev
        else:
            self.modified_z_scores = np.zeros_like(self.y)

        # Conclusiones toxicológicas
        self.n_excede_who = int(np.sum(self.y > self.who_threshold)) if self.who_threshold else 0
        self.n_excede_epa = int(np.sum(self.y > self.epa_threshold)) if self.epa_threshold else 0
        self.n_excede_cdc = int(np.sum(self.y > self.cdc_threshold)) if self.cdc_threshold else 0
        self.has_univariate_outliers = (self.g_max >= self.g_crit) or (self.max_val > self.tukey_upper) or (self.min_val < self.tukey_lower)

    def _compute_multivariate(self) -> None:
        """Cálculo quimiométrico multivariante en espacio latente PLS-1."""
        X_df = self.sub_df[self.feature_cols].copy()
        # Imputar medianas si quedase algún dato faltante residual
        X_df = X_df.apply(lambda col: pd.to_numeric(col, errors="coerce").fillna(col.median()))
        X_mat = X_df.values.astype(float)

        n, p = X_mat.shape
        k_comp = min(self.n_components, p, n - 1)
        if k_comp < 1:
            k_comp = 1
        self.k_comp = k_comp

        # Auto-escalado analítico (media 0, varianza 1)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_mat)

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(self.y[:, None]).flatten()

        # Ajuste de regresión PLS-1
        self.pls_model = PLSRegression(n_components=k_comp, scale=False)
        self.pls_model.fit(X_scaled, y_scaled)

        T = self.pls_model.x_scores_  # (n x k)
        P = self.pls_model.x_loadings_ # (p x k)
        W = self.pls_model.x_weights_  # (p x k)
        self.scores_T = T
        self.loadings_P = P
        self.weights_W = W

        # 1. Hotelling T² y su límite F de Snedecor al (1 - alpha)
        var_t = np.var(T, axis=0, ddof=1)
        var_t[var_t == 0] = 1.0
        self.T2_Hotelling = np.sum((T**2) / var_t, axis=1)

        f_crit = stats.f.ppf(1.0 - self.alpha, dfn=k_comp, dfd=n - k_comp)
        self.T2_limit = float((k_comp * (n - 1) / (n - k_comp)) * f_crit)

        # 2. Residuales Q (SPE: Squared Prediction Error)
        X_recon = T @ P.T
        E = X_scaled - X_recon
        self.Q_residuals = np.sum(E**2, axis=1)

        # Límite Q al percentil 95 o aproximación Jackson-Mudholkar
        theta1 = float(np.trace(E.T @ E) / (n - 1)) if n > 1 else 1.0
        self.Q_limit = float(np.percentile(self.Q_residuals, 95)) if len(self.Q_residuals) > 0 else 1.0

        # 3. Apalancamiento (Leverage hii en espacio T)
        # H = T (T' T)^(-1) T'
        inv_TT = np.linalg.pinv(T.T @ T)
        self.leverage = np.sum((T @ inv_TT) * T, axis=1)
        self.leverage_limit = float(2.0 * k_comp / n)

        # 4. Residuales del modelo y Distancia de Cook Di
        y_pred_scaled = self.pls_model.predict(X_scaled).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled[:, None]).flatten()
        self.y_pred = y_pred
        residuals = self.y - y_pred
        self.residuals = residuals

        # Error estándar residual
        df_res = max(1, n - k_comp - 1)
        se_res = np.sqrt(np.sum(residuals**2) / df_res) if df_res > 0 else 1.0

        # Residuales estudentizados
        lev_safe = np.clip(self.leverage, 0.0, 0.999)
        self.studentized_residuals = residuals / (se_res * np.sqrt(1.0 - lev_safe))

        # Distancia de Cook
        self.cooks_D = (self.studentized_residuals**2 / k_comp) * (lev_safe / (1.0 - lev_safe))
        self.cooks_threshold = float(4.0 / n)

        self._multivariate_computed = True

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Construye una tabla consolidada con los parámetros de diagnóstico de cada muestra.
        """
        records = []
        for i in range(self.n_obs):
            sid = self.ids[i]
            y_val = float(self.y[i])
            z_val = float(self.z_scores[i])
            mod_z = float(self.modified_z_scores[i])

            is_tukey = (y_val > self.tukey_upper) or (y_val < self.tukey_lower)
            is_tukey_ext = (y_val > self.tukey_upper_extreme) or (y_val < self.tukey_lower_extreme)

            rec = {
                "Muestra": sid,
                "Concentracion": y_val,
                "Z_Score": z_val,
                "Modified_Z": mod_z,
                "Tukey_Outlier": "Atípico (1.5 IQR)" if is_tukey else "Normal",
            }

            if self._multivariate_computed:
                t2_val = float(self.T2_Hotelling[i])
                q_val = float(self.Q_residuals[i])
                lev_val = float(self.leverage[i])
                ck_val = float(self.cooks_D[i])
                r_stud = float(self.studentized_residuals[i])

                is_t2_out = t2_val > self.T2_limit
                is_high_lev = lev_val > self.leverage_limit
                is_cook_alert = ck_val > self.cooks_threshold

                # Clasificación analítica integral
                if is_t2_out and is_high_lev:
                    status = "Punto Palanca Extremo"
                elif is_high_lev:
                    status = "Alto Apalancamiento"
                elif is_t2_out:
                    status = "Atípico Multivariante T²"
                elif is_cook_alert:
                    status = "Observación Influyente"
                else:
                    status = "Homogéneo"

                rec.update({
                    "T1_Score": float(self.scores_T[i, 0]),
                    "T2_Score": float(self.scores_T[i, 1]) if self.k_comp > 1 else 0.0,
                    "Hotelling_T2": t2_val,
                    "Q_Residual": q_val,
                    "Leverage": lev_val,
                    "Cooks_D": ck_val,
                    "Residual_Estudentizado": r_stud,
                    "Diagnostico": status,
                })
            records.append(rec)

        return pd.DataFrame(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Alias para get_summary_dataframe()."""
        return self.get_summary_dataframe()

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Exporta la tabla completa de diagnóstico quimiométrico a HTML."""
        df = self.to_dataframe()
        html_code = render_html_table(
            df=df,
            title=f"Diagnóstico de Consistencia y Valores Atípicos: {self.target_col}",
            subtitle=f"Evaluación univariante y multivariante (N = {self.n_obs})",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

    def get_kpis(self) -> Dict[str, Any]:

        """Retorna los indicadores analíticos clave de la cohorte."""
        kpis = {
            "n_obs": self.n_obs,
            "mean": self.mean,
            "std": self.std,
            "cv_percent": self.cv_percent,
            "median": self.median,
            "iqr": self.iqr,
            "min": self.min_val,
            "max": self.max_val,
            "ci_95": (self.ci_lower, self.ci_upper),
            "tukey_fences": (self.tukey_lower, self.tukey_upper),
            "grubbs": {
                "g_max": self.g_max,
                "g_min": self.g_min,
                "g_crit": self.g_crit,
                "reject_h0": self.g_max >= self.g_crit
            },
            "toxicology": {
                "who_threshold": self.who_threshold,
                "excede_who": self.n_excede_who,
                "epa_threshold": self.epa_threshold,
                "excede_epa": self.n_excede_epa,
                "cdc_threshold": self.cdc_threshold,
                "excede_cdc": self.n_excede_cdc,
            }
        }
        if self._multivariate_computed:
            kpis["multivariate"] = {
                "k_components": self.k_comp,
                "T2_limit": self.T2_limit,
                "leverage_limit": self.leverage_limit,
                "cooks_threshold": self.cooks_threshold,
                "n_t2_outliers": int(np.sum(self.T2_Hotelling > self.T2_limit)),
                "n_high_leverage": int(np.sum(self.leverage > self.leverage_limit)),
                "n_cooks_alert": int(np.sum(self.cooks_D > self.cooks_threshold)),
            }
        return kpis
