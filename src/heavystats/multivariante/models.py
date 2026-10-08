"""
Modelos multivariantes regularizados y basados en variables latentes:
- LassoModeler: Regresión lineal con penalización L1 y selección parsimoniosa de variables (LassoCV / LassoLarsIC).
- PlsModeler: Regresión por Mínimos Cuadrados Parciales (PLS-R) con cálculo del vector de Importancia de Variable en la Proyección (VIP).
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LassoLarsIC
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def prepare_features_matrix(
    data: pd.DataFrame, 
    features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transforma un subconjunto de columnas en una matriz numérica procesada,
    manejando variables binarias (SI/NO, 1/0) y variables categóricas nominales
    mediante codificación dummy (one-hot) si es necesario.
    """
    X_df = pd.DataFrame(index=data.index)
    col_mapping: List[str] = []

    for col in features:
        if col not in data.columns:
            continue
        s = data[col]
        # Verificar si es binaria booleana o de texto SI/NO
        str_vals = s.dropna().astype(str).str.upper()
        if set(str_vals.unique()).issubset({"NO", "SI", "0", "1", "FALSE", "TRUE"}):
            X_df[col] = str_vals.isin({"SI", "1", "TRUE"}).astype(float)
            col_mapping.append(col)
        elif s.dtype == object or str(s.dtype) == "category":
            # Dummy encoding para politómicas nominales
            dummies = pd.get_dummies(s, prefix=col, drop_first=True, dtype=float)
            for c in dummies.columns:
                X_df[c] = dummies[c]
                col_mapping.append(c)
        else:
            X_df[col] = pd.to_numeric(s, errors="coerce").astype(float)
            col_mapping.append(col)

    # Imputar medianas si quedase algún valor faltante residual
    X_df = X_df.apply(lambda col: col.fillna(col.median()))
    return X_df, col_mapping


class LassoModeler:
    """
    Ajustador de Regresión LASSO (L1 penalizado).
    
    Aplica una penalización L1 que contrae los coeficientes de variables redundantes
    hacia exactamente cero, reteniendo únicamente el subconjunto parsimonioso de
    determinantes estadísticamente influyentes.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        alphas: Optional[np.ndarray] = None,
        criterion: str = "cv",  # 'cv', 'aic', 'bic'
        random_state: int = 42,
    ):
        self.cv_folds = int(cv_folds)
        self.alphas = alphas if alphas is not None else np.logspace(-4, 0.5, 200)
        self.criterion = criterion.lower().strip()
        self.random_state = int(random_state)

        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_: List[str] = []
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        self.alpha_: float = 0.0
        self.r2_: float = 0.0
        self.mse_: float = 0.0

    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "LassoModeler":
        """
        Ajusta el modelo LASSO sobre predictores estandarizados.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values.astype(float)
        else:
            X_mat = np.asarray(X, dtype=float)
            self.feature_names_ = list(feature_names) if feature_names else [f"X{i}" for i in range(X_mat.shape[1])]

        y_vec = np.asarray(y, dtype=float)
        n_samples = len(y_vec)

        # Estandarización de predictores
        X_scaled = self.scaler.fit_transform(X_mat)

        if self.criterion in ("aic", "bic"):
            # Optimización rápida basada en criterios de información
            ic_model = LassoLarsIC(criterion=self.criterion, random_state=self.random_state)
            ic_model.fit(X_scaled, y_vec)
            self.model = ic_model
            self.alpha_ = float(ic_model.alpha_)
            self.coef_ = ic_model.coef_.copy()
            self.intercept_ = float(ic_model.intercept_)
        else:
            # Validación cruzada (KFold o Leave-One-Out según n)
            k = max(2, min(self.cv_folds, n_samples))
            cv = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
            cv_model = LassoCV(
                alphas=self.alphas,
                cv=cv,
                random_state=self.random_state,
                max_iter=10000
            )
            cv_model.fit(X_scaled, y_vec)
            self.model = cv_model
            self.alpha_ = float(cv_model.alpha_)
            self.coef_ = cv_model.coef_.copy()
            self.intercept_ = float(cv_model.intercept_)

        y_pred = self.predict(X_mat)
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        ss_res = np.sum((y_vec - y_pred) ** 2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        self.mse_ = float(np.mean((y_vec - y_pred) ** 2))

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_mat = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_mat)
        return X_scaled @ self.coef_ + self.intercept_

    @property
    def summary_df(self) -> pd.DataFrame:
        """
        Retorna una tabla resumen con los coeficientes estandarizados beta y clasificación del efecto.
        """
        records = []
        for name, coef in zip(self.feature_names_, self.coef_):
            c_val = float(coef)
            if abs(c_val) < 1e-5:
                eff = "Descartado (β = 0)"
            elif c_val > 0:
                eff = "Aporte (↑)"
            else:
                eff = "Atenuación (↓)"

            records.append({
                "Variable": name,
                "Beta_Estandarizado": c_val,
                "Efecto_Lasso": eff,
                "Retenido": abs(c_val) >= 1e-5
            })

        df = pd.DataFrame(records)
        return df.sort_values(by="Beta_Estandarizado", ascending=False).reset_index(drop=True)


class PlsModeler:
    """
    Ajustador de Regresión por Mínimos Cuadrados Parciales (PLS-R).
    
    Proyecta tanto la matriz de covariables estandarizadas X como la respuesta y
    hacia un espacio de variables latentes ortogonales que maximizan la covarianza explicada.
    Calcula de forma exacta el estadístico VIP (Variable Importance in Projection).
    """

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
    ):
        self.n_components = int(n_components)
        self.scale = bool(scale)
        self.scaler = StandardScaler()

        self.model: Optional[PLSRegression] = None
        self.feature_names_: List[str] = []
        self.vip_: np.ndarray = np.array([])
        self.coef_: np.ndarray = np.array([])
        self.r2_: float = 0.0
        self.explained_variance_ratio_x_: np.ndarray = np.array([])

    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "PlsModeler":
        """
        Ajusta la regresión PLS y calcula el vector VIP.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values.astype(float)
        else:
            X_mat = np.asarray(X, dtype=float)
            self.feature_names_ = list(feature_names) if feature_names else [f"X{i}" for i in range(X_mat.shape[1])]

        y_vec = np.asarray(y, dtype=float)
        if y_vec.ndim == 1:
            y_2d = y_vec[:, None]
        else:
            y_2d = y_vec

        n_samples, p_features = X_mat.shape
        n_comp = min(self.n_components, p_features, n_samples - 1)
        if n_comp < 1:
            n_comp = 1
        self.n_components = n_comp

        if self.scale:
            X_scaled = self.scaler.fit_transform(X_mat)
        else:
            X_scaled = X_mat

        self.model = PLSRegression(n_components=n_comp, scale=True)
        self.model.fit(X_scaled, y_2d)

        # Coeficientes de regresión estandarizados
        self.coef_ = self.model.coef_.flatten().copy()

        # Cálculo riguroso del vector VIP (Variable Importance in Projection)
        # T: x_scores (n x A), W: x_weights (p x A), Q: y_loadings (q x A)
        T = self.model.x_scores_
        W = self.model.x_weights_
        Q = self.model.y_loadings_
        A = n_comp

        # Suma de cuadrados explicada por componente: SS_a = (q_a^2) * (t_a^T t_a)
        # Para respuesta 1D: Q[0, a]
        q_vec = Q[0, :] if Q.ndim > 1 else Q
        ss_a = np.sum(T**2, axis=0) * (q_vec**2)
        total_ss = np.sum(ss_a)

        if total_ss > 0:
            # Normalizar pesos W por componente
            w_norm = np.linalg.norm(W, axis=0)
            w_norm[w_norm == 0] = 1.0
            w_normalized = W / w_norm

            # VIP_j = sqrt( p * sum_a [ SS_a * (w_{aj} / ||w_a||)^2 ] / total_ss )
            weighted_w = np.sum(ss_a * (w_normalized**2), axis=1)
            self.vip_ = np.sqrt(p_features * weighted_w / total_ss)
        else:
            self.vip_ = np.ones(p_features, dtype=float)

        # Métrica R2
        y_pred = self.model.predict(X_scaled).flatten()
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        ss_res = np.sum((y_vec - y_pred) ** 2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_mat = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_mat) if self.scale else X_mat
        return self.model.predict(X_scaled).flatten()

    @property
    def summary_df(self) -> pd.DataFrame:
        """
        Retorna la tabla resumen con VIP, coeficientes PLS y clasificación de relevancia.
        """
        records = []
        for name, vip_val, coef in zip(self.feature_names_, self.vip_, self.coef_):
            is_critical = float(vip_val) >= 1.0
            records.append({
                "Variable": name,
                "VIP": float(vip_val),
                "Coef_PLS": float(coef),
                "Influencia_PLS": "Crítica (VIP ≥ 1.0)" if is_critical else "Moderada/Baja (VIP < 1.0)",
                "Es_Critico": is_critical
            })

        df = pd.DataFrame(records)
        return df.sort_values(by="VIP", ascending=False).reset_index(drop=True)
