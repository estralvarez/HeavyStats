"""
Modelos multivariantes regularizados y basados en variables latentes:
- LassoModeler: Regresión lineal con penalización L1 y selección parsimoniosa de variables (LassoCV / LassoLarsIC).
- PlsModeler: Regresión por Mínimos Cuadrados Parciales (PLS-R) con cálculo del vector de Importancia de Variable en la Proyección (VIP).
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LassoLarsIC
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from heavystats.html_utils import BaseReport, render_html_table


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


def prepare_pls_dataframe(
    data: Union[pd.DataFrame, str, Path, List[Any]],
    target_metal: str = "Mercurio_ug_L",
    id_col: Optional[str] = "Muestra_Codificada",
    include_domains_totals: bool = True,
    include_individual_features: bool = True,
    normalize_diet: bool = True,
    log_target: bool = False,
) -> pd.DataFrame:
    """
    Construye un nuevo DataFrame estructurado y 100% numérico para análisis PLS (Partial Least Squares)
    y modelado multivariante, transformando sistemáticamente las variables de la encuesta por dominio:

    1. Hábitos Dietarios (Alim_*): Frecuencias ordinales estandarizadas [0.0 - 1.0] (división entre 4.0 si normalize_diet=True).
    2. Hábitos y Salud en el Hogar (Salud_Fuma, Salud_Actividad, Salud_Bombillos, Salud_Techo, Salud_Joyeria, etc.): Codificación binaria (1.0 / 0.0).
    3. Movilidad Urbana (Salud_Transporte_*): Binarización (1.0 / 0.0).
    4. Fuentes Hídricas (Salud_Agua_*): Binarización (1.0 / 0.0).
    5. Exposición Antrópica (Exposicion_Talleres_*, Exposicion_Lugares_*, Exposicion_Industrias_*): Binarización (1.0 / 0.0).
    6. Indicadores de Riesgo (Exposicion_Cualquier_*, Salud_Cualquier_*): Binarización (1.0 / 0.0).
    7. Totales Consolidados por Dominio [0.0 - 1.0] (si include_domains_totals=True):
       - Tot_Dieta_Norm, Tot_Habitos_Norm, Tot_Movilidad_Norm, Tot_Agua_Norm,
         Tot_Exp_Antropica_Norm, Tot_Riesgo_Norm, Exposicion_Total_Norm.
    8. Sociodemográficas cuantitativas y factores generales (Edad, Sexo, Es_Expuesto, Score_Riesgo) si están presentes.
    9. Concentración del metal objetivo (target_metal) y columna identificadora (id_col).

    Parameters
    ----------
    data : Union[pd.DataFrame, str, Path, List[Any]]
        Dataset analítico original con cuestionario y concentraciones biológicas,
        ruta al archivo (.csv) o lista de rutas candidatas.
    target_metal : str, default="Mercurio_ug_L"
        Nombre de la variable biológica respuesta ('Mercurio_ug_L', 'Plomo_ug_dL', etc.).
    id_col : str, optional, default="Muestra_Codificada"
        Columna identificadora de la muestra/individuo a retener como referencia.
    include_domains_totals : bool, default=True
        Si True, calcula e incorpora los indicadores agregados y normalizados por dominio.
    include_individual_features : bool, default=True
        Si True, incluye las variables individuales transformadas de cada bloque.
    normalize_diet : bool, default=True
        Si True, divide las frecuencias dietarias (0 a 4) entre 4.0 para escala [0.0 - 1.0].
    log_target : bool, default=False
        Si True, transforma la variable respuesta con logaritmo natural: ln(target_metal).

    Returns
    -------
    pd.DataFrame
        Nuevo DataFrame listo para entrenar modelos PLS (PlsModeler), LASSO o MultivariatePipeline.
    """
    def _read_data_file(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p, sep=None, engine="python")
        except Exception:
            return pd.read_csv(p)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos especificado en: {p}")
        df = _read_data_file(p)
    elif isinstance(data, (list, tuple)):
        found_path = None
        for item in data:
            if isinstance(item, (str, Path)) and Path(item).exists():
                found_path = Path(item)
                break
        if found_path is not None:
            df = _read_data_file(found_path)
        else:
            raise TypeError(
                f"El argumento 'data' recibió una lista de {len(data)} elementos, pero ninguno es una ruta válida existente. "
                "Asegúrese de pasar el DataFrame 'df_hg' o una ruta válida."
            )
    else:
        raise TypeError(
            f"El argumento 'data' debe ser un pd.DataFrame o una ruta válida, recibido: {type(data).__name__}"
        )

    out_df = pd.DataFrame(index=df.index)

    # 1. Identificador si existe
    if id_col and id_col in df.columns:
        out_df[id_col] = df[id_col]

    # 2. Resolver metal objetivo
    metal_candidates = [target_metal]
    target_lower = str(target_metal).lower()
    if "mercurio" in target_lower or "hg" in target_lower:
        metal_candidates.extend(["Mercurio_ug_L", "Mercurio", "hg", "Mercurio_Sangre"])
    elif "plomo" in target_lower or "pb" in target_lower:
        metal_candidates.extend(["Plomo_ug_dL", "Plomo", "pb", "Plomo_Sangre"])
    elif "cadmio" in target_lower or "cd" in target_lower:
        metal_candidates.extend(["Cadmio_ug_L", "Cadmio", "cd", "Cadmio_Sangre"])

    resolved_metal = None
    for cand in metal_candidates:
        if cand in df.columns:
            resolved_metal = cand
            break

    # 3. Sociodemográficas y clínicas
    if "Edad" in df.columns:
        out_df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    if "Sexo" in df.columns:
        s_str = df["Sexo"].astype(str).str.lower()
        out_df["Sexo"] = (s_str.isin(["masculino", "m", "1", "hombre", "true"])).astype(float)
    if "Es_Expuesto" in df.columns:
        exp_str = df["Es_Expuesto"].astype(str).str.upper()
        out_df["Es_Expuesto"] = (exp_str.isin(["SI", "1", "TRUE", "S"])).astype(float)
    if "Score_Riesgo" in df.columns:
        out_df["Score_Riesgo"] = pd.to_numeric(df["Score_Riesgo"], errors="coerce")
    if "IMC" in df.columns:
        out_df["IMC"] = pd.to_numeric(df["IMC"], errors="coerce")

    # 4. Detección de bloques de encuesta
    cols_alim = [c for c in df.columns if c.startswith("Alim_")]
    cols_hab = [c for c in ["Salud_Fuma", "Salud_Actividad", "Salud_Bombillos", "Salud_Techo", "Salud_Joyeria"] if c in df.columns]
    cols_mov = [c for c in ["Salud_Transporte_caminar", "Salud_Transporte_publico", "Salud_Transporte_vehiculo"] if c in df.columns]
    cols_agua = [c for c in ["Salud_Agua_pozo_profundo", "Salud_Agua_filtrada", "Salud_Agua_mineral_embotellada"] if c in df.columns]
    cols_antropica = [c for c in [
        "Exposicion_Talleres_carpinteria", "Exposicion_Talleres_latoneria", "Exposicion_Talleres_mecanico",
        "Exposicion_Lugares_canales", "Exposicion_Lugares_estacion_gasolina", "Exposicion_Lugares_llenadora_gas_natural",
        "Exposicion_Lugares_rectificadora_motores", "Exposicion_Lugares_rios", "Exposicion_Industrias_fabrica_metales",
        "Exposicion_Industrias_fabrica_productos_quimicos"
    ] if c in df.columns]
    cols_riesgo = [c for c in [
        "Exposicion_Cualquier_Taller", "Exposicion_Cualquier_Industria",
        "Exposicion_Cualquier_Lugar_Riesgo", "Salud_Cualquier_Agua_Riesgo"
    ] if c in df.columns]

    # Bloque 1: Dieta
    diet_dict: Dict[str, pd.Series] = {}
    for c in cols_alim:
        c_num = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        c_norm = c_num / 4.0 if normalize_diet else c_num
        diet_dict[c] = c_norm
        if include_individual_features:
            out_df[c] = c_norm

    # Bloque 2: Hábitos
    hab_dict: Dict[str, pd.Series] = {}
    for c in cols_hab:
        s_val = (df[c].astype(str).str.upper().isin(["SI", "1", "TRUE"])).astype(float)
        hab_dict[c] = s_val
        if include_individual_features:
            out_df[c] = s_val

    # Bloque 3: Movilidad
    mov_dict: Dict[str, pd.Series] = {}
    for c in cols_mov:
        s_val = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        mov_dict[c] = s_val
        if include_individual_features:
            out_df[c] = s_val

    # Bloque 4: Fuentes de agua
    agua_dict: Dict[str, pd.Series] = {}
    for c in cols_agua:
        s_val = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        agua_dict[c] = s_val
        if include_individual_features:
            out_df[c] = s_val

    # Bloque 5: Exposición antrópica
    ant_dict: Dict[str, pd.Series] = {}
    for c in cols_antropica:
        s_val = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        ant_dict[c] = s_val
        if include_individual_features:
            out_df[c] = s_val

    # Bloque 6: Indicadores agregados de riesgo
    rie_dict: Dict[str, pd.Series] = {}
    for c in cols_riesgo:
        s_val = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        rie_dict[c] = s_val
        if include_individual_features:
            out_df[c] = s_val

    # Bloque 7: Indicadores consolidados por dominio [0.0 - 1.0]
    if include_domains_totals:
        if diet_dict:
            out_df["Tot_Dieta_Norm"] = pd.DataFrame(diet_dict).mean(axis=1)
        if hab_dict:
            out_df["Tot_Habitos_Norm"] = pd.DataFrame(hab_dict).mean(axis=1)
        if mov_dict:
            out_df["Tot_Movilidad_Norm"] = pd.DataFrame(mov_dict).mean(axis=1)
        if agua_dict:
            out_df["Tot_Agua_Norm"] = pd.DataFrame(agua_dict).mean(axis=1)
        if ant_dict:
            out_df["Tot_Exp_Antropica_Norm"] = pd.DataFrame(ant_dict).mean(axis=1)
        if rie_dict:
            out_df["Tot_Riesgo_Norm"] = pd.DataFrame(rie_dict).mean(axis=1)

        # Carga de exposición global
        all_features_dicts = {}
        all_features_dicts.update(diet_dict)
        all_features_dicts.update(hab_dict)
        all_features_dicts.update(mov_dict)
        all_features_dicts.update(agua_dict)
        all_features_dicts.update(ant_dict)
        all_features_dicts.update(rie_dict)
        if all_features_dicts:
            out_df["Exposicion_Total_Norm"] = pd.DataFrame(all_features_dicts).mean(axis=1)

    # Imputar medianas a cualquier NaN residual en columnas numéricas
    for col in out_df.columns:
        if col != id_col:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
            if out_df[col].isna().any():
                med_val = out_df[col].median()
                out_df[col] = out_df[col].fillna(0.0 if pd.isna(med_val) else med_val)

    # 8. Variable biológica respuesta
    if resolved_metal is not None:
        y_vals = pd.to_numeric(df[resolved_metal], errors="coerce")
        if log_target:
            out_df[resolved_metal] = np.log(y_vals)
        else:
            out_df[resolved_metal] = y_vals

    return out_df


class LassoModeler(BaseReport):
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

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna la tabla resumen de coeficientes LASSO en un DataFrame."""
        return self.summary_df

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Exporta la tabla resumen de LASSO a HTML de calidad editorial."""
        df = self.to_dataframe()
        html_code = render_html_table(
            df=df,
            title="Selección y Coeficientes Estandarizados (Regresión LASSO)",
            subtitle=f"Modelo L1 con penalización α = {self.alpha_:.4f} (R² = {self.r2_:.3f})",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code



class PlsModeler(BaseReport):
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
                "Influencia_PLS": "Crítica (VIP >= 1.0)" if is_critical else "Moderada/Baja (VIP < 1.0)",
                "Es_Critico": is_critical
            })

        df = pd.DataFrame(records)
        return df.sort_values(by="VIP", ascending=False).reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna la tabla resumen con los puntajes VIP y coeficientes PLS."""
        return self.summary_df

    def to_html(self, filepath: Optional[str] = None, full_page: bool = False) -> str:
        """Exporta la tabla de resultados PLS a formato HTML de calidad editorial."""
        df = self.to_dataframe()
        html_code = render_html_table(
            df=df,
            title="Importancia de Variables en la Proyección (PLS-R VIP)",
            subtitle=f"Modelo PLS-R con {self.n_components} componentes latentes (R² = {self.r2_:.3f})",
            full_page=full_page
        )
        if filepath:
            self._write_file(filepath, html_code)
        return html_code

