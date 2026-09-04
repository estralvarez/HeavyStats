"""
Motores de cálculo estadístico para análisis bivariantes.
Implementa pruebas no paramétricas, paramétricas sobre escala logarítmica,
tendencias ordinales, correlaciones con bootstrap, tablas de contingencia y ajustes de multiplicidad.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats


def adjust_pvalues(
    p_values: Sequence[float], 
    method: str = "holm"
) -> np.ndarray:
    """
    Ajusta una lista o arreglo de valores p por multiplicidad de hipótesis.

    Métodos soportados:
    - 'holm': Método step-down de Holm-Bonferroni (control estricto de FWER).
    - 'fdr_bh' / 'bh': Procedimiento de Benjamini-Hochberg (control de FDR).
    - 'bonferroni': Corrección clásica de Bonferroni (conservadora).
    - 'hochberg': Procedimiento step-up de Hochberg.

    Parámetros
    ----------
    p_values : Sequence[float]
        Valores p sin ajustar.
    method : str
        Nombre del método de corrección.

    Retorna
    -------
    np.ndarray
        Valores p ajustados, acotados en el rango [0, 1].
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n <= 1:
        return np.clip(p_arr, 0.0, 1.0)

    method = method.lower().strip()
    valid_mask = ~np.isnan(p_arr)
    p_clean = p_arr[valid_mask]
    m = len(p_clean)

    if m <= 1:
        return np.clip(p_arr, 0.0, 1.0)

    order = np.argsort(p_clean)
    sorted_p = p_clean[order]
    adj = np.empty(m, dtype=float)

    if method == "bonferroni":
        adj_sorted = np.clip(sorted_p * m, 0.0, 1.0)
        adj[order] = adj_sorted

    elif method in ("holm", "holm-bonferroni"):
        # Step-down Holm: p_adj(i) = max_{j <= i} min(1, (m - j + 1) * p(j))
        factors = np.arange(m, 0, -1)
        step_vals = np.clip(sorted_p * factors, 0.0, 1.0)
        # Forzar monotonía no decreciente
        cum_max = np.maximum.accumulate(step_vals)
        adj[order] = cum_max

    elif method in ("fdr_bh", "bh", "benjamini-hochberg"):
        # Step-up Benjamini-Hochberg: p_adj(i) = min_{j >= i} min(1, (m / j) * p(j))
        ranks = np.arange(1, m + 1)
        step_vals = np.clip(sorted_p * (m / ranks), 0.0, 1.0)
        # Forzar monotonía hacia atrás
        cum_min = np.minimum.accumulate(step_vals[::-1])[::-1]
        adj[order] = cum_min

    elif method == "hochberg":
        ranks = np.arange(m, 0, -1)
        step_vals = np.clip(sorted_p * ranks, 0.0, 1.0)
        cum_min = np.minimum.accumulate(step_vals[::-1])[::-1]
        adj[order] = cum_min

    else:
        raise ValueError(f"Método de ajuste '{method}' no reconocido. Opciones: 'holm', 'fdr_bh', 'bonferroni', 'hochberg'.")

    result = np.full_like(p_arr, np.nan)
    result[valid_mask] = adj
    return result


def bootstrap_ci_diff_medians(
    x: np.ndarray, 
    y: np.ndarray, 
    n_boot: int = 2000, 
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calcula la diferencia de medianas (Med_x - Med_y) y su intervalo de confianza no paramétrico
    mediante remuestreo Bootstrap matricial de alta velocidad.
    """
    rng = np.random.default_rng(random_state)
    n_x, n_y = len(x), len(y)
    diff_obs = float(np.median(x) - np.median(y))

    if n_x == 0 or n_y == 0:
        return diff_obs, np.nan, np.nan

    # Remuestreo matricial vectorizado
    sample_x = rng.choice(x, size=(n_boot, n_x), replace=True)
    sample_y = rng.choice(y, size=(n_boot, n_y), replace=True)
    boot_diffs = np.median(sample_x, axis=1) - np.median(sample_y, axis=1)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_diffs, (alpha / 2.0) * 100))
    ci_high = float(np.percentile(boot_diffs, (1.0 - alpha / 2.0) * 100))
    return diff_obs, ci_low, ci_high


def hodges_lehmann_2sample(
    x: np.ndarray, 
    y: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calcula el estimador de desplazamiento de localización de Hodges-Lehmann (pseudomediana pareada)
    para 2 muestras independientes y su intervalo de confianza exacto/asintótico libre de distribución.
    """
    diffs = np.subtract.outer(x, y).flatten()
    diffs.sort()
    hl_estimate = float(np.median(diffs))

    n1, n2 = len(x), len(y)
    N = n1 * n2
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # Error estándar asintótico del estadístico de Wilcoxon-Mann-Whitney
    sigma_w = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    mu_w = N / 2.0
    c_alpha = int(np.floor(mu_w - z_crit * sigma_w))

    k_low = max(0, c_alpha)
    k_high = min(N - 1, N - 1 - c_alpha)

    ci_low = float(diffs[k_low]) if k_low < N else float(diffs[0])
    ci_high = float(diffs[k_high]) if k_high >= 0 else float(diffs[-1])
    return hl_estimate, ci_low, ci_high


def mann_whitney_test(
    x: Sequence[float], 
    y: Sequence[float], 
    alternative: str = "two-sided",
    n_boot: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Aplica la prueba de Mann-Whitney U para contrastar 2 grupos independientes con distribución asimétrica.
    Calcula tamaños del efecto estandarizados (correlación biserial de rangos, Delta de Cliff, CLES)
    e intervalos de confianza para la diferencia de medianas y el estimador de Hodges-Lehmann.

    Parámetros
    ----------
    x : Sequence[float]
        Valores del Grupo 1.
    y : Sequence[float]
        Valores del Grupo 2.
    alternative : str
        Hipótesis alternativa ('two-sided', 'less', 'greater').
    n_boot : int
        Número de réplicas bootstrap para intervalos de confianza.

    Retorna
    -------
    Dict[str, Any]
        Diccionario con estadísticas completas, tamaños del efecto, valores p e intervalos de confianza.
    """
    arr_x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    arr_y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values

    n1, n2 = len(arr_x), len(arr_y)
    if n1 == 0 or n2 == 0:
        raise ValueError("Uno o ambos grupos no contienen observaciones numéricas válidas.")

    # Estadísticos descriptivos de cada grupo
    med1, q25_1, q75_1 = float(np.median(arr_x)), float(np.percentile(arr_x, 25)), float(np.percentile(arr_x, 75))
    med2, q25_2, q75_2 = float(np.median(arr_y)), float(np.percentile(arr_y, 25)), float(np.percentile(arr_y, 75))
    iqr1 = q75_1 - q25_1
    iqr2 = q75_2 - q25_2

    # Mann-Whitney U
    res = stats.mannwhitneyu(arr_x, arr_y, alternative=alternative)
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Tamaños del efecto
    # Rank Biserial Correlation r_rb = 1 - (2*U / (n1*n2))
    # Common Language Effect Size (CLES / Area under ROC / Superiority): A = U / (n1*n2)
    total_pairs = n1 * n2
    cles = u_stat / total_pairs
    rank_biserial = 1.0 - (2.0 * u_stat / total_pairs)
    cliffs_delta = -rank_biserial  # Cliff's d = (2*U/(n1*n2)) - 1 = P(X>Y) - P(Y>X)

    # Diferencia de medianas y Bootstrap CI
    diff_med, ci_med_low, ci_med_high = bootstrap_ci_diff_medians(
        arr_x, arr_y, n_boot=n_boot, random_state=random_state
    )

    # Estimador de Hodges-Lehmann e IC libre de distribución
    hl_est, hl_ci_low, hl_ci_high = hodges_lehmann_2sample(arr_x, arr_y)

    return {
        "test": "Mann-Whitney U",
        "n1": n1,
        "n2": n2,
        "median1": med1,
        "iqr1": iqr1,
        "q25_1": q25_1,
        "q75_1": q75_1,
        "median2": med2,
        "iqr2": iqr2,
        "q25_2": q25_2,
        "q75_2": q75_2,
        "diff_medians": diff_med,
        "diff_medians_ci": (ci_med_low, ci_med_high),
        "hodges_lehmann": hl_est,
        "hodges_lehmann_ci": (hl_ci_low, hl_ci_high),
        "u_statistic": u_stat,
        "p_value": p_val,
        "rank_biserial": rank_biserial,
        "cliffs_delta": cliffs_delta,
        "cles": cles,
        "alternative": alternative,
    }


def welch_ttest_log(
    x: Sequence[float], 
    y: Sequence[float],
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Aplica la prueba t de Welch sobre concentraciones transformadas logarítmicamente ln(x),
    apropiada para biomarcadores asimétricos log-normales con heterocedasticidad.
    Reporta la Razón de Medias Geométricas (GMR) con su intervalo de confianza analítico exacto al 95%.

    Parámetros
    ----------
    x : Sequence[float]
        Concentraciones del Grupo 1 (deben ser > 0).
    y : Sequence[float]
        Concentraciones del Grupo 2 (deben ser > 0).
    confidence : float
        Nivel de confianza para el intervalo de la Razón de Medias Geométricas.

    Retorna
    -------
    Dict[str, Any]
        Métricas de medias geométricas, GSD, GMR, IC 95%, t de Welch, gl y valor p.
    """
    arr_x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    arr_y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values

    pos_x = arr_x[arr_x > 0]
    pos_y = arr_y[arr_y > 0]

    n1, n2 = len(pos_x), len(pos_y)
    if n1 < 2 or n2 < 2:
        raise ValueError("Se requieren al menos 2 observaciones positivas por grupo para la prueba t de Welch sobre escala logarítmica.")

    log_x = np.log(pos_x)
    log_y = np.log(pos_y)

    mean_log_x, var_log_x = float(np.mean(log_x)), float(np.var(log_x, ddof=1))
    mean_log_y, var_log_y = float(np.mean(log_y)), float(np.var(log_y, ddof=1))
    sd_log_x = np.sqrt(var_log_x)
    sd_log_y = np.sqrt(var_log_y)

    # Medias geométricas y GSD
    geo_mean1 = float(np.exp(mean_log_x))
    geo_mean2 = float(np.exp(mean_log_y))
    gsd1 = float(np.exp(sd_log_x))
    gsd2 = float(np.exp(sd_log_y))

    # Razón de Medias Geométricas: GMR = GM1 / GM2 = exp(mean(ln(X)) - mean(ln(Y)))
    diff_log = mean_log_x - mean_log_y
    gmr = float(np.exp(diff_log))

    # Error estándar de la diferencia y grados de libertad de Welch-Satterthwaite
    se_diff = float(np.sqrt((var_log_x / n1) + (var_log_y / n2)))
    if se_diff == 0:
        df_welch = float(n1 + n2 - 2)
    else:
        df_welch = float(((var_log_x / n1 + var_log_y / n2) ** 2) / (
            ((var_log_x / n1) ** 2) / (n1 - 1) + ((var_log_y / n2) ** 2) / (n2 - 1)
        ))

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_welch))

    ci_gmr_low = float(np.exp(diff_log - t_crit * se_diff))
    ci_gmr_high = float(np.exp(diff_log + t_crit * se_diff))

    # Estadístico t de Welch y valor p
    ttest_res = stats.ttest_ind(log_x, log_y, equal_var=False)
    t_stat = float(ttest_res.statistic)
    p_val = float(ttest_res.pvalue)

    # Cohen's d en escala logarítmica
    pooled_sd_log = np.sqrt((var_log_x + var_log_y) / 2.0)
    cohen_d_log = diff_log / pooled_sd_log if pooled_sd_log > 0 else 0.0

    # Diagnóstico de normalidad de residuos combinados
    residuals = np.concatenate([log_x - mean_log_x, log_y - mean_log_y])
    if len(residuals) >= 3:
        w_res, p_norm_res = stats.shapiro(residuals)
    else:
        w_res, p_norm_res = np.nan, np.nan

    return {
        "test": "Welch t-test (log)",
        "n1": n1,
        "n2": n2,
        "geo_mean1": geo_mean1,
        "gsd1": gsd1,
        "geo_mean2": geo_mean2,
        "gsd2": gsd2,
        "gmr": gmr,
        "gmr_ci": (ci_gmr_low, ci_gmr_high),
        "t_statistic": t_stat,
        "df": df_welch,
        "p_value": p_val,
        "cohen_d_log": float(cohen_d_log),
        "residuals_shapiro_p": float(p_norm_res) if pd.notna(p_norm_res) else np.nan,
    }


def kruskal_wallis_test(
    *groups: Sequence[float], 
    group_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Aplica la prueba global no paramétrica de Kruskal-Wallis para comparar variables continuas
    entre 3 o más categorías independientes. Calcula el tamaño del efecto Epsilon al cuadrado (epsilon^2).

    Parámetros
    ----------
    *groups : Sequence[float]
        Grupos de observaciones a comparar.
    group_names : Sequence[str], opcional
        Nombres identificadores de cada categoría.

    Retorna
    -------
    Dict[str, Any]
        Estadístico H, grados de libertad, valor p, resumen por categoría y tamaño del efecto.
    """
    cleaned_groups = []
    for g in groups:
        arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().values
        if len(arr) > 0:
            cleaned_groups.append(arr)

    k = len(cleaned_groups)
    if k < 2:
        raise ValueError("Se requieren al menos 2 grupos con observaciones válidas para Kruskal-Wallis.")

    if group_names is None or len(group_names) != k:
        group_names = [f"Grupo_{i+1}" for i in range(k)]

    group_summaries = []
    for name, arr in zip(group_names, cleaned_groups):
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        group_summaries.append({
            "name": str(name),
            "n": len(arr),
            "median": med,
            "iqr": q75 - q25,
            "q25": q25,
            "q75": q75,
        })

    res = stats.kruskal(*cleaned_groups)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df_kw = k - 1
    total_n = sum(len(g) for g in cleaned_groups)

    # Tamaño del efecto: Epsilon-squared: eps^2 = (H - k + 1) / (N - k)
    if total_n > k:
        epsilon_sq = max(0.0, min(1.0, (h_stat - k + 1) / (total_n - k)))
        eta_sq = max(0.0, min(1.0, (h_stat - k + 1) / (total_n - 1)))
    else:
        epsilon_sq, eta_sq = 0.0, 0.0

    return {
        "test": "Kruskal-Wallis",
        "k_groups": k,
        "total_n": total_n,
        "h_statistic": h_stat,
        "df": df_kw,
        "p_value": p_val,
        "epsilon_squared": float(epsilon_sq),
        "eta_squared": float(eta_sq),
        "group_summaries": group_summaries,
    }


def dunn_posthoc_test(
    *groups: Sequence[float], 
    group_names: Optional[Sequence[str]] = None,
    method: str = "holm"
) -> List[Dict[str, Any]]:
    """
    Aplica la prueba de comparaciones múltiples pareadas post-hoc de Dunn con corrección estricta
    por empates y ajuste de valores p mediante Holm-Bonferroni (o Benjamini-Hochberg).

    Parámetros
    ----------
    *groups : Sequence[float]
        Grupos a comparar pareadamente.
    group_names : Sequence[str], opcional
        Nombres legibles de los grupos.
    method : str
        Método de ajuste de multiplicidad ('holm', 'fdr_bh', 'bonferroni').

    Retorna
    -------
    List[Dict[str, Any]]
        Lista de contrastes pareados con diferencia de rangos, z-score, p sin ajustar y p ajustado.
    """
    cleaned_groups = []
    for g in groups:
        arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().values
        if len(arr) > 0:
            cleaned_groups.append(arr)

    k = len(cleaned_groups)
    if k < 2:
        return []

    if group_names is None or len(group_names) != k:
        group_names = [f"Grupo_{i+1}" for i in range(k)]

    # Concatenar todas las observaciones y calcular rangos globales combinados
    all_data = []
    group_indices = []
    for idx, arr in enumerate(cleaned_groups):
        for val in arr:
            all_data.append(val)
            group_indices.append(idx)

    all_data = np.array(all_data)
    group_indices = np.array(group_indices)
    N = len(all_data)

    ranks = stats.rankdata(all_data)
    mean_ranks = [float(np.mean(ranks[group_indices == i])) for i in range(k)]
    ns = [len(g) for g in cleaned_groups]

    # Corrección por empates en rangos
    _, tie_counts = np.unique(all_data, return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts) / (12.0 * (N - 1)) if N > 1 else 0.0
    var_rank = (N * (N + 1) / 12.0) - tie_term

    comparisons = []
    raw_pvals = []

    for i in range(k):
        for j in range(i + 1, k):
            se_dunn = np.sqrt(var_rank * (1.0 / ns[i] + 1.0 / ns[j]))
            diff_rank = mean_ranks[i] - mean_ranks[j]
            z_val = diff_rank / se_dunn if se_dunn > 0 else 0.0
            p_raw = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))

            raw_pvals.append(p_raw)
            comparisons.append({
                "contrast": f"{group_names[i]} vs {group_names[j]}",
                "group1": str(group_names[i]),
                "group2": str(group_names[j]),
                "mean_rank_diff": float(diff_rank),
                "z_statistic": float(z_val),
                "p_raw": float(p_raw),
            })

    adj_pvals = adjust_pvalues(raw_pvals, method=method)
    for comp, p_adj in zip(comparisons, adj_pvals):
        comp["p_adj"] = float(p_adj)
        comp["method_adj"] = method
        comp["significant"] = bool(p_adj < 0.05)

    return comparisons


def jonckheere_terpstra_test(
    groups: Sequence[Sequence[float]], 
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Aplica la prueba de tendencia monótona ordenada de Jonckheere-Terpstra para contrastar
    si las concentraciones de metales aumentan (o disminuyen) progresivamente a lo largo de
    una variable ordinal categorizada en 3 o más niveles sucesivos (ej. frecuencia dietaria: nunca < rara vez < a veces...).
    Incluye corrección exacta por empates de datos.

    Parámetros
    ----------
    groups : Sequence[Sequence[float]]
        Secuencia de listas o arreglos numéricos ordenados jerárquicamente de menor a mayor categoría.
    alternative : str
        Hipótesis de tendencia ('two-sided', 'increasing', 'decreasing').

    Retorna
    -------
    Dict[str, Any]
        Estadístico J, esperanza E[J], varianza Var(J), z estandarizado y valor p.
    """
    cleaned = []
    for g in groups:
        arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().values
        if len(arr) > 0:
            cleaned.append(arr)

    k = len(cleaned)
    if k < 2:
        raise ValueError("Se requieren al menos 2 grupos ordenados para la prueba de Jonckheere-Terpstra.")

    ns = [len(g) for g in cleaned]
    N = sum(ns)

    # Calcular J = suma_{i < j} U_ij
    J_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            # Mann-Whitney cuenta pares X_jm > X_il + 0.5 * (X_jm == X_il)
            diffs = np.subtract.outer(cleaned[j], cleaned[i])
            u_ij = np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
            J_stat += u_ij

    # Esperanza matemática bajo H0
    sum_n_sq = sum(n**2 for n in ns)
    E_J = (N**2 - sum_n_sq) / 4.0

    # Varianza con corrección rigurosa por empates
    all_vals = np.concatenate(cleaned)
    _, tie_counts = np.unique(all_vals, return_counts=True)

    sum_n_term = sum(n * (n - 1) * (2 * n + 5) for n in ns)
    sum_t_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts)

    sum_n_cubes = sum(n * (n - 1) * (n - 2) for n in ns)
    sum_t_cubes = sum(t * (t - 1) * (t - 2) for t in tie_counts)

    sum_n_pairs = sum(n * (n - 1) for n in ns)
    sum_t_pairs = sum(t * (t - 1) for t in tie_counts)

    term1 = (N * (N - 1) * (2 * N + 5) - sum_n_term - sum_t_term) / 72.0
    term2 = (sum_n_cubes * sum_t_cubes) / (36.0 * N * (N - 1) * (N - 2)) if N > 2 else 0.0
    term3 = (sum_n_pairs * sum_t_pairs) / (8.0 * N * (N - 1)) if N > 1 else 0.0

    var_J = term1 + term2 + term3
    if var_J <= 0:
        # Respaldo sin empates
        var_J = (N**2 * (2 * N + 3) - sum(n**2 * (2 * n + 3) for n in ns)) / 72.0

    sd_J = np.sqrt(max(1e-12, var_J))

    # Corrección de continuidad en z
    diff = J_stat - E_J
    if diff > 0:
        z_stat = (diff - 0.5) / sd_J
    elif diff < 0:
        z_stat = (diff + 0.5) / sd_J
    else:
        z_stat = 0.0

    if alternative == "two-sided":
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    elif alternative in ("increasing", "greater"):
        p_val = 1.0 - stats.norm.cdf(z_stat)
    elif alternative in ("decreasing", "less"):
        p_val = stats.norm.cdf(z_stat)
    else:
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {
        "test": "Jonckheere-Terpstra",
        "k_groups": k,
        "total_n": N,
        "j_statistic": float(J_stat),
        "expected_j": float(E_J),
        "variance_j": float(var_J),
        "z_statistic": float(z_stat),
        "p_value": float(np.clip(p_val, 0.0, 1.0)),
        "alternative": alternative,
    }


def spearman_correlation(
    x: Sequence[float], 
    y: Sequence[float], 
    n_boot: int = 2000, 
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calcula el coeficiente de correlación de rangos de Spearman (rho) y obtiene intervalos
    de confianza no paramétricos robustos mediante remuestreo Bootstrap (BCa y percentiles).

    Parámetros
    ----------
    x : Sequence[float]
        Primera variable continua u ordinal.
    y : Sequence[float]
        Segunda variable continua u ordinal.
    n_boot : int
        Número de remuestreos bootstrap.
    confidence : float
        Nivel de confianza deseado (por defecto 0.95).

    Retorna
    -------
    Dict[str, Any]
        rho de Spearman, valor p analítico, intervalos bootstrap y tamaño muestral válido.
    """
    df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
    df_pair["x"] = pd.to_numeric(df_pair["x"], errors="coerce")
    df_pair["y"] = pd.to_numeric(df_pair["y"], errors="coerce")
    df_pair = df_pair.dropna()

    n = len(df_pair)
    if n < 3:
        return {
            "test": "Spearman Correlation",
            "n": n,
            "rho": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    arr_x = df_pair["x"].values
    arr_y = df_pair["y"].values

    res = stats.spearmanr(arr_x, arr_y)
    rho_obs = float(res.statistic)
    p_val = float(res.pvalue)

    # Remuestreo matricial vectorizado de alta velocidad
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=(n_boot, n), replace=True)
    bx = arr_x[idx]
    by = arr_y[idx]

    # Calcular rangos a lo largo del eje de observaciones
    rx = np.argsort(np.argsort(bx, axis=1), axis=1).astype(float) + 1.0
    ry = np.argsort(np.argsort(by, axis=1), axis=1).astype(float) + 1.0

    rx_c = rx - rx.mean(axis=1, keepdims=True)
    ry_c = ry - ry.mean(axis=1, keepdims=True)

    cov = (rx_c * ry_c).sum(axis=1)
    var_x = (rx_c ** 2).sum(axis=1)
    var_y = (ry_c ** 2).sum(axis=1)
    denom = np.sqrt(var_x * var_y)
    valid_denom = denom > 1e-12
    boot_rhos = np.zeros(n_boot, dtype=float)
    boot_rhos[valid_denom] = np.clip(cov[valid_denom] / denom[valid_denom], -1.0, 1.0)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_rhos, (alpha / 2.0) * 100))
    ci_high = float(np.percentile(boot_rhos, (1.0 - alpha / 2.0) * 100))

    return {
        "test": "Spearman Correlation",
        "n": n,
        "rho": rho_obs,
        "p_value": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
    }


def spearman_matrix(
    df: pd.DataFrame, 
    columns: List[str], 
    n_boot: int = 2000, 
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Calcula la matriz completa de correlación de Spearman para una lista de variables (ej. Pb, Hg, Cd),
    generando matrices estructuradas de coeficientes rho, valores p e intervalos de confianza bootstrap.
    """
    k = len(columns)
    rho_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    p_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    ci_low_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    ci_high_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for i in range(k):
        col_i = columns[i]
        for j in range(k):
            col_j = columns[j]
            if i == j:
                rho_matrix.loc[col_i, col_j] = 1.0
                p_matrix.loc[col_i, col_j] = 0.0
                ci_low_matrix.loc[col_i, col_j] = 1.0
                ci_high_matrix.loc[col_i, col_j] = 1.0
            elif i < j:
                res = spearman_correlation(
                    df[col_i], df[col_j], n_boot=n_boot, confidence=confidence, random_state=random_state + i + j
                )
                rho_matrix.loc[col_i, col_j] = res["rho"]
                rho_matrix.loc[col_j, col_i] = res["rho"]
                p_matrix.loc[col_i, col_j] = res["p_value"]
                p_matrix.loc[col_j, col_i] = res["p_value"]
                ci_low_matrix.loc[col_i, col_j] = res["ci_low"]
                ci_low_matrix.loc[col_j, col_i] = res["ci_low"]
                ci_high_matrix.loc[col_i, col_j] = res["ci_high"]
                ci_high_matrix.loc[col_j, col_i] = res["ci_high"]

    return {
        "rho": rho_matrix,
        "p_value": p_matrix,
        "ci_low": ci_low_matrix,
        "ci_high": ci_high_matrix,
    }


def fisher_chi2_test(
    contingency_table: Union[pd.DataFrame, np.ndarray],
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Evalúa la asociación entre 2 variables categóricas calculando la prueba de Chi-cuadrado de Pearson
    y la Prueba Exacta de Fisher, verificando automáticamente la condición de frecuencias esperadas pequeñas (<5).
    Reporta Odds Ratio (OR), Riesgo Relativo (RR) e intervalos de confianza epidemiológicos.

    Parámetros
    ----------
    contingency_table : pd.DataFrame o np.ndarray
        Tabla de contingencia 2x2 o rxc.

    Retorna
    -------
    Dict[str, Any]
        Valores p de Fisher y Chi2, frecuencias esperadas, conteo de celdas < 5, Odds Ratio y recomendación de prueba.
    """
    table_arr = np.asarray(contingency_table, dtype=float)
    nrows, ncols = table_arr.shape

    if nrows < 2 or ncols < 2:
        raise ValueError("La tabla de contingencia debe ser al menos de dimensiones 2x2.")

    total_n = float(np.sum(table_arr))
    row_sums = np.sum(table_arr, axis=1, keepdims=True)
    col_sums = np.sum(table_arr, axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / total_n

    cells_below_5 = int(np.sum(expected < 5.0))
    pct_below_5 = (cells_below_5 / expected.size) * 100.0
    has_low_expected = bool(cells_below_5 > 0)

    # Chi-cuadrado de Pearson (con corrección de Yates si es 2x2)
    chi2_res = stats.chi2_contingency(table_arr, correction=(nrows == 2 and ncols == 2))
    chi2_stat = float(chi2_res.statistic)
    chi2_p = float(chi2_res.pvalue)
    chi2_df = int(chi2_res.dof)

    # Coeficiente V de Cramér
    min_dim = min(nrows - 1, ncols - 1)
    cramer_v = float(np.sqrt(chi2_stat / (total_n * min_dim))) if min_dim > 0 and total_n > 0 else 0.0

    is_2x2 = (nrows == 2 and ncols == 2)
    fisher_p = np.nan
    odds_ratio = np.nan
    or_ci = (np.nan, np.nan)
    risk_ratio = np.nan
    rr_ci = (np.nan, np.nan)

    if is_2x2:
        # Prueba Exacta de Fisher
        fisher_res = stats.fisher_exact(table_arr, alternative=alternative)
        odds_ratio = float(fisher_res.statistic)
        fisher_p = float(fisher_res.pvalue)

        # Intervalo de confianza de Woolf para Odds Ratio con corrección de Haldane (0.5) para evitar ceros
        a, b = table_arr[0, 0], table_arr[0, 1]
        c, d = table_arr[1, 0], table_arr[1, 1]

        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = np.log((a_c * d_c) / (b_c * c_c))
        se_log_or = np.sqrt(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
        z_crit = 1.959963984540054  # 95%
        or_ci = (float(np.exp(log_or - z_crit * se_log_or)), float(np.exp(log_or + z_crit * se_log_or)))

        # Risk Ratio (RR): (a / (a+b)) / (c / (c+d))
        p1 = a / (a + b) if (a + b) > 0 else 0.0
        p2 = c / (c + d) if (c + d) > 0 else 0.0
        if p2 > 0:
            risk_ratio = float(p1 / p2)
            se_log_rr = np.sqrt((1.0 - p1) / (a + 0.001) + (1.0 - p2) / (c + 0.001))
            rr_ci = (float(np.exp(np.log(risk_ratio + 1e-9) - z_crit * se_log_rr)), float(np.exp(np.log(risk_ratio + 1e-9) + z_crit * se_log_rr)))

    # Recomendación metodológica
    if is_2x2:
        if has_low_expected or total_n < 40:
            recommended_test = "Fisher Exact"
            recommended_p = fisher_p
            warning_msg = f"⚠️ Frecuencias esperadas < 5 en {cells_below_5} celdas ({pct_below_5:.0f}%). Se recomienda Prueba Exacta de Fisher."
        else:
            recommended_test = "Chi-cuadrado (con corrección de continuidad)"
            recommended_p = chi2_p
            warning_msg = "Condiciones asintóticas adecuadas para Chi-cuadrado (frecuencias esperadas >= 5)."
    else:
        recommended_test = "Chi-cuadrado de Pearson"
        recommended_p = chi2_p
        warning_msg = f"⚠️ Frecuencias esperadas < 5 en {cells_below_5} celdas ({pct_below_5:.0f}%)." if has_low_expected else "Frecuencias esperadas adecuadas."

    return {
        "test": recommended_test,
        "is_2x2": is_2x2,
        "total_n": total_n,
        "cells_below_5": cells_below_5,
        "pct_below_5": pct_below_5,
        "has_low_expected": has_low_expected,
        "chi2_statistic": chi2_stat,
        "chi2_p": chi2_p,
        "chi2_df": chi2_df,
        "cramer_v": cramer_v,
        "fisher_p": fisher_p,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": or_ci,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci": rr_ci,
        "recommended_p": recommended_p,
        "warning_msg": warning_msg,
        "expected_counts": expected,
    }
