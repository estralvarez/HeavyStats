"""
Motores de cálculo estadístico para análisis bivariantes.
Implementa pruebas no paramétricas con tamaños de efecto e intervalos de confianza Bootstrap
vectorizados en NumPy para máximo rendimiento y precisión matemática:
- Correlación de Spearman con IC Bootstrap al 95%.
- Mann-Whitney U con Rank Biserial Correlation (r_rb), Hodges-Lehmann y diferencia de medianas.
- Kruskal-Wallis con Epsilon Cuadrado (epsilon^2) y prueba post-hoc de Dunn con ajuste múltiple.
- Tendencia ordinal de Jonckheere-Terpstra.
- Control de multiplicidad Benjamini-Hochberg FDR y Holm-Bonferroni.
- Algoritmo de ranking multicriterio de variables (Prioridad Alta, Intermedia, Baja).
- Diagnóstico de colinealidad y redundancia.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats


def adjust_pvalues(
    p_values: Sequence[float], 
    method: str = "fdr_bh"
) -> np.ndarray:
    """
    Ajusta una lista o arreglo de valores p por multiplicidad de hipótesis.

    Métodos soportados:
    - 'fdr_bh' / 'bh' / 'fdr': Procedimiento de Benjamini-Hochberg (control de FDR).
    - 'holm' / 'holm-bonferroni': Método step-down de Holm-Bonferroni (control de FWER).
    - 'bonferroni': Corrección clásica de Bonferroni.
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

    method_clean = method.lower().strip()
    valid_mask = ~np.isnan(p_arr)
    p_clean = p_arr[valid_mask]
    m = len(p_clean)

    if m <= 1:
        return np.clip(p_arr, 0.0, 1.0)

    order = np.argsort(p_clean)
    sorted_p = p_clean[order]
    adj = np.empty(m, dtype=float)

    if method_clean in ("fdr_bh", "bh", "fdr", "benjamini-hochberg"):
        ranks = np.arange(1, m + 1)
        step_vals = np.clip(sorted_p * (m / ranks), 0.0, 1.0)
        cum_min = np.minimum.accumulate(step_vals[::-1])[::-1]
        adj[order] = cum_min

    elif method_clean in ("holm", "holm-bonferroni"):
        factors = np.arange(m, 0, -1)
        step_vals = np.clip(sorted_p * factors, 0.0, 1.0)
        cum_max = np.maximum.accumulate(step_vals)
        adj[order] = cum_max

    elif method_clean == "bonferroni":
        adj_sorted = np.clip(sorted_p * m, 0.0, 1.0)
        adj[order] = adj_sorted

    elif method_clean == "hochberg":
        ranks = np.arange(m, 0, -1)
        step_vals = np.clip(sorted_p * ranks, 0.0, 1.0)
        cum_min = np.minimum.accumulate(step_vals[::-1])[::-1]
        adj[order] = cum_min

    else:
        raise ValueError(f"Método de ajuste '{method}' no reconocido. Opciones: 'fdr_bh', 'holm', 'bonferroni', 'hochberg'.")

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
    Calcula la diferencia de medianas (Med_x - Med_y) y su intervalo de confianza al 95%
    mediante remuestreo Bootstrap matricial no paramétrico.
    """
    x_clean = np.asarray(x, dtype=float)[~np.isnan(x)]
    y_clean = np.asarray(y, dtype=float)[~np.isnan(y)]
    n_x, n_y = len(x_clean), len(y_clean)

    if n_x == 0 or n_y == 0:
        return np.nan, np.nan, np.nan

    diff_obs = float(np.median(x_clean) - np.median(y_clean))
    if n_x < 2 or n_y < 2:
        return diff_obs, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    sample_x = rng.choice(x_clean, size=(n_boot, n_x), replace=True)
    sample_y = rng.choice(y_clean, size=(n_boot, n_y), replace=True)
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
    x_clean = np.asarray(x, dtype=float)[~np.isnan(x)]
    y_clean = np.asarray(y, dtype=float)[~np.isnan(y)]
    n1, n2 = len(x_clean), len(y_clean)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan

    diffs = np.subtract.outer(x_clean, y_clean).flatten()
    diffs.sort()
    hl_estimate = float(np.median(diffs))

    N = n1 * n2
    if N < 4:
        return hl_estimate, float(diffs[0]), float(diffs[-1])

    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    sigma_w = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    mu_w = N / 2.0
    c_alpha = int(np.floor(mu_w - z_crit * sigma_w))

    k_low = max(0, c_alpha)
    k_high = min(N - 1, N - 1 - c_alpha)

    ci_low = float(diffs[k_low]) if k_low < N else float(diffs[0])
    ci_high = float(diffs[k_high]) if k_high >= 0 else float(diffs[-1])
    return hl_estimate, ci_low, ci_high


def mann_whitney_test(
    group1_vals: Sequence[float], 
    group0_vals: Sequence[float], 
    alternative: str = "two-sided",
    n_boot: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Ejecuta la prueba no paramétrica de Mann-Whitney U para dos grupos independientes,
    calculando medianas, IQR, tamaño de efecto biserial por rangos (r_rb con IC Bootstrap vectorizado),
    estimador de Hodges-Lehmann y diferencia de medianas.
    """
    g1 = pd.to_numeric(pd.Series(group1_vals), errors="coerce").dropna().to_numpy()
    g0 = pd.to_numeric(pd.Series(group0_vals), errors="coerce").dropna().to_numpy()
    n1, n0 = len(g1), len(g0)

    res: Dict[str, Any] = {
        "n1": n1,
        "n0": n0,
        "n_total": n1 + n0,
        "median1": float(np.median(g1)) if n1 > 0 else np.nan,
        "median0": float(np.median(g0)) if n0 > 0 else np.nan,
        "iqr1": (float(np.percentile(g1, 25)), float(np.percentile(g1, 75))) if n1 > 0 else (np.nan, np.nan),
        "iqr0": (float(np.percentile(g0, 25)), float(np.percentile(g0, 75))) if n0 > 0 else (np.nan, np.nan),
        "u_stat": np.nan,
        "p_val": np.nan,
        "r_rb": np.nan,
        "r_rb_ci": (np.nan, np.nan),
        "diff_medians": np.nan,
        "diff_medians_ci": (np.nan, np.nan),
        "hl_shift": np.nan,
        "hl_shift_ci": (np.nan, np.nan),
    }

    if n1 == 0 or n0 == 0:
        return res

    try:
        mw = stats.mannwhitneyu(g1, g0, alternative=alternative)
        res["u_stat"] = float(mw.statistic)
        res["p_val"] = float(mw.pvalue)

        # Correlación biserial por rangos: r_rb = (2*U / (n1*n0)) - 1
        u1 = float(mw.statistic)
        r_rb = float((2.0 * u1) / (n1 * n0) - 1.0)
        res["r_rb"] = r_rb

        # Bootstrap Vectorizado ultrarrápido para r_rb
        if n1 >= 2 and n0 >= 2 and n_boot > 0:
            rng = np.random.default_rng(random_state)
            sample_1 = rng.choice(g1, size=(n_boot, n1), replace=True)
            sample_0 = rng.choice(g0, size=(n_boot, n0), replace=True)

            # Diferencias cruzadas matriciales: shape (n_boot, n1, n0)
            diffs_boot = sample_1[:, :, None] - sample_0[:, None, :]
            u_boot = np.sum(diffs_boot > 0, axis=(1, 2)) + 0.5 * np.sum(diffs_boot == 0, axis=(1, 2))
            r_boot = (2.0 * u_boot) / (n1 * n0) - 1.0

            alpha = 1.0 - confidence
            res["r_rb_ci"] = (
                float(np.percentile(r_boot, (alpha / 2.0) * 100)),
                float(np.percentile(r_boot, (1.0 - alpha / 2.0) * 100))
            )
    except Exception:
        pass

    # Diferencia de medianas con IC Bootstrap
    diff_m, d_low, d_high = bootstrap_ci_diff_medians(g1, g0, n_boot=n_boot, confidence=confidence, random_state=random_state)
    res["diff_medians"] = diff_m
    res["diff_medians_ci"] = (d_low, d_high)

    # Estimador de desplazamiento Hodges-Lehmann con IC
    hl, hl_low, hl_high = hodges_lehmann_2sample(g1, g0, confidence=confidence)
    res["hl_shift"] = hl
    res["hl_shift_ci"] = (hl_low, hl_high)

    return res


def kruskal_wallis_test(
    groups_data: Sequence[Sequence[float]], 
    group_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Ejecuta la prueba de Kruskal-Wallis para k grupos independientes,
    calculando el estadístico H, valor p, tamaño de efecto Epsilon Cuadrado (epsilon^2)
    y estadísticas por grupo (n, mediana, IQR).
    """
    cleaned_groups = []
    names = list(group_names) if group_names is not None else [f"Grupo_{i+1}" for i in range(len(groups_data))]

    group_stats = []
    for i, g in enumerate(groups_data):
        clean_arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy()
        if len(clean_arr) > 0:
            cleaned_groups.append(clean_arr)
            name = names[i] if i < len(names) else f"Grupo_{i+1}"
            group_stats.append({
                "group": name,
                "n": len(clean_arr),
                "median": float(np.median(clean_arr)),
                "q25": float(np.percentile(clean_arr, 25)),
                "q75": float(np.percentile(clean_arr, 75)),
            })

    k = len(cleaned_groups)
    total_n = sum(len(g) for g in cleaned_groups)

    res: Dict[str, Any] = {
        "k": k,
        "n_total": total_n,
        "h_stat": np.nan,
        "p_val": np.nan,
        "epsilon_sq": np.nan,
        "group_stats": group_stats,
    }

    if k < 2 or total_n <= k:
        return res

    try:
        kw = stats.kruskal(*cleaned_groups)
        h_stat = float(kw.statistic)
        p_val = float(kw.pvalue)
        res["h_stat"] = h_stat
        res["p_val"] = p_val

        # Epsilon cuadrado: epsilon^2 = (H - k + 1) / (N - k)
        if total_n > k:
            eps_sq = (h_stat - k + 1.0) / (total_n - k)
            res["epsilon_sq"] = float(np.clip(eps_sq, 0.0, 1.0))
    except Exception:
        pass

    return res


def dunn_posthoc_test(
    groups_data: Sequence[Sequence[float]], 
    group_names: Sequence[str],
    p_adjust: str = "holm"
) -> pd.DataFrame:
    """
    Ejecuta comparaciones múltiples por pares mediante la prueba post-hoc de Dunn
    sobre sumas de rangos globales con ajuste de valores p.
    """
    valid_groups = []
    valid_names = []
    for g, name in zip(groups_data, group_names):
        c = pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy()
        if len(c) > 0:
            valid_groups.append(c)
            valid_names.append(name)

    k = len(valid_groups)
    if k < 2:
        return pd.DataFrame(columns=["Comparación", "Grupo 1", "Grupo 2", "Z", "p", "p_adj"])

    all_vals = np.concatenate(valid_groups)
    all_ranks = stats.rankdata(all_vals)
    N = len(all_vals)

    start_idx = 0
    group_mean_ranks = []
    group_ns = []
    for g in valid_groups:
        n_g = len(g)
        ranks_g = all_ranks[start_idx : start_idx + n_g]
        group_mean_ranks.append(float(np.mean(ranks_g)))
        group_ns.append(n_g)
        start_idx += n_g

    _, tie_counts = np.unique(all_vals, return_counts=True)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    c_tie = 1.0 - (tie_sum / (N**3 - N)) if N > 1 else 1.0

    records = []
    p_values = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = group_mean_ranks[i] - group_mean_ranks[j]
            se = np.sqrt((N * (N + 1) / 12.0) * c_tie * (1.0 / group_ns[i] + 1.0 / group_ns[j]))
            z = diff / se if se > 0 else 0.0
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

            records.append({
                "Comparación": f"{valid_names[i]} vs {valid_names[j]}",
                "Grupo 1": valid_names[i],
                "Grupo 2": valid_names[j],
                "Dif_Rangos": float(diff),
                "Z": float(z),
                "p": float(p),
            })
            p_values.append(p)

    adj_p = adjust_pvalues(p_values, method=p_adjust)
    for rec, padj in zip(records, adj_p):
        rec["p_adj"] = float(padj)

    return pd.DataFrame(records)


def jonckheere_terpstra_test(
    groups_data: Sequence[Sequence[float]],
    alternative: str = "increasing"
) -> Dict[str, Any]:
    """
    Calcula la prueba de Jonckheere-Terpstra para contrastar tendencias ordenadas
    monotónicas a través de k grupos (ej. Bajo < Medio < Alto).
    """
    valid_groups = [
        pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy()
        for g in groups_data
    ]
    valid_groups = [g for g in valid_groups if len(g) > 0]
    k = len(valid_groups)

    res = {
        "j_stat": np.nan,
        "z_stat": np.nan,
        "p_val": np.nan,
        "k": k,
    }
    if k < 2:
        return res

    j_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            mw = stats.mannwhitneyu(valid_groups[j], valid_groups[i], alternative="two-sided")
            j_stat += mw.statistic

    ns = [len(g) for g in valid_groups]
    N = sum(ns)

    e_j = (N**2 - sum(n**2 for n in ns)) / 4.0
    var_j = (
        N**2 * (2 * N + 3)
        - sum(n**2 * (2 * n + 3) for n in ns)
    ) / 72.0

    se_j = np.sqrt(var_j) if var_j > 0 else 1.0
    z_stat = (j_stat - e_j) / se_j

    if alternative == "increasing":
        p_val = 1.0 - stats.norm.cdf(z_stat)
    elif alternative == "decreasing":
        p_val = stats.norm.cdf(z_stat)
    else:  # two-sided
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    res["j_stat"] = float(j_stat)
    res["z_stat"] = float(z_stat)
    res["p_val"] = float(np.clip(p_val, 0.0, 1.0))
    return res


def spearman_correlation(
    x: Sequence[float], 
    y: Sequence[float], 
    alternative: str = "two-sided",
    n_boot: int = 2000, 
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calcula el coeficiente de correlación por rangos de Spearman (rho_s),
    su significancia asintótica (p-value) e intervalo de confianza al 95%
    mediante remuestreo Bootstrap no paramétrico vectorizado ultrarrápido.
    """
    s_x = pd.to_numeric(pd.Series(x), errors="coerce")
    s_y = pd.to_numeric(pd.Series(y), errors="coerce")
    valid_mask = s_x.notna() & s_y.notna()

    arr_x = s_x[valid_mask].to_numpy()
    arr_y = s_y[valid_mask].to_numpy()
    n = len(arr_x)

    res = {
        "rho": np.nan,
        "p_val": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n_valid": n,
    }

    if n < 3:
        return res

    try:
        sp = stats.spearmanr(arr_x, arr_y, alternative=alternative)
        rho_val = float(sp.statistic)
        p_val = float(sp.pvalue)
        res["rho"] = rho_val
        res["p_val"] = p_val

        # Intervalo Bootstrap al 95% Vectorizado
        if n >= 4 and n_boot > 0:
            rng = np.random.default_rng(random_state)
            boot_idx = rng.choice(n, size=(n_boot, n), replace=True)

            bx = arr_x[boot_idx]
            by = arr_y[boot_idx]

            # Rango en cada muestra bootstrap
            # argsort(argsort) genera los rangos en tiempo O(n log n)
            rx = np.argsort(np.argsort(bx, axis=1), axis=1).astype(float) + 1.0
            ry = np.argsort(np.argsort(by, axis=1), axis=1).astype(float) + 1.0

            rx_m = rx - rx.mean(axis=1, keepdims=True)
            ry_m = ry - ry.mean(axis=1, keepdims=True)

            cov = np.sum(rx_m * ry_m, axis=1)
            var_x = np.sum(rx_m**2, axis=1)
            var_y = np.sum(ry_m**2, axis=1)
            denom = np.sqrt(var_x * var_y)

            valid_boot = denom > 0
            boot_rhos = np.zeros(n_boot)
            boot_rhos[valid_boot] = cov[valid_boot] / denom[valid_boot]

            clean_rhos = boot_rhos[valid_boot]
            if len(clean_rhos) > 20:
                alpha = 1.0 - confidence
                res["ci_low"] = float(np.percentile(clean_rhos, (alpha / 2.0) * 100))
                res["ci_high"] = float(np.percentile(clean_rhos, (1.0 - alpha / 2.0) * 100))
    except Exception:
        pass

    return res


def spearman_matrix(
    df: pd.DataFrame, 
    columns: Optional[Sequence[str]] = None, 
    n_boot: int = 1000, 
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Calcula matrices completas de correlación de Spearman inter-variables:
    - Matriz de coeficientes (rho)
    - Matriz de valores p (p-values)
    - Matriz de intervalos de confianza al 95%
    - Matriz de tamaños muestrales efectivos (n)
    """
    cols = list(columns) if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    k = len(cols)

    rho_mat = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    ci_low_mat = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    ci_high_mat = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    n_mat = pd.DataFrame(np.zeros((k, k), dtype=int), index=cols, columns=cols)

    for i in range(k):
        col_i = cols[i]
        n_mat.loc[col_i, col_i] = int(df[col_i].dropna().shape[0])
        for j in range(i + 1, k):
            col_j = cols[j]
            res = spearman_correlation(df[col_i], df[col_j], n_boot=n_boot, random_state=random_state)
            rho = res["rho"]
            p = res["p_val"]
            ci_l = res["ci_low"]
            ci_h = res["ci_high"]
            n_ij = res["n_valid"]

            rho_mat.loc[col_i, col_j] = rho_mat.loc[col_j, col_i] = rho
            p_mat.loc[col_i, col_j] = p_mat.loc[col_j, col_i] = p
            ci_low_mat.loc[col_i, col_j] = ci_low_mat.loc[col_j, col_i] = ci_l
            ci_high_mat.loc[col_i, col_j] = ci_high_mat.loc[col_j, col_i] = ci_h
            n_mat.loc[col_i, col_j] = n_mat.loc[col_j, col_i] = n_ij

    return {
        "rho": rho_mat,
        "p_values": p_mat,
        "ci_low": ci_low_mat,
        "ci_high": ci_high_mat,
        "n": n_mat,
    }


def fisher_chi2_test(
    table_2x2: Sequence[Sequence[int]]
) -> Dict[str, Any]:
    """
    Ejecuta la prueba exacta de Fisher y Chi-cuadrado con corrección de Yates
    para tablas de contingencia 2x2, calculando también Odds Ratio, Riesgo Relativo y V de Cramér.
    """
    arr = np.asarray(table_2x2, dtype=int)
    res: Dict[str, Any] = {
        "fisher_odds_ratio": np.nan,
        "fisher_p": np.nan,
        "chi2_stat": np.nan,
        "chi2_p": np.nan,
        "odds_ratio": np.nan,
        "or_ci": (np.nan, np.nan),
        "relative_risk": np.nan,
        "rr_ci": (np.nan, np.nan),
        "cramers_v": np.nan,
    }
    if arr.shape != (2, 2):
        return res

    try:
        fo, fp = stats.fisher_exact(arr)
        res["fisher_odds_ratio"] = float(fo)
        res["fisher_p"] = float(fp)
    except Exception:
        pass

    try:
        c2, cp, _, _ = stats.chi2_contingency(arr, correction=True)
        res["chi2_stat"] = float(c2)
        res["chi2_p"] = float(cp)
        n = arr.sum()
        if n > 0:
            res["cramers_v"] = float(np.sqrt(c2 / n))
    except Exception:
        pass

    # Odds ratio con IC
    try:
        if hasattr(stats.contingency, "odds_ratio"):
            or_obj = stats.contingency.odds_ratio(arr)
            res["odds_ratio"] = float(or_obj.statistic)
            ci_obj = or_obj.confidence_interval(confidence_level=0.95)
            res["or_ci"] = (float(ci_obj.low), float(ci_obj.high))
        else:
            a, b = arr[0, 0] + 0.5, arr[0, 1] + 0.5
            c, d = arr[1, 0] + 0.5, arr[1, 1] + 0.5
            or_h = (a * d) / (b * c)
            res["odds_ratio"] = float(or_h)
            se = np.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
            res["or_ci"] = (float(np.exp(np.log(or_h) - 1.96 * se)), float(np.exp(np.log(or_h) + 1.96 * se)))
    except Exception:
        pass

    # Riesgo Relativo con IC
    try:
        a, b = arr[0, 0], arr[0, 1]
        c, d = arr[1, 0], arr[1, 1]
        n1 = a + b
        n0 = c + d
        if n1 > 0 and n0 > 0:
            a_h = a + 0.5
            c_h = c + 0.5
            p1_h = a_h / (n1 + 0.5)
            p0_h = c_h / (n0 + 0.5)
            rr_val = p1_h / p0_h
            res["relative_risk"] = float(rr_val)
            se_rr = np.sqrt((1.0/a_h - 1.0/(n1 + 0.5)) + (1.0/c_h - 1.0/(n0 + 0.5)))
            res["rr_ci"] = (float(np.exp(np.log(rr_val) - 1.96 * se_rr)), float(np.exp(np.log(rr_val) + 1.96 * se_rr)))
    except Exception:
        pass

    return res


def qualitative_association_test(
    x: Sequence[Any], 
    y: Sequence[Any],
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calcula la asociación bivariante completa entre dos variables cualitativas (2x2 o RxC).
    Devuelve tabla de contingencia, prueba exacta de Fisher, Chi-cuadrado (Yates y Pearson),
    Odds Ratio (OR con IC 95%), Riesgo Relativo (RR con IC 95%), Diferencia de Riesgo y V de Cramér.
    """
    s_x = pd.Series(x).dropna()
    s_y = pd.Series(y).dropna()
    common_idx = s_x.index.intersection(s_y.index)
    s_x = s_x.loc[common_idx]
    s_y = s_y.loc[common_idx]
    n = len(s_x)

    res: Dict[str, Any] = {
        "n": n,
        "contingency_table": pd.DataFrame(),
        "is_2x2": False,
        "fisher_p": np.nan,
        "odds_ratio": np.nan,
        "or_ci": (np.nan, np.nan),
        "relative_risk": np.nan,
        "rr_ci": (np.nan, np.nan),
        "risk_diff": np.nan,
        "chi2_stat": np.nan,
        "chi2_p": np.nan,
        "chi2_yates_stat": np.nan,
        "chi2_yates_p": np.nan,
        "cramers_v": np.nan,
        "phi": np.nan,
    }

    if n == 0:
        return res

    ctab = pd.crosstab(s_x, s_y)
    res["contingency_table"] = ctab

    r, c = ctab.shape
    if r < 2 or c < 2:
        return res

    # Chi-cuadrado estándar
    try:
        chi2_raw, p_raw, dof, _ = stats.chi2_contingency(ctab.to_numpy(), correction=False)
        res["chi2_stat"] = float(chi2_raw)
        res["chi2_p"] = float(p_raw)
        min_dim = min(r, c)
        if n > 0 and (min_dim - 1) > 0:
            res["cramers_v"] = float(np.sqrt(chi2_raw / (n * (min_dim - 1))))
    except Exception:
        pass

    # Si es tabla 2x2
    if r == 2 and c == 2:
        res["is_2x2"] = True
        arr_2x2 = ctab.to_numpy()
        a, b = int(arr_2x2[0, 0]), int(arr_2x2[0, 1])
        c_val, d = int(arr_2x2[1, 0]), int(arr_2x2[1, 1])

        # Phi
        if pd.notna(res["chi2_stat"]) and n > 0:
            res["phi"] = float(np.sqrt(res["chi2_stat"] / n))

        # Chi2 Yates
        try:
            c2_y, p_y, _, _ = stats.chi2_contingency(arr_2x2, correction=True)
            res["chi2_yates_stat"] = float(c2_y)
            res["chi2_yates_p"] = float(p_y)
        except Exception:
            pass

        # Fisher Exact
        try:
            _, fp = stats.fisher_exact(arr_2x2)
            res["fisher_p"] = float(fp)
        except Exception:
            pass

        # Odds Ratio con IC
        try:
            if hasattr(stats.contingency, "odds_ratio"):
                or_obj = stats.contingency.odds_ratio(arr_2x2)
                res["odds_ratio"] = float(or_obj.statistic)
                ci_obj = or_obj.confidence_interval(confidence_level=confidence)
                res["or_ci"] = (float(ci_obj.low), float(ci_obj.high))
            else:
                a_h, b_h, c_h, d_h = a + 0.5, b + 0.5, c_val + 0.5, d + 0.5
                or_val = (a_h * d_h) / (b_h * c_h)
                res["odds_ratio"] = float(or_val)
                se_ln_or = np.sqrt(1.0/a_h + 1.0/b_h + 1.0/c_h + 1.0/d_h)
                z = stats.norm.ppf(1.0 - (1.0 - confidence)/2.0)
                res["or_ci"] = (float(np.exp(np.log(or_val) - z * se_ln_or)), float(np.exp(np.log(or_val) + z * se_ln_or)))
        except Exception:
            pass

        # Riesgo Relativo y Diferencia de Riesgo
        try:
            n_exp = a + b
            n_noexp = c_val + d
            if n_exp > 0 and n_noexp > 0:
                p1 = a / n_exp
                p0 = c_val / n_noexp
                res["risk_diff"] = float(p1 - p0)

                p1_h = (a + 0.5) / (n_exp + 0.5)
                p0_h = (c_val + 0.5) / (n_noexp + 0.5)
                rr_val = p1_h / p0_h
                res["relative_risk"] = float(rr_val)
                se_ln_rr = np.sqrt((1.0/(a + 0.5) - 1.0/(n_exp + 0.5)) + (1.0/(c_val + 0.5) - 1.0/(n_noexp + 0.5)))
                z = stats.norm.ppf(1.0 - (1.0 - confidence)/2.0)
                res["rr_ci"] = (float(np.exp(np.log(rr_val) - z * se_ln_rr)), float(np.exp(np.log(rr_val) + z * se_ln_rr)))
        except Exception:
            pass

    # Aliases de conveniencia
    res["p_val_fisher"] = res.get("fisher_p")
    res["p_val_chi2"] = res.get("chi2_p")
    res["cramer_v"] = res.get("cramers_v")

    return res



def independent_t_test(
    group1_vals: Sequence[float], 
    group0_vals: Sequence[float], 
    equal_var: bool = False,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Ejecuta la prueba t de Student o t de Welch para dos grupos independientes,
    calculando medias, desviación estándar por grupo, diferencia de medias con IC al 95%,
    prueba de Levene de homocedasticidad, d de Cohen y g de Hedges (corrección para n pequeña).
    """
    g1 = pd.to_numeric(pd.Series(group1_vals), errors="coerce").dropna().to_numpy()
    g0 = pd.to_numeric(pd.Series(group0_vals), errors="coerce").dropna().to_numpy()
    n1, n0 = len(g1), len(g0)

    res: Dict[str, Any] = {
        "n1": n1,
        "n0": n0,
        "n_total": n1 + n0,
        "mean1": float(np.mean(g1)) if n1 > 0 else np.nan,
        "mean0": float(np.mean(g0)) if n0 > 0 else np.nan,
        "std1": float(np.std(g1, ddof=1)) if n1 > 1 else np.nan,
        "std0": float(np.std(g0, ddof=1)) if n0 > 1 else np.nan,
        "diff_means": np.nan,
        "diff_means_ci": (np.nan, np.nan),
        "t_stat": np.nan,
        "p_val": np.nan,
        "df": np.nan,
        "levene_stat": np.nan,
        "levene_p": np.nan,
        "cohen_d": np.nan,
        "hedges_g": np.nan,
    }

    if n1 < 2 or n0 < 2:
        return res

    mean1 = res["mean1"]
    mean0 = res["mean0"]
    s1 = res["std1"]
    s0 = res["std0"]
    diff = mean1 - mean0
    res["diff_means"] = diff

    # Prueba de Levene
    try:
        lev = stats.levene(g1, g0, center="median")
        res["levene_stat"] = float(lev.statistic)
        res["levene_p"] = float(lev.pvalue)
    except Exception:
        pass

    # Prueba t (Welch o Student)
    try:
        tt = stats.ttest_ind(g1, g0, equal_var=equal_var)
        res["t_stat"] = float(tt.statistic)
        res["p_val"] = float(tt.pvalue)
        if hasattr(tt, "df"):
            res["df"] = float(tt.df)
    except Exception:
        pass

    # Intervalo de confianza de la diferencia de medias
    alpha = 1.0 - confidence
    if equal_var:
        df = n1 + n0 - 2
        sp2 = ((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / df if df > 0 else np.nan
        se_diff = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n0)) if pd.notna(sp2) else np.nan
    else:
        se_diff = np.sqrt((s1**2 / n1) + (s0**2 / n0))
        num = (s1**2 / n1 + s0**2 / n0)**2
        den = (s1**4 / (n1**2 * (n1 - 1))) + (s0**4 / (n0**2 * (n0 - 1)))
        df = num / den if den > 0 else (n1 + n0 - 2)

    if np.isnan(res["df"]):
        res["df"] = float(df)

    if pd.notna(se_diff) and se_diff > 0 and df > 0:
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        res["diff_means_ci"] = (float(diff - tcrit * se_diff), float(diff + tcrit * se_diff))

    # Tamaños de efecto: d de Cohen y g de Hedges
    sp2_pool = ((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2) if (n1 + n0 - 2) > 0 else 0.0
    s_pool = np.sqrt(sp2_pool) if sp2_pool > 0 else np.nan
    if s_pool and s_pool > 0:
        d = diff / s_pool
        res["cohen_d"] = float(d)
        j = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0)) if (n1 + n0) > 2 else 1.0
        res["hedges_g"] = float(d * j)

    return res


def anova_oneway_test(
    groups_data: Sequence[Sequence[float]], 
    group_names: Optional[Sequence[str]] = None,
    equal_var: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta el análisis de varianza (ANOVA) de una vía (clásico o de Welch),
    calculando medias y desviaciones estándar por grupo, tamaño de efecto Eta cuadrado (eta^2)
    y Omega cuadrado (omega^2), prueba de Levene de homocedasticidad y comparaciones post-hoc (Tukey HSD).
    """
    cleaned_groups = []
    names = list(group_names) if group_names is not None else [f"Grupo_{i+1}" for i in range(len(groups_data))]

    group_stats = []
    for i, g in enumerate(groups_data):
        clean_arr = pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy()
        if len(clean_arr) > 0:
            cleaned_groups.append(clean_arr)
            name = names[i] if i < len(names) else f"Grupo_{i+1}"
            m = float(np.mean(clean_arr))
            s = float(np.std(clean_arr, ddof=1)) if len(clean_arr) > 1 else 0.0
            group_stats.append({
                "group": name,
                "n": len(clean_arr),
                "mean": m,
                "std": s,
                "sem": s / np.sqrt(len(clean_arr)) if len(clean_arr) > 0 else 0.0,
            })

    k = len(cleaned_groups)
    total_n = sum(len(g) for g in cleaned_groups)

    res: Dict[str, Any] = {
        "k": k,
        "n_total": total_n,
        "f_stat": np.nan,
        "p_val": np.nan,
        "eta_sq": np.nan,
        "omega_sq": np.nan,
        "levene_stat": np.nan,
        "levene_p": np.nan,
        "group_stats": group_stats,
        "posthoc": None
    }

    if k < 2 or total_n <= k:
        return res

    # Prueba de Levene
    try:
        lev = stats.levene(*cleaned_groups, center="median")
        res["levene_stat"] = float(lev.statistic)
        res["levene_p"] = float(lev.pvalue)
    except Exception:
        pass

    # ANOVA clásico
    try:
        f_res = stats.f_oneway(*cleaned_groups)
        f_val = float(f_res.statistic)
        p_val = float(f_res.pvalue)
        res["f_stat"] = f_val
        res["p_val"] = p_val

        # Cálculo de Sumas de Cuadrados para eta^2 y omega^2
        all_vals = np.concatenate(cleaned_groups)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in cleaned_groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in cleaned_groups)
        ss_total = ss_between + ss_within

        df_between = k - 1
        df_within = total_n - k
        ms_within = ss_within / df_within if df_within > 0 else np.nan

        if ss_total > 0:
            res["eta_sq"] = float(np.clip(ss_between / ss_total, 0.0, 1.0))
            if ms_within is not None and (ss_total + ms_within) > 0:
                om_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)
                res["omega_sq"] = float(np.clip(om_sq, 0.0, 1.0))

        # Comparaciones múltiples Tukey HSD
        try:
            if hasattr(stats, "tukey_hsd"):
                thsd = stats.tukey_hsd(*cleaned_groups)
                p_records = []
                for i in range(k):
                    for j in range(i + 1, k):
                        diff_m = group_stats[i]["mean"] - group_stats[j]["mean"]
                        p_val_pair = float(thsd.pvalue[i, j])
                        p_records.append({
                            "Comparación": f"{group_stats[i]['group']} vs {group_stats[j]['group']}",
                            "Grupo 1": group_stats[i]["group"],
                            "Grupo 2": group_stats[j]["group"],
                            "Dif_Medias": float(diff_m),
                            "p_adj": p_val_pair
                        })
                res["posthoc"] = pd.DataFrame(p_records)
        except Exception:
            pass

    except Exception:
        pass

    return res


def pearson_correlation(
    x: Sequence[float], 
    y: Sequence[float], 
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calcula el coeficiente de correlación lineal de Pearson (r), su significancia (p-value),
    intervalo de confianza al 95% mediante la transformación z de Fisher,
    y parámetros de regresión lineal simple (R^2, pendiente, intercepto, error estándar).
    """
    s_x = pd.to_numeric(pd.Series(x), errors="coerce")
    s_y = pd.to_numeric(pd.Series(y), errors="coerce")
    valid_mask = s_x.notna() & s_y.notna()

    arr_x = s_x[valid_mask].to_numpy()
    arr_y = s_y[valid_mask].to_numpy()
    n = len(arr_x)

    res = {
        "r": np.nan,
        "p_val": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "r_squared": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "std_err": np.nan,
        "n_valid": n,
    }

    if n < 3:
        return res

    try:
        pr = stats.pearsonr(arr_x, arr_y)
        r_val = float(pr.statistic)
        p_val = float(pr.pvalue)
        res["r"] = r_val
        res["p_val"] = p_val
        res["r_squared"] = float(r_val**2)

        # Intervalo de confianza mediante transformación z de Fisher
        if n >= 4 and abs(r_val) < 1.0:
            alpha = 1.0 - confidence
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            z = np.arctanh(r_val)
            se_z = 1.0 / np.sqrt(n - 3)
            res["ci_low"] = float(np.tanh(z - z_crit * se_z))
            res["ci_high"] = float(np.tanh(z + z_crit * se_z))

        # Regresión lineal simple
        lr = stats.linregress(arr_x, arr_y)
        res["slope"] = float(lr.slope)
        res["intercept"] = float(lr.intercept)
        res["std_err"] = float(lr.stderr)
    except Exception:
        pass

    return res


def kendall_correlation(
    x: Sequence[float], 
    y: Sequence[float]
) -> Dict[str, Any]:
    """
    Calcula el coeficiente de correlación Tau-b de Kendall (tau),
    especialmente robusto ante muestras pequeñas (n=20) y presencia de empates numéricos.
    """
    s_x = pd.to_numeric(pd.Series(x), errors="coerce")
    s_y = pd.to_numeric(pd.Series(y), errors="coerce")
    valid_mask = s_x.notna() & s_y.notna()

    arr_x = s_x[valid_mask].to_numpy()
    arr_y = s_y[valid_mask].to_numpy()
    n = len(arr_x)

    res = {
        "tau": np.nan,
        "p_val": np.nan,
        "n_valid": n,
    }

    if n < 3:
        return res

    try:
        kt = stats.kendalltau(arr_x, arr_y)
        res["tau"] = float(kt.statistic)
        res["p_val"] = float(kt.pvalue)
    except Exception:
        pass

    return res


def rank_bivariate_associations(
    df_results: pd.DataFrame,
    effect_col: str = "Efecto_Num",
    p_raw_col: str = "p_Raw",
    p_fdr_col: str = "p_FDR",
    n_pos_col: str = "n_Positivo",
    min_effect_high: float = 0.30,
    min_effect_med: float = 0.18,
    fdr_alpha: float = 0.10,
    p_alpha: float = 0.05,
    min_n_pos: int = 2
) -> pd.DataFrame:
    """
    Clasifica las variables analizadas en un Ranking de Prioridad Multicriterio (Etapa 18):
    - 'Prioridad Alta': Efecto sustancial (|efecto| >= min_effect_high) + Frecuencia suficiente (n_pos >= min_n_pos)
      + Evidencia estadística (p_FDR < fdr_alpha o p < p_alpha).
    - 'Prioridad Intermedia': Efecto moderado (|efecto| >= min_effect_med) o evidencia exploratoria razonable.
    - 'Prioridad Baja': Frecuencia insuficiente (n_pos < min_n_pos), efecto despreciable o evidencia muy inestable.
    """
    df_ranked = df_results.copy()
    priorities = []
    justifications = []

    for _, row in df_ranked.iterrows():
        eff = abs(float(row.get(effect_col, 0.0))) if pd.notna(row.get(effect_col)) else 0.0
        p_raw = float(row.get(p_raw_col, 1.0)) if pd.notna(row.get(p_raw_col)) else 1.0
        p_fdr = float(row.get(p_fdr_col, 1.0)) if pd.notna(row.get(p_fdr_col)) else 1.0
        n_pos = int(row.get(n_pos_col, 99)) if pd.notna(row.get(n_pos_col)) else 99

        if n_pos < min_n_pos:
            priorities.append("Prioridad Baja")
            justifications.append(f"Frecuencia insuficiente en expuestos (n={n_pos} < {min_n_pos}).")
            continue

        if eff >= min_effect_high and (p_fdr < fdr_alpha or p_raw < p_alpha):
            priorities.append("Prioridad Alta")
            justifications.append(f"Efecto consistente (|E|={eff:.2f} >= {min_effect_high}), n={n_pos} y p={p_raw:.4f} (FDR={p_fdr:.4f}).")
        elif eff >= min_effect_high:
            priorities.append("Prioridad Intermedia")
            justifications.append(f"Efecto notable (|E|={eff:.2f}) con significancia marginal (p={p_raw:.4f}).")
        elif eff >= min_effect_med and p_raw < p_alpha:
            priorities.append("Prioridad Intermedia")
            justifications.append(f"Efecto moderado (|E|={eff:.2f}) y evidencia exploratoria preliminar (p={p_raw:.4f}).")
        elif eff >= min_effect_med:
            priorities.append("Prioridad Intermedia")
            justifications.append(f"Efecto moderado (|E|={eff:.2f}) sin significancia formal (p={p_raw:.4f}).")
        else:
            priorities.append("Prioridad Baja")
            justifications.append(f"Efecto débil (|E|={eff:.2f} < {min_effect_med}) e incertidumbre elevada.")

    df_ranked["Prioridad"] = priorities
    df_ranked["Justificacion"] = justifications

    prio_order = {"Prioridad Alta": 0, "Prioridad Intermedia": 1, "Prioridad Baja": 2}
    df_ranked["_prio_val"] = df_ranked["Prioridad"].map(prio_order)
    df_ranked["_eff_abs"] = df_ranked[effect_col].abs()
    df_ranked = df_ranked.sort_values(by=["_prio_val", "_eff_abs"], ascending=[True, False]).drop(columns=["_prio_val", "_eff_abs"])

    return df_ranked.reset_index(drop=True)


def collinearity_matrix(
    df: pd.DataFrame, 
    columns: Sequence[str],
    method: str = "spearman"
) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre predictores para diagnosticar colinealidad y redundancias (Etapa 19).
    """
    valid_cols = [c for c in columns if c in df.columns]
    sub_df = df[valid_cols].apply(pd.to_numeric, errors="coerce")
    return sub_df.corr(method=method)


__all__ = [
    "adjust_pvalues",
    "bootstrap_ci_diff_medians",
    "hodges_lehmann_2sample",
    "mann_whitney_test",
    "independent_t_test",
    "kruskal_wallis_test",
    "anova_oneway_test",
    "dunn_posthoc_test",
    "jonckheere_terpstra_test",
    "spearman_correlation",
    "pearson_correlation",
    "kendall_correlation",
    "spearman_matrix",
    "fisher_chi2_test",
    "qualitative_association_test",
    "rank_bivariate_associations",
    "collinearity_matrix",
]
