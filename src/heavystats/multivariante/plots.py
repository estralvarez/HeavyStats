"""
Módulo de visualizaciones para Análisis Multivariante Quimiométrico (PLS-R y LASSO).
Genera figuras a 300 DPI con calidad de publicación para tesis y revistas científicas.
"""

from typing import Optional, Tuple, Any, List, Union, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from heavystats.multivariante.models import PlsModeler, LassoModeler


class MultivariatePlots:
    """
    Motor de generación gráfica para modelos multivariantes (PLS-1, LASSO).
    """

    def __init__(
        self,
        pls_model: Optional[PlsModeler] = None,
        lasso_model: Optional[LassoModeler] = None,
        labels_map: Optional[Dict[str, str]] = None,
        target_label: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        pls_model : PlsModeler, optional
            Modelo PLS ajustado.
        lasso_model : LassoModeler, optional
            Modelo LASSO ajustado.
        labels_map : Dict[str, str], optional
            Diccionario de mapeo de nombres técnicos de columnas a nombres legibles en español.
        target_label : str, optional
            Etiqueta del biomarcador para los ejes gráficos (p. ej. 'Mercurio Plasmático [µg/L]').
        """
        self.pls = pls_model
        self.lasso = lasso_model
        self.labels_map = labels_map or {}
        self.target_label = target_label or "Concentración Biológica"

    def _get_label(self, var_name: str) -> str:
        return self.labels_map.get(var_name, var_name)

    def _save_fig(self, fig: plt.Figure, filepath: Optional[Union[str, Path]]) -> None:
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")

    def plot_vip_ranking(
        self,
        top_n: int = 15,
        threshold: float = 1.0,
        figsize: Tuple[float, float] = (10.0, 6.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 1: Ranking horizontal de importancia de variable en la proyección (VIP).
        Diferencia en color los factores críticos (VIP >= 1.0) de los secundarios.
        """
        if self.pls is None or len(self.pls.vip_) == 0:
            raise ValueError("Se requiere un modelo PlsModeler ajustado con vector VIP.")

        df_vip = self.pls.summary_df.copy()
        df_top = df_vip.head(top_n).sort_values(by="VIP", ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(df_top))
        vip_vals = df_top["VIP"].values
        var_names = [self._get_label(v) for v in df_top["Variable"]]

        colors = ["#b91c1c" if v >= threshold else "#64748b" for v in vip_vals]

        bars = ax.barh(y_pos, vip_vals, color=colors, height=0.65, edgecolor="#0f172a", linewidth=0.8)

        # Anotación del valor numérico
        for idx, (val, bar) in enumerate(zip(vip_vals, bars)):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#0f172a"
            )

        ax.axvline(threshold, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Umbral Crítico VIP = {threshold:.1f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Puntaje VIP (Variable Importance in Projection)", fontsize=10, fontweight="bold")
        ax.set_title(f"Determinantes Críticos de Exposición: Ranking VIP (Top {top_n})", fontsize=11.5, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_weights_biplot(
        self,
        w1_idx: int = 0,
        w2_idx: int = 1,
        top_labels: int = 12,
        figsize: Tuple[float, float] = (8.5, 7.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 2: Biplot de pesos PLS (w1 vs w2) con vector de respuesta biológica.
        """
        if self.pls is None or self.pls.model is None:
            raise ValueError("Se requiere un modelo PlsModeler ajustado.")

        W = self.pls.model.x_weights_
        p_features = W.shape[0]
        if W.shape[1] < 2:
            raise ValueError("Se requieren al menos 2 componentes latentes para graficar el Biplot.")

        fig, ax = plt.subplots(figsize=figsize)
        w1 = W[:, w1_idx]
        w2 = W[:, w2_idx]

        # Puntos de covariables
        ax.scatter(w1, w2, color="#0284c7", s=70, edgecolor="#0f172a", linewidth=0.8, alpha=0.85, zorder=3)

        # Vector de respuesta Y
        q = self.pls.model.y_loadings_
        q1 = q[0, w1_idx] if q.ndim > 1 else q[w1_idx]
        q2 = q[0, w2_idx] if q.ndim > 1 else q[w2_idx]
        norm_q = np.sqrt(q1**2 + q2**2)
        if norm_q > 0:
            scale_fac = 0.8 * max(np.max(np.abs(w1)), np.max(np.abs(w2))) / norm_q
            q1_scaled, q2_scaled = q1 * scale_fac, q2 * scale_fac
        else:
            q1_scaled, q2_scaled = 0.2, 0.2

        ax.arrow(0, 0, q1_scaled, q2_scaled, head_width=0.03, head_length=0.04, fc="#b91c1c", ec="#b91c1c", linewidth=2.0, zorder=5)
        ax.text(q1_scaled * 1.15, q2_scaled * 1.15, f"{self.target_label} (Y)", color="#b91c1c", fontsize=10, fontweight="bold", zorder=6)

        # Etiquetar las variables más extremas
        dist = np.sqrt(w1**2 + w2**2)
        top_indices = np.argsort(dist)[-top_labels:]
        for idx in top_indices:
            name = self._get_label(self.pls.feature_names_[idx])
            ax.annotate(
                name,
                (w1[idx], w2[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.85)
            )

        ax.axhline(0, color="#94a3b8", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#94a3b8", linestyle=":", linewidth=1.0)
        ax.set_xlabel(f"Pesos PLS Componente {w1_idx + 1} (w₁)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Pesos PLS Componente {w2_idx + 1} (w₂)", fontsize=10, fontweight="bold")
        ax.set_title(f"Biplot de Pesos PLS-1: Covariables vs {self.target_label}", fontsize=11.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_calibration(
        self,
        y_obs: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        figsize: Tuple[float, float] = (7.5, 6.0),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 3: Curva de calibración: Concentraciones Observadas vs Predichas por el modelo.
        """
        y_o = np.asarray(y_obs, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_o, y_p, color="#047857", s=75, edgecolor="#064e3b", linewidth=0.9, alpha=0.85, zorder=3)

        min_val = min(float(np.min(y_o)), float(np.min(y_p))) * 0.9
        max_val = max(float(np.max(y_o)), float(np.max(y_p))) * 1.1

        # Línea de identidad 1:1
        ax.plot([min_val, max_val], [min_val, max_val], color="#64748b", linestyle="--", linewidth=1.5, label="Identidad 1:1")

        # Regresión OLS observados vs predichos
        slope, intercept, r_val, p_val, _ = stats.linregress(y_o, y_p)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, slope * x_line + intercept, color="#059669", linewidth=2.0, label=f"Ajuste OLS (R² = {r_val**2:.3f})")

        rmse = float(np.sqrt(np.mean((y_o - y_p) ** 2)))
        stats_text = f"R² = {r_val**2:.3f}\nRMSE = {rmse:.3f}\np < 0.001"
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9)
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel(f"{self.target_label} Observado", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{self.target_label} Predicho", fontsize=10, fontweight="bold")
        ax.set_title("Validación Interna del Modelo: Observados vs Predichos", fontsize=11.5, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_impact_quadrants(
        self,
        figsize: Tuple[float, float] = (9.5, 6.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 4: Cuadrantes epidemiológicos de impacto (Coeficiente beta_PLS vs Puntaje VIP).
        """
        if self.pls is None or len(self.pls.vip_) == 0:
            raise ValueError("Se requiere un modelo PlsModeler ajustado.")

        df = self.pls.summary_df.copy()
        fig, ax = plt.subplots(figsize=figsize)

        betas = df["Coef_PLS"].values
        vips = df["VIP"].values

        # Colorear según cuadrante
        colors = []
        for b, v in zip(betas, vips):
            if v >= 1.0 and b > 0:
                colors.append("#b91c1c") # Aporte crítico
            elif v >= 1.0 and b <= 0:
                colors.append("#0284c7") # Atenuación crítica
            else:
                colors.append("#94a3b8") # Secundario

        ax.scatter(betas, vips, c=colors, s=80, edgecolor="#0f172a", linewidth=0.8, alpha=0.85, zorder=3)

        ax.axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1.4, label="Corte Crítico VIP = 1.0")
        ax.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.2)

        # Etiquetar factores críticos
        for i, row in df.iterrows():
            if row["VIP"] >= 1.0:
                name = self._get_label(row["Variable"])
                ax.annotate(
                    name,
                    (row["Coef_PLS"], row["VIP"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#0f172a",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.85)
                )

        ax.set_xlabel("Coeficiente Estandarizado β (Dirección del Efecto)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Importancia de Variable en Proyección (VIP)", fontsize=10, fontweight="bold")
        ax.set_title("Cuadrantes Epidemiológicos de Impacto: Coeficiente β vs Puntaje VIP", fontsize=11.5, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax
