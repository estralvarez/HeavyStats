"""
Visualizaciones gráficas científicas y quimiométricas de diagnóstico de valores atípicos.
Genera figuras a 300 DPI con estándares editoriales (FACYT / Universidad de Carabobo).
"""

from typing import Optional, Tuple, Any, List, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from heavystats.atipicos.diagnostics import OutlierDiagnostics


class OutlierPlots:
    """
    Generador de figuras científicas para el diagnóstico de valores atípicos y puntos palanca.
    """

    def __init__(self, diagnostics: OutlierDiagnostics):
        """
        Parameters
        ----------
        diagnostics : OutlierDiagnostics
            Instancia configurada y ajustada del motor de diagnóstico.
        """
        self.diag = diagnostics

    def _save_fig(self, fig: plt.Figure, filepath: Optional[Union[str, Path]]) -> None:
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")

    def plot_univariate_fences(
        self,
        metal_label: Optional[str] = None,
        figsize: Tuple[float, float] = (10.0, 6.0),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 1: Distribución univariante compuesta (Boxplot horizontal + Densidad KDE)
        con vallas de Tukey (1.5 IQR), mediana y contraste de límites toxicológicos.
        """
        lbl = metal_label or f"[{self.diag.target_col}] (µg/L)"
        y = self.diag.y

        fig, (ax_box, ax_dist) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [0.32, 0.68]}
        )

        # 1. Boxplot Superior
        bp = ax_box.boxplot(
            y,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#bae6fd", color="#0284c7", linewidth=1.5),
            whiskerprops=dict(color="#0284c7", linewidth=1.5, linestyle="--"),
            capprops=dict(color="#0284c7", linewidth=1.5),
            medianprops=dict(color="#0369a1", linewidth=2.0),
            showfliers=False,
        )
        sns.stripplot(
            x=y,
            ax=ax_box,
            color="#0f172a",
            size=6,
            jitter=0.15,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_box.axvline(self.diag.tukey_upper, color="#ef4444", linestyle=":", linewidth=1.6, label="Valla Sup. Tukey (Q3 + 1.5 IQR)")
        ax_box.set_yticks([])
        ax_box.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax_box.set_title(f"Diagnóstico Univariante y Vallas de Tukey: {lbl} (N = {self.diag.n_obs})", fontsize=11, fontweight="bold", pad=8)
        sns.despine(ax=ax_box, left=True)

        # 2. Histograma y Densidad KDE Inferior
        sns.histplot(
            y,
            kde=True,
            ax=ax_dist,
            color="#0284c7",
            bins=8,
            stat="density",
            alpha=0.35,
            edgecolor="#0284c7",
            line_kws=dict(linewidth=2.0),
        )
        ax_dist.axvline(self.diag.mean, color="#0f172a", linestyle="-", linewidth=1.5, label=f"Media ({self.diag.mean:.2f})")
        ax_dist.axvline(self.diag.median, color="#0284c7", linestyle="--", linewidth=1.5, label=f"Mediana ({self.diag.median:.2f})")
        ax_dist.axvline(self.diag.tukey_upper, color="#ef4444", linestyle=":", linewidth=1.6, label=f"Valla Sup. ({self.diag.tukey_upper:.2f})")

        if self.diag.who_threshold:
            ax_dist.axvline(self.diag.who_threshold, color="#b91c1c", linestyle="-.", linewidth=1.6, label=f"Límite OMS ({self.diag.who_threshold:.1f})")

        ax_dist.set_xlabel(lbl, fontsize=10, fontweight="bold")
        ax_dist.set_ylabel("Densidad de Probabilidad", fontsize=10, fontweight="bold")
        ax_dist.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax_dist.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_dist)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, (ax_box, ax_dist)

    def plot_latent_scores(
        self,
        figsize: Tuple[float, float] = (7.0, 5.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 2: Espacio de puntuaciones latentes (t1 vs t2) con elipse de confianza de Hotelling T².
        """
        if not getattr(self.diag, "_multivariate_computed", False):
            raise RuntimeError("Se requieren predictores multivariantes para graficar el espacio latente.")

        fig, ax = plt.subplots(figsize=figsize)
        T = self.diag.scores_T
        t1, t2 = T[:, 0], T[:, 1] if self.diag.k_comp > 1 else np.zeros_like(t1)

        # Dispersión con color según concentración
        sc = ax.scatter(
            t1, t2,
            c=self.diag.y,
            cmap="viridis",
            s=80,
            edgecolors="#0f172a",
            linewidths=1.0,
            alpha=0.9,
            zorder=4
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"[{self.diag.target_col}] (µg/L)", fontsize=9.5, fontweight="bold")

        # Identificar y etiquetar atípicos o casos influyentes
        t2_out = self.diag.T2_Hotelling > self.diag.T2_limit
        for i, sid in enumerate(self.diag.ids):
            if t2_out[i] or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                ax.annotate(
                    sid,
                    (t1[i], t2[i]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
                    zorder=5
                )

        ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#64748b", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Puntuación Latente 1 (t₁)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Puntuación Latente 2 (t₂)", fontsize=10, fontweight="bold")
        ax.set_title("Espacio Latente PLS-1: Puntuaciones (t₁ vs t₂)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_influence_space(
        self,
        figsize: Tuple[float, float] = (7.5, 5.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 3: Diagrama de Influencia Quimiométrica (Hotelling T² vs Residuales Q / SPE).
        """
        if not getattr(self.diag, "_multivariate_computed", False):
            raise RuntimeError("Se requieren predictores multivariantes para graficar la influencia.")

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            self.diag.T2_Hotelling,
            self.diag.Q_residuals,
            c=self.diag.y,
            cmap="viridis",
            s=80,
            edgecolors="#0f172a",
            linewidths=1.0,
            alpha=0.9,
            zorder=4
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"[{self.diag.target_col}] (µg/L)", fontsize=9.5, fontweight="bold")

        ax.axvline(self.diag.T2_limit, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Límite T² 95% ({self.diag.T2_limit:.2f})")
        ax.axhline(self.diag.Q_limit, color="#d97706", linestyle="--", linewidth=1.5, label=f"Límite Q 95% ({self.diag.Q_limit:.2f})")

        # Etiquetar muestras en cuadrantes atípicos
        for i, sid in enumerate(self.diag.ids):
            if self.diag.T2_Hotelling[i] > self.diag.T2_limit or self.diag.Q_residuals[i] > self.diag.Q_limit:
                ax.annotate(
                    sid,
                    (self.diag.T2_Hotelling[i], self.diag.Q_residuals[i]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
                    zorder=5
                )

        ax.set_xlabel("Distancia de Hotelling T²", fontsize=10, fontweight="bold")
        ax.set_ylabel("Residuales Q en Predictores (SPE)", fontsize=10, fontweight="bold")
        ax.set_title("Diagrama de Influencia Quimiométrica (T² vs Q)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_leverage_cook(
        self,
        figsize: Tuple[float, float] = (7.5, 5.5),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Figura 4: Diagrama de Apalancamiento (Leverage hii) vs Distancia de Cook (Di).
        """
        if not getattr(self.diag, "_multivariate_computed", False):
            raise RuntimeError("Se requieren predictores multivariantes para graficar apalancamiento vs Cook.")

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            self.diag.leverage,
            self.diag.cooks_D,
            c=self.diag.y,
            cmap="viridis",
            s=85,
            edgecolors="#0f172a",
            linewidths=1.0,
            alpha=0.9,
            zorder=4
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"[{self.diag.target_col}] (µg/L)", fontsize=9.5, fontweight="bold")

        ax.axvline(self.diag.leverage_limit, color="#d97706", linestyle="--", linewidth=1.5, label=f"Límite Leverage 2p/n ({self.diag.leverage_limit:.2f})")
        ax.axhline(self.diag.cooks_threshold, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Umbral Cook ({self.diag.cooks_threshold:.2f})")

        for i, sid in enumerate(self.diag.ids):
            if (self.diag.leverage[i] > self.diag.leverage_limit) or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                ax.annotate(
                    sid,
                    (self.diag.leverage[i], self.diag.cooks_D[i]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
                    zorder=5
                )

        ax.set_xlabel("Apalancamiento de la Muestra (Leverage h_ii)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Distancia de Cook (D_i)", fontsize=10, fontweight="bold")
        ax.set_title("Diagrama de Apalancamiento vs Distancia de Cook", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()
        self._save_fig(fig, filepath)
        return fig, ax

    def plot_dashboard_2x2(
        self,
        metal_label: Optional[str] = None,
        figsize: Tuple[float, float] = (14.0, 11.0),
        filepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, Any]:
        """
        Dashboard consolidado de 4 paneles (2x2) a 300 DPI:
        (A) Boxplot + Distribución y Valla de Tukey.
        (B) Espacio de Puntuaciones Latentes (t1 vs t2).
        (C) Diagrama de Influencia (T² de Hotelling vs Residuales Q).
        (D) Diagrama de Apalancamiento vs Distancia de Cook.
        """
        lbl = metal_label or f"[{self.diag.target_col}] (µg/L)"
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Panel (A): Univariante
        ax_a = axes[0, 0]
        sns.histplot(
            self.diag.y,
            kde=True,
            ax=ax_a,
            color="#0284c7",
            bins=8,
            stat="density",
            alpha=0.35,
            edgecolor="#0284c7"
        )
        ax_a.axvline(self.diag.mean, color="#0f172a", linestyle="-", linewidth=1.5, label=f"Media ({self.diag.mean:.2f})")
        ax_a.axvline(self.diag.median, color="#0284c7", linestyle="--", linewidth=1.5, label=f"Mediana ({self.diag.median:.2f})")
        ax_a.axvline(self.diag.tukey_upper, color="#ef4444", linestyle=":", linewidth=1.6, label=f"Valla Sup. ({self.diag.tukey_upper:.2f})")
        if self.diag.who_threshold:
            ax_a.axvline(self.diag.who_threshold, color="#b91c1c", linestyle="-.", linewidth=1.6, label=f"Límite OMS ({self.diag.who_threshold:.1f})")
        ax_a.set_title("(A) Distribución Univariante y Vallas de Tukey", fontsize=11, fontweight="bold")
        ax_a.set_xlabel(lbl, fontsize=9.5, fontweight="bold")
        ax_a.set_ylabel("Densidad", fontsize=9.5, fontweight="bold")
        ax_a.legend(loc="upper right", fontsize=8, frameon=True)
        ax_a.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_a)

        # Si no hay multivariante, ocultar los otros 3
        if not getattr(self.diag, "_multivariate_computed", False):
            for r in [0, 1]:
                for c in [0, 1]:
                    if (r, c) != (0, 0):
                        axes[r, c].set_visible(False)
            plt.tight_layout()
            self._save_fig(fig, filepath)
            return fig, axes

        # Panel (B): Espacio Latente
        ax_b = axes[0, 1]
        T = self.diag.scores_T
        t1, t2 = T[:, 0], T[:, 1] if self.diag.k_comp > 1 else np.zeros_like(T[:, 0])
        sc_b = ax_b.scatter(t1, t2, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        for i, sid in enumerate(self.diag.ids):
            if (self.diag.T2_Hotelling[i] > self.diag.T2_limit) or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                ax_b.annotate(sid, (t1[i], t2[i]), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_b.axhline(0, color="#64748b", linestyle=":", linewidth=0.9)
        ax_b.axvline(0, color="#64748b", linestyle=":", linewidth=0.9)
        ax_b.set_title("(B) Espacio de Puntuaciones Latentes PLS-1 (t₁ vs t₂)", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Puntuación Latente 1 (t₁)", fontsize=9.5, fontweight="bold")
        ax_b.set_ylabel("Puntuación Latente 2 (t₂)", fontsize=9.5, fontweight="bold")
        ax_b.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_b)

        # Panel (C): Influencia
        ax_c = axes[1, 0]
        sc_c = ax_c.scatter(self.diag.T2_Hotelling, self.diag.Q_residuals, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        ax_c.axvline(self.diag.T2_limit, color="#b91c1c", linestyle="--", linewidth=1.4, label=f"Límite T² ({self.diag.T2_limit:.2f})")
        ax_c.axhline(self.diag.Q_limit, color="#d97706", linestyle="--", linewidth=1.4, label=f"Límite Q ({self.diag.Q_limit:.2f})")
        for i, sid in enumerate(self.diag.ids):
            if (self.diag.T2_Hotelling[i] > self.diag.T2_limit) or (self.diag.Q_residuals[i] > self.diag.Q_limit):
                ax_c.annotate(sid, (self.diag.T2_Hotelling[i], self.diag.Q_residuals[i]), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_c.set_title("(C) Diagrama de Influencia Quimiométrica (T² vs Q)", fontsize=11, fontweight="bold")
        ax_c.set_xlabel("Distancia de Hotelling T²", fontsize=9.5, fontweight="bold")
        ax_c.set_ylabel("Residuales Q (SPE)", fontsize=9.5, fontweight="bold")
        ax_c.legend(loc="upper right", fontsize=8, frameon=True)
        ax_c.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_c)

        # Panel (D): Apalancamiento vs Cook
        ax_d = axes[1, 1]
        sc_d = ax_d.scatter(self.diag.leverage, self.diag.cooks_D, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        ax_d.axvline(self.diag.leverage_limit, color="#d97706", linestyle="--", linewidth=1.4, label=f"Límite 2p/n ({self.diag.leverage_limit:.2f})")
        ax_d.axhline(self.diag.cooks_threshold, color="#b91c1c", linestyle="--", linewidth=1.4, label=f"Umbral Cook ({self.diag.cooks_threshold:.2f})")
        for i, sid in enumerate(self.diag.ids):
            if (self.diag.leverage[i] > self.diag.leverage_limit) or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                ax_d.annotate(sid, (self.diag.leverage[i], self.diag.cooks_D[i]), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_d.set_title("(D) Diagrama de Apalancamiento vs Distancia de Cook", fontsize=11, fontweight="bold")
        ax_d.set_xlabel("Apalancamiento (Leverage h_ii)", fontsize=9.5, fontweight="bold")
        ax_d.set_ylabel("Distancia de Cook (D_i)", fontsize=9.5, fontweight="bold")
        ax_d.legend(loc="upper right", fontsize=8, frameon=True)
        ax_d.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_d)

        # Barra de color común para concentración
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
        cb = fig.colorbar(sc_d, cax=cbar_ax)
        cb.set_label(lbl, fontsize=10, fontweight="bold")

        fig.suptitle(f"Dashboard Integral de Diagnóstico Quimiométrico y Valores Atípicos: {self.diag.target_col}", fontsize=13, fontweight="bold", y=0.98)
        self._save_fig(fig, filepath)
        return fig, axes
