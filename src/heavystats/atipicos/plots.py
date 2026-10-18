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
        box_kwargs = dict(
            patch_artist=True,
            boxprops=dict(facecolor="#bae6fd", color="#0284c7", linewidth=1.5),
            whiskerprops=dict(color="#0284c7", linewidth=1.5, linestyle="--"),
            capprops=dict(color="#0284c7", linewidth=1.5),
            medianprops=dict(color="#0369a1", linewidth=2.0),
            showfliers=False,
        )
        try:
            bp = ax_box.boxplot(y, orientation="horizontal", **box_kwargs)
        except TypeError:
            bp = ax_box.boxplot(y, vert=False, **box_kwargs)
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

    def _get_target_label(self, metal_label: Optional[str] = None) -> str:
        if metal_label:
            return metal_label
        tc = self.diag.target_col
        tc_lower = tc.lower()
        if "mercurio" in tc_lower or "hg" in tc_lower:
            return r"[$\mathrm{Hg}$] Plasmático ($\mu\mathrm{g/L}$)"
        elif "plomo" in tc_lower or "pb" in tc_lower:
            return r"[$\mathrm{Pb}$] Sanguíneo ($\mu\mathrm{g/dL}$)"
        elif "cadmio" in tc_lower or "cd" in tc_lower:
            return r"[$\mathrm{Cd}$] Sanguíneo ($\mu\mathrm{g/L}$)"
        clean = tc.replace("_ug_L", "").replace("_ug_dL", "").replace("_", " ")
        return f"[{clean}] (µg/L)"

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
        t1 = T[:, 0]
        t2 = T[:, 1] if self.diag.k_comp > 1 else np.zeros_like(t1)

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
        cbar.set_label(self._get_target_label(), fontsize=9.5, fontweight="bold")

        # Headroom en ejes
        max_t1, min_t1 = float(np.max(t1)), float(np.min(t1))
        span_t1 = max(1.0, max_t1 - min_t1)
        ax.set_xlim(min_t1 - span_t1 * 0.12, max_t1 + span_t1 * 0.12)

        if self.diag.k_comp > 1:
            max_t2, min_t2 = float(np.max(t2)), float(np.min(t2))
            span_t2 = max(1.0, max_t2 - min_t2)
            ax.set_ylim(min_t2 - span_t2 * 0.12, max_t2 + span_t2 * 0.12)

        # Identificar y etiquetar atípicos o casos influyentes
        t2_out = self.diag.T2_Hotelling > self.diag.T2_limit
        for i, sid in enumerate(self.diag.ids):
            if t2_out[i] or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                x_off = -22 if t1[i] > max_t1 - span_t1 * 0.25 else 7
                y_off = -16 if (self.diag.k_comp > 1 and t2[i] > max_t2 - span_t2 * 0.25) else 7
                ax.annotate(
                    sid,
                    (t1[i], t2[i]),
                    xytext=(x_off, y_off),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.92),
                    zorder=5
                )

        ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#64748b", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"Puntuación Latente 1 ($t_1$)", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Puntuación Latente 2 ($t_2$)", fontsize=10, fontweight="bold")
        ax.set_title(r"Espacio Latente PLS-1: Puntuaciones ($t_1$ vs $t_2$)", fontsize=11, fontweight="bold")
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
        cbar.set_label(self._get_target_label(), fontsize=9.5, fontweight="bold")

        ax.axvline(self.diag.T2_limit, color="#b91c1c", linestyle="--", linewidth=1.5, label=rf"Límite Hotelling $T^2$ 95% ({self.diag.T2_limit:.2f})")
        ax.axhline(self.diag.Q_limit, color="#d97706", linestyle="--", linewidth=1.5, label=rf"Límite Residual $Q$ 95% ({self.diag.Q_limit:.2f})")

        # Headroom en ejes
        max_t2 = max(float(np.max(self.diag.T2_Hotelling)), self.diag.T2_limit)
        max_q = max(float(np.max(self.diag.Q_residuals)), self.diag.Q_limit)
        ax.set_xlim(-0.02 * max_t2, max_t2 * 1.18)
        ax.set_ylim(-0.02 * max_q, max_q * 1.20)

        # Etiquetar muestras en cuadrantes atípicos con offsets inteligentes
        for i, sid in enumerate(self.diag.ids):
            t2_val = self.diag.T2_Hotelling[i]
            q_val = self.diag.Q_residuals[i]
            if t2_val > self.diag.T2_limit or q_val > self.diag.Q_limit:
                x_off = -24 if t2_val > max_t2 * 0.75 else 7
                y_off = -18 if q_val > max_q * 0.75 else 7
                ax.annotate(
                    sid,
                    (t2_val, q_val),
                    xytext=(x_off, y_off),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.92),
                    zorder=5
                )

        ax.set_xlabel(r"Distancia de Hotelling $T^2$", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Residuales $Q$ en Predictores ($\mathrm{SPE}$)", fontsize=10, fontweight="bold")
        ax.set_title(r"Diagrama de Influencia Quimiométrica ($T^2$ vs $Q$)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")
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
        cbar.set_label(self._get_target_label(), fontsize=9.5, fontweight="bold")

        ax.axvline(
            self.diag.leverage_limit,
            color="#d97706",
            linestyle="--",
            linewidth=1.5,
            label=rf"Límite Apalancamiento $h^* = 2p/n$ ({self.diag.leverage_limit:.2f})"
        )
        ax.axhline(
            self.diag.cooks_threshold,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.5,
            label=rf"Umbral Influencia Cook $D^* = 4/n$ ({self.diag.cooks_threshold:.2f})"
        )

        # Headroom para evitar colisión de puntos extremos con bordes y leyenda
        max_lev = max(float(np.max(self.diag.leverage)), self.diag.leverage_limit)
        max_cook = max(float(np.max(self.diag.cooks_D)), self.diag.cooks_threshold)
        ax.set_xlim(-0.015, max_lev * 1.18)
        ax.set_ylim(-0.015, max_cook * 1.20)

        for i, sid in enumerate(self.diag.ids):
            lev_i = self.diag.leverage[i]
            ck_i = self.diag.cooks_D[i]
            if (lev_i > self.diag.leverage_limit) or (ck_i > self.diag.cooks_threshold):
                # Desplazar hacia la izquierda/abajo si el punto está cerca del extremo superior o derecho
                x_off = -24 if lev_i > max_lev * 0.75 else 7
                y_off = -18 if ck_i > max_cook * 0.75 else 7
                ax.annotate(
                    sid,
                    (lev_i, ck_i),
                    xytext=(x_off, y_off),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b91c1c",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.92),
                    zorder=5
                )

        ax.set_xlabel(r"Apalancamiento de la Muestra ($h_{ii}$)", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Distancia de Cook ($D_i$)", fontsize=10, fontweight="bold")
        ax.set_title(r"Diagrama de Apalancamiento ($h_{ii}$) vs Distancia de Cook ($D_i$)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")
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
        lbl = self._get_target_label(metal_label)
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
        ax_a.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")
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
        t1 = T[:, 0]
        t2 = T[:, 1] if self.diag.k_comp > 1 else np.zeros_like(T[:, 0])
        sc_b = ax_b.scatter(t1, t2, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        
        max_t1_b = float(np.max(t1))
        span_t1_b = max(1.0, max_t1_b - float(np.min(t1)))
        for i, sid in enumerate(self.diag.ids):
            if (self.diag.T2_Hotelling[i] > self.diag.T2_limit) or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                x_off = -22 if t1[i] > max_t1_b - span_t1_b * 0.25 else 5
                ax_b.annotate(sid, (t1[i], t2[i]), xytext=(x_off, 5), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_b.axhline(0, color="#64748b", linestyle=":", linewidth=0.9)
        ax_b.axvline(0, color="#64748b", linestyle=":", linewidth=0.9)
        ax_b.set_title(r"(B) Espacio de Puntuaciones Latentes PLS-1 ($t_1$ vs $t_2$)", fontsize=11, fontweight="bold")
        ax_b.set_xlabel(r"Puntuación Latente 1 ($t_1$)", fontsize=9.5, fontweight="bold")
        ax_b.set_ylabel(r"Puntuación Latente 2 ($t_2$)", fontsize=9.5, fontweight="bold")
        ax_b.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_b)

        # Panel (C): Influencia
        ax_c = axes[1, 0]
        sc_c = ax_c.scatter(self.diag.T2_Hotelling, self.diag.Q_residuals, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        ax_c.axvline(self.diag.T2_limit, color="#b91c1c", linestyle="--", linewidth=1.4, label=rf"Límite $T^2$ ({self.diag.T2_limit:.2f})")
        ax_c.axhline(self.diag.Q_limit, color="#d97706", linestyle="--", linewidth=1.4, label=rf"Límite $Q$ ({self.diag.Q_limit:.2f})")
        
        max_t2_c = max(float(np.max(self.diag.T2_Hotelling)), self.diag.T2_limit)
        max_q_c = max(float(np.max(self.diag.Q_residuals)), self.diag.Q_limit)
        ax_c.set_xlim(-0.02 * max_t2_c, max_t2_c * 1.18)
        ax_c.set_ylim(-0.02 * max_q_c, max_q_c * 1.20)

        for i, sid in enumerate(self.diag.ids):
            if (self.diag.T2_Hotelling[i] > self.diag.T2_limit) or (self.diag.Q_residuals[i] > self.diag.Q_limit):
                x_off = -22 if self.diag.T2_Hotelling[i] > max_t2_c * 0.75 else 5
                y_off = -16 if self.diag.Q_residuals[i] > max_q_c * 0.75 else 5
                ax_c.annotate(sid, (self.diag.T2_Hotelling[i], self.diag.Q_residuals[i]), xytext=(x_off, y_off), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_c.set_title(r"(C) Diagrama de Influencia Quimiométrica ($T^2$ vs $Q$)", fontsize=11, fontweight="bold")
        ax_c.set_xlabel(r"Distancia de Hotelling $T^2$", fontsize=9.5, fontweight="bold")
        ax_c.set_ylabel(r"Residuales $Q$ ($\mathrm{SPE}$)", fontsize=9.5, fontweight="bold")
        ax_c.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")
        ax_c.grid(True, linestyle=":", alpha=0.5)
        sns.despine(ax=ax_c)

        # Panel (D): Apalancamiento vs Cook
        ax_d = axes[1, 1]
        sc_d = ax_d.scatter(self.diag.leverage, self.diag.cooks_D, c=self.diag.y, cmap="viridis", s=75, edgecolors="#0f172a", linewidths=0.9, alpha=0.9)
        ax_d.axvline(self.diag.leverage_limit, color="#d97706", linestyle="--", linewidth=1.4, label=rf"Límite $2p/n$ ({self.diag.leverage_limit:.2f})")
        ax_d.axhline(self.diag.cooks_threshold, color="#b91c1c", linestyle="--", linewidth=1.4, label=rf"Umbral Cook ({self.diag.cooks_threshold:.2f})")
        
        max_lev_d = max(float(np.max(self.diag.leverage)), self.diag.leverage_limit)
        max_cook_d = max(float(np.max(self.diag.cooks_D)), self.diag.cooks_threshold)
        ax_d.set_xlim(-0.015, max_lev_d * 1.18)
        ax_d.set_ylim(-0.015, max_cook_d * 1.20)

        for i, sid in enumerate(self.diag.ids):
            if (self.diag.leverage[i] > self.diag.leverage_limit) or (self.diag.cooks_D[i] > self.diag.cooks_threshold):
                x_off = -22 if self.diag.leverage[i] > max_lev_d * 0.75 else 5
                y_off = -16 if self.diag.cooks_D[i] > max_cook_d * 0.75 else 5
                ax_d.annotate(sid, (self.diag.leverage[i], self.diag.cooks_D[i]), xytext=(x_off, y_off), textcoords="offset points", fontsize=8, fontweight="bold", color="#b91c1c")
        ax_d.set_title(r"(D) Diagrama de Apalancamiento ($h_{ii}$) vs Distancia de Cook ($D_i$)", fontsize=11, fontweight="bold")
        ax_d.set_xlabel(r"Apalancamiento de la Muestra ($h_{ii}$)", fontsize=9.5, fontweight="bold")
        ax_d.set_ylabel(r"Distancia de Cook ($D_i$)", fontsize=9.5, fontweight="bold")
        ax_d.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")
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
