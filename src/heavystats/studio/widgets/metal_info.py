"""
Widget para visualización de parámetros toxicológicos y normativos del metal activo.
"""

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static


class MetalInfoWidget(Static):
    """
    Panel reactivo con la ficha técnica toxicológica del metal seleccionado.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def actualizar(self, cfg):
        self.cfg = cfg
        self.refresh()

    def render(self):
        if not self.cfg:
            return Panel("Ningún metal seleccionado.", title="Ficha Técnica")

        tabla = Table.grid(padding=(0, 2))
        tabla.add_column("Propiedad", style="bold cyan")
        tabla.add_column("Valor", style="white")

        tabla.add_row("Metal Analítico:", f"{self.cfg['nombre']} ({self.cfg['simbolo']})")
        tabla.add_row("Unidad de Medida:", f"{self.cfg['unidad']}")
        tabla.add_row("Metal Analizado:", f"[bold green]{self.cfg['col_concentracion']}[/]")
        tabla.add_row("Variable de Riesgo:", f"{self.cfg['col_riesgo']}")
        tabla.add_row("Límite de Riesgo:", f"[bold red]{self.cfg['limite_permisible']} {self.cfg['unidad']}[/]")
        tabla.add_row("Referencia Internacional:", f"{self.cfg['entidad_referencia']}")
        tabla.add_row("LOD Instrumental:", f"{self.cfg['lod']}")
        tabla.add_row("Dataset Destino:", f"datos/{self.cfg['archivo_datos']}")

        return Panel(
            tabla,
            title=f"[bold {self.cfg.get('color', 'cyan')}] Parámetros Toxicológicos: {self.cfg['nombre']} [/]",
            border_style=self.cfg.get("color", "cyan")
        )
