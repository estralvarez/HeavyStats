"""
Widget para diagnóstico y auditoría en tiempo real del archivo de datos CSV.
"""

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static
from heavystats.studio.filesystem import auditar_dataset, verificar_proyecto_existente


class DataStatusWidget(Static):
    """
    Panel interactivo que informa el estado de salud y completitud del dataset.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.audit = auditar_dataset(cfg) if cfg else None

    def actualizar(self, cfg):
        self.cfg = cfg
        self.audit = auditar_dataset(cfg)
        self.refresh()

    def render(self):
        if not self.audit:
            return Panel("Sin datos auditados.", title="Estado del Dataset")

        tabla = Table.grid(padding=(0, 2))
        tabla.add_column("Aspecto", style="bold")
        tabla.add_column("Detalle", style="white")

        estado = self.audit["estado"]
        if estado == "optimo":
            badge = "[bold green]LISTO PARA INFERENCIA[/]"
            borde = "green"
        elif estado == "incompleto":
            badge = "[bold yellow]INCOMPLETO (0 DATOS EN COLUMNA)[/]"
            borde = "yellow"
        elif estado == "ausente":
            badge = "[bold red]ARCHIVO NO ENCONTRADO[/]"
            borde = "red"
        else:
            badge = "[bold red]ERROR[/]"
            borde = "red"

        tabla.add_row("Estado Analítico:", badge)
        ya_generado = verificar_proyecto_existente(self.cfg) if self.cfg else False
        badge_proyecto = "[bold green]LISTO (CARPETAS CREADAS)[/]" if ya_generado else "[bold yellow]PENDIENTE DE GENERAR[/]"
        tabla.add_row("Estado Proyecto:", badge_proyecto)
        tabla.add_row("Archivo en datos/:", f"[italic]{self.audit['nombre_archivo']}[/]")
        tabla.add_row("Población Total (N):", f"{self.audit['n_total']} participantes")
        tabla.add_row(
            f"Muestra con {self.audit['metal']} (n):",
            f"[bold cyan]{self.audit['n_metal']} determinaciones[/]"
        )
        tabla.add_row("Diagnóstico:", f"{self.audit['mensaje']}")

        return Panel(
            tabla,
            title="[bold] Auditoría de Datos en Tiempo Real [/]",
            border_style=borde
        )
