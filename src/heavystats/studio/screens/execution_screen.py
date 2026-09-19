"""
Pantalla de ejecución con barra de progreso y log en vivo para HeavyStats Studio.
"""

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ProgressBar, RichLog
from heavystats.studio.pipeline import ejecutar_pipeline


class ExecutionScreen(ModalScreen):
    """
    Pantalla modal que muestra el avance en tiempo real de la generación de cuadernos.
    """

    DEFAULT_CSS = """
    ExecutionScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    #exec-dialog {
        width: 85%;
        max-width: 90;
        height: 80%;
        max-height: 28;
        background: #1e1e2e;
        border: thick $primary;
        padding: 1 2;
    }

    #lbl-titulo {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #lbl-status {
        margin-top: 1;
        margin-bottom: 1;
        color: $text;
        text-style: italic;
    }

    #log-console {
        height: 12;
        background: #11111b;
        border: solid #45475a;
        margin-bottom: 1;
    }

    #box-btn-finalizar {
        align: center middle;
        height: auto;
        width: 100%;
        margin-top: 1;
    }

    #btn-finalizar {
        min-width: 28;
        display: none;
    }
    """

    def __init__(self, cfg, on_complete_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.on_complete_callback = on_complete_callback
        self.resultado = None

    def compose(self) -> ComposeResult:
        with Container(id="exec-dialog"):
            yield Label(
                f"Generando Proyecto y Cuadernos: {self.cfg['nombre']} ({self.cfg['simbolo']})",
                id="lbl-titulo"
            )
            yield ProgressBar(total=8, show_percentage=True, id="pbar")
            yield Label("Iniciando pipeline analítico...", id="lbl-status")
            yield RichLog(id="log-console", highlight=True, markup=True)
            with Horizontal(id="box-btn-finalizar"):
                yield Button("Continuar", variant="success", id="btn-finalizar")

    def on_mount(self) -> None:
        self.ejecutar_tarea()

    @work(thread=True)
    def ejecutar_tarea(self) -> None:
        """Worker en hilo secundario para evitar congelar el loop de eventos de la TUI."""
        def callback(paso, total, mensaje):
            self.app.call_from_thread(self._actualizar_progreso, paso, total, mensaje)

        try:
            self.resultado = ejecutar_pipeline(self.cfg, progress_callback=callback)
            self.app.call_from_thread(self._notificar_exito)
        except Exception as e:
            self.app.call_from_thread(self._notificar_error, str(e))

    def _actualizar_progreso(self, paso: int, total: int, mensaje: str) -> None:
        pbar = self.query_one("#pbar", ProgressBar)
        lbl = self.query_one("#lbl-status", Label)
        log = self.query_one("#log-console", RichLog)

        pbar.update(progress=paso)
        lbl.update(f"Paso {paso}/{total}: {mensaje}")
        log.write(f"[bold cyan][{paso}/{total}][/] {mensaje}")

    def _notificar_exito(self) -> None:
        lbl = self.query_one("#lbl-status", Label)
        log = self.query_one("#log-console", RichLog)
        btn = self.query_one("#btn-finalizar", Button)

        lbl.update("[bold green]Generación completada con 100% de éxito![/]")
        log.write("\n[bold green][OK] Todos los cuadernos (.ipynb) han sido construidos.[/]")
        log.write("[bold green][OK] Directorios de salidas listos para reportes y gráficos.[/]")
        btn.styles.display = "block"
        btn.focus()

    def _notificar_error(self, err_msg: str) -> None:
        lbl = self.query_one("#lbl-status", Label)
        log = self.query_one("#log-console", RichLog)
        btn = self.query_one("#btn-finalizar", Button)

        lbl.update("[bold red]Ocurrió un error en la generación.[/]")
        log.write(f"\n[bold red]ERROR:[/] {err_msg}")
        btn.label = "Cerrar"
        btn.variant = "error"
        btn.styles.display = "block"
        btn.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-finalizar":
            self.dismiss(self.resultado)
