"""
Pantalla modal de confirmación tras finalizar la generación del proyecto.
Muestra un mensaje claro de éxito y las opciones para continuar con otro metal o salir.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class SummaryScreen(ModalScreen):
    """
    Modal de confirmación limpio tras finalizar la generación del proyecto.
    """

    DEFAULT_CSS = """
    SummaryScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.75);
    }

    #summary-dialog {
        width: 66;
        max-width: 90%;
        height: auto;
        background: #1e1e2e;
        border: thick $success;
        padding: 2 3;
    }

    #lbl-sum-icono {
        text-align: center;
        text-style: bold;
        color: $success;
        margin-bottom: 1;
        width: 100%;
    }

    #lbl-sum-mensaje {
        text-align: center;
        color: #cdd6f4;
        margin-bottom: 1;
        width: 100%;
    }

    #lbl-sum-pregunta {
        text-align: center;
        color: #a6adc8;
        text-style: italic;
        margin-bottom: 2;
        width: 100%;
    }

    #box-botones {
        align: center middle;
        height: auto;
        width: 100%;
    }

    #box-botones Button {
        margin: 0 1;
        min-width: 22;
    }
    """

    def __init__(self, resultado=None, **kwargs):
        super().__init__(**kwargs)
        self.resultado = resultado or {}

    def compose(self) -> ComposeResult:
        metal_nom = self.resultado.get("metal", {}).get("nombre", "Metal")
        simbolo = self.resultado.get("metal", {}).get("simbolo", "")
        ya_existia = self.resultado.get("ya_existia", False)

        with Container(id="summary-dialog"):
            if ya_existia:
                yield Label(
                    "[OK] ¡PROYECTO YA GENERADO!",
                    id="lbl-sum-icono"
                )
                yield Label(
                    f"Las carpetas y cuadernos analíticos para [bold cyan]{metal_nom} ({simbolo})[/] ya fueron creados previamente y están listos.",
                    id="lbl-sum-mensaje"
                )
            else:
                yield Label(
                    "[OK] ¡PROYECTO GENERADO CON ÉXITO!",
                    id="lbl-sum-icono"
                )
                yield Label(
                    f"El análisis y los cuadernos para [bold cyan]{metal_nom} ({simbolo})[/] han sido construidos correctamente.",
                    id="lbl-sum-mensaje"
                )
            yield Label(
                "¿Desea generar el análisis para otro metal o salir de la aplicación?",
                id="lbl-sum-pregunta"
            )
            with Horizontal(id="box-botones"):
                yield Button("Generar otro metal", variant="primary", id="btn-otro")
                yield Button("Salir", variant="default", id="btn-salir-sum")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-otro":
            self.dismiss(True)
        elif event.button.id == "btn-salir-sum":
            self.app.exit()
