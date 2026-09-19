"""
Pantalla principal de HeavyStats Studio: Selección interactiva de metal y auditoría.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, RadioButton, RadioSet, Static
from heavystats.studio.config import METALES_CONFIG, DEFAULT_METAL
from heavystats.studio.filesystem import verificar_proyecto_existente
from heavystats.studio.widgets.data_status import DataStatusWidget
from heavystats.studio.widgets.metal_info import MetalInfoWidget
from heavystats.studio.screens.execution_screen import ExecutionScreen
from heavystats.studio.screens.summary_screen import SummaryScreen


class MainScreen(Screen):
    """
    Pantalla interactiva de bienvenida y selección toxicológica.
    """

    BINDINGS = [
        ("1", "select_metal('plomo')", "Plomo (Pb)"),
        ("2", "select_metal('mercurio')", "Mercurio (Hg)"),
        ("3", "select_metal('cadmio')", "Cadmio (Cd)"),
        ("enter", "generar_proyecto", "Generar Proyecto"),
        ("f5", "refrescar_auditoria", "Re-auditar Datos"),
        ("q", "app.quit", "Salir"),
    ]

    DEFAULT_CSS = """
    MainScreen {
        background: #11111b;
        color: #cdd6f4;
    }

    #header-banner {
        height: auto;
        background: #1e1e2e;
        border-bottom: heavy $accent;
        padding: 0 1;
        text-align: center;
    }

    #banner-titulo {
        text-style: bold;
        color: $accent;
    }

    #banner-subtitulo {
        color: #a6adc8;
        text-style: italic;
    }

    #main-container {
        padding: 0 1;
        height: 1fr;
    }

    #cols-container {
        height: 1fr;
    }

    #left-column {
        width: 48%;
        height: 1fr;
        margin-right: 1;
    }

    #right-column {
        width: 52%;
        height: 1fr;
    }

    #box-selector {
        height: auto;
        background: #181825;
        border: solid #45475a;
        padding: 0 1;
        margin-bottom: 1;
    }

    #lbl-selector-title {
        text-style: bold;
        color: $primary;
        margin-top: 0;
        margin-bottom: 0;
    }

    RadioSet {
        height: auto;
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }

    RadioButton {
        height: 1;
        padding: 0 1;
        margin: 0;
    }

    #widget-metal-info {
        height: auto;
    }

    #widget-data-status {
        height: auto;
        margin-bottom: 1;
    }

    #box-cuadernos {
        height: auto;
        background: #181825;
        border: solid #45475a;
        padding: 0 1;
    }

    #lbl-cuadernos-title {
        text-style: bold;
        color: $accent;
        margin-top: 0;
        margin-bottom: 0;
    }

    .item-cuaderno {
        color: #bac2de;
        padding-left: 1;
    }

    #actions-bar {
        height: 3;
        margin-top: 0;
        align: center middle;
    }

    #actions-bar Button {
        margin: 0 1;
    }

    #btn-generar {
        min-width: 32;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metal_actual = METALES_CONFIG[DEFAULT_METAL]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="header-banner"):
            yield Label("HEAVYSTATS STUDIO — BIOMONITOREO DE METALES PESADOS", id="banner-titulo")
            yield Label("Generador Automatizado de Proyectos Bioestadísticos y Epidemiológicos", id="banner-subtitulo")

        with Container(id="main-container"):
            with Horizontal(id="cols-container"):
                # Columna Izquierda: Selector de Metal + Ficha Técnica
                with Vertical(id="left-column"):
                    with Container(id="box-selector"):
                        yield Label("1. Seleccione el Metal a Analizar:", id="lbl-selector-title")
                        with RadioSet(id="radio-metales"):
                            yield RadioButton(
                                "Plomo (Pb, µg/dL) — CDC BLRV: 3.5 [Por defecto]",
                                value=True,
                                id="rb-plomo"
                            )
                            yield RadioButton(
                                "Mercurio (Hg, µg/L) — EPA/OMS: 5.0",
                                value=False,
                                id="rb-mercurio"
                            )
                            yield RadioButton(
                                "Cadmio (Cd, µg/L) — OMS/ATSDR: 1.0",
                                value=False,
                                id="rb-cadmio"
                            )
                    yield MetalInfoWidget(cfg=self.metal_actual, id="widget-metal-info")

                # Columna Derecha: Auditoría de Datos + Lista de Cuadernos
                with Vertical(id="right-column"):
                    yield DataStatusWidget(cfg=self.metal_actual, id="widget-data-status")
                    with Container(id="box-cuadernos"):
                        yield Label("2. Cuadernos que se Construirán Automáticamente:", id="lbl-cuadernos-title")
                        yield Label("• validacion.ipynb (Auditoría de Datos)", classes="item-cuaderno")
                        yield Label("• filtros.ipynb (Preprocesamiento de muestra analítica)", classes="item-cuaderno")
                        yield Label("• univariante/01_tablas.ipynb (Resumen Estadístico Toxicológico)", classes="item-cuaderno")
                        yield Label("• univariante/02_graficos.ipynb (Graficos Estadísticos Descriptivos)", classes="item-cuaderno")
                        yield Label("• bivariante/01_analisis_estadistico.ipynb (Analisis Estadístico Bivariante)", classes="item-cuaderno")
                        yield Label("• bivariante/02_graficos_bivariantes.ipynb (Graficos Estadísticos Bivariantes)", classes="item-cuaderno")

            with Horizontal(id="actions-bar"):
                yield Button("Generar Proyecto y Cuadernos (Enter)", variant="primary", id="btn-generar")
                yield Button("Re-auditar Datos (F5)", variant="default", id="btn-auditar")
                yield Button("Salir (q)", variant="error", id="btn-salir")

        yield Footer()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Sincroniza la selección visual de metales con los paneles informativos."""
        idx = event.radio_set.pressed_index
        mapa = {0: "plomo", 1: "mercurio", 2: "cadmio"}
        metal_key = mapa.get(idx, "plomo")
        self._cambiar_metal(metal_key)

    def action_select_metal(self, metal_key: str) -> None:
        """Atajo de teclado numérico (1=Pb, 2=Hg, 3=Cd)."""
        try:
            rb = self.query_one(f"#rb-{metal_key}", RadioButton)
            rb.value = True
        except Exception:
            pass
        self._cambiar_metal(metal_key)

    def _cambiar_metal(self, metal_key: str) -> None:
        self.metal_actual = METALES_CONFIG[metal_key]
        self.query_one("#widget-metal-info", MetalInfoWidget).actualizar(self.metal_actual)
        self.query_one("#widget-data-status", DataStatusWidget).actualizar(self.metal_actual)

    def action_refrescar_auditoria(self) -> None:
        self.query_one("#widget-data-status", DataStatusWidget).actualizar(self.metal_actual)

    def action_generar_proyecto(self) -> None:
        self._iniciar_generacion()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-generar":
            self._iniciar_generacion()
        elif event.button.id == "btn-auditar":
            self.action_refrescar_auditoria()
        elif event.button.id == "btn-salir":
            self.app.exit()

    def _iniciar_generacion(self) -> None:
        def on_summary_complete(generar_otro):
            if generar_otro:
                try:
                    self.query_one("#radio-metales").focus()
                except Exception:
                    pass

        # Si las carpetas y cuadernos del metal ya están creadas, no ejecutar la función de generación
        if verificar_proyecto_existente(self.metal_actual):
            self.app.push_screen(
                SummaryScreen(
                    resultado={
                        "metal": self.metal_actual,
                        "ya_existia": True,
                    }
                ),
                callback=on_summary_complete,
            )
            return

        def on_execution_complete(resultado):
            if resultado:
                self.app.push_screen(
                    SummaryScreen(resultado=resultado),
                    callback=on_summary_complete
                )

        self.app.push_screen(
            ExecutionScreen(cfg=self.metal_actual),
            callback=on_execution_complete
        )
