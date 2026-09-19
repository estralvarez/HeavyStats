"""
Aplicación Textual principal para HeavyStats Studio.
"""

from textual.app import App
from heavystats.studio.screens.main_screen import MainScreen
import heavystats as hs


class HeavyStatsApp(App):
    """
    Aplicación interactiva de terminal para HeavyStats Studio.
    """

    TITLE = "HeavyStats Studio"
    SUB_TITLE = f"v{hs.__version__} — Biomonitoreo de Metales Pesados"
    CSS = """
    Screen {
        background: #11111b;
    }

    Button {
        height: 3;
        min-width: 18;
        background: #313244;
        color: #cdd6f4;
        border: tall #585b70;
        text-style: bold;
    }

    Button:hover {
        background: #45475a;
        color: #ffffff;
        border: tall #89b4fa;
        text-style: bold;
    }

    Button:focus {
        border: tall #ffffff;
        text-style: bold;
    }

    /* Botones primarios (Generar Proyecto, Generar otro metal) */
    Button.-primary {
        background: #1e66f5;
        color: #ffffff;
        border: tall #3b82f6;
    }

    Button.-primary:hover {
        background: #2563eb;
        color: #ffffff;
        border: tall #93c5fd;
        text-style: bold;
    }

    Button.-primary:focus {
        background: #1d4ed8;
        border: tall #ffffff;
        color: #ffffff;
    }

    /* Botones de éxito (Continuar) */
    Button.-success {
        background: #16a34a;
        color: #ffffff;
        border: tall #22c55e;
    }

    Button.-success:hover {
        background: #15803d;
        color: #ffffff;
        border: tall #86efac;
        text-style: bold;
    }

    Button.-success:focus {
        background: #166534;
        border: tall #ffffff;
        color: #ffffff;
    }

    /* Botones de peligro / salir (Salir) */
    Button.-error {
        background: #dc2626;
        color: #ffffff;
        border: tall #ef4444;
    }

    Button.-error:hover {
        background: #b91c1c;
        color: #ffffff;
        border: tall #fca5a5;
        text-style: bold;
    }

    Button.-error:focus {
        background: #991b1b;
        border: tall #ffffff;
        color: #ffffff;
    }

    /* Botones por defecto / secundarios (Re-auditar, Salir en modal) */
    Button.-default {
        background: #313244;
        color: #cdd6f4;
        border: tall #585b70;
    }

    Button.-default:hover {
        background: #45475a;
        color: #ffffff;
        border: tall #bac2de;
        text-style: bold;
    }

    Button.-default:focus {
        background: #45475a;
        border: tall #ffffff;
        color: #ffffff;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(MainScreen())


def iniciar_tui():
    """
    Punto de entrada para ejecutar la interfaz de usuario de terminal Textual.
    """
    app = HeavyStatsApp()
    return app.run()
