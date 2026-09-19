"""
Punto de entrada CLI y ejecutor unificado para HeavyStats Studio.
Permite ejecutar la aplicación mediante Terminal User Interface (TUI con Textual)
o por línea de comandos (CLI / Headless / Fallback de consola).
"""

import argparse
import sys
import heavystats as hs
from heavystats.studio.config import METALES_CONFIG, DEFAULT_METAL, resolver_metal
from heavystats.studio.pipeline import ejecutar_pipeline


def menu_interactivo_terminal():
    """Menú interactivo por terminal estándar (fallback si se deshabilita la TUI)."""
    print("\n" + "=" * 62)
    print("   HeavyStats Studio - Generador de Proyectos Bioestadísticos")
    print("=" * 62)
    print("Seleccione el metal que desea analizar:")
    print("  [1] Plomo (Pb, µg/dL)       [Por defecto - Presione Enter]")
    print("  [2] Mercurio (Hg, µg/L)")
    print("  [3] Cadmio (Cd, µg/L)")
    print("-" * 62)
    try:
        entrada = input("Ingrese opción (1/2/3, nombre o símbolo) [1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        entrada = ""
    return resolver_metal(entrada)


def build_parser():
    """Construye el parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        prog="heavystats",
        description="HeavyStats Studio: Generador automatizado de proyectos y cuadernos bioestadísticos."
    )
    parser.add_argument(
        "-m", "--metal",
        type=str,
        default=None,
        help="Metal a analizar: plomo (pb), mercurio (hg), cadmio (cd), arsenico (as). Por defecto: plomo."
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Deshabilita la interfaz visual Textual y ejecuta en modo consola estándar."
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"HeavyStats v{hs.__version__}"
    )
    return parser


def main():
    """Punto de entrada principal para el comando de consola 'heavystats'."""
    parser = build_parser()
    args = parser.parse_args()

    # Si se especificó un metal explícito por CLI o flag --no-tui
    if args.metal is not None or args.no_tui:
        cfg = resolver_metal(args.metal) if args.metal else menu_interactivo_terminal()
        print(f"\n[CLI] Metal seleccionado: {cfg['nombre']} ({cfg['simbolo']}, {cfg['unidad']})")
        res = ejecutar_pipeline(cfg)
        print("\n" + "=" * 62)
        if res.get("ya_existia"):
            print(f"   [AVISO] ¡EL PROYECTO PARA {cfg['nombre'].upper()} ({cfg['simbolo']}) YA EXISTE!")
            print("   Las carpetas y cuadernos ya están listos en el espacio de trabajo.")
        else:
            print(f"   ¡PROYECTO PARA {cfg['nombre'].upper()} ({cfg['simbolo']}) GENERADO CON ÉXITO!")
            print("=" * 62)
            for nb in res.get("cuadernos", []):
                print(f"  [OK] {nb}")
            if "dataset_audit" in res and "ruta" in res["dataset_audit"]:
                print(f"Dataset de trabajo: {res['dataset_audit']['ruta']}")
        print("=" * 62 + "\n")
        return

    # Modo interactivo por defecto: Textual TUI
    if not sys.stdout.isatty():
        print("[INFO] Terminal no interactivo detectado. Ejecutando en modo estándar...")
        cfg = resolver_metal(args.metal or DEFAULT_METAL)
        ejecutar_pipeline(cfg)
        return

    try:
        from heavystats.studio.app import iniciar_tui
        iniciar_tui()
    except Exception as e:
        print(f"[AVISO] No se pudo inicializar la TUI gráfica: {e}")
        print("[INFO] Cambiando a modo interactivo por consola estándar...")
        cfg = menu_interactivo_terminal()
        ejecutar_pipeline(cfg)


if __name__ == "__main__":
    main()
