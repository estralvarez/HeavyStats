"""
Constructor y serializador de cuadernos interactivos Jupyter Notebook (.ipynb v4).
"""

import json
from pathlib import Path
import heavystats as hs


def crear_cuaderno(ruta, titulo, descripcion="", celdas=None, celdas_codigo=None):
    """
    Genera un archivo Jupyter Notebook (.ipynb) válido y estructurado en nbformat v4.

    Parameters
    ----------
    ruta : str or Path
        Ruta destino del cuaderno (e.g. 'validacion' o 'univariante/01_tablas').
    titulo : str
        Título de cabecera en Markdown.
    descripcion : str
        Resumen metodológico inicial.
    celdas : list of dict, optional
        Lista estructurada de celdas: [{"tipo": "markdown" | "code", "contenido": str}, ...]
    celdas_codigo : list of str, optional
        Lista simple de bloques de código (compatibilidad).

    Returns
    -------
    Path
        Ruta al archivo .ipynb generado.
    """
    path = Path(ruta)
    if path.suffix != ".ipynb":
        path = path.with_suffix(".ipynb")

    path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {titulo}\n",
                "\n",
                f"{descripcion}\n\n",
                f"> Protocolo bioestadístico y epidemiológico de biomonitoreo con **HeavyStats** v{hs.__version__}."
            ]
        }
    ]

    if celdas:
        for c in celdas:
            tipo = c.get("tipo", "code")
            contenido = c.get("contenido", "")
            lines = [line + "\n" for line in contenido.strip().split("\n")]
            if tipo == "markdown":
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": lines
                })
            else:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines
                })
    elif celdas_codigo:
        for codigo in celdas_codigo:
            lines = [line + "\n" for line in codigo.strip().split("\n")]
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return path
