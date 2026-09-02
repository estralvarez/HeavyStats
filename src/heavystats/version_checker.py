"""
Módulo de comprobación automática de versiones para HeavyStats.

Permite notificar al usuario de forma no intrusiva y no bloqueante
cuando exista una nueva versión disponible para su actualización.
"""

import os
import json
import time
import re
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

PACKAGE_NAME = "heavystats"
CACHE_DIR = Path.home() / ".heavystats"
CACHE_FILE = CACHE_DIR / "version_cache.json"
CHECK_INTERVAL_SECONDS = 86400  # 24 horas


def _parse_version_tuple(v_str: str) -> Tuple[int, ...]:
    """Convierte una cadena de versión semántica (ej. '0.2.0.dev1' o '1.3.4') en tupla numérica comparable."""
    clean = re.sub(r"[^\d.]", "", v_str.split(".dev")[0].split("a")[0].split("b")[0].split("rc")[0])
    parts = [int(p) for p in clean.split(".") if p.isdigit()]
    return tuple(parts) if parts else (0,)


def _is_newer_version(latest_str: str, current_str: str) -> bool:
    """Retorna True si latest_str es estrictamente mayor que current_str."""
    try:
        t_latest = _parse_version_tuple(latest_str)
        t_current = _parse_version_tuple(current_str)
        if t_latest > t_current:
            return True
        # Si las partes numéricas base son iguales, comparar sufijos de desarrollo
        if t_latest == t_current:
            if "dev" in current_str and "dev" not in latest_str:
                return True
        return False
    except Exception:
        return False


def _display_notification(current_version: str, latest_version: str) -> None:
    """Muestra la notificación en Jupyter Notebook o en consola según el entorno."""
    # Comprobar si estamos en Jupyter / IPython interactivo
    in_jupyter = False
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None and "IPKernelApp" in ip.config:
            in_jupyter = True
    except Exception:
        in_jupyter = False

    if in_jupyter:
        try:
            from IPython.display import display, HTML  # type: ignore
            html_msg = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6;
                        border-radius: 6px; padding: 12px 16px; margin: 12px 0; color: #1e3a8a; font-size: 13px;">
                <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                    <span>🚀</span> <span>Nueva versión de HeavyStats disponible: <strong>v{latest_version}</strong></span>
                    <span style="font-size: 12px; color: #64748b; font-weight: normal;">(versión instalada: v{current_version})</span>
                </div>
                <div style="color: #334155; margin-top: 4px;">
                    Para actualizar a la última versión, ejecuta:
                    <code style="background: #ffffff; padding: 2px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-family: monospace; font-weight: 600; color: #0f172a;">
                        pip install --upgrade {PACKAGE_NAME}
                    </code>
                </div>
            </div>
            """
            display(HTML(html_msg))
            return
        except Exception:
            pass

    # Fallback a consola de texto plano seguro con cualquier codificación de terminal
    msg = (
        f"\n[heavystats] Nueva version disponible: v{latest_version} (instalada: v{current_version})\n"
        f"   Para actualizar ejecuta: pip install --upgrade {PACKAGE_NAME}\n"
    )
    try:
        print(msg)
    except Exception:
        pass


def _check_remote_version(current_version: str) -> None:
    """Consulta la versión en PyPI de forma segura y actualiza la caché local."""
    now = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verificar si ya consultamos en las últimas 24 horas
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            last_check = cache_data.get("last_check", 0)
            if now - last_check < CHECK_INTERVAL_SECONDS:
                cached_latest = cache_data.get("latest_version")
                if cached_latest and _is_newer_version(cached_latest, current_version):
                    _display_notification(current_version, cached_latest)
                return
        except Exception:
            pass

    # 2. Consultar API de PyPI con timeout corto de 0.8s
    try:
        url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"heavystats/{current_version} (Python automated version check)"}
        )
        with urllib.request.urlopen(req, timeout=0.8) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                latest_version = data.get("info", {}).get("version", current_version)
                
                # Guardar en caché
                try:
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump({"last_check": now, "latest_version": latest_version}, f)
                except Exception:
                    pass

                if _is_newer_version(latest_version, current_version):
                    _display_notification(current_version, latest_version)
    except Exception:
        # Fallo silencioso ante falta de conexión o paquete no publicado en PyPI aún
        pass


def check_for_updates(current_version: Optional[str] = None, async_check: bool = True) -> None:
    """
    Verifica si existe una versión más reciente de HeavyStats.
    
    Parámetros
    ----------
    current_version : str, opcional
        Versión actual instalada. Si es None, se lee de heavystats.__version__.
    async_check : bool, opcional (por defecto True)
        Si True, ejecuta la comprobación en un hilo en segundo plano (no bloquea el notebook).
    """
    if os.environ.get("HEAVYSTATS_NO_UPDATE_CHECK") == "1":
        return

    if current_version is None:
        try:
            import heavystats
            current_version = heavystats.__version__
        except Exception:
            current_version = "0.2.0.dev1"

    if async_check:
        thread = threading.Thread(target=_check_remote_version, args=(current_version,), daemon=True)
        thread.start()
    else:
        _check_remote_version(current_version)
