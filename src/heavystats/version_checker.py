"""
Módulo de comprobación automática de versiones para HeavyStats.

Permite notificar al usuario de forma no intrusiva y no bloqueante
cuando exista una nueva versión disponible en GitHub o PyPI.
"""

import os
import json
import time
import re
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

PACKAGE_NAME = "heavystats"
GITHUB_REPO = "estralvarez/HeavyStats"
GITHUB_INSTALL_URL = f"git+https://github.com/{GITHUB_REPO}.git"

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
        if t_latest == t_current:
            if "dev" in current_str and "dev" not in latest_str:
                return True
        return False
    except Exception:
        return False


def _display_notification(current_version: str, latest_version: str) -> None:
    """Muestra la notificación en Jupyter Notebook o en consola según el entorno."""
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
                    <span style="font-size: 12px; color: #64748b; font-weight: normal;">(versión actual: v{current_version})</span>
                </div>
                <div style="color: #334155; margin-top: 6px; font-size: 12px;">
                    Para actualizar desde GitHub, ejecuta en tu terminal:
                    <div style="margin-top: 4px;">
                        <code style="background: #ffffff; padding: 3px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-family: monospace; font-weight: 600; color: #0f172a; display: inline-block;">
                            pip install --upgrade {GITHUB_INSTALL_URL}
                        </code>
                    </div>
                </div>
            </div>
            """
            display(HTML(html_msg))
            return
        except Exception:
            pass

    # Fallback a consola de texto plano seguro
    msg = (
        f"\n[heavystats] Nueva version disponible: v{latest_version} (instalada: v{current_version})\n"
        f"   Para actualizar desde GitHub ejecuta: pip install --upgrade {GITHUB_INSTALL_URL}\n"
    )
    try:
        print(msg)
    except Exception:
        pass


def _fetch_github_version(token: Optional[str] = None) -> Optional[str]:
    """Intenta consultar la versión más reciente en GitHub mediante la API de releases/tags o pyproject."""
    headers = {
        "User-Agent": f"heavystats-version-check",
        "Accept": "application/vnd.github.v3+json"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 1. Probar GitHub Releases API
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                tag_name = data.get("tag_name", "").lstrip("v")
                if tag_name:
                    return tag_name
    except Exception:
        pass

    # 2. Probar GitHub Tags API
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/tags"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                tags_data = json.loads(resp.read().decode("utf-8"))
                if tags_data and isinstance(tags_data, list):
                    first_tag = tags_data[0].get("name", "").lstrip("v")
                    if first_tag:
                        return first_tag
    except Exception:
        pass

    # 3. Probar Raw pyproject.toml
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/master/pyproject.toml"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                content = resp.read().decode("utf-8")
                m = re.search(r'version\s*=\s*[\"\']([^\"\']+)[\"\']', content)
                if m:
                    return m.group(1)
    except Exception:
        pass

    return None


def _check_remote_version(current_version: str) -> None:
    """Consulta la versión en GitHub/PyPI de forma segura y actualiza la caché local."""
    now = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verificar caché local de 24 horas
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

    # 2. Consultar versión en GitHub
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    latest_version = _fetch_github_version(token)

    # 3. Si no respondió GitHub, consultar PyPI como respaldo
    if not latest_version:
        try:
            url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"heavystats/{current_version}"}
            )
            with urllib.request.urlopen(req, timeout=0.8) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    latest_version = data.get("info", {}).get("version")
        except Exception:
            pass

    # 4. Guardar en caché y notificar si hay actualización
    if latest_version:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"last_check": now, "latest_version": latest_version}, f)
        except Exception:
            pass

        if _is_newer_version(latest_version, current_version):
            _display_notification(current_version, latest_version)


def check_for_updates(current_version: Optional[str] = None, async_check: bool = True) -> None:
    """
    Verifica si existe una versión más reciente de HeavyStats en GitHub o PyPI.
    
    Parámetros
    ----------
    current_version : str, opcional
        Versión actual instalada. Si es None, se lee de heavystats.__version__.
    async_check : bool, opcional (por defecto True)
        Si True, ejecuta la comprobación en un hilo en segundo plano (no bloquea la sesión).
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
