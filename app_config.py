"""Read Neon CLI environment files without logging or exporting their secrets."""
from collections.abc import Mapping
import os
from pathlib import Path

from dotenv import dotenv_values


def get_database_url(secrets: Mapping | None = None, project_dir: Path | None = None):
    """Explicit environment, Streamlit secrets, .env.local, then .env.

    MODEL_DATABASE_URL remains supported; Neon's pooled DATABASE_URL works directly.
    Environment files are not copied into os.environ or sent to the browser.
    """
    for values in (os.environ, secrets or {}):
        for key in ("MODEL_DATABASE_URL", "DATABASE_URL"):
            if values.get(key):
                return values[key]
    root = project_dir if project_dir is not None else Path(__file__).resolve().parent
    # Prefer the whole higher-priority file, including Neon's standard DATABASE_URL.
    for name in (".env.local", ".env"):
        path = root / name
        if path.is_file():
            values = dotenv_values(path, interpolate=False)
            for key in ("MODEL_DATABASE_URL", "DATABASE_URL"):
                if values.get(key):
                    return values[key]
    return None
