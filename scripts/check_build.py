"""Offline build validation for the Python/Streamlit application."""
import ast
import importlib
from pathlib import Path
import sys
import tomllib

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
config = tomllib.loads((root / ".streamlit/config.toml").read_text(encoding="utf-8"))
for face in config["theme"]["fontFaces"]:
    path = root / face["url"].removeprefix("app/")
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing deployment font: {path.relative_to(root)}")
files = [*root.glob("*.py"), *root.joinpath("scripts").glob("*.py")]
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for dependency in ("streamlit", "pandas", "numpy", "sklearn", "plotly", "scipy",
                   "requests", "joblib", "psycopg", "dotenv", "app_config", "model_store"):
    importlib.import_module(dependency)
print(f"Python build passed: {len(files)} modules parsed; required runtime imports passed.")
