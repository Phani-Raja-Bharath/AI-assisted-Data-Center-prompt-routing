"""Offline build validation for the Python/Streamlit application."""
import ast
import importlib
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
files = [*root.glob("*.py"), *root.joinpath("scripts").glob("*.py")]
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for dependency in ("streamlit", "pandas", "numpy", "sklearn", "plotly", "scipy",
                   "requests", "joblib", "psycopg", "dotenv", "app_config", "model_store"):
    importlib.import_module(dependency)
print(f"Python build passed: {len(files)} modules parsed; required runtime imports passed.")
