from pathlib import Path

from src.utils.helpers import project_path


def output_path(*parts: str) -> Path:
    """Ruta dentro de output/ para reportes o figuras generadas."""
    return project_path("output", *parts)
