from pathlib import Path


def project_root() -> Path:
    """Devuelve la raiz del repositorio."""
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)
