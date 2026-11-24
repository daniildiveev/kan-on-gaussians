"""Filesystem locations shared by notebooks, training and tracking."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"


def project_path(path: str | Path) -> Path:
    """Interpret relative paths from the project root and preserve absolute paths."""
    return PROJECT_ROOT / Path(path).expanduser()
