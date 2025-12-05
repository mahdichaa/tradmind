import os


# Project base directory (repo root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Default config directory inside the project (repo_root/config)
DEFAULT_CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Allow overriding via environment variable CONFIG_DIR
# Accept absolute or relative path; if relative, make it relative to BASE_DIR
_env_cfg = os.getenv("CONFIG_DIR", DEFAULT_CONFIG_DIR)
CONFIG_DIR = _env_cfg if os.path.isabs(_env_cfg) else os.path.abspath(os.path.join(BASE_DIR, _env_cfg))


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def config_path(filename: str) -> str:
    """
    Return absolute path to a file inside the CONFIG_DIR, ensuring the directory exists.
    """
    ensure_dir(CONFIG_DIR)
    return os.path.join(CONFIG_DIR, filename)
