"""
Utility functions, SSL configuration, directory paths, and logging helpers.
"""

import logging
import os
import random
import ssl
from pathlib import Path
import certifi
import numpy as np

# Automatically configure Homebrew OpenMP library path for XGBoost on macOS if present
if os.path.exists("/opt/homebrew/opt/libomp/lib"):
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if "/opt/homebrew/opt/libomp/lib" not in current_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"/opt/homebrew/opt/libomp/lib:{current_dyld}".strip(":")

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RANDOM_STATE = 42


def setup_ssl() -> None:
    """Configure SSL context with certifi for reliable dataset downloading across platforms."""
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ssl_context
    except Exception as e:
        logging.warning(f"Failed to set custom SSL context: {e}")


def set_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Set random seed for reproducibility across standard libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "heart_disease_ml") -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
