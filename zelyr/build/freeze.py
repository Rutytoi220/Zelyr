"""PyInstaller Binary Freezing Script.

Invokes PyInstaller programmatically to produce two standalone executables:
  - zelyr-daemon  (headless background service)
  - zelyr-gui     (PySide6 graphical interface)
"""
import subprocess
import sys
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
SPEC_DIR = BUILD_DIR / "specs"

# Critical hidden imports that PyInstaller cannot auto-detect
HIDDEN_IMPORTS = [
    "faster_whisper",
    "sounddevice",
    "soundfile",
    "playwright",
    "cryptography",
    "anthropic",
    "sqlite3",
    "pynvml",
    "fusepy",
    "chromadb",
    "apscheduler",
    "websockets",
    "watchdog",
    "psutil",
    "numpy",
    "PIL",
    "cv2",
    "pyttsx3",
    "zelyr",
    "zelyr.kernel",
    "zelyr.kernel.zelyr_kernel",
    "zelyr.server.daemon",
    "zelyr.gui.app",
    # Legacy import shim for plugins that still import ``axiom``.
    "axiom",
]

# Static data assets to bundle
DATA_ASSETS = [
    ("zelyr/build/templates/zelyr.desktop", "templates"),
    ("zelyr/build/templates/zelyr.service", "templates"),
]


def build_hidden_import_args() -> list:
    args = []
    for imp in HIDDEN_IMPORTS:
        args.extend(["--hidden-import", imp])
    return args


def build_data_args() -> list:
    args = []
    for src, dest in DATA_ASSETS:
        full = PROJECT_ROOT / src
        if full.exists():
            args.extend(["--add-data", f"{full}{os.pathsep}{dest}"])
    return args


def freeze_daemon():
    """Freeze the headless daemon into a single binary."""
    logger.info("Freeze: Building zelyr-daemon …")
    entry = str(PROJECT_ROOT / "zelyr" / "server" / "daemon.py")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "zelyr-daemon",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR / "daemon"),
        "--specpath", str(SPEC_DIR),
        "--noconfirm",
        *build_hidden_import_args(),
        *build_data_args(),
        entry,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    logger.info(f"Freeze: zelyr-daemon → {DIST_DIR / 'zelyr-daemon'}")


def freeze_gui():
    """Freeze the PySide6 GUI into a single binary."""
    logger.info("Freeze: Building zelyr-gui …")
    entry = str(PROJECT_ROOT / "zelyr" / "gui" / "app.py")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "zelyr-gui",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR / "gui"),
        "--specpath", str(SPEC_DIR),
        "--noconfirm",
        *build_hidden_import_args(),
        *build_data_args(),
        entry,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    logger.info(f"Freeze: zelyr-gui → {DIST_DIR / 'zelyr-gui'}")


def freeze_all():
    """Build both executables."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    freeze_daemon()
    freeze_gui()
    logger.info("Freeze: All binaries built successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    freeze_all()
