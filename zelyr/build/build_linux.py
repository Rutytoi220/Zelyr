"""Linux Native Packaging — .deb, .rpm and AppImage.

Uses nfpm (Not Effing Package Manager) for .deb/.rpm and appimagetool
for the universal AppImage bundle.  Both work from the frozen PyInstaller
binaries in ``dist/``.
"""
import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
RELEASES_DIR = PROJECT_ROOT / "releases"
BUILD_DIR = PROJECT_ROOT / "build"
BUILD_TEMPLATES = Path(__file__).resolve().parent / "templates"
BUILD_SCRIPTS = Path(__file__).resolve().parent / "scripts"

ZELYR_VERSION = "5.2.0"


# ------------------------------------------------------------------
# nfpm YAML generator
# ------------------------------------------------------------------
def generate_nfpm_yaml(target: str) -> str:
    """Return the nfpm config as a YAML string for *target* (deb | rpm)."""
    return textwrap.dedent(f"""\
        name: zelyr
        arch: amd64
        platform: linux
        version: {ZELYR_VERSION}
        maintainer: rutytoi <charlesbbm@outlook.fr>
        description: Zelyr — Local-First AI Operating System for Linux
        vendor: Zelyr Project
        homepage: https://github.com/Rutytoi220/Zelyr
        license: MIT

        contents:
          - src: {DIST_DIR}/zelyr-daemon
            dst: /usr/bin/zelyr-daemon
            file_info:
              mode: 0755
          - src: {DIST_DIR}/zelyr-gui
            dst: /usr/bin/zelyr-gui
            file_info:
              mode: 0755
          - src: {BUILD_TEMPLATES}/zelyr.desktop
            dst: /usr/share/applications/Zelyr.desktop
          - src: {BUILD_TEMPLATES}/zelyr.service
            dst: /usr/lib/systemd/user/zelyr.service

        scripts:
          postinstall: {BUILD_SCRIPTS}/post_install.sh
    """)


def build_nfpm_package(target: str):
    """Build a .deb or .rpm via nfpm."""
    if not shutil.which("nfpm"):
        logger.error("nfpm binary not found. Install: https://nfpm.goreleaser.com")
        return None

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    nfpm_cfg = BUILD_DIR / "nfpm.yaml"
    nfpm_cfg.write_text(generate_nfpm_yaml(target))

    out_file = RELEASES_DIR / f"zelyr-{ZELYR_VERSION}-amd64.{target}"
    cmd = [
        "nfpm", "package",
        "--config", str(nfpm_cfg),
        "--packager", target,
        "--target", str(out_file),
    ]
    logger.info(f"nfpm: building {target} → {out_file}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    logger.info(f"nfpm: ✅  {out_file}")
    return out_file


# ------------------------------------------------------------------
# AppImage
# ------------------------------------------------------------------
def build_appimage():
    """Assemble an AppDir and invoke appimagetool."""
    if not shutil.which("appimagetool"):
        logger.error("appimagetool not found. Download from https://appimage.github.io")
        return None

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)

    appdir = DIST_DIR / "Zelyr.AppDir"
    appdir.mkdir(parents=True, exist_ok=True)

    usr_bin = appdir / "usr" / "bin"
    usr_bin.mkdir(parents=True, exist_ok=True)

    # Copy binaries
    for binary in ("zelyr-daemon", "zelyr-gui"):
        src = DIST_DIR / binary
        if src.exists():
            shutil.copy2(src, usr_bin / binary)

    # Desktop & icon at AppDir root
    desktop_src = BUILD_TEMPLATES / "zelyr.desktop"
    if desktop_src.exists():
        shutil.copy2(desktop_src, appdir / "Zelyr.desktop")

    # Create a minimal AppRun
    apprun = appdir / "AppRun"
    apprun.write_text("#!/bin/bash\nexec \"$(dirname \"$0\")/usr/bin/zelyr-gui\" \"$@\"\n")
    apprun.chmod(0o755)

    out = RELEASES_DIR / f"Zelyr-{ZELYR_VERSION}-x86_64.AppImage"
    cmd = ["appimagetool", str(appdir), str(out)]
    logger.info(f"AppImage: building → {out}")
    subprocess.run(cmd, check=True)
    logger.info(f"AppImage: ✅  {out}")
    return out
