"""Zelyr Master Package Engine.

Orchestrates the full freeze → package → checksum pipeline.

Usage (from CLI):
    python3 -m zelyr.api.cli package --target [freeze|deb|rpm|appimage|exe|dmg|all]
"""
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RELEASES_DIR = PROJECT_ROOT / "releases"


class PackageEngine:
    """Drives the entire build + packaging pipeline."""

    VALID_TARGETS = ("freeze", "deb", "rpm", "appimage", "exe", "dmg", "all")

    def __init__(self):
        RELEASES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, target: str):
        target = target.strip().lower()
        if target not in self.VALID_TARGETS:
            logger.error(f"PackageEngine: Unknown target '{target}'. "
                         f"Valid: {', '.join(self.VALID_TARGETS)}")
            return

        if target in ("freeze", "all"):
            self._freeze()

        if target in ("deb", "all"):
            self._build_linux("deb")

        if target in ("rpm", "all"):
            self._build_linux("rpm")

        if target in ("appimage", "all"):
            self._build_appimage()

        if target in ("exe", "all"):
            self._build_windows()

        if target in ("dmg", "all"):
            self._build_macos()

        # Generate SHA256 checksums for everything in releases/
        self._generate_checksums()
        logger.info("PackageEngine: ✅  Pipeline complete.")

    # ---- stages ----

    def _freeze(self):
        logger.info("PackageEngine: Stage → PyInstaller freeze")
        from zelyr.build.freeze import freeze_all
        freeze_all()

    def _build_linux(self, fmt: str):
        logger.info(f"PackageEngine: Stage → Linux {fmt}")
        from zelyr.build.build_linux import build_nfpm_package
        build_nfpm_package(fmt)

    def _build_appimage(self):
        logger.info("PackageEngine: Stage → AppImage")
        from zelyr.build.build_linux import build_appimage
        build_appimage()

    def _build_windows(self):
        logger.info("PackageEngine: Stage → Windows Installer")
        from zelyr.build.build_win import build_windows_installer
        build_windows_installer()

    def _build_macos(self):
        logger.info("PackageEngine: Stage → macOS DMG")
        from zelyr.build.build_mac import build_dmg
        build_dmg()

    def _generate_checksums(self):
        """Write sha256sums.txt for all release artefacts."""
        sums_file = RELEASES_DIR / "sha256sums.txt"
        lines = []
        for f in sorted(RELEASES_DIR.iterdir()):
            if f.is_file() and f.name != "sha256sums.txt":
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                lines.append(f"{h}  {f.name}")
                logger.info(f"Checksum: {h}  {f.name}")
        sums_file.write_text("\n".join(lines) + "\n")
        logger.info(f"PackageEngine: SHA256 checksums → {sums_file}")
