"""macOS Packaging — .app bundle & .dmg disk image generator.

Creates a Zelyr.app bundle directory, wires the daemon into a launchd
user agent plist, and outputs a drag-and-drop .dmg via hdiutil or
create-dmg.
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
ZELYR_VERSION = "1.0.0"


def _generate_info_plist() -> str:
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>CFBundleName</key>          <string>Zelyr</string>
            <key>CFBundleDisplayName</key>   <string>Zelyr</string>
            <key>CFBundleIdentifier</key>    <string>com.zelyr.gui</string>
            <key>CFBundleVersion</key>       <string>{ZELYR_VERSION}</string>
            <key>CFBundleExecutable</key>    <string>zelyr-gui</string>
            <key>CFBundlePackageType</key>   <string>APPL</string>
            <key>LSMinimumSystemVersion</key><string>12.0</string>
        </dict>
        </plist>
    """)


def _generate_launchd_plist() -> str:
    return textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>                <string>com.zelyr.daemon</string>
            <key>ProgramArguments</key>
            <array>
                <string>/Applications/Zelyr.app/Contents/MacOS/zelyr-daemon</string>
            </array>
            <key>RunAtLoad</key>            <true/>
            <key>KeepAlive</key>            <true/>
        </dict>
        </plist>
    """)


def build_app_bundle() -> Path:
    """Assemble Zelyr.app directory structure."""
    app = DIST_DIR / "Zelyr.app"
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    # Info.plist
    (app / "Contents" / "Info.plist").write_text(_generate_info_plist())

    # Copy binaries
    for name in ("zelyr-daemon", "zelyr-gui"):
        src = DIST_DIR / name
        if src.exists():
            shutil.copy2(src, macos / name)

    # Launchd plist → Resources (user can install to ~/Library/LaunchAgents)
    (resources / "com.zelyr.daemon.plist").write_text(_generate_launchd_plist())

    logger.info(f"macOS: Built Zelyr.app → {app}")
    return app


def build_dmg() -> Path | None:
    """Wrap Zelyr.app in a .dmg disk image."""
    app = build_app_bundle()
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    out = RELEASES_DIR / f"Zelyr-{ZELYR_VERSION}.dmg"

    create_dmg = shutil.which("create-dmg")
    hdiutil = shutil.which("hdiutil")

    if create_dmg:
        cmd = [
            create_dmg,
            "--volname", "Zelyr",
            "--window-size", "600", "400",
            "--app-drop-link", "450", "200",
            str(out), str(app),
        ]
        subprocess.run(cmd, check=True)
    elif hdiutil:
        cmd = [
            hdiutil, "create",
            "-volname", "Zelyr",
            "-srcfolder", str(app),
            "-ov", "-format", "UDZO",
            str(out),
        ]
        subprocess.run(cmd, check=True)
    else:
        logger.warning("macOS: Neither create-dmg nor hdiutil found — .app bundle only.")
        return None

    logger.info(f"macOS: ✅  {out}")
    return out
