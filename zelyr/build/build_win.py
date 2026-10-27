"""Windows Packaging — Inno Setup Script Generator.

Produces ``zelyr_setup.iss`` that compiles into ``Zelyr_Setup_v5.2.0.exe``
from the frozen PyInstaller binaries.
"""
import logging
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
RELEASES_DIR = PROJECT_ROOT / "releases"
BUILD_DIR = PROJECT_ROOT / "build"
ZELYR_VERSION = "5.2.0"


def generate_inno_setup_script() -> str:
    return textwrap.dedent(f"""\
        [Setup]
        AppName=Zelyr
        AppVersion={ZELYR_VERSION}
        DefaultDirName={{autopf}}\\Zelyr
        DefaultGroupName=Zelyr
        OutputDir={RELEASES_DIR}
        OutputBaseFilename=Zelyr_Setup_v{ZELYR_VERSION}
        Compression=lzma2
        SolidCompression=yes
        WizardStyle=modern
        ArchitecturesInstallIn64BitMode=x64compatible

        [Files]
        Source: "{DIST_DIR}\\zelyr-daemon.exe"; DestDir: "{{app}}"; Flags: ignoreversion
        Source: "{DIST_DIR}\\zelyr-gui.exe"; DestDir: "{{app}}"; Flags: ignoreversion

        [Icons]
        Name: "{{group}}\\Zelyr"; Filename: "{{app}}\\zelyr-gui.exe"
        Name: "{{commondesktop}}\\Zelyr"; Filename: "{{app}}\\zelyr-gui.exe"

        [Run]
        Filename: "schtasks"; Parameters: "/Create /TN \\"Zelyr Daemon\\" /TR \\"{{app}}\\zelyr-daemon.exe\\" /SC ONLOGON /RL HIGHEST /F"; Flags: runhidden

        [UninstallRun]
        Filename: "schtasks"; Parameters: "/Delete /TN \\"Zelyr Daemon\\" /F"; Flags: runhidden
    """)


def build_windows_installer():
    """Write the .iss file and optionally compile with ISCC."""
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    iss_path = BUILD_DIR / "zelyr_setup.iss"
    iss_path.write_text(generate_inno_setup_script())
    logger.info(f"Windows: Generated Inno Setup script → {iss_path}")

    # Attempt compilation if ISCC is on PATH (Wine + Inno on Linux, or native Windows)
    import shutil
    iscc = shutil.which("ISCC") or shutil.which("iscc")
    if iscc:
        import subprocess
        subprocess.run([iscc, str(iss_path)], check=True)
        logger.info("Windows: ✅  Installer compiled.")
    else:
        logger.info("Windows: ISCC not found — .iss script generated but not compiled.")

    return iss_path
