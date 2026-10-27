"""Identity checks for the native packaging helpers and their templates."""

from __future__ import annotations

import plistlib
from pathlib import Path


def test_package_generators_use_zelyr_identity() -> None:
    from zelyr.build.build_linux import generate_nfpm_yaml
    from zelyr.build.build_mac import _generate_info_plist, _generate_launchd_plist
    from zelyr.build.build_win import generate_inno_setup_script
    from zelyr.build.freeze import build_data_args

    nfpm = generate_nfpm_yaml("deb")
    assert "name: zelyr" in nfpm
    assert "homepage: https://github.com/Rutytoi220/Zelyr" in nfpm
    assert "/usr/bin/zelyr-daemon" in nfpm
    assert "/usr/share/applications/Zelyr.desktop" in nfpm
    assert "axiom" not in nfpm.lower()

    windows = generate_inno_setup_script()
    assert "AppName=Zelyr" in windows
    assert "Zelyr_Setup_v" in windows
    assert "zelyr-daemon.exe" in windows
    assert "axiom" not in windows.lower()

    info_plist = plistlib.loads(_generate_info_plist().encode())
    launchd_plist = plistlib.loads(_generate_launchd_plist().encode())
    assert info_plist["CFBundleName"] == "Zelyr"
    assert info_plist["CFBundleIdentifier"] == "com.zelyr.gui"
    assert launchd_plist["Label"] == "com.zelyr.daemon"
    assert "Zelyr.app" in launchd_plist["ProgramArguments"][0]
    assert "axiom" not in str(info_plist).lower()
    assert "axiom" not in str(launchd_plist).lower()

    assets = build_data_args()
    assert any("zelyr.desktop" in value for value in assets)
    assert any("zelyr.service" in value for value in assets)
    assert all("axiom" not in value.lower() for value in assets)


def test_packaging_templates_use_zelyr_identity() -> None:
    templates = Path(__file__).resolve().parent.parent / "zelyr" / "build" / "templates"
    desktop = (templates / "zelyr.desktop").read_text(encoding="utf-8")
    service = (templates / "zelyr.service").read_text(encoding="utf-8")

    assert "Name=Zelyr" in desktop
    assert "Exec=/usr/bin/zelyr-gui" in desktop
    assert "StartupWMClass=Zelyr" in desktop
    assert "axiom" not in desktop.lower()

    assert "Description=Zelyr" in service
    assert "ExecStart=/usr/bin/zelyr-daemon" in service
    assert "axiom" not in service.lower()
