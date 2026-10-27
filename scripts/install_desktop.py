#!/usr/bin/env python3
"""Automated Linux Desktop Launcher Installer for Zelyr."""

import os
import shutil
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / "zelyr" / "gui" / "assets"
    
    # 1. Create system icon directory
    icon_dir = Path.home() / ".local" / "share" / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Copy asset icon
    logo_svg = assets_dir / "logo.svg"
    logo_png = assets_dir / "logo.png"
    
    if logo_svg.exists():
        icon_path = icon_dir / "zelyr.svg"
        shutil.copy2(logo_svg, icon_path)
        print(f"Installed icon to {icon_path}")
    elif logo_png.exists():
        icon_path = icon_dir / "zelyr.png"
        shutil.copy2(logo_png, icon_path)
        print(f"Installed icon to {icon_path}")
    else:
        print(f"Warning: No logo found in {assets_dir}. Icon will not be installed.")
        
    # 3. Ensure applications directory exists
    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Generate desktop entry
    desktop_file = apps_dir / "Zelyr.desktop"
    desktop_content = f"""[Desktop Entry]
Name=Zelyr Desktop
Comment=Local-First AI Orchestration Framework
Exec=python3 -m zelyr.gui.app
Path={project_root.absolute()}
Icon=zelyr
Terminal=false
Type=Application
Categories=Development;Utility;System;
Keywords=AI;LLM;Orchestration;Ollama;Linux;PySide6;
StartupWMClass=Zelyr
"""
    desktop_file.write_text(desktop_content, encoding="utf-8")
    print(f"Installed desktop entry to {desktop_file}")
    
    # 5. Execute update-desktop-database
    try:
        subprocess.run(["update-desktop-database", str(apps_dir)], check=False)
        print("Updated desktop database.")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
