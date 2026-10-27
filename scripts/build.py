#!/usr/bin/env python3
"""
Build script for packaging Zelyr into a standalone executable using PyInstaller.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=== Zelyr Standalone Packager ===")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        import PyInstaller.__main__
        
    root_dir = Path(__file__).resolve().parent.parent
    main_py = root_dir / "main.py"
    
    if not main_py.exists():
        print(f"Error: {main_py} not found.")
        sys.exit(1)
        
    # Build the pyinstaller arguments
    args = [
        str(main_py),
        '--name=zelyr',
        '--onefile',
        '--windowed', # No console window (can be omitted if CLI output is desired)
        '--clean',
        '--noconfirm',
        f'--distpath={root_dir / "dist"}',
        f'--workpath={root_dir / "build"}',
        f'--specpath={root_dir}',
    ]
    
    # Add assets
    assets_dir = root_dir / "zelyr" / "gui" / "assets"
    if assets_dir.exists():
        # For cross-platform compatibility, use os.pathsep
        args.append(f'--add-data={assets_dir}{os.pathsep}zelyr/gui/assets')
        
    # Add generic hidden imports for robustness
    args.extend([
        '--hidden-import=PySide6',
        '--hidden-import=pydantic',
        '--hidden-import=chromadb',
        '--hidden-import=aiohttp',
    ])
    
    print(f"Running PyInstaller...")
    PyInstaller.__main__.run(args)
    
    print(f"Packaging complete! Check the '{root_dir / 'dist'}' directory.")

if __name__ == "__main__":
    main()
