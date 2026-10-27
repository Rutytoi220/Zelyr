import os
import sys
import shutil
import subprocess
import urllib.request
import tarfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Packager:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.dist_dir = self.root_dir / "dist"
        self.zelyr_dist = self.dist_dir / "Zelyr"
        self.version = "11.4.0" # Assuming version 11.4

    def build(self, target: str):
        print(f"[INFO] Starting packager for target: {target}")
        os.chdir(self.root_dir)
        
        # 1. Base PyInstaller Build
        self._build_pyinstaller()
        
        targets = [target] if target != 'all' else ['deb', 'appimage', 'rpm']
        
        for t in targets:
            if t == 'deb':
                self._build_deb()
            elif t == 'appimage':
                self._build_appimage()
            elif t == 'rpm':
                self._build_rpm()

    def _build_pyinstaller(self):
        print("[INFO] Running PyInstaller Base Build...")
        if not (self.root_dir / "zelyr.spec").exists():
            print("[ERROR] zelyr.spec not found!")
            sys.exit(1)
        
        result = subprocess.run([sys.executable, "-m", "PyInstaller", "zelyr.spec", "--clean", "--noconfirm"], cwd=self.root_dir)
        if result.returncode != 0:
            print("[ERROR] PyInstaller build failed.")
            sys.exit(result.returncode)

    def _build_deb(self):
        print("[INFO] Building DEB package...")
        deb_root = self.dist_dir / "zelyr_amd64"
        if deb_root.exists():
            shutil.rmtree(deb_root)
            
        deb_root.mkdir(parents=True)
        
        # Structure
        (deb_root / "DEBIAN").mkdir()
        (deb_root / "opt" / "zelyr").mkdir(parents=True)
        (deb_root / "usr" / "bin").mkdir(parents=True)
        (deb_root / "usr" / "share" / "applications").mkdir(parents=True)
        (deb_root / "usr" / "share" / "pixmaps").mkdir(parents=True)
        
        # Copy binaries
        print("[INFO] Copying binaries to /opt/zelyr")
        for item in self.zelyr_dist.iterdir():
            if item.is_dir():
                shutil.copytree(item, deb_root / "opt" / "zelyr" / item.name)
            else:
                shutil.copy2(item, deb_root / "opt" / "zelyr" / item.name)
                
        # Create launcher
        launcher_path = deb_root / "usr" / "bin" / "zelyr"
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("exec /opt/zelyr/Zelyr \"$@\"\n")
        os.chmod(launcher_path, 0o755)
        
        # Create control file
        control_path = deb_root / "DEBIAN" / "control"
        with open(control_path, "w") as f:
            f.write(f"Package: zelyr\n")
            f.write(f"Version: {self.version}\n")
            f.write("Section: utils\n")
            f.write("Priority: optional\n")
            f.write("Architecture: amd64\n")
            f.write("Depends: libgl1, libglib2.0-0, bash\n")
            f.write("Maintainer: Zelyr Contributors <admin@zelyr.local>\n")
            f.write("Description: A Local-First AI Operating System for Linux.\n")
            
        # Assets
        desktop_content = """[Desktop Entry]
Name=Zelyr Pro
Exec=/usr/bin/zelyr
Icon=zelyr-logo
Type=Application
Categories=Utility;Development;
"""
        with open(deb_root / "usr" / "share" / "applications" / "zelyr.desktop", "w") as f:
            f.write(desktop_content)
            
        logo_path = self.root_dir / "assets" / "zelyr-logo.png"
        if logo_path.exists():
            shutil.copy2(logo_path, deb_root / "usr" / "share" / "pixmaps" / "zelyr-logo.png")
        else:
            import base64
            dummy = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
            with open(deb_root / "usr" / "share" / "pixmaps" / "zelyr-logo.png", "wb") as f:
                f.write(dummy)
                
        # dpkg-deb build
        if shutil.which("dpkg-deb"):
            result = subprocess.run(["dpkg-deb", "--build", str(deb_root)], cwd=self.dist_dir)
            if result.returncode == 0:
                print("[SUCCESS] DEB package generated successfully.")
            else:
                print("[ERROR] Failed to build DEB package.")
        else:
            print("[INFO] dpkg-deb not found, using Python tarfile + ar fallback...")
            control_tar_path = self.dist_dir / "control.tar.gz"
            with tarfile.open(control_tar_path, "w:gz") as tar:
                tar.add(deb_root / "DEBIAN" / "control", arcname="./control")
            
            data_tar_path = self.dist_dir / "data.tar.gz"
            with tarfile.open(data_tar_path, "w:gz") as tar:
                for item in deb_root.iterdir():
                    if item.name != "DEBIAN":
                        tar.add(item, arcname=f"./{item.name}")
                        
            debian_binary_path = self.dist_dir / "debian-binary"
            with open(debian_binary_path, "w") as f:
                f.write("2.0\n")
                
            deb_file = self.dist_dir / "zelyr_amd64.deb"
            if deb_file.exists():
                deb_file.unlink()
                
            subprocess.run(["ar", "qc", str(deb_file), "debian-binary", "control.tar.gz", "data.tar.gz"], cwd=self.dist_dir)
            print("[SUCCESS] DEB package generated successfully (via fallback).")

    def _build_appimage(self):
        print("[INFO] Building AppImage...")
        appdir = self.dist_dir / "Zelyr.AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
            
        appdir.mkdir(parents=True)
        (appdir / "usr" / "bin").mkdir(parents=True)
        (appdir / "usr" / "share" / "applications").mkdir(parents=True)
        
        for item in self.zelyr_dist.iterdir():
            if item.is_dir():
                shutil.copytree(item, appdir / "usr" / "bin" / item.name)
            else:
                shutil.copy2(item, appdir / "usr" / "bin" / item.name)
                
        desktop_content = """[Desktop Entry]
Name=Zelyr Pro
Exec=AppRun
Icon=zelyr-logo
Type=Application
Categories=Utility;Development;
"""
        with open(appdir / "zelyr.desktop", "w") as f:
            f.write(desktop_content)
        shutil.copy2(appdir / "zelyr.desktop", appdir / "usr" / "share" / "applications" / "zelyr.desktop")
        
        logo_path = self.root_dir / "assets" / "zelyr-logo.png"
        if logo_path.exists():
            shutil.copy2(logo_path, appdir / "zelyr-logo.png")
        else:
            import base64
            dummy = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
            with open(appdir / "zelyr-logo.png", "wb") as f:
                f.write(dummy)
                
        apprun_path = appdir / "AppRun"
        with open(apprun_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("HERE=\"$(dirname \"$(readlink -f \"${0}\")\")\"\n")
            f.write("exec \"${HERE}/usr/bin/Zelyr\" \"$@\"\n")
        os.chmod(apprun_path, 0o755)
        
        appimagetool = self.root_dir / "appimagetool-x86_64.AppImage"
        if not appimagetool.exists():
            print("[INFO] Downloading appimagetool...")
            urllib.request.urlretrieve(
                "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage",
                str(appimagetool)
            )
            os.chmod(appimagetool, 0o755)
            
        result = subprocess.run([str(appimagetool), str(appdir), f"Zelyr-v{self.version}-x86_64.AppImage"], cwd=self.dist_dir)
        if result.returncode == 0:
            print("[SUCCESS] AppImage generated successfully.")
            shutil.rmtree(appdir)
        else:
            print("[ERROR] Failed to build AppImage.")

    def _build_rpm(self):
        print("[INFO] Building RPM package...")
        # Check if alien is installed
        alien_path = shutil.which("alien")
        if alien_path:
            deb_file = self.dist_dir / "zelyr_amd64.deb"
            if not deb_file.exists():
                print("[INFO] RPM requires DEB file first. Building DEB...")
                self._build_deb()
            print("[INFO] Running alien to convert DEB to RPM...")
            result = subprocess.run([alien_path, "-r", "-g", str(deb_file)], cwd=self.dist_dir)
            if result.returncode == 0:
                # Actually alien generates an rpm, wait, -r -g generates a directory. Just use -r
                result = subprocess.run([alien_path, "-r", str(deb_file)], cwd=self.dist_dir)
                if result.returncode == 0:
                    print("[SUCCESS] RPM generated via Alien.")
                    return
        
        print("[WARNING] alien tool not found or failed. RPM build skipped or fallback required.")

def build(target: str):
    packager = Packager()
    packager.build(target)

