"""Native Git Update Manager.

Checks the GitHub repository API for new tags.
Performs a background 'git pull origin main' to synchronize local code changes
without a full binary reinstall.
"""
import logging
import asyncio
import subprocess
import urllib.request
import json
import os

logger = logging.getLogger(__name__)

class ZelyrUpdateManager:
    """Manages OTA software updates via Git."""
    
    def __init__(self, current_version: str = "v6.0.0"):
        self.current_version = current_version
        self.repo_api_url = "https://api.github.com/repos/Rutytoi220/Zelyr/releases/latest"
        
    async def check_for_updates(self) -> dict:
        """Checks GitHub API for the latest release tag."""
        logger.info("ZelyrUpdateManager: Checking for updates...")
        
        def _fetch():
            try:
                req = urllib.request.Request(self.repo_api_url, headers={'User-Agent': 'Zelyr-Updater'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    return data.get("tag_name", self.current_version)
            except Exception as e:
                logger.error(f"ZelyrUpdateManager: Failed to fetch updates - {e}")
                return self.current_version

        latest_version = await asyncio.to_thread(_fetch)
        
        # Simple semantic version check (e.g. 'v6.1.0' > 'v6.0.0')
        update_available = latest_version != self.current_version and latest_version > self.current_version
        
        if update_available:
            logger.info(f"ZelyrUpdateManager: New update available! {self.current_version} -> {latest_version}")
        else:
            logger.info("ZelyrUpdateManager: System is up to date.")
            
        return {
            "update_available": update_available,
            "latest_version": latest_version,
            "current_version": self.current_version
        }

    async def perform_update(self, mock_mode: bool = False) -> bool:
        """Executes Over-The-Air Update using native Linux packages."""
        logger.info("ZelyrUpdateManager: Initiating Over-The-Air Update...")
        
        def _get_package_type():
            import platform
            sys_name = platform.system()
            if sys_name == "Windows":
                return "exe"
            elif sys_name == "Darwin":
                return "dmg"
            elif sys_name == "Linux":
                try:
                    with open("/etc/os-release") as f:
                        content = f.read().lower()
                        if "debian" in content or "ubuntu" in content or "pop" in content:
                            return "deb"
                        elif "fedora" in content or "centos" in content or "rhel" in content or "bazzite" in content:
                            return "rpm"
                except Exception as e:
                    logger.error(f"ZelyrUpdateManager: Could not read /etc/os-release - {e}")
            return None

        def _do_update():
            if mock_mode:
                logger.info("ZelyrUpdateManager: [MOCK] Update successful.")
                return True
                
            pkg_type = _get_package_type()
            if not pkg_type:
                logger.error("ZelyrUpdateManager: Unsupported OS. Only Debian/Fedora/Windows/macOS are supported for OTA updates.")
                return False
                
            try:
                # 1. Get the latest release assets
                req = urllib.request.Request(self.repo_api_url, headers={'User-Agent': 'Zelyr-Updater'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    assets = data.get("assets", [])
                    
                download_url = None
                for asset in assets:
                    if asset.get("name", "").endswith(f".{pkg_type}"):
                        download_url = asset.get("browser_download_url")
                        break
                        
                if not download_url:
                    logger.error(f"ZelyrUpdateManager: No .{pkg_type} package found in the latest release assets.")
                    return False
                    
                # 2. Download the package
                import tempfile
                import platform
                
                tmp_dir = tempfile.gettempdir()
                tmp_file = os.path.join(tmp_dir, f"zelyr_latest.{pkg_type}")
                
                logger.info(f"ZelyrUpdateManager: Downloading {download_url} to {tmp_file}...")
                urllib.request.urlretrieve(download_url, tmp_file)
                
                # 3. Install the package
                logger.info("ZelyrUpdateManager: Requesting privileges to install the package...")
                
                if pkg_type == "deb":
                    cmd = ["pkexec", "dpkg", "-i", tmp_file]
                elif pkg_type == "rpm":
                    cmd = ["pkexec", "rpm", "-Uvh", tmp_file]
                elif pkg_type == "exe":
                    # Launch silent installer
                    cmd = [tmp_file, "/S"]
                elif pkg_type == "dmg":
                    # Mount DMG, copy app, unmount
                    cmd = f"hdiutil attach {tmp_file} && cp -R /Volumes/Zelyr/Zelyr.app /Applications/ && hdiutil detach /Volumes/Zelyr"
                    
                if pkg_type == "dmg":
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                if result.returncode != 0:
                    logger.error(f"ZelyrUpdateManager: Installation failed.\n{result.stderr}")
                    return False
                    
                logger.info("ZelyrUpdateManager: Package installed successfully.")
                
                # 4. Restart Daemon
                logger.info("ZelyrUpdateManager: Restarting background daemon...")
                if platform.system() == "Linux":
                    subprocess.run(["systemctl", "--user", "restart", "zelyr.service"], check=False)
                elif platform.system() == "Darwin":
                    # Launchd logic
                    subprocess.run(["launchctl", "stop", "com.zelyr.daemon"], check=False)
                    subprocess.run(["launchctl", "start", "com.zelyr.daemon"], check=False)
                elif platform.system() == "Windows":
                    # Windows service logic (or just inform user)
                    logger.info("Windows OTA completed. Please restart Zelyr manually.")
                return True
                
            except urllib.error.HTTPError as e:
                if e.code in [403, 429]:
                    logger.error("ZelyrUpdateManager: GitHub API rate limit exceeded. Please try again later.")
                else:
                    logger.error(f"ZelyrUpdateManager: HTTP Error during update - {e}")
                return False
            except Exception as e:
                logger.error(f"ZelyrUpdateManager: Exception during update - {e}")
                return False

        success = await asyncio.to_thread(_do_update)
        return success

# Legacy AXIOM API compatibility aliases
AxiomUpdateManager = ZelyrUpdateManager
