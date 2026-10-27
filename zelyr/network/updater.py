import urllib.request
import json
import logging
from packaging.version import Version, InvalidVersion
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class UpdateManager:
    def __init__(self, repo: str = "ollama/ollama"):
        # using a known active repo for fallback test if ChienGPT doesn't exist
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{self.repo}/releases/latest"

    def check_for_updates(self, current_version: str) -> Optional[Dict]:
        try:
            req = urllib.request.Request(
                self.api_url, 
                headers={'User-Agent': 'Zelyr-Updater'}
            )
            with urllib.request.urlopen(req, timeout=10.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            remote_tag = data.get("tag_name", "").lstrip("v")
            current = current_version.lstrip("v")
            
            try:
                if Version(remote_tag) > Version(current):
                    # Find first asset url if available
                    assets = data.get("assets", [])
                    download_url = assets[0]["browser_download_url"] if assets else None
                    
                    return {
                        "version": remote_tag,
                        "body": data.get("body", "No changelog provided."),
                        "download_url": download_url
                    }
            except InvalidVersion:
                logger.warning(f"Invalid version format during comparison: local={current}, remote={remote_tag}")
                
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            
        return None
