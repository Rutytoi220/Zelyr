import logging
import asyncio
import aiohttp
from zelyr import __version__

logger = logging.getLogger(__name__)

class UpdateManager:
    def __init__(self):
        self.current_version = __version__
        self.repo_api_url = "https://api.github.com/repos/Rutytoi220/Zelyr/releases/latest"

    async def check_for_updates(self) -> dict:
        """Checks GitHub API for the latest release tag."""
        logger.info("UpdateManager: Checking for updates...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.repo_api_url, headers={'User-Agent': 'Zelyr-Updater'}, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        latest_version = data.get("tag_name", self.current_version)
                    else:
                        logger.error(f"UpdateManager: Failed to fetch updates - Status {response.status}")
                        latest_version = self.current_version
        except Exception as e:
            logger.error(f"UpdateManager: Exception during update check - {e}")
            latest_version = self.current_version

        # Remove 'v' prefix if exists for comparison
        clean_latest = latest_version.lstrip('v')
        clean_current = self.current_version.lstrip('v')
        
        # Simple semantic version check
        # Using packaging.version if possible, else string fallback
        try:
            from packaging.version import parse
            update_available = parse(clean_latest) > parse(clean_current)
        except ImportError:
            # Fallback string comparison for simplicity if packaging isn't available
            update_available = clean_latest != clean_current and clean_latest > clean_current

        if update_available:
            logger.info(f"UpdateManager: New update available! {self.current_version} -> {latest_version}")
        else:
            logger.info("UpdateManager: System is up to date.")

        return {
            "update_available": update_available,
            "latest_version": latest_version,
            "current_version": self.current_version
        }
