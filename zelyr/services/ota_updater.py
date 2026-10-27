"""Zelyr Over-The-Air (OTA) Update Engine.

Queries GitHub Releases API for the latest version, compares semantically
against the running build, and orchestrates the download/verify/install pipeline.
"""
from zelyr.compat import data_dir
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

ZELYR_VERSION = "5.0.0"
AXIOM_VERSION = ZELYR_VERSION  # legacy public constant
GITHUB_API_URL = "https://api.github.com/repos/Rutytoi220/Zelyr/releases/latest"
STAGING_DIR = data_dir() / "staging"


class OTAUpdateEngine:
    """Polls GitHub Releases for new versions and drives the update pipeline."""

    def __init__(self):
        self.current_version = ZELYR_VERSION
        self.latest_release: Optional[Dict[str, Any]] = None
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_version(tag: str) -> tuple:
        """Turn 'v5.1.0' or '5.1.0' into a comparable tuple (5, 1, 0)."""
        tag = tag.lstrip("vV")
        parts = []
        for segment in tag.split("."):
            try:
                parts.append(int(segment))
            except ValueError:
                parts.append(0)
        return tuple(parts)

    def is_newer(self, remote_tag: str) -> bool:
        return self._parse_version(remote_tag) > self._parse_version(self.current_version)

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------
    async def check_for_update(self) -> Optional[Dict[str, Any]]:
        """Query the GitHub Releases API (non-blocking via executor)."""
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_latest_release),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OTA: GitHub API request timed out after 10 s.")
            return None
        except Exception as exc:
            logger.error(f"OTA: Failed to query GitHub API — {exc}")
            return None

        if data is None:
            return None

        tag = data.get("tag_name", "")
        if not tag:
            logger.info("OTA: Remote release has no tag_name; skipping.")
            return None

        if self.is_newer(tag):
            logger.info(f"OTA: New release detected — {tag} (current: v{self.current_version})")
            self.latest_release = data
            return data

        logger.info(f"OTA: Already on latest (v{self.current_version} >= {tag}).")
        return None

    def _fetch_latest_release(self) -> Optional[Dict[str, Any]]:
        req = Request(GITHUB_API_URL, headers={"Accept": "application/vnd.github+json"})
        try:
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            logger.error(f"OTA: Network error — {exc}")
            return None

    # ------------------------------------------------------------------
    # Asset download
    # ------------------------------------------------------------------
    async def download_asset(self, asset_url: str, filename: str) -> Optional[Path]:
        """Download a release asset to the staging directory."""
        dest = STAGING_DIR / filename
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._download_file, asset_url, str(dest))
            logger.info(f"OTA: Downloaded {filename} → {dest}")
            return dest
        except Exception as exc:
            logger.error(f"OTA: Download failed — {exc}")
            return None

    @staticmethod
    def _download_file(url: str, dest: str):
        req = Request(url, headers={"Accept": "application/octet-stream"})
        with urlopen(req, timeout=60) as resp, open(dest, "wb") as fp:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                fp.write(chunk)

    # ------------------------------------------------------------------
    # Convenience: extract changelog from release body
    # ------------------------------------------------------------------
    def get_changelog(self) -> str:
        if self.latest_release:
            return self.latest_release.get("body", "No changelog available.")
        return ""
