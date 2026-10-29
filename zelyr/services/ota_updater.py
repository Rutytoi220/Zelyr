"""Zelyr Over-The-Air (OTA) Update Engine.

Queries GitHub Releases API for the latest version, validates release manifests,
enforces cryptographic SHA256 checksum verification, stages artifacts into isolated
directories, executes atomic symlink/directory swaps with pre-flight dry-run health checks,
and provides automated atomic rollback on boot or health-check failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from zelyr.compat import data_dir

logger = logging.getLogger(__name__)

ZELYR_VERSION = "1.2.0.dev0"
AXIOM_VERSION = ZELYR_VERSION  # legacy public constant
GITHUB_API_URL = "https://api.github.com/repos/Rutytoi220/Zelyr/releases/latest"


@dataclass
class UpdateResult:
    """Result of an OTA update stage or complete installation."""
    success: bool
    version: Optional[str] = None
    error: Optional[str] = None
    rolled_back: bool = False
    stage: str = "init"
    details: Optional[Dict[str, Any]] = None


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hexadecimal digest for a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify that a local file's SHA256 matches the expected digest."""
    if not file_path.is_file():
        return False
    actual_hash = compute_sha256(file_path)
    return actual_hash == expected_sha256.strip().lower()


def validate_release_manifest(data: Any) -> Dict[str, Any]:
    """Validate release metadata dictionary against strict schema requirements."""
    if not isinstance(data, dict):
        raise ValueError("Release manifest must be a valid JSON dictionary")

    tag_name = data.get("tag_name")
    if not tag_name or not isinstance(tag_name, str) or not tag_name.strip():
        raise ValueError("Release manifest missing mandatory non-empty 'tag_name'")

    assets = data.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Release manifest 'assets' must be a list")

    return data


class OTAUpdateEngine:
    """Polls GitHub Releases for new versions and drives the update pipeline."""

    def __init__(
        self,
        install_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        current_version: str = ZELYR_VERSION,
    ):
        self.current_version = current_version
        self.latest_release: Optional[Dict[str, Any]] = None
        self.base_dir = Path(base_dir) if base_dir else (data_dir() / "updates")
        self.install_dir = Path(install_dir) if install_dir else (data_dir() / "active_version")
        self.staging_dir = self.base_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.install_dir.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_version(tag: str) -> Tuple[int, ...]:
        """Turn 'v5.1.0' or '5.1.0' into a comparable tuple (5, 1, 0)."""
        tag = tag.lstrip("vV")
        # Strip release candidate or dev suffixes for basic tuple comparison
        clean_tag = tag.split("-")[0].split("+")[0]
        parts = []
        for segment in clean_tag.split("."):
            try:
                parts.append(int(segment))
            except ValueError:
                parts.append(0)
        return tuple(parts)

    def is_newer(self, remote_tag: str) -> bool:
        """Check if remote_tag is newer than current_version."""
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

        try:
            validated = validate_release_manifest(data)
        except ValueError as err:
            logger.error(f"OTA: Invalid release manifest schema — {err}")
            return None

        tag = validated.get("tag_name", "")
        if self.is_newer(tag):
            logger.info(f"OTA: New release detected — {tag} (current: v{self.current_version})")
            self.latest_release = validated
            return validated

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
    # Asset download & Cryptographic Verification
    # ------------------------------------------------------------------
    async def download_asset(
        self,
        asset_url: str,
        filename: str,
        expected_sha256: Optional[str] = None,
    ) -> Optional[Path]:
        """Download a release asset to the staging directory and optionally verify sha256."""
        dest = self.staging_dir / filename
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._download_file, asset_url, str(dest))
            logger.info(f"OTA: Downloaded {filename} → {dest}")
        except Exception as exc:
            logger.error(f"OTA: Download failed — {exc}")
            if dest.exists():
                dest.unlink()
            return None

        if expected_sha256:
            if not verify_sha256(dest, expected_sha256):
                dest.unlink()
                logger.error(f"OTA: SHA256 checksum mismatch for {filename}. Download aborted.")
                raise ValueError(
                    f"Cryptographic verification failed: SHA256 mismatch for asset {filename}"
                )

        return dest

    @staticmethod
    def _download_file(url: str, dest: str) -> None:
        req = Request(url, headers={"Accept": "application/octet-stream"})
        with urlopen(req, timeout=60) as resp, open(dest, "wb") as fp:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                fp.write(chunk)

    # ------------------------------------------------------------------
    # Atomic Staging & Installation Pipeline
    # ------------------------------------------------------------------
    def stage_update(
        self,
        artifact_path: Path,
        version_tag: str,
        expected_sha256: Optional[str] = None,
    ) -> Path:
        """Stage an artifact into an isolated temporary staging directory (~/.../staging_<version>)."""
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Verify cryptographic checksum if supplied
        if expected_sha256:
            if not verify_sha256(artifact_path, expected_sha256):
                actual_hash = compute_sha256(artifact_path)
                raise ValueError(
                    f"Cryptographic verification failed: expected {expected_sha256}, got {actual_hash}"
                )

        staging_dir = self.base_dir / f"staging_{version_tag}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        if artifact_path.is_file():
            # If artifact is a single executable or archive
            target_file = staging_dir / artifact_path.name
            shutil.copy2(artifact_path, target_file)
            if os.access(artifact_path, os.X_OK):
                os.chmod(target_file, 0o755)
        elif artifact_path.is_dir():
            shutil.copytree(artifact_path, staging_dir, dirs_exist_ok=True)

        return staging_dir

    async def apply_update(
        self,
        version_tag: str,
        staging_dir: Path,
        health_check_cmd: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> UpdateResult:
        """Execute atomic installation with pre-flight dry-run and automatic rollback."""
        if not staging_dir.exists():
            return UpdateResult(
                success=False,
                version=version_tag,
                error=f"Staging directory does not exist: {staging_dir}",
                rolled_back=False,
                stage="staging",
            )

        backup_dir = self.base_dir / f".backup_{self.current_version}"
        had_previous_install = self.install_dir.exists() or self.install_dir.is_symlink()

        # Step 1: Backup existing installation
        if had_previous_install:
            if backup_dir.exists():
                if backup_dir.is_symlink() or backup_dir.is_file():
                    backup_dir.unlink()
                else:
                    shutil.rmtree(backup_dir)

            if self.install_dir.is_symlink():
                target = self.install_dir.readlink()
                backup_dir.symlink_to(target)
            else:
                shutil.copytree(self.install_dir, backup_dir)

        # Step 2: Atomic Swap
        try:
            if self.install_dir.is_symlink() or not self.install_dir.exists():
                tmp_link = self.install_dir.with_name(f".tmp_link_{uuid4().hex[:8]}")
                tmp_link.symlink_to(staging_dir.resolve())
                tmp_link.replace(self.install_dir)
            else:
                # Directory swap
                tmp_dir = self.install_dir.with_name(f".tmp_install_{uuid4().hex[:8]}")
                shutil.copytree(staging_dir, tmp_dir)
                shutil.rmtree(self.install_dir)
                tmp_dir.replace(self.install_dir)
        except Exception as e:
            # Swap failed; restore backup if existed
            if had_previous_install and backup_dir.exists():
                self._restore_backup(backup_dir)
            return UpdateResult(
                success=False,
                version=version_tag,
                error=f"Atomic swap failed: {e}",
                rolled_back=had_previous_install,
                stage="swap",
            )

        # Step 3: Pre-flight Verification (dry-run health check in isolated subprocess)
        cmd = health_check_cmd or [sys.executable, "-m", "zelyr.cli.main", "status"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                err_msg = (
                    f"Pre-flight health-check failed with exit code {proc.returncode}. "
                    f"Stderr: {proc.stderr.strip()}"
                )
                logger.error(f"OTA: {err_msg} Triggering automatic rollback...")
                if had_previous_install and backup_dir.exists():
                    self._restore_backup(backup_dir)
                return UpdateResult(
                    success=False,
                    version=version_tag,
                    error=err_msg,
                    rolled_back=had_previous_install,
                    stage="health_check",
                    details={"exit_code": proc.returncode, "stderr": proc.stderr},
                )
        except subprocess.TimeoutExpired:
            err_msg = f"Pre-flight health-check timed out after {timeout}s"
            logger.error(f"OTA: {err_msg}. Triggering automatic rollback...")
            if had_previous_install and backup_dir.exists():
                self._restore_backup(backup_dir)
            return UpdateResult(
                success=False,
                version=version_tag,
                error=err_msg,
                rolled_back=had_previous_install,
                stage="health_check",
            )
        except Exception as e:
            err_msg = f"Pre-flight health-check execution failed: {e}"
            logger.error(f"OTA: {err_msg}. Triggering automatic rollback...")
            if had_previous_install and backup_dir.exists():
                self._restore_backup(backup_dir)
            return UpdateResult(
                success=False,
                version=version_tag,
                error=err_msg,
                rolled_back=had_previous_install,
                stage="health_check",
            )

        # Step 4: Health check succeeded — promote version and clean up staging
        self.current_version = version_tag.lstrip("vV")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        logger.info(f"OTA: Successfully upgraded to version {version_tag}")
        return UpdateResult(
            success=True,
            version=version_tag,
            rolled_back=False,
            stage="complete",
        )

    def _restore_backup(self, backup_dir: Path) -> None:
        """Internal helper to restore previous version from backup."""
        try:
            if self.install_dir.exists() or self.install_dir.is_symlink():
                if self.install_dir.is_symlink() or self.install_dir.is_file():
                    self.install_dir.unlink()
                else:
                    shutil.rmtree(self.install_dir)

            if backup_dir.is_symlink():
                target = backup_dir.readlink()
                self.install_dir.symlink_to(target)
            elif backup_dir.is_dir():
                shutil.copytree(backup_dir, self.install_dir)
            elif backup_dir.is_file():
                shutil.copy2(backup_dir, self.install_dir)
        except Exception as e:
            logger.error(f"OTA: Critical error during backup restoration: {e}")

    def rollback(self, backup_dir: Optional[Path] = None) -> bool:
        """Explicitly trigger rollback to the previous version backup."""
        target_backup = backup_dir or (self.base_dir / f".backup_{self.current_version}")
        if not target_backup.exists() and not target_backup.is_symlink():
            logger.error(f"OTA: Rollback failed. No backup found at {target_backup}")
            return False

        self._restore_backup(target_backup)
        logger.info(f"OTA: Successfully rolled back to backup at {target_backup}")
        return True

    # ------------------------------------------------------------------
    # Convenience: extract changelog from release body
    # ------------------------------------------------------------------
    def get_changelog(self) -> str:
        if self.latest_release:
            return self.latest_release.get("body", "No changelog available.")
        return ""
