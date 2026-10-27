"""Cryptographic Release Artifact Verifier.

Validates downloaded update tarballs against SHA256 checksums to prevent
supply-chain attacks before any installation step proceeds.
"""
from zelyr.compat import data_dir
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from zelyr.engine.audit_ledger import AuditLedger

logger = logging.getLogger(__name__)

STAGING_DIR = data_dir() / "staging"


class ChecksumMismatchError(Exception):
    """Raised when the computed hash does not match the expected checksum."""


class ReleaseSecurityVerifier:
    """Verifies SHA256 checksums of downloaded release artifacts."""

    def __init__(self):
        self.ledger = AuditLedger()

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------
    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Compute the SHA256 hex-digest of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as fp:
            for chunk in iter(lambda: fp.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_artifact(
        self, artifact_path: Path, expected_hash: str
    ) -> bool:
        """Compare artifact SHA256 against the expected hash.

        Raises ChecksumMismatchError on failure and wipes the staging
        directory to prevent any tainted data from persisting.
        """
        computed = self.compute_sha256(artifact_path)
        logger.info(
            f"ReleaseVerifier: Computed SHA256 = {computed}  |  Expected = {expected_hash}"
        )

        if computed != expected_hash.strip().lower():
            logger.critical(
                "ReleaseVerifier: ⚠️  CHECKSUM MISMATCH — possible supply-chain compromise!"
            )
            self.ledger.log_execution(
                "ReleaseSecurityVerifier",
                "verify_artifact",
                {
                    "file": str(artifact_path),
                    "expected": expected_hash,
                    "computed": computed,
                },
                "CRITICAL",
                "REJECTED",
            )
            self._wipe_staging()
            raise ChecksumMismatchError(
                f"Hash mismatch for {artifact_path.name}: "
                f"expected {expected_hash}, got {computed}"
            )

        logger.info("ReleaseVerifier: ✅  Checksum verified successfully.")
        self.ledger.log_execution(
            "ReleaseSecurityVerifier",
            "verify_artifact",
            {"file": str(artifact_path), "hash": computed},
            "LOW",
            "VERIFIED",
        )
        return True

    # ------------------------------------------------------------------
    # Parse a standard sha256sums.txt file
    # ------------------------------------------------------------------
    @staticmethod
    def parse_checksums_file(checksums_path: Path) -> dict:
        """Parse a ``sha256sums.txt`` file into ``{filename: hash}``."""
        result = {}
        with open(checksums_path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    hash_val, fname = parts
                    # Some tools use ``*filename`` for binary mode
                    fname = fname.lstrip("*").strip()
                    result[fname] = hash_val.lower()
        return result

    # ------------------------------------------------------------------
    # Staging cleanup
    # ------------------------------------------------------------------
    @staticmethod
    def _wipe_staging():
        """Remove all files from the staging directory."""
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR, ignore_errors=True)
            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("ReleaseVerifier: Staging directory wiped.")
