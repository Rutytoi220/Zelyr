#!/usr/bin/env python3
"""Validate exact versioned release artifact filenames and non-empty payloads.

Ensures that before PyPI publication or GitHub Release creation:
1. Every expected artifact for the release tag exists and is non-empty.
2. Exact normalized versions match across Python (PEP 440), DEB, RPM, AppImage,
   macOS DMG, and Windows ZIP conventions.
3. No extra, duplicate, or wrong-version artifacts are present in the staging directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_release_tag import normalize_tag_to_pep440


def expected_asset_filenames(tag: str, mode: str) -> list[str]:
    """Return the exact sorted list of expected artifact filenames for ``tag`` and ``mode``."""
    pep440_version = normalize_tag_to_pep440(tag)
    raw_version = tag[1:]
    rpm_version = raw_version.replace("-", "_")

    python_assets = [
        f"zelyr_core-{pep440_version}-py3-none-any.whl",
        f"zelyr_core-{pep440_version}.tar.gz",
    ]
    if mode == "pypi":
        return sorted(python_assets)
    if mode == "release":
        return sorted(
            [
                *python_assets,
                f"Zelyr-v{raw_version}-x86_64.AppImage",
                f"zelyr_{raw_version}_amd64.deb",
                f"zelyr-{rpm_version}-1.x86_64.rpm",
                f"Zelyr-v{raw_version}-macos.dmg",
                f"Zelyr-v{raw_version}-windows-x64.zip",
            ]
        )
    raise ValueError(f"Unsupported validation mode {mode!r}; expected 'pypi' or 'release'.")


def validate_release_assets(assets_dir: Path, tag: str, mode: str) -> list[Path]:
    """Validate that ``assets_dir`` contains the exact expected non-empty artifacts."""
    expected_names = expected_asset_filenames(tag, mode)
    if not assets_dir.is_dir():
        raise ValueError(f"Assets directory not found: {assets_dir}")

    present_files = sorted(path for path in assets_dir.iterdir() if path.is_file())
    present_names = [path.name for path in present_files]

    expected_set = set(expected_names)
    present_set = set(present_names)

    missing = sorted(expected_set - present_set)
    unexpected = sorted(present_set - expected_set)

    errors: list[str] = []
    if missing:
        errors.append(f"Missing expected artifacts for {tag}: {', '.join(missing)}")
    if unexpected:
        errors.append(f"Unexpected or wrong-version artifacts in {assets_dir}: {', '.join(unexpected)}")

    empty = sorted(path.name for path in present_files if path.name in expected_set and path.stat().st_size == 0)
    if empty:
        errors.append(f"Zero-byte artifacts detected: {', '.join(empty)}")

    if errors:
        raise ValueError("; ".join(errors))

    return [assets_dir / name for name in expected_names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate exact versioned release artifact filenames and non-empty sizes."
    )
    tag_group = parser.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--tag", help="Release tag (e.g. v1.4.0)")
    tag_group.add_argument(
        "--tag-env",
        help="Environment variable containing the release tag",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded artifacts to validate",
    )
    parser.add_argument(
        "--mode",
        choices=("pypi", "release"),
        required=True,
        help="Validation scope: 'pypi' (wheel + sdist) or 'release' (all 7 release assets)",
    )
    args = parser.parse_args(argv)

    tag = os.environ.get(args.tag_env, "") if args.tag_env else args.tag
    try:
        verified = validate_release_assets(args.assets_dir, tag, args.mode)
        names = ", ".join(p.name for p in verified)
        print(f"[check_release_assets] Verified {len(verified)} {args.mode} artifacts for {tag}: {names}")
    except ValueError as exc:
        print(f"[check_release_assets] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
