#!/usr/bin/env python3
"""Package a complete PyInstaller --onedir Windows bundle into a versioned zip archive.

PyInstaller builds ``dist/Zelyr/`` with ``exclude_binaries=True`` and ``COLLECT``
(see ``zelyr.spec``), meaning ``Zelyr.exe`` requires its sibling runtime files
(such as ``_internal/``) to execute. Uploading only ``Zelyr.exe`` produces a
non-functional binary.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_release_tag import normalize_tag_to_pep440


def package_windows_bundle(
    source_dir: Path,
    outdir: Path,
    tag: str,
) -> Path:
    """Validate and zip a complete PyInstaller ``--onedir`` Windows directory."""
    # Validate tag format (e.g., v1.4.0 -> version string 1.4.0; preserves raw suffix after 'v').
    normalize_tag_to_pep440(tag)
    version_label = tag[1:]

    if not source_dir.is_dir():
        raise ValueError(f"PyInstaller bundle directory not found: {source_dir}")

    exe_path = source_dir / "Zelyr.exe"
    if not exe_path.is_file() or exe_path.stat().st_size == 0:
        raise ValueError(f"Missing or empty executable in bundle: {exe_path}")

    bundle_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    companion_files = [path for path in bundle_files if path != exe_path]
    if not companion_files:
        raise ValueError(
            f"Bundle at {source_dir} contains only Zelyr.exe with no sibling runtime "
            "dependencies (expected PyInstaller --onedir files such as _internal/)."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    archive_path = outdir / f"Zelyr-v{version_label}-windows-x64.zip"
    root_folder = source_dir.name

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in bundle_files:
            arcname = Path(root_folder) / file_path.relative_to(source_dir)
            archive.write(file_path, arcname.as_posix())

    # Re-read archive to verify completeness.
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        expected_exe = f"{root_folder}/Zelyr.exe"
        if expected_exe not in names or len(names) < 2:
            raise ValueError(
                f"Generated archive {archive_path} failed completeness check (entries: {len(names)})."
            )

    return archive_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package the complete Windows PyInstaller --onedir bundle into a versioned zip."
    )
    tag_group = parser.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--tag", help="Release tag (e.g. v1.4.0)")
    tag_group.add_argument(
        "--tag-env",
        help="Environment variable containing the release tag",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dist/Zelyr"),
        help="Path to the PyInstaller --onedir output directory (default: dist/Zelyr)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory for the zip archive (default: .)",
    )
    args = parser.parse_args(argv)

    tag = os.environ.get(args.tag_env, "") if args.tag_env else args.tag
    try:
        archive = package_windows_bundle(args.source_dir, args.outdir, tag)
        print(f"[package_windows_bundle] Created verified Windows bundle: {archive}")
    except ValueError as exc:
        print(f"[package_windows_bundle] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
