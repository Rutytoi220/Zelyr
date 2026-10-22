#!/usr/bin/env python3
"""Build AXIOM distributions from an isolated copy of the source tree.

Setuptools' ``bdist_wheel`` command reuses ``build/lib`` inside the project
root.  If a module is deleted from the source and the wheel is rebuilt without
cleaning that directory, the deleted module is silently packaged again.

This script copies only the inputs consumed by the setuptools build into a
temporary directory, so ``build/lib`` always starts empty and the artifacts
reflect the current source tree.  It also verifies that every ``.py`` file in
the wheel exists in the source copy before reporting success.

Usage::

    python scripts/build_dist.py                 # sdist + wheel in dist/
    python scripts/build_dist.py --wheel         # wheel only
    python scripts/build_dist.py --no-isolation  # reuse the current environment
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Top-level inputs consumed by the setuptools build.  ``axiom`` is the only
# package matched by ``[tool.setuptools.packages.find] include = ["axiom*"]``.
BUILD_INPUTS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "COPYING",
    "axiom",
)


def _copy_inputs(destination: Path) -> None:
    copied = False
    for name in BUILD_INPUTS:
        src = ROOT / name
        if not src.exists():
            continue
        dst = destination / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            shutil.copy2(src, dst)
        copied = True
    if not copied:
        raise SystemExit("No build inputs found; run this from the repository root.")


def _verify_wheel(wheel: Path, source_root: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        missing = [
            name
            for name in archive.namelist()
            if name.endswith(".py")
            and not name.startswith(("axiom_core-", "axiom-"))
            and not (source_root / name).is_file()
        ]
    if missing:
        raise SystemExit(
            "Wheel contains files absent from the current source tree:\n  "
            + "\n  ".join(sorted(missing))
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build sdist/wheel from a pristine copy of the source tree."
    )
    parser.add_argument(
        "--outdir", default="dist", help="artifact directory (default: dist)"
    )
    parser.add_argument("--wheel", action="store_true", help="build only the wheel")
    parser.add_argument("--sdist", action="store_true", help="build only the sdist")
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="reuse the current environment instead of creating an isolated one",
    )
    parser.add_argument(
        "--keep-src", action="store_true", help="keep the temporary source copy"
    )
    args = parser.parse_args()
    if args.wheel and args.sdist:
        parser.error("--wheel and --sdist are mutually exclusive")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix="axiom-build-src-"))
    source = temp_root / "src"
    started = time.time()
    try:
        # Keep directory creation inside the try block so the temporary tree
        # is removed even if copying or the build itself fails.
        source.mkdir()
        _copy_inputs(source)

        cmd = [sys.executable, "-m", "build", "--outdir", str(outdir)]
        if args.wheel:
            cmd.append("--wheel")
        elif args.sdist:
            cmd.append("--sdist")
        if args.no_isolation:
            cmd.append("--no-isolation")

        print(f"[build_dist] Building from isolated source: {source}")
        subprocess.run(cmd, cwd=source, check=True)

        wheels = [
            path for path in outdir.glob("*.whl") if path.stat().st_mtime >= started
        ]
        for wheel in wheels:
            _verify_wheel(wheel, source)
            print(f"[build_dist] Verified wheel contents: {wheel}")

        artifacts = sorted(
            path for path in outdir.iterdir() if path.stat().st_mtime >= started
        )
        if not artifacts:
            raise SystemExit("Build completed but no new artifact was found.")
        for artifact in artifacts:
            print(f"[build_dist] Artifact: {artifact}")
    finally:
        if args.keep_src:
            print(f"[build_dist] Temporary source kept at {source}")
        else:
            # Only the mkdtemp tree is removed; ``outdir`` is never touched.
            shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
