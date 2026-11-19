#!/usr/bin/env python3
"""Verify release tag formatting, git checkout integrity, and pyproject.toml version parity.

Used by release workflows to ensure:
1. A release tag has a valid leading-``v`` version format (no branches or shell metacharacters).
2. The checked-out ``HEAD`` commit matches ``refs/tags/<tag>^{commit}`` exactly.
3. The tag version matches ``[project].version`` in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent

# Strict version tag pattern: v<major>.<minor>.<patch> with optional prerelease suffix.
TAG_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)(?:-(rc|a|b|alpha|beta)\.?(\d+))?$")
PEP440_PRERELEASE_MAP = {
    "a": "a",
    "alpha": "a",
    "b": "b",
    "beta": "b",
    "rc": "rc",
}


def normalize_tag_to_pep440(tag: str) -> str:
    """Validate a git release tag and return its normalized PEP 440 version string."""
    if not tag or not isinstance(tag, str):
        raise ValueError("Release tag must be a non-empty string.")
    match = TAG_PATTERN.fullmatch(tag.strip())
    if not match or tag != tag.strip():
        raise ValueError(
            f"Invalid release tag {tag!r}. Expected format like 'v1.4.0' or 'v1.4.0-rc1'."
        )
    base, pre_kind, pre_num = match.groups()
    if pre_kind is not None:
        return f"{base}{PEP440_PRERELEASE_MAP[pre_kind]}{pre_num}"
    return base


def _parse_project_version_fallback(text: str) -> str:
    """Minimal stdlib-only parser for ``[project].version`` when tomllib/tomli is unavailable."""
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if in_project:
            match = re.match(r'^version\s*=\s*"([^"]+)"', line)
            if match:
                return match.group(1).strip()
    raise ValueError("Missing or empty [project].version in pyproject.toml")


def read_project_version(pyproject_path: Path) -> str:
    """Read ``[project].version`` from ``pyproject.toml``."""
    if not pyproject_path.is_file():
        raise ValueError(f"pyproject.toml not found at {pyproject_path}")
    if tomllib is not None:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
        version = data.get("project", {}).get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Missing or empty [project].version in {pyproject_path}")
        return version.strip()
    return _parse_project_version_fallback(pyproject_path.read_text(encoding="utf-8"))


def verify_version_match(tag: str, pyproject_path: Path = ROOT / "pyproject.toml") -> str:
    """Verify that ``tag`` matches ``[project].version`` in ``pyproject.toml``."""
    expected_version = normalize_tag_to_pep440(tag)
    project_version = read_project_version(pyproject_path)
    if expected_version != project_version:
        raise ValueError(
            f"Release tag {tag!r} (normalized: {expected_version!r}) does not match "
            f"pyproject.toml version {project_version!r}."
        )
    return project_version


def _git_rev_parse(ref: str, repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise ValueError(f"Failed to resolve git ref {ref!r}: {detail}")
    return proc.stdout.strip()


def verify_checkout_matches_tag(tag: str, repo_dir: Path = ROOT) -> str:
    """Verify that ``refs/tags/<tag>`` exists and points to the exact ``HEAD`` commit."""
    normalize_tag_to_pep440(tag)
    tag_commit = _git_rev_parse(f"refs/tags/{tag}^{{commit}}", repo_dir)
    head_commit = _git_rev_parse("HEAD", repo_dir)
    if head_commit != tag_commit:
        raise ValueError(
            f"Checked-out HEAD ({head_commit}) does not match tag {tag!r} commit ({tag_commit})."
        )
    return head_commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify release tag format, git checkout commit, and pyproject.toml version."
    )
    tag_group = parser.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--tag", help="Release tag string (e.g. v1.4.0)")
    tag_group.add_argument(
        "--tag-env",
        help="Environment variable name holding the release tag string",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=ROOT,
        help="Repository root directory (default: script parent repo)",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=None,
        help="Path to pyproject.toml (default: <repo-dir>/pyproject.toml)",
    )
    parser.add_argument(
        "--check-version",
        action="store_true",
        help="Verify tag version matches [project].version in pyproject.toml",
    )
    parser.add_argument(
        "--check-checkout",
        action="store_true",
        help="Verify HEAD commit matches refs/tags/<tag>^{commit}",
    )
    args = parser.parse_args(argv)

    tag = os.environ.get(args.tag_env, "") if args.tag_env else args.tag
    pyproject_path = args.pyproject or (args.repo_dir / "pyproject.toml")

    try:
        normalized = normalize_tag_to_pep440(tag)
        if args.check_version:
            project_version = verify_version_match(tag, pyproject_path)
            print(
                f"[check_release_tag] Version match OK: tag={tag} "
                f"(normalized={normalized}) == pyproject.toml ({project_version})"
            )
        if args.check_checkout:
            commit = verify_checkout_matches_tag(tag, args.repo_dir)
            print(f"[check_release_tag] Checkout match OK: HEAD == refs/tags/{tag} ({commit})")
        if not args.check_version and not args.check_checkout:
            print(f"[check_release_tag] Tag format OK: {tag} (normalized={normalized})")
    except ValueError as exc:
        print(f"[check_release_tag] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
