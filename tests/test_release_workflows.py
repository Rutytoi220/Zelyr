"""Offline verification tests for release workflow integrity, tag guards, and bundle packaging."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from email.utils import getaddresses
from pathlib import Path

import pytest
import yaml

from scripts import check_release_assets, check_release_tag, package_windows_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _load_workflow(filename: str) -> dict:
    raw = (WORKFLOWS_DIR / filename).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def _get_on_block(workflow: dict) -> dict:
    # PyYAML 1.1 parses bare `on:` as boolean True.
    on_block = workflow.get("on", workflow.get(True))
    assert isinstance(on_block, dict)
    return on_block


class TestReleaseWorkflowInvariants:
    """Structural security and single-owner release-gating invariants."""

    def test_only_publish_workflow_triggers_on_tag_push(self) -> None:
        publish = _load_workflow("publish.yml")
        release = _load_workflow("release.yml")
        attach = _load_workflow("attach-appimage.yml")
        build_all = _load_workflow("build-all.yml")

        publish_on = _get_on_block(publish)
        assert "push" in publish_on
        assert publish_on["push"]["tags"] == ["v*.*.*"]

        # Reusable builders must not trigger directly on push or dispatch.
        release_on = _get_on_block(release)
        assert list(release_on.keys()) == ["workflow_call"]

        build_all_on = _get_on_block(build_all)
        assert list(build_all_on.keys()) == ["workflow_call"]

        # Manual AppImage attachment lives in its own workflow_dispatch-only workflow.
        attach_on = _get_on_block(attach)
        assert list(attach_on.keys()) == ["workflow_dispatch"]

    def test_publish_workflow_fails_closed_and_gates_github_release(self) -> None:
        publish = _load_workflow("publish.yml")
        assert publish["permissions"] == {"contents": "read"}
        jobs = publish["jobs"]

        # Build job uses Python 3.11 (compatible with requires-python = ">=3.10, <3.12").
        build_job = jobs["validate-and-build-dist"]
        assert build_job["permissions"] == {"contents": "read"}
        build_steps = build_job["steps"]
        setup_py = next(s for s in build_steps if s.get("uses", "").startswith("actions/setup-python@"))
        assert setup_py["with"]["python-version"] == "3.11"
        assert any("check_release_tag.py" in s.get("run", "") for s in build_steps)

        # Automatic AppImage & cross-platform callers must receive strictly contents: read.
        assert jobs["appimage"]["permissions"] == {"contents": "read"}
        assert jobs["cross-platform"]["permissions"] == {"contents": "read"}

        # PyPI publish job must fail closed (no continue-on-error) and wait for tests/builds.
        pypi_job = jobs["pypi-publish"]
        assert set(pypi_job["needs"]) == {"validate-and-build-dist", "appimage", "cross-platform"}
        assert pypi_job["permissions"] == {"id-token": "write", "contents": "read"}
        assert pypi_job["environment"]["name"] == "pypi"
        assert any(
            "check_release_assets.py" in s.get("run", "") and "--mode pypi" in s.get("run", "")
            for s in pypi_job["steps"]
        )

        pypi_step = next(
            s for s in pypi_job["steps"] if s.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
        )
        assert pypi_step.get("continue-on-error") is not True

        diag_step = next(s for s in pypi_job["steps"] if "Diagnostics" in s.get("name", ""))
        assert "failure()" in diag_step["if"]

        # GitHub Release job must be the sole release writer in publish.yml and validate exact assets.
        gh_job = jobs["github-release"]
        assert set(gh_job["needs"]) == {
            "validate-and-build-dist",
            "appimage",
            "cross-platform",
            "pypi-publish",
        }
        assert gh_job["permissions"] == {"contents": "write"}
        assert any(
            "check_release_assets.py" in s.get("run", "") and "--mode release" in s.get("run", "")
            for s in gh_job["steps"]
        )
        gh_release_step = next(
            s for s in gh_job["steps"] if s.get("uses", "").startswith("softprops/action-gh-release@")
        )
        assert gh_release_step["uses"] == "softprops/action-gh-release@v2"
        assert gh_release_step["with"]["fail_on_unmatched_files"] is True

    def test_reusable_appimage_workflow_is_strictly_read_only_and_pins_tag(self) -> None:
        release = _load_workflow("release.yml")
        assert release["permissions"] == {"contents": "read"}
        jobs = release["jobs"]
        assert set(jobs.keys()) == {"test", "build-appimage"}

        for job_name in ("test", "build-appimage"):
            job = jobs[job_name]
            assert job["permissions"] == {"contents": "read"}
            assert not any("action-gh-release" in s.get("uses", "") for s in job["steps"])
            checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@"))
            assert checkout["with"]["ref"] == "refs/tags/${{ inputs.tag }}"
            assert checkout["with"]["fetch-depth"] == 0
            assert any("check_release_tag.py" in s.get("run", "") for s in job["steps"])

    def test_manual_attach_workflow_isolates_write_permission_and_uses_gh_release_upload(self) -> None:
        attach = _load_workflow("attach-appimage.yml")
        assert attach["permissions"] == {"contents": "read"}
        jobs = attach["jobs"]
        assert set(jobs.keys()) == {"build", "attach-appimage"}

        # Build caller must not grant or inherit contents: write.
        build_job = jobs["build"]
        assert build_job["uses"] == "./.github/workflows/release.yml"
        assert build_job["with"]["tag"] == "${{ inputs.tag }}"
        assert build_job["permissions"] == {"contents": "read"}

        # Only attach-appimage holds contents: write, runs without an unnecessary checkout,
        # and uses `gh release upload` (never `softprops/action-gh-release` or `gh release create`).
        attach_job = jobs["attach-appimage"]
        assert attach_job["needs"] == "build"
        assert attach_job["permissions"] == {"contents": "write"}

        steps = attach_job["steps"]
        assert not any(s.get("uses", "").startswith("actions/checkout@") for s in steps)
        assert not any("action-gh-release" in s.get("uses", "") for s in steps)
        assert not any("gh release create" in s.get("run", "") for s in steps)

        upload_steps = [s for s in steps if "gh release upload" in s.get("run", "")]
        assert len(upload_steps) == 1
        upload_step = upload_steps[0]
        assert upload_step["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
        assert upload_step["env"]["GH_REPO"] == "${{ github.repository }}"
        assert upload_step["env"]["RELEASE_TAG"] == "${{ inputs.tag }}"

    def test_manual_attach_step_fails_closed_without_creating_release(self, tmp_path: Path) -> None:
        """Execute the actual attach-appimage shell step against a mocked `gh` CLI."""
        attach = _load_workflow("attach-appimage.yml")
        upload_step = next(
            s for s in attach["jobs"]["attach-appimage"]["steps"] if "gh release upload" in s.get("run", "")
        )
        script = upload_step["run"]

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log_file = tmp_path / "gh_calls.log"
        fake_gh = bin_dir / "gh"
        fake_gh.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$*" >> "{log_file}"\n'
            'if [ "${1:-}" = "release" ] && [ "${2:-}" = "view" ]; then\n'
            '  exit "${MOCK_VIEW_EXIT:-0}"\n'
            'fi\n'
            'if [ "${1:-}" = "release" ] && [ "${2:-}" = "upload" ]; then\n'
            '  exit "${MOCK_UPLOAD_EXIT:-0}"\n'
            'fi\n'
            'exit 99\n',
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)

        work_dir = tmp_path / "work"
        assets_dir = work_dir / "release-assets"
        assets_dir.mkdir(parents=True)
        appimage = assets_dir / "Zelyr-v1.4.0-x86_64.AppImage"
        appimage.write_bytes(b"fake-appimage-payload")

        base_env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "GH_TOKEN": "mock-token",
            "GH_REPO": "Rutytoi220/Zelyr",
            "RELEASE_TAG": "v1.4.0",
        }

        # 1. Missing release: `gh release view` fails -> script fails without invoking upload or create.
        log_file.write_text("", encoding="utf-8")
        missing = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            env={**base_env, "MOCK_VIEW_EXIT": "1", "MOCK_UPLOAD_EXIT": "0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert missing.returncode != 0
        calls_missing = log_file.read_text(encoding="utf-8").splitlines()
        assert calls_missing == ["release view v1.4.0 --repo Rutytoi220/Zelyr"]
        assert not any("create" in c for c in calls_missing)

        # 2. TOCTOU race (release deleted between view and upload): `gh release upload` fails -> no release created.
        log_file.write_text("", encoding="utf-8")
        toctou = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            env={**base_env, "MOCK_VIEW_EXIT": "0", "MOCK_UPLOAD_EXIT": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert toctou.returncode != 0
        calls_toctou = log_file.read_text(encoding="utf-8").splitlines()
        assert calls_toctou == [
            "release view v1.4.0 --repo Rutytoi220/Zelyr",
            "release upload v1.4.0 release-assets/Zelyr-v1.4.0-x86_64.AppImage --repo Rutytoi220/Zelyr",
        ]
        assert not any("create" in c for c in calls_toctou)

        # 3. Existing release + valid AppImage: succeeds with exact tag, file, and repo.
        log_file.write_text("", encoding="utf-8")
        ok = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            env={**base_env, "MOCK_VIEW_EXIT": "0", "MOCK_UPLOAD_EXIT": "0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert ok.returncode == 0, ok.stderr
        calls_ok = log_file.read_text(encoding="utf-8").splitlines()
        assert calls_ok == [
            "release view v1.4.0 --repo Rutytoi220/Zelyr",
            "release upload v1.4.0 release-assets/Zelyr-v1.4.0-x86_64.AppImage --repo Rutytoi220/Zelyr",
        ]

        # 4. Zero-byte AppImage: fails before invoking `gh` at all.
        appimage.write_bytes(b"")
        log_file.write_text("", encoding="utf-8")
        empty = subprocess.run(
            ["bash", "-c", script],
            cwd=work_dir,
            env={**base_env, "MOCK_VIEW_EXIT": "0", "MOCK_UPLOAD_EXIT": "0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert empty.returncode != 0
        assert log_file.read_text(encoding="utf-8") == ""

    def test_build_all_workflow_is_read_only_and_packages_complete_bundles(self) -> None:
        build_all = _load_workflow("build-all.yml")
        assert build_all["permissions"] == {"contents": "read"}
        build_job = build_all["jobs"]["build"]
        assert build_job["permissions"] == {"contents": "read"}
        steps = build_job["steps"]

        # Must never directly create or mutate GitHub Releases.
        assert not any("action-gh-release" in s.get("uses", "") for s in steps)

        # Must check out the exact requested tag and verify it.
        checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["ref"] == "refs/tags/${{ inputs.tag }}"
        assert checkout["with"]["fetch-depth"] == 0
        assert any("check_release_tag.py" in s.get("run", "") for s in steps)

        # macOS DMG filename must include version instead of zelyr_latest.dmg.
        raw = (WORKFLOWS_DIR / "build-all.yml").read_text(encoding="utf-8")
        assert "zelyr_latest.dmg" not in raw
        assert "Zelyr-v${VERSION}-macos.dmg" in raw

        # Windows must package the complete --onedir bundle via package_windows_bundle.py.
        assert "package_windows_bundle.py" in raw
        assert "Zelyr-v*-windows-x64.zip" in raw

    def test_build_appimage_pins_appimagetool_with_sha256_verification(self) -> None:
        script_text = (REPO_ROOT / "scripts" / "build_appimage.sh").read_text(encoding="utf-8")
        assert "AppImageKit/releases/download/continuous" not in script_text
        assert "AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage" in script_text
        assert 'APPIMAGETOOL_VERSION="1.9.1"' in script_text
        assert 'APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"' in script_text
        assert "sha256sum -c" in script_text


class TestCheckReleaseAssets:
    """Behavior tests for exact versioned artifact validation (wheel, sdist, DEB, RPM, AppImage, DMG, ZIP)."""

    @staticmethod
    def _populate_files(directory: Path, names: list[str], payload: bytes = b"artifact-bytes") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(payload)

    def test_accepts_exact_v1_4_0_release_and_pypi_assets(self, tmp_path: Path) -> None:
        pypi_dir = tmp_path / "dist"
        self._populate_files(
            pypi_dir,
            [
                "zelyr_core-1.4.0-py3-none-any.whl",
                "zelyr_core-1.4.0.tar.gz",
            ],
        )
        verified_pypi = check_release_assets.validate_release_assets(pypi_dir, "v1.4.0", "pypi")
        assert len(verified_pypi) == 2

        release_dir = tmp_path / "release-assets"
        self._populate_files(
            release_dir,
            [
                "zelyr_core-1.4.0-py3-none-any.whl",
                "zelyr_core-1.4.0.tar.gz",
                "Zelyr-v1.4.0-x86_64.AppImage",
                "zelyr_1.4.0_amd64.deb",
                "zelyr-1.4.0-1.x86_64.rpm",
                "Zelyr-v1.4.0-macos.dmg",
                "Zelyr-v1.4.0-windows-x64.zip",
            ],
        )
        verified_release = check_release_assets.validate_release_assets(release_dir, "v1.4.0", "release")
        assert len(verified_release) == 7

    def test_accepts_prerelease_version_conventions(self, tmp_path: Path) -> None:
        release_dir = tmp_path / "rc-assets"
        self._populate_files(
            release_dir,
            [
                "zelyr_core-1.2.0rc1-py3-none-any.whl",
                "zelyr_core-1.2.0rc1.tar.gz",
                "Zelyr-v1.2.0-rc1-x86_64.AppImage",
                "zelyr_1.2.0-rc1_amd64.deb",
                "zelyr-1.2.0_rc1-1.x86_64.rpm",
                "Zelyr-v1.2.0-rc1-macos.dmg",
                "Zelyr-v1.2.0-rc1-windows-x64.zip",
            ],
        )
        verified = check_release_assets.validate_release_assets(release_dir, "v1.2.0-rc1", "release")
        assert len(verified) == 7

    @pytest.mark.parametrize(
        ("bad_file", "replace_target"),
        [
            ("zelyr_core-1.3.0-py3-none-any.whl", "zelyr_core-1.4.0-py3-none-any.whl"),
            ("zelyr_core-1.3.0.tar.gz", "zelyr_core-1.4.0.tar.gz"),
            ("zelyr_1.3.0_amd64.deb", "zelyr_1.4.0_amd64.deb"),
            ("zelyr-1.3.0-1.x86_64.rpm", "zelyr-1.4.0-1.x86_64.rpm"),
        ],
    )
    def test_rejects_wrong_version_wheel_sdist_deb_or_rpm(
        self, tmp_path: Path, bad_file: str, replace_target: str
    ) -> None:
        assets = [
            "zelyr_core-1.4.0-py3-none-any.whl",
            "zelyr_core-1.4.0.tar.gz",
            "Zelyr-v1.4.0-x86_64.AppImage",
            "zelyr_1.4.0_amd64.deb",
            "zelyr-1.4.0-1.x86_64.rpm",
            "Zelyr-v1.4.0-macos.dmg",
            "Zelyr-v1.4.0-windows-x64.zip",
        ]
        mutated = [bad_file if name == replace_target else name for name in assets]
        release_dir = tmp_path / "bad-release"
        self._populate_files(release_dir, mutated)

        with pytest.raises(ValueError, match="Missing expected artifacts.*Unexpected or wrong-version"):
            check_release_assets.validate_release_assets(release_dir, "v1.4.0", "release")

    def test_rejects_empty_and_duplicate_artifacts(self, tmp_path: Path) -> None:
        pypi_dir = tmp_path / "empty-dist"
        self._populate_files(pypi_dir, ["zelyr_core-1.4.0-py3-none-any.whl"], payload=b"ok")
        (pypi_dir / "zelyr_core-1.4.0.tar.gz").write_bytes(b"")

        with pytest.raises(ValueError, match="Zero-byte artifacts"):
            check_release_assets.validate_release_assets(pypi_dir, "v1.4.0", "pypi")

        (pypi_dir / "zelyr_core-1.4.0.tar.gz").write_bytes(b"ok")
        (pypi_dir / "zelyr_core-1.3.0-py3-none-any.whl").write_bytes(b"stale")
        with pytest.raises(ValueError, match="Unexpected or wrong-version"):
            check_release_assets.validate_release_assets(pypi_dir, "v1.4.0", "pypi")

    def test_cli_invocation_works_directly_without_pythonpath(self, tmp_path: Path) -> None:
        pypi_dir = tmp_path / "dist"
        self._populate_files(
            pypi_dir,
            [
                "zelyr_core-1.4.0-py3-none-any.whl",
                "zelyr_core-1.4.0.tar.gz",
            ],
        )
        ok_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_release_assets.py"),
                "--tag-env",
                "RELEASE_TAG",
                "--assets-dir",
                str(pypi_dir),
                "--mode",
                "pypi",
            ],
            cwd=tmp_path,
            env={**os.environ, "RELEASE_TAG": "v1.4.0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert ok_proc.returncode == 0, ok_proc.stderr

        fail_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_release_assets.py"),
                "--tag-env",
                "RELEASE_TAG",
                "--assets-dir",
                str(pypi_dir),
                "--mode",
                "pypi",
            ],
            cwd=tmp_path,
            env={**os.environ, "RELEASE_TAG": "v1.4.1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert fail_proc.returncode == 1


class TestVerifyReleaseTag:
    """Unit tests for tag formatting, version parity, and git checkout verification."""

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("v1.4.0", "1.4.0"),
            ("v1.2.0-rc1", "1.2.0rc1"),
            ("v2.0.0-alpha.1", "2.0.0a1"),
            ("v1.4.0-beta2", "1.4.0b2"),
        ],
    )
    def test_normalize_valid_tags(self, tag: str, expected: str) -> None:
        assert check_release_tag.normalize_tag_to_pep440(tag) == expected

    @pytest.mark.parametrize(
        "bad_tag",
        [
            "",
            "1.4.0",
            "main",
            "refs/tags/v1.4.0",
            "v1.4",
            "v1.4.0; echo pwned",
            " v1.4.0",
        ],
    )
    def test_reject_invalid_tags(self, bad_tag: str) -> None:
        with pytest.raises(ValueError):
            check_release_tag.normalize_tag_to_pep440(bad_tag)

    def test_version_guard_matches_and_rejects_mismatches(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "zelyr-core"\nversion = "1.4.0"\n', encoding="utf-8")

        assert check_release_tag.verify_version_match("v1.4.0", pyproject) == "1.4.0"

        with pytest.raises(ValueError, match="does not match"):
            check_release_tag.verify_version_match("v1.4.1", pyproject)

        with pytest.raises(ValueError, match="does not match"):
            check_release_tag.verify_version_match("v1.4.0-rc1", pyproject)

    def test_fallback_toml_parser_reads_project_version_only(self, tmp_path: Path, monkeypatch) -> None:
        sample = '[build-system]\nversion = "9.9.9"\n[project]\nname = "zelyr-core"\nversion = "1.4.0"\n[tool.foo]\nversion = "0.0.1"\n'
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(sample, encoding="utf-8")
        monkeypatch.setattr(check_release_tag, "tomllib", None)
        assert check_release_tag.read_project_version(pyproject) == "1.4.0"

    def test_checkout_verification_against_real_ephemeral_git_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> str:
            res = subprocess.run(
                ["git", *args],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "Test",
                    "GIT_AUTHOR_EMAIL": "test@example.com",
                    "GIT_COMMITTER_NAME": "Test",
                    "GIT_COMMITTER_EMAIL": "test@example.com",
                },
            )
            return res.stdout.strip()

        git("init", "-b", "main")
        (repo / "pyproject.toml").write_text('[project]\nname = "zelyr-core"\nversion = "1.4.0"\n', encoding="utf-8")
        git("add", "pyproject.toml")
        git("commit", "-m", "v1.4.0 commit")
        tag_sha = git("rev-parse", "HEAD")
        git("tag", "-a", "v1.4.0", "-m", "Release v1.4.0")
        git("tag", "v1.4.0-rc1")

        # Checked out on exact tagged commit -> passes for both annotated and lightweight tags.
        assert check_release_tag.verify_checkout_matches_tag("v1.4.0", repo) == tag_sha
        assert check_release_tag.verify_checkout_matches_tag("v1.4.0-rc1", repo) == tag_sha

        # Advance main past the tag -> fails closed (prevents building from main for an older tag).
        (repo / "README.md").write_text("later commit on main\n", encoding="utf-8")
        git("add", "README.md")
        git("commit", "-m", "post-tag commit on main")
        with pytest.raises(ValueError, match="does not match tag"):
            check_release_tag.verify_checkout_matches_tag("v1.4.0", repo)

        # Nonexistent tag -> fails closed.
        with pytest.raises(ValueError, match="Failed to resolve git ref"):
            check_release_tag.verify_checkout_matches_tag("v9.9.9", repo)

        # CLI --tag-env end-to-end check on detached tag checkout.
        git("checkout", "--detach", "refs/tags/v1.4.0")
        cli_ok = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_release_tag.py"),
                "--tag-env",
                "RELEASE_TAG",
                "--repo-dir",
                str(repo),
                "--check-checkout",
                "--check-version",
            ],
            env={**os.environ, "RELEASE_TAG": "v1.4.0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert cli_ok.returncode == 0, cli_ok.stderr


class TestPackageWindowsBundle:
    """Unit tests for packaging the complete PyInstaller --onedir Windows bundle."""

    def test_packages_complete_onedir_bundle_preserving_folder_structure(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "dist" / "Zelyr"
        internal = source_dir / "_internal" / "zelyr" / "memory"
        internal.mkdir(parents=True)
        (source_dir / "Zelyr.exe").write_bytes(b"MZ-fake-pe-header")
        (source_dir / "_internal" / "python311.dll").write_bytes(b"fake-dll")
        (internal / "schema.sql").write_text("CREATE TABLE t (id INT);\n", encoding="utf-8")

        outdir = tmp_path / "out"
        archive = package_windows_bundle.package_windows_bundle(source_dir, outdir, "v1.4.0")

        assert archive == outdir / "Zelyr-v1.4.0-windows-x64.zip"
        assert archive.is_file()
        with zipfile.ZipFile(archive, "r") as zf:
            names = set(zf.namelist())
        assert names == {
            "Zelyr/Zelyr.exe",
            "Zelyr/_internal/python311.dll",
            "Zelyr/_internal/zelyr/memory/schema.sql",
        }

    def test_rejects_standalone_exe_without_onedir_dependencies(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "dist" / "Zelyr"
        source_dir.mkdir(parents=True)
        (source_dir / "Zelyr.exe").write_bytes(b"MZ-only-exe")

        with pytest.raises(ValueError, match="no sibling runtime dependencies"):
            package_windows_bundle.package_windows_bundle(source_dir, tmp_path / "out", "v1.4.0")

    def test_rejects_missing_executable(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "dist" / "Zelyr"
        (source_dir / "_internal").mkdir(parents=True)
        (source_dir / "_internal" / "python311.dll").write_bytes(b"dll")

        with pytest.raises(ValueError, match="Missing or empty executable"):
            package_windows_bundle.package_windows_bundle(source_dir, tmp_path / "out", "v1.4.0")

    def test_cli_invocation_works_directly_without_pythonpath(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "dist" / "Zelyr"
        (source_dir / "_internal").mkdir(parents=True)
        (source_dir / "Zelyr.exe").write_bytes(b"MZ-exe")
        (source_dir / "_internal" / "python311.dll").write_bytes(b"dll")
        outdir = tmp_path / "out"

        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "package_windows_bundle.py"),
                "--tag-env",
                "RELEASE_TAG",
                "--source-dir",
                str(source_dir),
                "--outdir",
                str(outdir),
            ],
            cwd=tmp_path,
            env={**os.environ, "RELEASE_TAG": "v1.4.0"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert (outdir / "Zelyr-v1.4.0-windows-x64.zip").is_file()


RESERVED_EMAIL_SUFFIXES = (".local", ".invalid", ".test", ".localhost", ".example")
RESERVED_EMAIL_DOMAINS = {"localhost", "example.com", "example.net", "example.org"}


def _assert_no_reserved_email_domains(metadata_text: str) -> None:
    """Parse RFC 5322 Core Metadata and reject reserved or special-use email domains."""
    msg = Parser().parsestr(metadata_text)
    for header in ("Author-email", "Maintainer-email"):
        values = msg.get_all(header, [])
        for _display_name, addr in getaddresses(values):
            if not addr:
                continue
            assert "@" in addr, f"Malformed email address in {header}: {addr!r}"
            domain = addr.rsplit("@", 1)[1].strip().lower().rstrip(".")
            if domain in RESERVED_EMAIL_DOMAINS or any(domain.endswith(suffix) for suffix in RESERVED_EMAIL_SUFFIXES):
                raise AssertionError(
                    f"Reserved or special-use email domain in {header}: {addr!r} (domain={domain!r})"
                )


class TestDistributionMetadataIntegrity:
    """Regression tests ensuring PyPI distribution metadata never uses reserved email domains."""

    @pytest.mark.parametrize(
        "bad_email",
        [
            "admin@zelyr.local",
            "dev@zelyr.invalid",
            "ci@zelyr.test",
            "root@host.localhost",
            "user@localhost",
            "maintainer@zelyr.example",
            "maintainer@example.com",
        ],
    )
    def test_rejects_reserved_and_special_use_email_domains(self, bad_email: str) -> None:
        metadata = f"Metadata-Version: 2.1\nName: zelyr-core\nVersion: 1.4.1\nAuthor-email: Zelyr Contributors <{bad_email}>\n"
        with pytest.raises(AssertionError, match="Reserved or special-use email domain"):
            _assert_no_reserved_email_domains(metadata)

    def test_accepts_author_without_email_and_valid_public_email(self) -> None:
        _assert_no_reserved_email_domains(
            "Metadata-Version: 2.1\nName: zelyr-core\nVersion: 1.4.1\nAuthor: Zelyr Contributors\n"
        )
        _assert_no_reserved_email_domains(
            "Metadata-Version: 2.1\nName: zelyr-core\nVersion: 1.4.1\nAuthor-email: Zelyr Contributors <dev@zelyr.org>\n"
        )

    def test_pyproject_authors_have_no_reserved_emails(self) -> None:
        """Parse pyproject.toml and validate every declared contact email (if any)."""
        if check_release_tag.tomllib is None:
            pytest.skip("neither tomllib nor tomli is available to parse pyproject.toml")

        data = check_release_tag.tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        assert project["name"] == "zelyr-core"
        assert any(entry.get("name") == "Zelyr Contributors" for entry in project["authors"])

        for field in ("authors", "maintainers"):
            for entry in project.get(field, []):
                email = entry.get("email")
                if email:
                    # A valid public address may be added in the future; reserved ones must never be.
                    _assert_no_reserved_email_domains(f"Author-email: {entry.get('name', '')} <{email}>\n")

    def test_built_distribution_metadata_is_pypi_compliant(self, tmp_path: Path) -> None:
        """Build actual distribution metadata and verify no reserved email domains are emitted."""
        expected_version = check_release_tag.read_project_version(REPO_ROOT / "pyproject.toml")
        metadata_payloads: list[tuple[str, str]] = []

        if importlib.util.find_spec("build") is not None:
            outdir = tmp_path / "dist"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_dist.py"),
                    "--no-isolation",
                    "--outdir",
                    str(outdir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, f"build_dist.py failed:\n{proc.stdout}\n{proc.stderr}"

            wheel = outdir / f"zelyr_core-{expected_version}-py3-none-any.whl"
            sdist = outdir / f"zelyr_core-{expected_version}.tar.gz"
            assert wheel.is_file(), f"Expected wheel not built: {wheel}"
            assert sdist.is_file(), f"Expected sdist not built: {sdist}"

            with zipfile.ZipFile(wheel, "r") as zf:
                meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
                metadata_payloads.append(("wheel:METADATA", zf.read(meta_name).decode("utf-8")))

            with tarfile.open(sdist, "r:gz") as tf:
                pkg_info_member = next(m for m in tf.getmembers() if m.name.endswith("/PKG-INFO"))
                extracted = tf.extractfile(pkg_info_member)
                assert extracted is not None
                metadata_payloads.append(("sdist:PKG-INFO", extracted.read().decode("utf-8")))
        else:
            import setuptools.build_meta as build_meta

            meta_dir = tmp_path / "meta"
            meta_dir.mkdir()
            prev_cwd = os.getcwd()
            try:
                os.chdir(REPO_ROOT)
                dist_info = build_meta.prepare_metadata_for_build_wheel(str(meta_dir))
            finally:
                os.chdir(prev_cwd)
            metadata_text = (meta_dir / dist_info / "METADATA").read_text(encoding="utf-8")
            metadata_payloads.append(("dist-info:METADATA", metadata_text))

        assert metadata_payloads, "No distribution metadata was generated for inspection"
        for label, payload in metadata_payloads:
            msg = Parser().parsestr(payload)
            assert msg["Name"] == "zelyr-core", f"{label} Name mismatch"
            assert msg["Version"] == expected_version, f"{label} Version mismatch"
            author_headers = " ".join(msg.get(h, "") for h in ("Author", "Author-email"))
            assert "Zelyr Contributors" in author_headers, f"{label} lost the author name"
            # Only inspect metadata headers: the long description (README body) is not validated by PyPI.
            header_text = "\n".join(f"{key}: {value}" for key, value in msg.items())
            assert "admin@zelyr.local" not in header_text, f"{label} still contains admin@zelyr.local"
            _assert_no_reserved_email_domains(payload)
