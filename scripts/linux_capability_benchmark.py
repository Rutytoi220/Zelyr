"""Reproducible Linux Agent Capability Matrix benchmark runner & validator."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = REPO_ROOT / ".venv" / "bin"
ZELYR_BIN = VENV_BIN / "zelyr"
PYTEST_BIN = VENV_BIN / "pytest"
PYTHON_BIN = VENV_BIN / "python"
OLLAMA_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 300
DEFAULT_ARTIFACT_DIR = Path(
    "/home/rutytoi/.gemini/antigravity-cli/brain/4565cef2-0c66-4842-9048-81590ec71565"
)

ALL_TASKS = (
    "task1_filesystem",
    "task2_repo_inspection",
    "task3_failure_diagnosis",
    "task4_multi_file_coding",
    "task5_verification_recovery",
    "task6_execution_prohibition",
)

COMPAT_SUBSET_TASKS = (
    "task1_filesystem",
    "task2_repo_inspection",
    "task3_failure_diagnosis",
    "task6_execution_prohibition",
)


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b""
        req_json: Optional[Dict[str, Any]] = None
        if raw_body:
            try:
                req_json = json.loads(raw_body.decode("utf-8"))
            except Exception:
                req_json = None

        upstream_req = urllib.request.Request(
            f"{OLLAMA_UPSTREAM}{self.path}",
            data=raw_body,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            method="POST",
        )
        turn_record: Optional[Dict[str, Any]] = None
        if self.path.endswith("/api/chat") and isinstance(req_json, dict):
            turn_record = {
                "timestamp": time.time(),
                "model": req_json.get("model"),
                "options": req_json.get("options"),
                "tools_count": len(req_json.get("tools") or []),
                "tool_names": [
                    (t.get("function", {}).get("name") or t.get("name"))
                    for t in (req_json.get("tools") or [])
                    if isinstance(t, dict)
                ],
                "messages_count": len(req_json.get("messages") or []),
                "response_thinking": [],
                "response_content": [],
                "response_tool_calls": [],
                "done_reason": None,
                "prompt_eval_count": None,
                "eval_count": None,
            }

        try:
            with urllib.request.urlopen(upstream_req, timeout=DEFAULT_TIMEOUT_S + 30) as resp:
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/x-ndjson"))
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except BrokenPipeError:
                        break
                    if turn_record is not None:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            msg = chunk.get("message") or {}
                            if msg.get("thinking"):
                                turn_record["response_thinking"].append(msg["thinking"])
                            if msg.get("content"):
                                turn_record["response_content"].append(msg["content"])
                            if msg.get("tool_calls"):
                                turn_record["response_tool_calls"].extend(msg["tool_calls"])
                            if chunk.get("done"):
                                turn_record["done_reason"] = chunk.get("done_reason")
                                turn_record["prompt_eval_count"] = chunk.get("prompt_eval_count")
                                turn_record["eval_count"] = chunk.get("eval_count")
                        except Exception:
                            pass
        except Exception as exc:
            err_payload = json.dumps({"error": str(exc)}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_payload)))
                self.end_headers()
                self.wfile.write(err_payload)
            except BrokenPipeError:
                pass
            if turn_record is not None:
                turn_record["proxy_error"] = str(exc)

        if turn_record is not None:
            turn_record["response_thinking_len"] = sum(len(s) for s in turn_record.pop("response_thinking"))
            turn_record["response_content"] = "".join(turn_record["response_content"])
            self.server.records.append(turn_record)  # type: ignore[attr-defined]


class _ProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.records: List[Dict[str, Any]] = []


def unload_all_ollama_models() -> None:
    """Unload any loaded Ollama models cleanly between sequential trials."""
    try:
        ps = json.loads(urllib.request.urlopen(f"{OLLAMA_UPSTREAM}/api/ps", timeout=5).read().decode("utf-8"))
        for m in ps.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                req = urllib.request.Request(
                    f"{OLLAMA_UPSTREAM}/api/generate",
                    data=json.dumps({"model": name, "keep_alive": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    urllib.request.urlopen(req, timeout=15).read()
                except Exception:
                    pass
        time.sleep(1.5)
    except Exception:
        pass


def get_ollama_inventory() -> Dict[str, Any]:
    """Query ollama CLI version and local /api/tags inventory."""
    ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    tags = json.loads(urllib.request.urlopen(f"{OLLAMA_UPSTREAM}/api/tags", timeout=5).read().decode("utf-8"))
    by_name = {m["name"]: m for m in tags.get("models", [])}
    return {"ollama_version": ver, "models": by_name}


def git_init(repo: Path) -> None:
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "bench@zelyr.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Zelyr Bench"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def snapshot_hashes(repo: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for p in sorted(repo.rglob("*")):
        if ".git" in p.parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        if p.is_file():
            hashes[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _run_bounded(args: Sequence[str], cwd: Path, timeout: int = 20) -> Tuple[int, str]:
    try:
        proc = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def setup_task_workspace(task_id: str, ws_dir: Path) -> Tuple[str, List[str]]:
    """Create disposable git workspace for one capability task and return (prompt, extra_cli_args)."""
    shutil.rmtree(ws_dir, ignore_errors=True)
    ws_dir.mkdir(parents=True, exist_ok=True)

    if task_id == "task1_filesystem":
        (ws_dir / "README.md").write_text("# Workspace\nClean directory.\n", encoding="utf-8")
        git_init(ws_dir)
        return (
            "Create a directory named capability_test inside the current workspace "
            "and create a file named hello.txt inside it containing exact text: Hello from Zelyr!",
            [],
        )

    if task_id == "task2_repo_inspection":
        (ws_dir / "src" / "greeter").mkdir(parents=True)
        (ws_dir / "tests").mkdir()
        (ws_dir / "pyproject.toml").write_text(
            '[project]\nname = "greeter"\nversion = "0.1.0"\n'
            '[project.scripts]\ngreeter = "greeter.cli:main"\n'
            '[tool.pytest.ini_options]\npythonpath = ["src"]\ntestpaths = ["tests"]\n',
            encoding="utf-8",
        )
        (ws_dir / "README.md").write_text("# Greeter CLI\n\nRun tests with:\n```\npytest -v\n```\n", encoding="utf-8")
        (ws_dir / "src" / "greeter" / "__init__.py").write_text("", encoding="utf-8")
        (ws_dir / "src" / "greeter" / "cli.py").write_text(
            'import sys\ndef main(argv=None):\n    args = argv or sys.argv[1:]\n'
            '    name = args[0] if args else "world"\n    print(f"Hello, {name}!")\n    return 0\n\n'
            'if __name__ == "__main__":\n    raise SystemExit(main())\n',
            encoding="utf-8",
        )
        (ws_dir / "tests" / "test_cli.py").write_text(
            'from greeter.cli import main\ndef test_main(capsys):\n    assert main(["Ada"]) == 0\n'
            '    assert capsys.readouterr().out.strip() == "Hello, Ada!"\n',
            encoding="utf-8",
        )
        git_init(ws_dir)
        return (
            "Inspect the current repository and explain what language it uses, what the entrypoint is, "
            "what test framework is used, and how to run the tests without modifying anything.",
            [],
        )

    if task_id == "task3_failure_diagnosis":
        (ws_dir / "tests").mkdir(parents=True)
        (ws_dir / "pyproject.toml").write_text(
            '[project]\nname = "calc"\nversion = "0.1.0"\n[tool.pytest.ini_options]\npythonpath = ["."]\n',
            encoding="utf-8",
        )
        (ws_dir / "calc.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a - b  # intentional bug: subtracts instead of adds\n",
            encoding="utf-8",
        )
        (ws_dir / "tests" / "test_calc.py").write_text(
            "from calc import add\n\ndef test_add_positive():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        git_init(ws_dir)
        return (
            "Inspect this project and run the existing test suite using pytest -v. Explain whether the tests pass "
            "or fail and why (whether it is a project bug or an environment issue), without modifying any files.",
            [],
        )

    if task_id == "task4_multi_file_coding":
        (ws_dir / "pkg").mkdir(parents=True)
        (ws_dir / "tests").mkdir()
        (ws_dir / "pyproject.toml").write_text(
            '[project]\nname = "kvparser"\nversion = "0.1.0"\n[tool.pytest.ini_options]\npythonpath = ["."]\n',
            encoding="utf-8",
        )
        (ws_dir / "README.md").write_text(
            "# KV Parser\n\nBug: `parse_line` currently crashes when a value contains `=` "
            "(for example `url=https://example.com?a=1&b=2`). Values should preserve `=` characters.\n",
            encoding="utf-8",
        )
        (ws_dir / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (ws_dir / "pkg" / "parser.py").write_text(
            "def parse_line(line: str) -> tuple[str, str]:\n"
            "    if '=' not in line:\n"
            "        raise ValueError('Missing equals sign')\n"
            "    key, value = line.split('=')\n"
            "    return key.strip(), value.strip()\n",
            encoding="utf-8",
        )
        (ws_dir / "tests" / "test_parser.py").write_text(
            "from pkg.parser import parse_line\n\n"
            "def test_basic():\n"
            "    assert parse_line('host=localhost') == ('host', 'localhost')\n",
            encoding="utf-8",
        )
        git_init(ws_dir)
        return (
            "Inspect this repository, fix the key=value parsing bug in pkg/parser.py so values containing '=' preserve "
            "'=' characters, add a regression test in tests/test_parser.py for values containing '=', and run pytest -v "
            "to verify tests pass.",
            [],
        )

    if task_id == "task5_verification_recovery":
        (ws_dir / "tests").mkdir(parents=True)
        (ws_dir / "pyproject.toml").write_text(
            '[project]\nname = "versioning"\nversion = "0.1.0"\n[tool.pytest.ini_options]\npythonpath = ["."]\n',
            encoding="utf-8",
        )
        (ws_dir / "README.md").write_text(
            "# Version Parser\n\nFix `parse_version` in `versioning.py` so `pytest -v` passes all acceptance tests.\n"
            "Requirements: handle optional leading 'v', 2 or 3 numeric components (pad patch=0), "
            "optional '-prerelease' and '+build' suffixes, and raise ValueError on invalid inputs "
            "(empty string, 1 component, >3 components, non-numeric parts).\n",
            encoding="utf-8",
        )
        (ws_dir / "versioning.py").write_text(
            "def parse_version(raw: str) -> tuple[int, int, int]:\n"
            "    # Naive baseline that fails 2-part and prerelease/build tags\n"
            "    parts = raw.lstrip('v').split('.')\n"
            "    return (int(parts[0]), int(parts[1]), int(parts[2]))\n",
            encoding="utf-8",
        )
        (ws_dir / "tests" / "test_versioning.py").write_text(
            "import pytest\nfrom versioning import parse_version\n\n"
            "def test_standard():\n    assert parse_version('v1.2.3') == (1, 2, 3)\n\n"
            "def test_two_part():\n    assert parse_version('2.4') == (2, 4, 0)\n\n"
            "def test_prerelease_and_build():\n    assert parse_version('1.5.0-beta.2+build99') == (1, 5, 0)\n\n"
            "def test_invalid():\n    for bad in ('', 'abc', '1', '1.2.3.4', '1.2.x'):\n"
            "        with pytest.raises(ValueError):\n            parse_version(bad)\n",
            encoding="utf-8",
        )
        git_init(ws_dir)
        return (
            "Run pytest -v on this project, inspect why the acceptance tests fail, "
            "fix versioning.py iteratively until all tests pass, and re-run pytest -v "
            "to confirm all tests pass.",
            ["--verify", "command_output:pytest -q:4 passed"],
        )

    if task_id == "task6_execution_prohibition":
        (ws_dir / "tests").mkdir(parents=True)
        (ws_dir / "app.py").write_text(
            "def divide_all(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += 100 / n  # ZeroDivisionError if n == 0\n"
            "    return total\n",
            encoding="utf-8",
        )
        (ws_dir / "tests" / "test_app.py").write_text(
            "from app import divide_all\ndef test_ok():\n    assert divide_all([2, 4]) == 75\n",
            encoding="utf-8",
        )
        git_init(ws_dir)
        return (
            "Inspect app.py and explain how to fix the bug, but do NOT modify any files or run any mutating commands.",
            [],
        )

    raise ValueError(f"Unknown task_id: {task_id}")


def validate_task_outcome(
    task_id: str,
    ws_dir: Path,
    cli_rc: int,
    stdout: str,
    stderr: str,
    before_hashes: Dict[str, str],
    exec_row: Dict[str, Any],
    diag_data: Dict[str, Any],
    wire_turns: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Independently validate task outcome on disk, git, pytest, and execution ledger."""
    after_hashes = snapshot_hashes(ws_dir)
    git_porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=ws_dir, capture_output=True, text=True).stdout
    changed_files = sorted(ln[3:].strip() for ln in git_porcelain.splitlines() if ln.strip())
    clean_repo = (before_hashes == after_hashes) and (not changed_files)
    lo = (stdout or "").lower()
    ledger_status = exec_row.get("status")
    ledger_ver = exec_row.get("verification_status")
    ledger_meta = exec_row.get("metadata") or {}

    if cli_rc == 124:
        return "TIMEOUT", {"clean_repo": clean_repo, "changed_files": changed_files, "reason": "Subprocess timed out"}

    if task_id == "task1_filesystem":
        dir_ok = (ws_dir / "capability_test").is_dir()
        hello = ws_dir / "capability_test" / "hello.txt"
        file_ok = hello.is_file()
        raw_content = hello.read_text(encoding="utf-8") if file_ok else None
        content_ok = raw_content is not None and raw_content.rstrip("\r\n") == "Hello from Zelyr!"
        ok = cli_rc == 0 and dir_ok and file_ok and content_ok and ledger_status != "failed"
        return ("PASS" if ok else "FAIL"), {
            "dir_exists": dir_ok,
            "file_exists": file_ok,
            "raw_content": raw_content,
            "exact_text_match": content_ok,
            "changed_files": changed_files,
        }

    if task_id == "task2_repo_inspection":
        sys.path.insert(0, str(REPO_ROOT))
        from zelyr.eval.validators import assert_response_semantic_facts

        sem_ok, sem_msg = assert_response_semantic_facts(stdout)
        run_info_ok = any(tok in lo for tok in ("pytest -v", "pytest", "python -m pytest"))
        ok = cli_rc == 0 and clean_repo and sem_ok and run_info_ok
        return ("PASS" if ok else "FAIL"), {
            "clean_repo": clean_repo,
            "changed_files": changed_files,
            "semantic_ok": sem_ok,
            "semantic_detail": sem_msg,
            "run_instructions_present": run_info_ok,
        }

    if task_id == "task3_failure_diagnosis":
        ledger_calls = (diag_data.get("tools") or {}).get("calls") or []
        ran_pytest_wire = any(
            (tc.get("function") or {}).get("name") == "execute_command"
            and "pytest" in str((tc.get("function") or {}).get("arguments", ""))
            for wt in wire_turns
            for tc in (wt.get("response_tool_calls") or [])
        )
        observed_test_fail_stderr = "1 failed" in stderr.lower() or "exit=1" in stderr.lower()
        diag_states = [c.get("execution_status") for c in ledger_calls if c.get("tool_id") == "execute_command"]
        mentions_fail = any(tok in lo for tok in ("fail", "failed", "assertionerror"))
        mentions_bug = any(tok in lo for tok in ("subtract", "a - b", "2 - 3", "minus", "project bug"))
        ok = cli_rc == 0 and clean_repo and ran_pytest_wire and observed_test_fail_stderr and mentions_fail and mentions_bug
        return ("PASS" if ok else "FAIL"), {
            "clean_repo": clean_repo,
            "changed_files": changed_files,
            "ran_pytest_wire": ran_pytest_wire,
            "observed_test_fail_stderr": observed_test_fail_stderr,
            "cmd_execution_statuses": diag_states,
            "mentions_failure": mentions_fail,
            "mentions_project_bug": mentions_bug,
        }

    if task_id == "task4_multi_file_coding":
        pt_rc, pt_out = _run_bounded([str(PYTEST_BIN), "-v"], ws_dir)
        chk_rc, chk_out = _run_bounded(
            [
                str(PYTHON_BIN),
                "-c",
                "from pkg.parser import parse_line; "
                "assert parse_line('host=localhost') == ('host', 'localhost'); "
                "assert parse_line('url=https://example.com?a=1&b=2') == ('url', 'https://example.com?a=1&b=2')",
            ],
            ws_dir,
        )
        diff_files = sorted(
            f for f in subprocess.run(["git", "diff", "--name-only"], cwd=ws_dir, capture_output=True, text=True).stdout.splitlines()
            if f.strip()
        )
        test_txt = (ws_dir / "tests" / "test_parser.py").read_text(encoding="utf-8")
        has_new_test = test_txt.count("def test_") >= 2 and ("=" in test_txt)
        only_intended = diff_files == ["pkg/parser.py", "tests/test_parser.py"] and changed_files == diff_files
        ok = cli_rc == 0 and pt_rc == 0 and chk_rc == 0 and only_intended and has_new_test and ledger_status != "failed"
        return ("PASS" if ok else "FAIL"), {
            "independent_pytest_rc": pt_rc,
            "independent_pytest_tail": pt_out.splitlines()[-3:] if pt_out else [],
            "independent_func_check_rc": chk_rc,
            "independent_func_check_err": chk_out if chk_rc != 0 else "",
            "changed_files": changed_files,
            "only_intended_files": only_intended,
            "has_regression_test": has_new_test,
        }

    if task_id == "task5_verification_recovery":
        pt_rc, pt_out = _run_bounded([str(PYTEST_BIN), "-v"], ws_dir)
        all_4_passed = (
            pt_rc == 0
            and "4 passed" in pt_out
            and all(
                tn in pt_out
                for tn in ("test_standard", "test_two_part", "test_prerelease_and_build", "test_invalid")
            )
        )
        tests_untouched = "tests/test_versioning.py" not in changed_files
        only_versioning = changed_files == ["versioning.py"]
        ver_diag = diag_data.get("verification") or {}
        unres = ledger_meta.get("unresolved_failure_count", 0)
        ledger_ok = ledger_status == "completed" and ledger_ver == "verified" and not unres
        if all_4_passed and cli_rc == 0 and tests_untouched and not ledger_ok:
            verdict = "UNVERIFIED"
        elif all_4_passed and cli_rc == 0 and only_versioning and ledger_ok:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        return verdict, {
            "cli_rc": cli_rc,
            "independent_pytest_rc": pt_rc,
            "all_4_passed": all_4_passed,
            "tests_untouched": tests_untouched,
            "changed_files": changed_files,
            "ledger_status": ledger_status,
            "ledger_verification_status": ledger_ver,
            "diag_verification": ver_diag,
            "independent_pytest_tail": pt_out.splitlines()[-6:] if pt_out else [],
        }

    if task_id == "task6_execution_prohibition":
        policy_comp = ledger_meta.get("policy_compliance") or (diag_data.get("policy") or {}).get("policy_compliance") or {}
        ledger_calls = (diag_data.get("tools") or {}).get("calls") or []
        mut_succeeded = [
            c.get("tool_id")
            for c in ledger_calls
            if c.get("may_have_side_effect") and c.get("execution_status") in ("succeeded_verified", "succeeded_unverified")
        ]
        explains_fix = any(tok in lo for tok in ("zero", "n == 0", "if n", "zerodivision"))
        policy_ok = (
            bool(policy_comp.get("has_prohibitions"))
            and bool(policy_comp.get("compliant"))
            and int(policy_comp.get("prohibited_violations") or 0) == 0
        )
        ok = cli_rc == 0 and clean_repo and explains_fix and policy_ok and not mut_succeeded
        return ("PASS" if ok else "FAIL"), {
            "clean_repo": clean_repo,
            "changed_files": changed_files,
            "explains_fix": explains_fix,
            "policy_compliance": policy_comp,
            "mutating_calls_succeeded": mut_succeeded,
        }

    return "ERROR", {"reason": f"Unhandled task {task_id}"}


def extract_tool_sequence(wire_turns: List[Dict[str, Any]], diag_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Correlate ordered LLM tool calls/arguments with ledger execution results."""
    calls_by_id = {
        c.get("call_id"): c
        for c in ((diag_data.get("tools") or {}).get("calls") or [])
        if c.get("call_id")
    }
    seq: List[Dict[str, Any]] = []
    for t_idx, wt in enumerate(wire_turns, 1):
        for tc in wt.get("response_tool_calls") or []:
            fn_obj = tc.get("function") or {}
            cid = tc.get("id")
            diag_call = calls_by_id.get(cid, {})
            seq.append({
                "llm_turn": t_idx,
                "call_id": cid,
                "tool": fn_obj.get("name") or tc.get("name"),
                "arguments": fn_obj.get("arguments"),
                "lifecycle_state": diag_call.get("lifecycle_state"),
                "execution_status": diag_call.get("execution_status"),
                "verification_status": diag_call.get("verification_status"),
                "error_type": diag_call.get("error_type"),
                "error": diag_call.get("error"),
            })
    return seq


def run_single_task(
    bench_root: Path,
    model: str,
    model_info: Dict[str, Any],
    ollama_version: str,
    task_id: str,
    timeout_s: int,
) -> Dict[str, Any]:
    """Run one CLI task in a completely disposable workspace with isolated ledger/XDG state."""
    slug = model.replace(":", "_").replace("/", "_")
    ws_dir = bench_root / "workspaces" / slug / task_id
    xdg_dir = bench_root / "xdg" / slug / task_id
    db_path = bench_root / "ledgers" / slug / f"{task_id}.db"
    shutil.rmtree(xdg_dir, ignore_errors=True)
    (xdg_dir / "config" / "zelyr").mkdir(parents=True, exist_ok=True)
    (xdg_dir / "data" / "zelyr").mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    prompt, extra_args = setup_task_workspace(task_id, ws_dir)
    before_hashes = snapshot_hashes(ws_dir)

    proxy = _ProxyServer()
    port = proxy.server_address[1]
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(xdg_dir / "config")
    env["XDG_DATA_HOME"] = str(xdg_dir / "data")
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["ZELYR_AUTOPILOT"] = "1"

    cmd = [
        str(ZELYR_BIN),
        "run",
        "--model",
        model,
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--db-path",
        str(db_path),
        *extra_args,
        prompt,
    ]
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=ws_dir, env=env, capture_output=True, text=True, timeout=timeout_s)
        cli_rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        cli_rc = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    duration_s = round(time.monotonic() - t0, 2)

    proxy.shutdown()
    proxy.server_close()

    exec_row: Dict[str, Any] = {}
    diag_data: Dict[str, Any] = {}
    exec_id: Optional[str] = None
    exec_proc = subprocess.run(
        [str(ZELYR_BIN), "executions", "--db-path", str(db_path), "--json"],
        cwd=ws_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    try:
        rows = (json.loads(exec_proc.stdout) or {}).get("executions", [])
        if rows:
            exec_row = rows[0]
            exec_id = exec_row.get("execution_id")
            d_proc = subprocess.run(
                [str(ZELYR_BIN), "diagnose", str(exec_id), "--db-path", str(db_path), "--json"],
                cwd=ws_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            diag_data = json.loads(d_proc.stdout)
    except Exception as exc:
        diag_data = {"parse_error": str(exc)}

    verdict, val_detail = validate_task_outcome(
        task_id, ws_dir, cli_rc, stdout, stderr, before_hashes, exec_row, diag_data, proxy.records
    )

    num_ctx_observed = next(
        ((t.get("options") or {}).get("num_ctx") for t in proxy.records if isinstance(t.get("options"), dict)),
        None,
    )
    return {
        "task_id": task_id,
        "model": model,
        "model_digest": model_info.get("digest"),
        "model_details": model_info.get("details"),
        "ollama_version": ollama_version,
        "context_size_measured": num_ctx_observed,
        "execution_limit_default": 12,
        "exact_cli_cmd": cmd,
        "cli_rc": cli_rc,
        "timed_out": timed_out,
        "duration_s": duration_s,
        "verdict": verdict,
        "validator_detail": val_detail,
        "execution_id": exec_id,
        "ledger_status": exec_row.get("status"),
        "ledger_verification_status": exec_row.get("verification_status"),
        "ledger_metadata": exec_row.get("metadata"),
        "ledger_transitions": diag_data.get("timeline_summary"),
        "tool_sequence": extract_tool_sequence(proxy.records, diag_data),
        "llm_turns": len(proxy.records),
        "max_prompt_eval_count": max((r.get("prompt_eval_count") or 0 for r in proxy.records), default=0),
        "sum_prompt_eval_count": sum((r.get("prompt_eval_count") or 0 for r in proxy.records)),
        "sum_eval_count": sum((r.get("eval_count") or 0 for r in proxy.records)),
        "wire_turns": proxy.records,
        "final_response": stdout.strip(),
        "stderr_log": stderr.strip(),
    }


async def run_isolated_browser_e2e() -> Dict[str, Any]:
    """Run headless isolated Chromium Native Messaging E2E test without altering user browser state."""
    from zelyr.engine.execution_policy import ExecutionPolicyConfig, ExecutionPolicyEngine, PolicyDecisionOutcome
    from zelyr.tools.browser_cdp import InteractWithBrowserTool
    from zelyr.tools.browser_native_host import (
        NativeBrowserClient,
        compute_unpacked_extension_id,
        install_native_messaging_host,
    )

    chromium_path = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if not chromium_path:
        return {"verdict": "UNAVAILABLE", "reason": "No Chromium executable installed on PATH."}

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    tmp = Path(tempfile.mkdtemp(prefix="zelyr_browser_e2e_"))
    httpd = None
    p1 = None
    p2 = None
    try:
        site = tmp / "site"
        site.mkdir()
        (site / "index.html").write_text(
            "<!doctype html><html><head><title>Zelyr Browser E2E</title></head><body>"
            "<main><h1>Zelyr Test Page</h1>"
            "<input id='name-box' type='text' placeholder='Enter label' />"
            "<button id='apply-btn' onclick=\"document.getElementById('status').textContent="
            "'APPLIED:' + document.getElementById('name-box').value\">Apply Label</button>"
            "<p id='status'>IDLE</p></main></body></html>",
            encoding="utf-8",
        )
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site)))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        test_url = f"http://127.0.0.1:{port}/index.html"

        ext_dir = REPO_ROOT / "zelyr" / "tools" / "extension"
        ext_id = compute_unpacked_extension_id(ext_dir)
        home_dir, prof_dir, sock = tmp / "home", tmp / "profile", tmp / "zelyr_native.sock"
        prof_dir.mkdir(parents=True)
        wrapper = tmp / "bin" / "zelyr-browser-host"
        install_native_messaging_host(
            extension_dir=ext_dir,
            hosts_dir=home_dir / ".config" / "chromium" / "NativeMessagingHosts",
            wrapper_path=wrapper,
            allowed_hosts=["127.0.0.1"],
        )
        install_native_messaging_host(
            extension_dir=ext_dir,
            user_data_dir=prof_dir,
            wrapper_path=wrapper,
            allowed_hosts=["127.0.0.1"],
        )
        wrapper.write_text(
            wrapper.read_text(encoding="utf-8").replace(
                "#!/bin/sh\n", f'#!/bin/sh\nexport ZELYR_BROWSER_NATIVE_SOCKET="{sock}"\n'
            ),
            encoding="utf-8",
        )
        os.chmod(wrapper, 0o755)

        env = {
            **os.environ,
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(home_dir / ".config"),
            "ZELYR_BROWSER_NATIVE_SOCKET": str(sock),
        }
        old_sock = os.environ.get("ZELYR_BROWSER_NATIVE_SOCKET")
        os.environ["ZELYR_BROWSER_NATIVE_SOCKET"] = str(sock)
        mock_bridge = MagicMock()
        mock_bridge.server = True
        mock_bridge.is_connected.return_value = False
        try:
            def _launch() -> subprocess.Popen[bytes]:
                return subprocess.Popen(
                    [
                        chromium_path,
                        "--headless=new",
                        "--no-sandbox",
                        "--disable-gpu",
                        f"--user-data-dir={prof_dir}",
                        f"--load-extension={ext_dir}",
                        "about:blank",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )

            p1 = _launch()
            for _ in range(100):
                if sock.exists():
                    break
                await asyncio.sleep(0.1)
            if not sock.exists():
                return {"verdict": "FAIL", "reason": "Extension failed to open AF_UNIX socket."}

            pol = ExecutionPolicyEngine(
                ExecutionPolicyConfig(policy_mode="autopilot", resource_restrictions={"allowed_hosts": ["127.0.0.1"]})
            )
            pol_allow = pol.evaluate("interact_with_browser", {"action": "open_tab", "url": test_url}, required_ring=0)
            pol_deny = pol.evaluate(
                "interact_with_browser", {"action": "open_tab", "url": "https://example.org/"}, required_ring=0
            )
            with patch("zelyr.tools.browser_extension.get_bridge", return_value=mock_bridge):
                tool = InteractWithBrowserTool()
                r_open = await tool.execute({"action": "open_tab", "url": test_url})
                await asyncio.sleep(0.7)
                r_tab = await tool.execute({"action": "get_active_tab"})
                r_snap = await tool.execute({"action": "get_page_snapshot"})
                r_type = await tool.execute({"action": "type", "selector": "#name-box", "text": "ZelyrE2E"})
                r_click = await tool.execute({"action": "click", "selector": "#apply-btn"})
                await asyncio.sleep(0.2)
                r_ext = await tool.execute({"action": "extract_page_content", "mode": "readable"})

                p1.terminate()
                p1.wait(timeout=5)
                p1 = None
                for _ in range(60):
                    if not sock.exists():
                        break
                    await asyncio.sleep(0.1)
                r_disc_client = await NativeBrowserClient(sock).send_command("get_active_tab")
                r_disc_tool = await tool.execute({"action": "get_active_tab", "port": 65432})

                p2 = _launch()
                for _ in range(100):
                    if sock.exists():
                        break
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.3)
                r_reopen = await tool.execute({"action": "open_tab", "url": test_url})
                await asyncio.sleep(0.6)
                r_retab = await tool.execute({"action": "get_active_tab"})

                checks = {
                    "policy_allow_localhost": pol_allow.outcome == PolicyDecisionOutcome.ALLOW,
                    "policy_deny_external": pol_deny.outcome == PolicyDecisionOutcome.DENY,
                    "open_tab": r_open.success,
                    "get_active_tab": r_tab.success and str(port) in str(r_tab.output),
                    "get_page_snapshot": r_snap.success and "Apply Label" in json.dumps(r_snap.output),
                    "dom_type_click_extract": r_type.success and r_click.success and "APPLIED:ZelyrE2E" in json.dumps(r_ext.output),
                    "disconnect_native_client_fails": not r_disc_client.success,
                    "disconnect_tool_fails": (not r_disc_tool.success) and ("No browser extension connected" in (r_disc_tool.error or "")),
                    "reconnect_recovered": r_reopen.success and r_retab.success and str(port) in str(r_retab.output),
                }
                verdict = "PASS" if all(checks.values()) else "FAIL"
                return {
                    "verdict": verdict,
                    "chromium": subprocess.run([chromium_path, "--version"], capture_output=True, text=True).stdout.strip(),
                    "extension_id": ext_id,
                    "checks": checks,
                }
        finally:
            if old_sock is None:
                os.environ.pop("ZELYR_BROWSER_NATIVE_SOCKET", None)
            else:
                os.environ["ZELYR_BROWSER_NATIVE_SOCKET"] = old_sock
    finally:
        for pr in (p1, p2):
            if pr is not None:
                pr.terminate()
                try:
                    pr.wait(timeout=3)
                except Exception:
                    pr.kill()
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


def _summarize_tool_seq(tool_seq: Sequence[Dict[str, Any]]) -> str:
    """Format compact ordered tool-call sequence string with execution outcomes."""
    parts: List[str] = []
    for c in tool_seq:
        tool = c.get("tool") or "?"
        args = c.get("arguments") or {}
        action = args.get("action") if isinstance(args, dict) else None
        status = c.get("execution_status") or c.get("lifecycle_state") or "unknown"
        label = f"{tool}({action})" if action else str(tool)
        parts.append(f"T{c.get('llm_turn')}:{label}[{status}]")
    return " -> ".join(parts) if parts else "none"


def render_markdown_summary(report: Dict[str, Any]) -> str:
    """Render Section-8 compliant Markdown capability matrix summary from JSON report."""
    cfg = report.get("execution_config") or {}
    lines = [
        "# Zelyr Full Linux Agent Capability Matrix Report",
        "",
        f"- **Git Commit:** `{report['git_commit']}` (`{report['git_branch']}`)",
        f"- **Ollama Version:** `{report['ollama_version']}`",
        f"- **Generated At:** `{report['generated_at']}`",
        (
            f"- **Execution Config (measured defaults):** `num_ctx={cfg.get('num_ctx_default')}`, "
            f"`max_steps={cfg.get('max_steps_default')}`, `timeout={cfg.get('timeout_per_task_s')}s`, "
            f"`autopilot={cfg.get('autopilot_env')}`, `sequential={cfg.get('sequential_execution')}`"
        ),
        "",
        "## Per-Model Summary",
        "",
        "| Model | Tier | Digest Prefix | Passed / Total | Total Duration |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ]
    for model_entry in report["results"]:
        tasks = model_entry.get("tasks") or []
        passed = sum(1 for t in tasks if t.get("verdict") == "PASS")
        total_dt = round(sum(float(t.get("duration_s") or 0.0) for t in tasks), 2)
        digest_pref = str(model_entry.get("digest") or "N/A")[:16]
        lines.append(
            f"| `{model_entry['model']}` | `{model_entry.get('tier', 'primary')}` | `{digest_pref}` | "
            f"**{passed}/{len(tasks)}** | {total_dt}s |"
        )

    lines.extend([
        "",
        "## Per-Task Outcomes, Verification & Tool Sequences",
        "",
        "| Model | Task | Verdict | CLI Exit | Ledger Status | Duration | Turns | Max Prompt Tok | Tool Sequence |",
        "| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
    ])
    for model_entry in report["results"]:
        model = model_entry["model"]
        for t in model_entry.get("tasks") or []:
            seq_str = _summarize_tool_seq(t.get("tool_sequence") or [])
            lines.append(
                f"| `{model}` | `{t['task_id']}` | **{t['verdict']}** | `{t.get('cli_rc')}` | "
                f"`{t.get('ledger_status')}/{t.get('ledger_verification_status')}` | {t.get('duration_s')}s | "
                f"{t.get('llm_turns')} | {t.get('max_prompt_eval_count')} | {seq_str} |"
            )

    browser = report.get("browser_e2e")
    if browser:
        lines.extend([
            "",
            "## Browser Native Messaging E2E",
            "",
            f"- **Verdict:** **{browser.get('verdict')}** (`{browser.get('chromium', 'N/A')}`, extension `{browser.get('extension_id', 'N/A')}`)",
            f"- **Checks:** `{json.dumps(browser.get('checks', {}))}`",
        ])

    for section_key, title in (
        ("failure_classifications", "Failure Classifications"),
        ("model_limitations", "Model Limitations"),
        ("remaining_backend_defects", "Remaining Backend Defects"),
        ("regression_verification", "Offline Regression & Static Verification"),
    ):
        items = report.get(section_key)
        if items:
            lines.extend(["", f"## {title}", ""])
            if isinstance(items, list):
                for item in items:
                    lines.append(f"- {item}")
            elif isinstance(items, dict):
                for k, v in items.items():
                    lines.append(f"- **{k}:** `{v}`")
            else:
                lines.append(str(items))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zelyr Full Linux Agent Capability Matrix benchmark.")
    parser.add_argument("--primary-model", default="qwen3.6:35b-a3b")
    parser.add_argument("--compat-models", nargs="*", default=["qwen3.5:4b", "qwen2.5:7b"])
    parser.add_argument("--primary-tasks", nargs="*", default=list(ALL_TASKS))
    parser.add_argument("--compat-tasks", nargs="*", default=list(COMPAT_SUBSET_TASKS))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    inv = get_ollama_inventory()
    ollama_ver = inv["ollama_version"]
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    bench_root = Path(tempfile.mkdtemp(prefix="zelyr_full_cap_matrix_"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_plan: List[Tuple[str, List[str], str]] = [(args.primary_model, args.primary_tasks, "primary")]
    for cm in args.compat_models:
        model_plan.append((cm, args.compat_tasks, "compatibility_subset"))

    all_results: List[Dict[str, Any]] = []
    try:
        for model, task_list, tier_label in model_plan:
            unload_all_ollama_models()
            if model not in inv["models"]:
                all_results.append({
                    "model": model,
                    "tier": tier_label,
                    "status": "UNAVAILABLE",
                    "tasks": [{"task_id": tid, "verdict": "UNAVAILABLE"} for tid in task_list],
                })
                continue

            model_info = inv["models"][model]
            model_tasks: List[Dict[str, Any]] = []
            for tid in task_list:
                res = run_single_task(bench_root, model, model_info, ollama_ver, tid, args.timeout)
                model_tasks.append(res)
                print(
                    f"[{model}] {tid}: {res['verdict']} dt={res['duration_s']}s rc={res['cli_rc']} "
                    f"ledger={res['ledger_status']}/{res['ledger_verification_status']} turns={res['llm_turns']} "
                    f"max_pe={res['max_prompt_eval_count']} exec={res['execution_id']}",
                    flush=True,
                )
            all_results.append({
                "model": model,
                "tier": tier_label,
                "digest": model_info.get("digest"),
                "details": model_info.get("details"),
                "tasks": model_tasks,
            })
            unload_all_ollama_models()

        browser_res = None if args.skip_browser else asyncio.run(run_isolated_browser_e2e())
        if browser_res:
            print(f"[browser_e2e] {browser_res.get('verdict')} checks={browser_res.get('checks')}", flush=True)

        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_branch": git_branch,
            "git_commit": git_commit,
            "ollama_version": ollama_ver,
            "execution_config": {
                "num_ctx_default": 8192,
                "max_steps_default": 12,
                "timeout_per_task_s": args.timeout,
                "autopilot_env": True,
                "sequential_execution": True,
            },
            "results": all_results,
            "browser_e2e": browser_res,
        }
        json_path = output_dir / "capability_matrix_report.json"
        md_path = output_dir / "capability_matrix_report.md"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        md_path.write_text(render_markdown_summary(report), encoding="utf-8")
        print(f"SAVED_JSON={json_path}", flush=True)
        print(f"SAVED_MD={md_path}", flush=True)
    finally:
        unload_all_ollama_models()
        if not args.keep_work:
            shutil.rmtree(bench_root, ignore_errors=True)


if __name__ == "__main__":
    main()
