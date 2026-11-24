"""Deterministic offline tests for Configuration Diagnostics & Validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List

from zelyr.config import AuthMode, ZelyrConfig
from zelyr.engine import (
    ConfigurationDiagnostics,
    diagnose_configuration,
    format_configuration_diagnostics,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_cli(args: List[str], *, config_dir_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    env["ZELYR_TESTING"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    if config_dir_path is not None:
        env["ZELYR_CONFIG_DIR"] = str(config_dir_path)
    return subprocess.run(
        [sys.executable, "-m", "zelyr.cli.main", *args],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_precedence_sources_and_env_shadowing_detected(tmp_path: Path) -> None:
    """Reports exact provenance (config_file, env, derived, default) and consumer divergence."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "auth_mode": "strict",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_model": "qwen3:8b",
            }
        ),
        encoding="utf-8",
    )

    rep = diagnose_configuration(
        config_path=cfg_path,
        env={
            "ZELYR_OLLAMA_BASE_URL": "http://10.0.0.5:11434",
            "OLLAMA_HOST": "http://10.0.0.9:11434",
            "ZELYR_NUM_CTX": "16384",
        },
    )
    assert rep.valid is True
    assert rep.effective_settings["auth_mode"]["value"] == "strict"
    assert rep.effective_settings["auth_mode"]["source"] == f"config_file:{cfg_path}"
    assert rep.effective_settings["execution_policy_mode"]["value"] == "strict"
    assert rep.effective_settings["execution_policy_mode"]["source"] == "derived:auth_mode:strict"
    assert rep.effective_settings["ollama_base_url"]["value"] == "http://10.0.0.5:11434"
    assert rep.effective_settings["ollama_base_url"]["source"] == "env:ZELYR_OLLAMA_BASE_URL"
    assert rep.effective_settings["ollama_num_ctx"]["value"] == 16384
    assert rep.effective_settings["ollama_num_ctx"]["source"] == "env:ZELYR_NUM_CTX"
    assert rep.effective_settings["ollama_keep_alive"]["source"] == "default"

    kinds = {a["kind"] for a in rep.structural_anomalies}
    assert "env_shadowing" in kinds
    assert "consumer_divergence" in kinds


def test_invalid_types_ranges_unknown_obsolete_and_ignored_fields_reported(tmp_path: Path) -> None:
    """Detects out-of-range fields, unknown keys, legacy keys, ignored settings, and invalid env values."""
    bad_cfg = tmp_path / "bad_config.json"
    bad_cfg.write_text(
        json.dumps(
            {
                "auth_mode": "nonexistent_mode",
                "execution_policy_mode": "unsafe_mode",
                "ollama_num_ctx": 128,
                "ollama_read_timeout": 1.5,
                "ollama_temperature": 5.0,
                "max_agents": -3,
                "graph_recovery_max_attempts": 99,
                "execution_policy_confirmation_ttl_seconds": 600.0,
                "execution_policy_max_confirmation_lifetime_seconds": 120.0,
                "sandbox_mode": False,
                "persona_tone": "concise",
                "olama_typo_key": "ignored",
            }
        ),
        encoding="utf-8",
    )

    rep = diagnose_configuration(
        config_path=bad_cfg,
        env={"ZELYR_NUM_CTX": "not_an_int"},
    )
    assert rep.valid is False
    assert "olama_typo_key" in rep.unknown_keys
    assert "persona_tone" in rep.obsolete_or_migrated_keys

    issue_fields = {i["field"]: i["code"] for i in rep.validation_issues}
    assert issue_fields["auth_mode"] == "invalid_enum"
    assert issue_fields["execution_policy_mode"] == "invalid_enum"
    assert issue_fields["ollama_num_ctx"] == "invalid_env_int"
    assert issue_fields["ollama_read_timeout"] == "below_minimum"
    assert issue_fields["ollama_temperature"] == "out_of_range"
    assert issue_fields["max_agents"] == "invalid_positive_int"
    assert issue_fields["graph_recovery_max_attempts"] == "out_of_range"

    anom_fields = {(a["kind"], a["field"]) for a in rep.structural_anomalies}
    assert ("ignored_setting", "sandbox_mode") in anom_fields
    assert ("unknown_key", "olama_typo_key") in anom_fields
    assert ("obsolete_key_migrated", "persona_tone") in anom_fields

    # Confirm legacy persona_tone still migrated cleanly into ZelyrConfig for backward compatibility
    loaded = ZelyrConfig.load(bad_cfg)
    assert loaded.persona["communication"]["tone"] == "concise"


def test_secret_redaction_and_cli_config_commands(tmp_path: Path) -> None:
    """Secrets in URLs or policy overrides are redacted; CLI show/validate enforce exit semantics."""
    cfg_dir = tmp_path / "cfg_home"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = ZelyrConfig(
        auth_mode=AuthMode.BASIC,
        execution_policy_mode="balanced",
        ollama_base_url="https://user:secretpw@ollama.internal:11434?api_key=tok_hidden_123",
        execution_policy_tool_overrides={"my_tool": {"outcome": "deny", "token": "secret_cli_token_xyz"}},
        execution_policy_resource_restrictions={"bearer_secret": "sk-super-secret-value-888"},
    )
    cfg_file = cfg_dir / "config.json"
    cfg.save(cfg_file)

    rep = ConfigurationDiagnostics.diagnose(config=cfg, env={})
    dumped = json.dumps(rep.to_dict()) + format_configuration_diagnostics(rep)
    for secret in ("secretpw", "tok_hidden_123", "secret_cli_token_xyz", "sk-super-secret-value-888"):
        assert secret not in dumped

    # Clean config: validate --json exits 0 cleanly (ignoring host env overrides in subprocess if clean)
    clean_file = tmp_path / "clean.json"
    clean_file.write_text(json.dumps({"auth_mode": "strict", "ollama_model": "qwen3:8b"}), encoding="utf-8")
    r_show = _run_cli(["config", "show", "--config-path", str(clean_file), "--json"], config_dir_path=cfg_dir)
    assert r_show.returncode == 0, r_show.stderr
    assert json.loads(r_show.stdout)["valid"] is True

    r_val_ok = _run_cli(["config", "validate", "--config-path", str(clean_file), "--json"], config_dir_path=cfg_dir)
    assert r_val_ok.returncode == 0, r_val_ok.stderr

    # Config with unknown key passes show, fails validate (--strict semantics)
    warn_file = tmp_path / "warn.json"
    warn_file.write_text(json.dumps({"auth_mode": "strict", "unknown_extra_key": 1}), encoding="utf-8")
    r_warn_show = _run_cli(["config", "show", "--config-path", str(warn_file)], config_dir_path=cfg_dir)
    assert r_warn_show.returncode == 0
    assert "unknown_extra_key" in r_warn_show.stdout
    r_warn_val = _run_cli(["config", "validate", "--config-path", str(warn_file), "--json"], config_dir_path=cfg_dir)
    assert r_warn_val.returncode != 0

    # Malformed / missing config file fails both show and validate
    missing_file = tmp_path / "does_not_exist.json"
    r_missing = _run_cli(["config", "show", "--config-path", str(missing_file), "--json"], config_dir_path=cfg_dir)
    assert r_missing.returncode != 0
