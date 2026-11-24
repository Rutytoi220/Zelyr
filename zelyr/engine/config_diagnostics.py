"""Read-only configuration diagnostics, source provenance, and validation for ZelyrConfig."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Union

from zelyr.compat import config_dir, config_file
from zelyr.config import (
    AuthMode,
    ZelyrConfig,
    get_config,
    get_config_dir,
    validate_ollama_base_url,
)


_LEGACY_MIGRATED_KEYS = frozenset({"persona_tone", "persona_complexity", "special_instructions"})
_SUPPORTED_POLICY_MODES = frozenset({"conservative", "strict", "balanced", "basic", "autopilot"})
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"})
_VALID_ENGINE_MODES = frozenset({"local", "remote", "fastapi"})
_VALID_THEMES = frozenset({"minimalist", "cyberpunk", "nothing"})
_VALID_THEME_MODES = frozenset({"system", "dark", "light"})
_VALID_PROFILE_LEVELS = frozenset({"standard", "advanced", "developer"})
_VALID_MODEL_MODES = frozenset({"auto", "manual"})


def _redact_config_val(value: Any, *, key_name: str = "") -> Any:
    """Recursively redact secret credentials/tokens from configuration values."""
    if isinstance(value, Mapping):
        return {str(k): _redact_config_val(v, key_name=str(k)) for k, v in value.items()}
    low = str(key_name or "").lower()
    if any(
        tok in low
        for tok in (
            "secret",
            "token",
            "password",
            "passwd",
            "api_key",
            "apikey",
            "credential",
            "private_key",
            "bearer",
            "authorization",
        )
    ):
        return "[REDACTED]"
    if isinstance(value, (list, tuple)):
        return [_redact_config_val(x) for x in value]
    if isinstance(value, str):
        scrubbed = re.sub(r"(?i)(https?://)([^/@\s]+@)", r"\1[REDACTED]@", value)
        scrubbed = re.sub(
            r"(?i)([?&](?:token|key|api_key|secret|password|auth|sig|access_token)=)([^&\s]+)",
            r"\1[REDACTED]",
            scrubbed,
        )
        scrubbed = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", scrubbed)
        return scrubbed
    return value


def _lookup_env(env_map: Mapping[str, str], suffix: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve ZELYR_<suffix> or AXIOM_<suffix> from env_map, returning (value, var_name)."""
    z_name = suffix if suffix.startswith("ZELYR_") else f"ZELYR_{suffix}"
    a_name = f"AXIOM_{z_name[len('ZELYR_'):]}"
    if z_name in env_map and env_map[z_name] != "":
        return env_map[z_name], z_name
    if a_name in env_map and env_map[a_name] != "":
        return env_map[a_name], a_name
    return None, None


@dataclass(frozen=True)
class ConfigurationDiagnosticReport:
    """Structured diagnostic report describing effective Zelyr configuration, sources, and anomalies."""

    valid: bool
    config_file_path: Optional[str]
    config_file_exists: bool
    legacy_axiom_path: bool
    parse_error: Optional[str] = None
    effective_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    unknown_keys: List[str] = field(default_factory=list)
    obsolete_or_migrated_keys: List[str] = field(default_factory=list)
    structural_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigurationDiagnostics:
    """Read-only inspector validating ZelyrConfig provenance, ranges, overrides, and consumer consistency."""

    @staticmethod
    def diagnose(
        *,
        config: Optional[ZelyrConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> ConfigurationDiagnosticReport:
        env_map: Mapping[str, str] = env if env is not None else os.environ
        issues: List[Dict[str, Any]] = []
        anomalies: List[Dict[str, Any]] = []
        raw_json: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        eff_path_str: Optional[str] = None
        file_exists = False

        if config_path is not None:
            p = Path(config_path).expanduser()
            eff_path_str = str(p)
            file_exists = p.exists()
            if not file_exists:
                parse_error = f"Specified config file does not exist: {p}"
                issues.append(
                    {"severity": "error", "field": "config_path", "code": "file_not_found", "message": parse_error}
                )
            else:
                try:
                    loaded_obj = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(loaded_obj, dict):
                        parse_error = f"Config file must contain a JSON object, got {type(loaded_obj).__name__}."
                        issues.append(
                            {"severity": "error", "field": "config_path", "code": "invalid_json_root", "message": parse_error}
                        )
                    else:
                        raw_json = loaded_obj
                        if config is None:
                            config = ZelyrConfig.from_dict(raw_json)
                            config._config_source_path = str(p)
                except Exception as exc:
                    parse_error = f"Failed to parse config file '{p}': {exc}"
                    issues.append(
                        {"severity": "error", "field": "config_path", "code": "json_parse_error", "message": parse_error}
                    )

        if config is None:
            config = get_config()

        if eff_path_str is None:
            if getattr(config, "_config_source_path", None):
                eff_path_str = str(config._config_source_path)
                file_exists = Path(eff_path_str).exists()
            else:
                base_dir = get_config_dir()
                for cand in dict.fromkeys([
                    config_file("config.json"),
                    base_dir / "config.json",
                    config_dir() / "config.json",
                    Path.home() / ".config" / "axiom" / "config.json",
                    Path.home() / ".config" / "ChienGPT" / "config.json",
                ]):
                    if cand.exists():
                        eff_path_str = str(cand)
                        file_exists = True
                        break

        if raw_json is None and eff_path_str and Path(eff_path_str).exists() and parse_error is None:
            try:
                parsed = json.loads(Path(eff_path_str).read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    raw_json = parsed
            except Exception as exc:
                parse_error = f"On-disk config file is unreadable/corrupt: {exc}"
                issues.append(
                    {"severity": "error", "field": "config_file", "code": "on_disk_corrupt", "message": parse_error}
                )

        legacy_axiom = bool(eff_path_str and "/.config/axiom" in eff_path_str.replace("\\", "/"))
        if legacy_axiom:
            anomalies.append(
                {
                    "kind": "legacy_path",
                    "field": "config_file",
                    "message": f"Reading legacy AXIOM config path '{eff_path_str}'.",
                }
            )

        defaults = ZelyrConfig()
        explicit_fields = set(getattr(config, "_explicit_fields", set()) or ())

        def _cfg_source(field_name: str, cur_val: Any, default_val: Any) -> str:
            if field_name in explicit_fields:
                return f"config_file:{eff_path_str}" if eff_path_str else "explicit"
            if cur_val != default_val:
                return "explicit"
            return "default"

        unknown_keys: List[str] = list(getattr(config, "_unknown_keys", []) or [])
        migrated_keys: List[str] = list(getattr(config, "_migrated_keys", []) or [])
        if raw_json is not None:
            unknown_keys = sorted(
                str(k)
                for k in raw_json.keys()
                if k not in ZelyrConfig.__dataclass_fields__
                and not str(k).startswith("_")
                and k not in _LEGACY_MIGRATED_KEYS
            )
            migrated_keys = sorted(str(k) for k in raw_json.keys() if k in _LEGACY_MIGRATED_KEYS)

        for uk in unknown_keys:
            anomalies.append(
                {
                    "kind": "unknown_key",
                    "field": uk,
                    "message": f"Unknown/unsupported config key '{uk}' is ignored during ZelyrConfig loading.",
                }
            )
        for mk in migrated_keys:
            anomalies.append(
                {
                    "kind": "obsolete_key_migrated",
                    "field": mk,
                    "message": f"Obsolete key '{mk}' is migrated into structured `persona` at load time.",
                }
            )

        # Validate raw JSON values BEFORE clamping if raw_json was read cleanly
        raw_check = raw_json if raw_json is not None else {}

        # 1. auth_mode
        if "auth_mode" in raw_check:
            raw_am = raw_check["auth_mode"]
            if not isinstance(raw_am, str) or raw_am not in {m.value for m in AuthMode}:
                issues.append(
                    {
                        "severity": "error",
                        "field": "auth_mode",
                        "code": "invalid_enum",
                        "message": f"Invalid auth_mode {raw_am!r}; valid values: strict, basic, autopilot (silently fell back to 'basic').",
                    }
                )
        am_val = getattr(getattr(config, "auth_mode", AuthMode.BASIC), "value", getattr(config, "auth_mode", "basic"))

        # 2. execution_policy_mode
        if "execution_policy_mode" in raw_check and raw_check["execution_policy_mode"] is not None:
            raw_pm = str(raw_check["execution_policy_mode"]).strip().lower()
            if raw_pm not in _SUPPORTED_POLICY_MODES:
                issues.append(
                    {
                        "severity": "error",
                        "field": "execution_policy_mode",
                        "code": "invalid_enum",
                        "message": f"Unsupported execution_policy_mode {raw_check['execution_policy_mode']!r}; fell back to 'conservative'.",
                    }
                )

        raw_pol_mode = getattr(config, "execution_policy_mode", None)
        if raw_pol_mode is not None and str(raw_pol_mode).strip():
            eff_policy_mode = str(raw_pol_mode).strip().lower()
            pol_mode_src = _cfg_source("execution_policy_mode", raw_pol_mode, None)
            if eff_policy_mode not in _SUPPORTED_POLICY_MODES:
                issues.append(
                    {
                        "severity": "error",
                        "field": "execution_policy_mode",
                        "code": "invalid_enum",
                        "message": f"Invalid execution_policy_mode {raw_pol_mode!r}.",
                    }
                )
        else:
            eff_policy_mode = am_val if am_val in _SUPPORTED_POLICY_MODES else "conservative"
            pol_mode_src = f"derived:auth_mode:{am_val}"

        if raw_pol_mode and str(raw_pol_mode).strip().lower() != am_val:
            anomalies.append(
                {
                    "kind": "precedence_override",
                    "field": "execution_policy_mode",
                    "message": f"execution_policy_mode='{raw_pol_mode}' overrides derived auth_mode='{am_val}'.",
                }
            )

        # 3. Ollama URL, CTX, KeepAlive, Read Timeout (with real env precedence check)
        z_url, z_url_var = _lookup_env(env_map, "OLLAMA_BASE_URL")
        ollama_host_env = env_map.get("OLLAMA_HOST")
        if z_url:
            eff_ollama_url = validate_ollama_base_url(z_url)
            url_src = f"env:{z_url_var}"
            if ollama_host_env:
                anomalies.append(
                    {
                        "kind": "env_shadowing",
                        "field": "ollama_base_url",
                        "message": f"{z_url_var} shadows OLLAMA_HOST={_redact_config_val(ollama_host_env)}.",
                    }
                )
        elif ollama_host_env:
            eff_ollama_url = validate_ollama_base_url(ollama_host_env)
            url_src = "env:OLLAMA_HOST"
        else:
            eff_ollama_url = validate_ollama_base_url(config.ollama_base_url)
            url_src = _cfg_source("ollama_base_url", config.ollama_base_url, defaults.ollama_base_url)

        if (z_url or ollama_host_env) and eff_ollama_url != validate_ollama_base_url(config.ollama_base_url):
            anomalies.append(
                {
                    "kind": "consumer_divergence",
                    "field": "ollama_base_url",
                    "message": (
                        f"Environment overrides ollama_base_url to '{eff_ollama_url}', while modules reading "
                        "`config.ollama_base_url` directly (tools/vision.py, memory/semantic.py, cli/repl.py) "
                        f"still see '{config.ollama_base_url}' unless `get_effective_ollama_base_url()` is called."
                    ),
                }
            )

        # num_ctx
        raw_ctx_source_val = raw_check.get("ollama_num_ctx", config.ollama_num_ctx)
        if isinstance(raw_ctx_source_val, bool) or (
            raw_ctx_source_val is not None
            and (
                not isinstance(raw_ctx_source_val, int)
                or raw_ctx_source_val < 1024
                or raw_ctx_source_val > 65536
            )
        ):
            issues.append(
                {
                    "severity": "warning",
                    "field": "ollama_num_ctx",
                    "code": "out_of_range",
                    "message": f"ollama_num_ctx={raw_ctx_source_val!r} is outside [1024, 65536]; clamped to 8192.",
                }
            )
        z_ctx, z_ctx_var = _lookup_env(env_map, "NUM_CTX")
        if z_ctx is not None:
            try:
                parsed_env_ctx = int(z_ctx)
                if parsed_env_ctx < 1024 or parsed_env_ctx > 65536:
                    issues.append(
                        {
                            "severity": "warning",
                            "field": "ollama_num_ctx",
                            "code": "env_out_of_range",
                            "message": f"{z_ctx_var}={z_ctx!r} outside [1024, 65536]; falls back to 8192.",
                        }
                    )
                    eff_ctx = 8192
                else:
                    eff_ctx = parsed_env_ctx
            except (TypeError, ValueError):
                issues.append(
                    {
                        "severity": "error",
                        "field": "ollama_num_ctx",
                        "code": "invalid_env_int",
                        "message": f"{z_ctx_var}={z_ctx!r} is not a valid integer; falls back to 8192.",
                    }
                )
                eff_ctx = 8192
            ctx_src = f"env:{z_ctx_var}"
        else:
            eff_ctx = config.get_effective_num_ctx()
            ctx_src = _cfg_source("ollama_num_ctx", config.ollama_num_ctx, defaults.ollama_num_ctx)

        # keep_alive
        z_ka, z_ka_var = _lookup_env(env_map, "OLLAMA_KEEP_ALIVE")
        if z_ka is not None and z_ka.strip():
            eff_ka: Any = 0 if z_ka.strip() == "0" else z_ka.strip()
            ka_src = f"env:{z_ka_var}"
        elif config.ollama_keep_alive not in (None, ""):
            eff_ka = config.ollama_keep_alive
            ka_src = _cfg_source("ollama_keep_alive", config.ollama_keep_alive, defaults.ollama_keep_alive)
        else:
            eff_ka = "5m"
            ka_src = "default"

        # read_timeout
        z_rt, z_rt_var = _lookup_env(env_map, "OLLAMA_READ_TIMEOUT")
        eff_rt = 300.0
        rt_src = "default"
        if z_rt is not None:
            try:
                f_rt = float(z_rt)
                if not isinstance(z_rt, bool) and f_rt >= 5.0:
                    eff_rt = f_rt
                    rt_src = f"env:{z_rt_var}"
                else:
                    issues.append(
                        {
                            "severity": "warning",
                            "field": "ollama_read_timeout",
                            "code": "below_minimum",
                            "message": f"{z_rt_var}={z_rt!r} < 5.0s; ignored in favor of config/default.",
                        }
                    )
            except (TypeError, ValueError):
                issues.append(
                    {
                        "severity": "warning",
                        "field": "ollama_read_timeout",
                        "code": "invalid_float",
                        "message": f"{z_rt_var}={z_rt!r} invalid; ignored.",
                    }
                )
        if rt_src == "default":
            cfg_rt = getattr(config, "ollama_read_timeout", 300.0)
            if isinstance(cfg_rt, bool) or not isinstance(cfg_rt, (int, float)) or float(cfg_rt) < 5.0:
                issues.append(
                    {
                        "severity": "warning",
                        "field": "ollama_read_timeout",
                        "code": "below_minimum",
                        "message": f"ollama_read_timeout={cfg_rt!r} < 5.0s is accepted by ZelyrConfig yet silently overridden to 300.0s by get_effective_read_timeout().",
                    }
                )
            else:
                eff_rt = float(cfg_rt)
                rt_src = _cfg_source("ollama_read_timeout", float(cfg_rt), defaults.ollama_read_timeout)

        # Temperature, bounds, enum checks
        temp = getattr(config, "ollama_temperature", 0.7)
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not (0.0 <= float(temp) <= 2.0):
            issues.append(
                {
                    "severity": "warning",
                    "field": "ollama_temperature",
                    "code": "out_of_range",
                    "message": f"ollama_temperature={temp!r} outside expected [0.0, 2.0].",
                }
            )

        for int_field in ("max_history", "max_agents", "max_tools", "event_history_limit", "max_vector_memories"):
            val = getattr(config, int_field, None)
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                issues.append(
                    {
                        "severity": "error",
                        "field": int_field,
                        "code": "invalid_positive_int",
                        "message": f"{int_field}={val!r} must be a positive integer.",
                    }
                )

        # Graph recovery limits
        g_att = getattr(config, "graph_recovery_max_attempts", 2)
        if isinstance(g_att, bool) or not isinstance(g_att, int) or g_att < 0 or g_att > 10:
            issues.append(
                {
                    "severity": "error",
                    "field": "graph_recovery_max_attempts",
                    "code": "out_of_range",
                    "message": f"graph_recovery_max_attempts={g_att!r} must be an integer in [0, 10].",
                }
            )
        g_base = getattr(config, "graph_recovery_backoff_base_seconds", 0.05)
        g_max = getattr(config, "graph_recovery_backoff_max_seconds", 2.0)
        if isinstance(g_base, bool) or not isinstance(g_base, (int, float)) or float(g_base) < 0:
            issues.append(
                {
                    "severity": "error",
                    "field": "graph_recovery_backoff_base_seconds",
                    "code": "invalid_nonneg_float",
                    "message": f"graph_recovery_backoff_base_seconds={g_base!r} must be >= 0.",
                }
            )
        elif not isinstance(g_max, bool) and isinstance(g_max, (int, float)) and float(g_base) > float(g_max):
            issues.append(
                {
                    "severity": "warning",
                    "field": "graph_recovery_backoff_base_seconds",
                    "code": "inconsistent_bounds",
                    "message": f"graph_recovery_backoff_base_seconds ({g_base}) exceeds graph_recovery_backoff_max_seconds ({g_max}).",
                }
            )

        # Confirmation TTL ordering vs raw JSON clamping
        raw_ttl = raw_check.get("execution_policy_confirmation_ttl_seconds", config.execution_policy_confirmation_ttl_seconds)
        raw_max_ttl = raw_check.get(
            "execution_policy_max_confirmation_lifetime_seconds",
            config.execution_policy_max_confirmation_lifetime_seconds,
        )
        if (
            not isinstance(raw_ttl, bool)
            and isinstance(raw_ttl, (int, float))
            and not isinstance(raw_max_ttl, bool)
            and isinstance(raw_max_ttl, (int, float))
            and float(raw_ttl) > float(raw_max_ttl)
        ):
            issues.append(
                {
                    "severity": "warning",
                    "field": "execution_policy_confirmation_ttl_seconds",
                    "code": "ttl_exceeds_max_lifetime",
                    "message": (
                        f"execution_policy_confirmation_ttl_seconds ({raw_ttl}) > "
                        f"max_confirmation_lifetime_seconds ({raw_max_ttl}); max lifetime clamped up automatically."
                    ),
                }
            )

        for field_name, allowed_vals in (
            ("log_level", _VALID_LOG_LEVELS),
            ("engine_mode", _VALID_ENGINE_MODES),
            ("theme", _VALID_THEMES),
            ("theme_mode", _VALID_THEME_MODES),
            ("ui_profile_level", _VALID_PROFILE_LEVELS),
            ("model_selection_mode", _VALID_MODEL_MODES),
        ):
            cur = str(getattr(config, field_name, "") or "").strip()
            cmp_val = cur.upper() if field_name == "log_level" else cur.lower()
            if cmp_val not in allowed_vals:
                issues.append(
                    {
                        "severity": "warning",
                        "field": field_name,
                        "code": "invalid_enum",
                        "message": f"Unexpected {field_name}={cur!r}; expected one of {sorted(allowed_vals)}.",
                    }
                )

        # 4. Empirically verified structural anomalies across consumers
        # Unconsumed security/resource config fields
        if getattr(config, "sandbox_mode", True) is False:
            anomalies.append(
                {
                    "kind": "ignored_setting",
                    "field": "sandbox_mode",
                    "message": "`sandbox_mode` is not consumed outside config.py; use `auth_mode` and `execution_policy_*` to govern execution boundaries.",
                }
            )
        if getattr(config, "allow_system_tools", True) is False:
            anomalies.append(
                {
                    "kind": "ignored_setting",
                    "field": "allow_system_tools",
                    "message": "`allow_system_tools` is not consumed outside config.py; use `execution_policy_tool_overrides` / `caller_ring` to restrict system tools.",
                }
            )
        for unconsumed_name, default_val in (
            ("max_agents", 10),
            ("max_tools", 100),
            ("max_history", 1000),
            ("event_history_limit", 1000),
            ("theme_mode", "dark"),
        ):
            if getattr(config, unconsumed_name, default_val) != default_val:
                anomalies.append(
                    {
                        "kind": "ignored_setting",
                        "field": unconsumed_name,
                        "message": f"`{unconsumed_name}={getattr(config, unconsumed_name)!r}` currently has no runtime consumer outside zelyr/config.py.",
                    }
                )

        if getattr(config, "persona_key", "default") != "default" and getattr(config, "active_persona", "zelyr_core") != "zelyr_core":
            if config.persona_key != config.active_persona:
                anomalies.append(
                    {
                        "kind": "duplicated_setting",
                        "field": "persona_key/active_persona",
                        "message": (
                            f"`persona_key` ({config.persona_key!r}, used by GUI/OOBE) and `active_persona` "
                            f"({config.active_persona!r}, used by OrchestratorAgent) are separate fields."
                        ),
                    }
                )

        settings_json = config_dir() / "settings.json"
        if settings_json.exists() and getattr(config, "disabled_plugins", None):
            anomalies.append(
                {
                    "kind": "split_plugin_config",
                    "field": "disabled_plugins",
                    "message": f"Plugin disablement is split between ZelyrConfig.disabled_plugins and '{settings_json}'.",
                }
            )

        effective_settings: Dict[str, Dict[str, Any]] = {
            "auth_mode": {
                "value": am_val,
                "source": _cfg_source("auth_mode", config.auth_mode, defaults.auth_mode),
            },
            "execution_policy_mode": {
                "value": eff_policy_mode,
                "source": pol_mode_src,
            },
            "ollama_base_url": {
                "value": _redact_config_val(eff_ollama_url),
                "source": url_src,
            },
            "ollama_model": {
                "value": config.ollama_model,
                "source": _cfg_source("ollama_model", config.ollama_model, defaults.ollama_model),
            },
            "vision_model": {
                "value": config.vision_model,
                "source": _cfg_source("vision_model", config.vision_model, defaults.vision_model),
            },
            "embedding_model": {
                "value": config.embedding_model,
                "source": _cfg_source("embedding_model", config.embedding_model, defaults.embedding_model),
            },
            "ollama_num_ctx": {
                "value": eff_ctx,
                "source": ctx_src,
            },
            "ollama_keep_alive": {
                "value": eff_ka,
                "source": ka_src,
            },
            "ollama_read_timeout": {
                "value": eff_rt,
                "source": rt_src,
            },
            "ollama_temperature": {
                "value": config.ollama_temperature,
                "source": _cfg_source("ollama_temperature", config.ollama_temperature, defaults.ollama_temperature),
            },
            "db_path": {
                "value": config.db_path,
                "source": _cfg_source("db_path", config.db_path, defaults.db_path),
            },
            "engine_mode": {
                "value": config.engine_mode,
                "source": _cfg_source("engine_mode", config.engine_mode, defaults.engine_mode),
            },
            "swarm_enabled": {
                "value": bool(config.swarm_enabled),
                "source": _cfg_source("swarm_enabled", config.swarm_enabled, defaults.swarm_enabled),
            },
            "graph_recovery_max_attempts": {
                "value": config.graph_recovery_max_attempts,
                "source": _cfg_source("graph_recovery_max_attempts", config.graph_recovery_max_attempts, defaults.graph_recovery_max_attempts),
            },
            "execution_policy_confirmation_ttl_seconds": {
                "value": config.execution_policy_confirmation_ttl_seconds,
                "source": _cfg_source(
                    "execution_policy_confirmation_ttl_seconds",
                    config.execution_policy_confirmation_ttl_seconds,
                    defaults.execution_policy_confirmation_ttl_seconds,
                ),
            },
            "execution_policy_max_confirmation_lifetime_seconds": {
                "value": config.execution_policy_max_confirmation_lifetime_seconds,
                "source": _cfg_source(
                    "execution_policy_max_confirmation_lifetime_seconds",
                    config.execution_policy_max_confirmation_lifetime_seconds,
                    defaults.execution_policy_max_confirmation_lifetime_seconds,
                ),
            },
            "execution_policy_tool_overrides": {
                "value": _redact_config_val(config.execution_policy_tool_overrides),
                "source": _cfg_source(
                    "execution_policy_tool_overrides",
                    config.execution_policy_tool_overrides,
                    defaults.execution_policy_tool_overrides,
                ),
            },
            "execution_policy_resource_restrictions": {
                "value": _redact_config_val(config.execution_policy_resource_restrictions),
                "source": _cfg_source(
                    "execution_policy_resource_restrictions",
                    config.execution_policy_resource_restrictions,
                    defaults.execution_policy_resource_restrictions,
                ),
            },
        }

        error_cnt = sum(1 for i in issues if i["severity"] == "error")
        warn_cnt = sum(1 for i in issues if i["severity"] == "warning")
        return ConfigurationDiagnosticReport(
            valid=(error_cnt == 0),
            config_file_path=eff_path_str,
            config_file_exists=file_exists,
            legacy_axiom_path=legacy_axiom,
            parse_error=parse_error,
            effective_settings=effective_settings,
            validation_issues=issues,
            unknown_keys=unknown_keys,
            obsolete_or_migrated_keys=migrated_keys,
            structural_anomalies=anomalies,
            summary={
                "error_count": error_cnt,
                "warning_count": warn_cnt,
                "unknown_key_count": len(unknown_keys),
                "obsolete_key_count": len(migrated_keys),
                "anomaly_count": len(anomalies),
            },
        )


def diagnose_configuration(
    *,
    config: Optional[ZelyrConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ConfigurationDiagnosticReport:
    """Inspect and validate Zelyr configuration provenance, ranges, and anomalies."""
    return ConfigurationDiagnostics.diagnose(
        config=config,
        config_path=config_path,
        env=env,
    )


def format_configuration_diagnostics(report: ConfigurationDiagnosticReport) -> str:
    """Format a ConfigurationDiagnosticReport into clean terminal output."""
    lines = [
        f"Configuration Diagnostics (valid={report.valid}, errors={report.summary.get('error_count', 0)}, warnings={report.summary.get('warning_count', 0)}):",
        f"  config_file: {report.config_file_path or 'none'} (exists={report.config_file_exists}, legacy_axiom={report.legacy_axiom_path})",
    ]
    if report.parse_error:
        lines.append(f"  parse_error: {report.parse_error}")
    lines.append("  effective_settings:")
    for key, info in report.effective_settings.items():
        val = info.get("value")
        val_str = json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else str(val)
        lines.append(f"    {key:<48} {val_str}  [source: {info.get('source')}]")
    if report.unknown_keys:
        lines.append(f"  unknown_keys: {report.unknown_keys}")
    if report.obsolete_or_migrated_keys:
        lines.append(f"  obsolete_or_migrated_keys: {report.obsolete_or_migrated_keys}")
    if report.validation_issues:
        lines.append("  validation_issues:")
        for iss in report.validation_issues:
            lines.append(f"    [{iss['severity'].upper()}] {iss['field']} ({iss['code']}): {iss['message']}")
    if report.structural_anomalies:
        lines.append("  structural_anomalies:")
        for anom in report.structural_anomalies:
            lines.append(f"    [{anom['kind']}] {anom['field']}: {anom['message']}")
    return "\n".join(lines)


__all__ = [
    "ConfigurationDiagnosticReport",
    "ConfigurationDiagnostics",
    "diagnose_configuration",
    "format_configuration_diagnostics",
]
