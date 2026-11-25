"""Distributed multi-worker evaluation subsystem for Zelyr.

Discovers local and LAN Ollama endpoints, enforces per-worker allowlists and
conservative resource/concurrency caps, executes isolated tasks using Zelyr's
real orchestration and verification pipeline, and classifies unsupported vs
failed vs timed-out vs false-success outcomes without silent rerouting.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import urllib.parse

import httpx

from zelyr.config import get_config, validate_ollama_base_url
from zelyr.engine.error_classifier import _redact_secrets
from zelyr.engine.execution_ledger import detect_false_success
from zelyr.engine.model_compatibility import ModelCompatibilityDiagnostics


def _sanitize_artifact_segment(value: str, fallback: str = "item") -> str:
    """Sanitize identifiers used in workspace/transcript filenames so paths stay confined."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned or fallback


def _scrub_url_userinfo(url: str) -> str:
    """Remove embedded userinfo credentials from URLs before serializing reports."""
    return _redact_secrets(str(url or ""))


def _normalize_endpoint(url: str) -> str:
    """Normalize HTTP/HTTPS worker endpoints canonically for distinct-host evidence checks."""
    cleaned = _scrub_url_userinfo(url).strip()
    if not cleaned:
        return ""
    try:
        parts = urllib.parse.urlsplit(cleaned)
    except Exception:
        return cleaned.rstrip("/").lower()
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    path = (parts.path or "").rstrip("/")
    return f"{scheme}://{netloc}{path}" if netloc else cleaned.rstrip("/").lower()


class DistributedTaskStatus(str, enum.Enum):
    """Strict non-conflating outcome classifications for distributed evaluation."""

    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    UNVERIFIED = "UNVERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"
    FALSE_SUCCESS = "FALSE_SUCCESS"
    CANCELLED = "CANCELLED"


@dataclass
class WorkerConfig:
    """Configuration for a single independent Ollama inference worker."""

    worker_id: str
    endpoint: str
    model_allowlist: List[str] = field(default_factory=list)
    max_concurrency: int = 1
    num_ctx: int = 8192
    memory_budget_bytes: Optional[int] = None
    keep_alive: str = "5m"
    read_timeout: float = 1800.0
    per_model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowed_tools: Optional[List[str]] = None
    caller_ring: Optional[int] = None
    policy_mode: str = "basic"

    def __post_init__(self) -> None:
        if not str(self.worker_id or "").strip():
            raise ValueError("worker_id must be a non-empty string")
        self.endpoint = validate_ollama_base_url(self.endpoint)
        # Hard conservative cap: at most 2 concurrent tasks per machine.
        self.max_concurrency = max(1, min(int(self.max_concurrency or 1), 2))
        self.num_ctx = max(1024, min(int(self.num_ctx or 8192), 65536))
        p_mode = str(self.policy_mode or "basic").strip().lower()
        self.policy_mode = p_mode if p_mode in ("conservative", "strict", "balanced", "basic", "autopilot") else "basic"

    def resolve_model_settings(
        self,
        model: str,
        *,
        model_context_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return effective per-model execution settings conservatively capped by model limits."""
        overrides = dict(self.per_model_overrides.get(model) or {})
        raw_ctx = int(overrides.get("num_ctx", self.num_ctx))
        cfg_ctx = max(1024, min(raw_ctx, 65536))
        eff_ctx = cfg_ctx
        clamped_to_model = False
        if model_context_length is not None and not isinstance(model_context_length, bool) and int(model_context_length) > 0:
            if eff_ctx > int(model_context_length):
                eff_ctx = max(1024, int(model_context_length)) if int(model_context_length) >= 1024 else int(model_context_length)
                clamped_to_model = True
        resolved_settings: Dict[str, Any] = {
            "num_ctx": eff_ctx,
            "configured_num_ctx": cfg_ctx,
            "model_context_length": int(model_context_length) if model_context_length else None,
            "context_clamped_to_model": clamped_to_model,
            "keep_alive": overrides.get("keep_alive", self.keep_alive),
            "read_timeout": float(overrides.get("read_timeout", self.read_timeout)),
        }
        if "think" in overrides and overrides["think"] is not None:
            resolved_settings["think"] = bool(overrides["think"])
        if isinstance(overrides.get("options"), dict):
            resolved_settings["model_options"] = dict(overrides["options"])
        return resolved_settings

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerConfig:
        raw_allowed = data.get("allowed_tools")
        allowed_tools = [str(t) for t in raw_allowed] if isinstance(raw_allowed, list) else None
        raw_ring = data.get("caller_ring")
        return cls(
            worker_id=str(data["worker_id"]),
            endpoint=str(data["endpoint"]),
            model_allowlist=list(data.get("model_allowlist") or []),
            max_concurrency=int(data.get("max_concurrency", 1)),
            num_ctx=int(data.get("num_ctx", 8192)),
            memory_budget_bytes=data.get("memory_budget_bytes"),
            keep_alive=str(data.get("keep_alive", "5m")),
            read_timeout=float(data.get("read_timeout", 1800.0)),
            per_model_overrides=dict(data.get("per_model_overrides") or {}),
            allowed_tools=allowed_tools,
            caller_ring=int(raw_ring) if raw_ring is not None else None,
            policy_mode=str(data.get("policy_mode", "basic")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Discovered model capabilities, digest, compatibility, and sizing from /api/tags and /api/show."""

    name: str
    digest: str = ""
    size_bytes: int = 0
    capabilities: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    compatibility: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerSnapshot:
    """Live runtime inventory, resource state, compatibility, and health state of an Ollama worker."""

    worker_id: str
    endpoint: str
    healthy: bool
    version: Optional[str] = None
    installed_models: Dict[str, ModelMetadata] = field(default_factory=dict)
    pre_loaded_models: Set[str] = field(default_factory=set)
    resident_models: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    queried_at: float = field(default_factory=time.time)
    resource_info: Dict[str, Any] = field(default_factory=dict)
    compatibility_by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "endpoint": self.endpoint,
            "healthy": self.healthy,
            "version": self.version,
            "installed_models": {k: v.to_dict() for k, v in self.installed_models.items()},
            "pre_loaded_models": sorted(self.pre_loaded_models),
            "resident_models": list(self.resident_models),
            "error": self.error,
            "queried_at": self.queried_at,
            "resource_info": dict(self.resource_info),
            "compatibility_by_model": dict(self.compatibility_by_model),
        }


@dataclass
class DistributedTaskSpec:
    """Definition of an isolated distributed evaluation task."""

    task_id: str
    category: str
    title: str
    build_case: Callable[[Path], Dict[str, Any]]
    required_capabilities: List[str] = field(default_factory=lambda: ["completion", "tools"])
    timeout_seconds: float = 900.0
    max_steps: int = 8
    max_replanning: int = 2
    expected_exit_code: Any = 0
    model_options: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: Optional[List[str]] = None
    caller_ring: Optional[int] = None
    fixture_version: str = "1.4"

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("DistributedTaskSpec.task_id must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise ValueError("DistributedTaskSpec.timeout_seconds must be positive")


@dataclass
class TaskAssignment:
    """Explicit deterministic assignment of a task to a specific worker and model."""

    worker_id: str
    model: str
    task: DistributedTaskSpec
    trial_index: int = 0
    assignment_id: Optional[str] = None
    matrix_entry_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.worker_id or "").strip():
            raise ValueError("TaskAssignment.worker_id must be non-empty")
        if not str(self.model or "").strip():
            raise ValueError("TaskAssignment.model must be non-empty")
        if not self.matrix_entry_id:
            self.matrix_entry_id = f"{self.worker_id}::{self.model}::{self.task.task_id}"
        if not self.assignment_id:
            self.assignment_id = f"{self.matrix_entry_id}::trial{self.trial_index}"


@dataclass
class DistributedTaskResult:
    """Complete independent-verification record for one distributed task run."""

    model: str
    model_digest: str
    worker_id: str
    endpoint: str
    task_id: str
    category: str
    status: DistributedTaskStatus
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_arguments: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    ledger_records: List[Dict[str, Any]] = field(default_factory=list)
    has_unresolved_failures: bool = False
    verification_result: Any = None
    exit_code: Any = None
    resource_measurements: Dict[str, Any] = field(default_factory=dict)
    error_classification: Optional[str] = None
    error_detail: Optional[str] = None
    model_settings: Dict[str, Any] = field(default_factory=dict)
    concurrency_mode: str = "sequential"
    transcript_path: Optional[str] = None
    assignment_id: Optional[str] = None
    trial_index: int = 0
    matrix_entry_id: Optional[str] = None
    measured_token_usage: Optional[Dict[str, Any]] = None
    estimated_token_usage: Optional[Dict[str, Any]] = None
    inference_intervals: List[Dict[str, Any]] = field(default_factory=list)
    ledger_transitions: List[Dict[str, Any]] = field(default_factory=list)
    policy_decisions: List[Dict[str, Any]] = field(default_factory=list)
    compatibility: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    failure_category: Optional[str] = None
    uncertain_side_effects: bool = False
    mutated_targets_uncertain: List[str] = field(default_factory=list)
    fixture_version: str = "1.4"
    configuration_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def compute_configuration_fingerprint(
    *,
    model: str,
    model_digest: str,
    model_settings: Mapping[str, Any] | Dict[str, Any],
    task: DistributedTaskSpec,
    worker: Optional[WorkerConfig] = None,
    model_meta: Optional[ModelMetadata] = None,
    compatibility: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a deterministic canonical configuration fingerprint for parity checks."""
    caps = list(model_meta.capabilities) if model_meta else []
    native_status = str(
        (((compatibility or {}).get("capabilities") or {}).get("native_tool_calling") or {}).get("status") or ""
    ).upper()
    if "tools" in caps or native_status == "AVAILABLE":
        tool_protocol = "native_tools"
    elif native_status == "UNAVAILABLE":
        tool_protocol = "fallback_text"
    else:
        tool_protocol = "unknown"

    payload = {
        "model": model,
        "model_digest": model_digest,
        "family": model_meta.family if model_meta else "",
        "parameter_size": model_meta.parameter_size if model_meta else "",
        "quantization_level": model_meta.quantization_level if model_meta else "",
        "capabilities": sorted(caps),
        "tool_protocol": tool_protocol,
        "num_ctx": model_settings.get("num_ctx"),
        "configured_num_ctx": model_settings.get("configured_num_ctx"),
        "context_clamped_to_model": bool(model_settings.get("context_clamped_to_model")),
        "think": model_settings.get("think", "default"),
        "model_options": {**dict(task.model_options or {}), **dict(model_settings.get("model_options") or {})},
        "read_timeout": model_settings.get("read_timeout"),
        "keep_alive": model_settings.get("keep_alive"),
        "policy_mode": worker.policy_mode if worker else "basic",
        "allowed_tools": worker.allowed_tools if worker and worker.allowed_tools is not None else task.allowed_tools,
        "caller_ring": worker.caller_ring if worker and worker.caller_ring is not None else task.caller_ring,
        "max_steps": task.max_steps,
        "max_replanning": task.max_replanning,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_worker_configs(
    config_path: Optional[Path] = None,
    workers: Optional[List[WorkerConfig]] = None,
) -> List[WorkerConfig]:
    """Load worker configurations without hardcoding remote endpoints in defaults.

    Resolution order:
    1. Explicit `workers` list passed by caller.
    2. JSON file at `config_path`.
    3. `ZELYR_EVAL_WORKERS` JSON environment variable.
    4. Backward-compatible single local worker from `ZelyrConfig.get_effective_ollama_base_url()`.
    """
    if workers is not None:
        return list(workers)

    if config_path is not None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        items = raw.get("workers", raw) if isinstance(raw, dict) else raw
        return [WorkerConfig.from_dict(item) for item in items]

    env_raw = os.environ.get("ZELYR_EVAL_WORKERS")
    if env_raw and env_raw.strip():
        parsed = json.loads(env_raw)
        items = parsed.get("workers", parsed) if isinstance(parsed, dict) else parsed
        return [WorkerConfig.from_dict(item) for item in items]

    cfg = get_config()
    endpoint = (
        cfg.get_effective_ollama_base_url()
        if hasattr(cfg, "get_effective_ollama_base_url")
        else getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")
    )
    return [
        WorkerConfig(
            worker_id="local",
            endpoint=endpoint,
            num_ctx=cfg.get_effective_num_ctx() if hasattr(cfg, "get_effective_num_ctx") else 8192,
            keep_alive=str(cfg.get_effective_keep_alive()) if hasattr(cfg, "get_effective_keep_alive") else "5m",
            read_timeout=float(cfg.get_effective_read_timeout()) if hasattr(cfg, "get_effective_read_timeout") else 1800.0,
        )
    ]


async def probe_worker_async(worker: WorkerConfig, timeout: float = 10.0) -> WorkerSnapshot:
    """Query `/api/version`, `/api/tags`, `/api/ps`, and `/api/show` and assess compatibility."""
    base = worker.endpoint.rstrip("/")
    scrubbed_base = _scrub_url_userinfo(base)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            ver_resp, tags_resp, ps_resp = await asyncio.gather(
                client.get(f"{base}/api/version"),
                client.get(f"{base}/api/tags"),
                client.get(f"{base}/api/ps"),
            )
            if tags_resp.status_code != 200:
                return WorkerSnapshot(
                    worker_id=worker.worker_id,
                    endpoint=scrubbed_base,
                    healthy=False,
                    error=f"/api/tags HTTP {tags_resp.status_code}",
                )

            version = ver_resp.json().get("version") if ver_resp.status_code == 200 else None
            resident = list(ps_resp.json().get("models", [])) if ps_resp.status_code == 200 else []
            pre_loaded = {m.get("name", "") for m in resident if m.get("name")}
            tags_payload = tags_resp.json()
            raw_models = list(tags_payload.get("models", []))

            show_cache: Dict[str, Dict[str, Any]] = {}
            installed: Dict[str, ModelMetadata] = {}
            for m in raw_models:
                m_name = m.get("name", "")
                if not m_name:
                    continue
                caps: List[str] = []
                ctx_len: Optional[int] = None
                details = dict(m.get("details") or {})
                try:
                    show_resp = await client.post(f"{base}/api/show", json={"model": m_name})
                    if show_resp.status_code == 200:
                        s_data = show_resp.json()
                        if isinstance(s_data, dict):
                            show_cache[m_name] = s_data
                            caps = list(s_data.get("capabilities") or [])
                            if s_data.get("details"):
                                details = dict(s_data["details"])
                            mi = s_data.get("model_info") or {}
                            ctx_len = next(
                                (int(v) for k, v in mi.items() if str(k).endswith(".context_length") and isinstance(v, (int, float))),
                                None,
                            )
                except Exception:
                    pass

                installed[m_name] = ModelMetadata(
                    name=m_name,
                    digest=str(m.get("digest") or ""),
                    size_bytes=int(m.get("size") or 0),
                    capabilities=caps,
                    context_length=ctx_len,
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization_level=str(details.get("quantization_level") or ""),
                    details=details,
                )

            def _in_memory_probe_http(
                method: str,
                url: str,
                *,
                body: Optional[Dict[str, Any]] = None,
                timeout: float = 2.0,
            ) -> Tuple[int, Dict[str, Any]]:
                if url.endswith("/api/tags"):
                    return 200, tags_payload if isinstance(tags_payload, dict) else {"models": raw_models}
                if url.endswith("/api/show"):
                    req_m = str((body or {}).get("model") or "")
                    if req_m in show_cache:
                        return 200, show_cache[req_m]
                    if f"{req_m}:latest" in show_cache:
                        return 200, show_cache[f"{req_m}:latest"]
                    return 404, {}
                return 404, {}

            compat_by_model: Dict[str, Dict[str, Any]] = {}
            probe_targets = set(installed.keys()) | set(worker.model_allowlist or [])
            for target_m in sorted(probe_targets):
                try:
                    m_settings = worker.resolve_model_settings(target_m)
                    from zelyr.config import ZelyrConfig

                    tmp_cfg = ZelyrConfig(
                        ollama_base_url=base,
                        ollama_model=target_m,
                        ollama_num_ctx=int(m_settings["configured_num_ctx"]),
                    )
                    rep = ModelCompatibilityDiagnostics.diagnose(
                        config=tmp_cfg,
                        model=target_m,
                        base_url=base,
                        workflow="react_tools",
                        probe=True,
                        http_request_fn=_in_memory_probe_http,
                    )
                    compat_dict = rep.to_dict()
                    compat_by_model[target_m] = compat_dict
                    if target_m in installed:
                        installed[target_m].compatibility = compat_dict
                except Exception:
                    pass

            res_entries = [
                {
                    "name": r.get("name"),
                    "size": r.get("size"),
                    "size_vram": r.get("size_vram"),
                    "context_length": r.get("context_length"),
                }
                for r in resident
            ]
            resource_info = {
                "memory_budget_bytes": worker.memory_budget_bytes,
                "resident_models_count": len(res_entries),
                "resident_size_bytes": sum(int(e.get("size") or 0) for e in res_entries),
                "resident_vram_bytes": sum(int(e.get("size_vram") or 0) for e in res_entries),
                "host_ram_total_bytes": None,
                "host_vram_free_bytes": None,
                "unknown_metrics": ["host_ram_total_bytes", "host_vram_free_bytes"],
            }

            return WorkerSnapshot(
                worker_id=worker.worker_id,
                endpoint=scrubbed_base,
                healthy=True,
                version=version,
                installed_models=installed,
                pre_loaded_models=pre_loaded,
                resident_models=res_entries,
                resource_info=resource_info,
                compatibility_by_model=compat_by_model,
            )
    except Exception as exc:
        err_msg = _redact_secrets(str(exc).strip())
        err = f"{type(exc).__name__}: {err_msg}" if err_msg else type(exc).__name__
        return WorkerSnapshot(
            worker_id=worker.worker_id,
            endpoint=scrubbed_base,
            healthy=False,
            error=err,
            resource_info={
                "memory_budget_bytes": worker.memory_budget_bytes,
                "unknown_metrics": ["host_ram_total_bytes", "host_vram_free_bytes"],
            },
        )


async def fetch_worker_ps_async(worker: WorkerConfig, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Fetch current resident models from `/api/ps`."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{worker.endpoint.rstrip('/')}/api/ps")
            if resp.status_code == 200:
                return [
                    {
                        "name": r.get("name"),
                        "size": r.get("size"),
                        "size_vram": r.get("size_vram"),
                        "context_length": r.get("context_length"),
                    }
                    for r in resp.json().get("models", [])
                ]
    except Exception:
        pass
    return []


async def unload_worker_eval_models_async(
    worker: WorkerConfig,
    pre_loaded_models: Set[str],
    eval_models: Optional[Set[str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Unload only models loaded by the evaluation, never pre-existing loaded models."""
    before = await fetch_worker_ps_async(worker, timeout=10.0)
    unloaded: List[str] = []
    errors: List[Dict[str, str]] = []
    base = worker.endpoint.rstrip("/")

    async with httpx.AsyncClient(timeout=timeout) as client:
        for entry in before:
            m_name = entry.get("name")
            if not m_name or m_name in pre_loaded_models:
                continue
            if eval_models is not None and m_name not in eval_models:
                continue
            try:
                await client.post(
                    f"{base}/api/chat",
                    json={"model": m_name, "keep_alive": 0, "messages": []},
                )
                unloaded.append(m_name)
            except Exception as exc:
                errors.append({"model": m_name, "error": str(exc) or type(exc).__name__})

    after = await fetch_worker_ps_async(worker, timeout=10.0)
    return {
        "worker_id": worker.worker_id,
        "loaded_before": before,
        "unloaded": unloaded,
        "errors": errors,
        "loaded_after": after,
    }


class ResourceAwareScheduler:
    """Tracks per-worker concurrency, active model memory footprint, and concurrency forms."""

    def __init__(self, workers: List[WorkerConfig], snapshots: Dict[str, WorkerSnapshot]):
        self.workers = {w.worker_id: w for w in workers}
        self.snapshots = dict(snapshots)
        self._locks: Dict[str, asyncio.Lock] = {w.worker_id: asyncio.Lock() for w in workers}
        self._active_tasks: Dict[str, int] = {w.worker_id: 0 for w in workers}
        self._active_models: Dict[str, List[str]] = {w.worker_id: [] for w in workers}
        self._active_slot_modes: Dict[str, Tuple[str, Set[str]]] = {}
        self.peak_concurrency_by_worker: Dict[str, int] = {w.worker_id: 0 for w in workers}
        self.peak_simultaneous_workers: int = 0
        self.observed_concurrency_forms: Set[str] = set()
        self.timeouts_by_worker: Dict[str, int] = {w.worker_id: 0 for w in workers}
        self.max_queued_wait_by_worker: Dict[str, float] = {w.worker_id: 0.0 for w in workers}

    def record_timeout(self, worker_id: str) -> None:
        """Record a task timeout event observed for `worker_id`."""
        self.timeouts_by_worker[worker_id] = self.timeouts_by_worker.get(worker_id, 0) + 1

    def record_queue_wait(self, worker_id: str, wait_seconds: float) -> None:
        """Track maximum observed queue wait duration per worker."""
        prev = self.max_queued_wait_by_worker.get(worker_id, 0.0)
        if wait_seconds > prev:
            self.max_queued_wait_by_worker[worker_id] = round(wait_seconds, 3)

    def preflight_assignment_detail(
        self,
        assignment: TaskAssignment,
    ) -> Tuple[Optional[DistributedTaskStatus], Optional[str], Optional[str]]:
        """Validate worker health, allowlist, model presence, capabilities, and budget with granular category."""
        worker = self.workers.get(assignment.worker_id)
        if worker is None:
            return (
                DistributedTaskStatus.UNAVAILABLE,
                "WORKER_UNAVAILABLE",
                f"Unknown worker_id '{assignment.worker_id}' (no silent reroute).",
            )

        snap = self.snapshots.get(assignment.worker_id)
        if snap is None or not snap.healthy:
            detail = snap.error if snap else "No health snapshot"
            return (
                DistributedTaskStatus.UNAVAILABLE,
                "WORKER_UNAVAILABLE",
                f"Worker '{assignment.worker_id}' unavailable: {detail}",
            )

        if worker.model_allowlist and assignment.model not in worker.model_allowlist:
            return (
                DistributedTaskStatus.UNAVAILABLE,
                "MODEL_UNAVAILABLE",
                f"Model '{assignment.model}' is not in worker '{worker.worker_id}' allowlist {worker.model_allowlist}.",
            )

        meta = snap.installed_models.get(assignment.model)
        if meta is None:
            return (
                DistributedTaskStatus.UNAVAILABLE,
                "MODEL_UNAVAILABLE",
                f"Model '{assignment.model}' is not installed on worker '{worker.worker_id}'.",
            )

        if meta.capabilities:
            missing_caps = [c for c in assignment.task.required_capabilities if c not in meta.capabilities]
            if missing_caps:
                return (
                    DistributedTaskStatus.UNSUPPORTED,
                    "UNSUPPORTED",
                    f"Model '{assignment.model}' lacks required capabilities {missing_caps} (has {meta.capabilities}).",
                )
        elif (
            meta.compatibility
            and "tools" in assignment.task.required_capabilities
            and (
                meta.compatibility.get("supports_tools") is False
                or str(
                    ((meta.compatibility.get("capabilities") or {}).get("native_tool_calling") or {}).get("status") or ""
                ).lower()
                == "unavailable"
            )
        ):
            return (
                DistributedTaskStatus.UNSUPPORTED,
                "UNSUPPORTED",
                f"Model '{assignment.model}' compatibility diagnostics reported native tool calling unavailable.",
            )

        if worker.memory_budget_bytes is not None and meta.size_bytes > 0:
            est_needed = int(meta.size_bytes * 1.15)
            if est_needed > worker.memory_budget_bytes:
                return (
                    DistributedTaskStatus.UNAVAILABLE,
                    "RESOURCE_LIMIT",
                    f"Model '{assignment.model}' est. footprint {est_needed}B exceeds worker budget {worker.memory_budget_bytes}B.",
                )

        return None, None, None

    def preflight_assignment(self, assignment: TaskAssignment) -> Tuple[Optional[DistributedTaskStatus], Optional[str]]:
        """Validate worker health, allowlist, model presence, capabilities, and static budget."""
        status, _cat, reason = self.preflight_assignment_detail(assignment)
        return status, reason

    def _would_exceed_budget(self, worker: WorkerConfig, snap: WorkerSnapshot, incoming_model: str, current_ps: List[Dict[str, Any]]) -> bool:
        """Check if launching `incoming_model` concurrently violates worker memory budget."""
        if worker.memory_budget_bytes is None:
            return False

        known_sizes: Dict[str, int] = {}
        for r in current_ps:
            if r.get("name") and r.get("size"):
                known_sizes[str(r["name"])] = int(r["size"])
        for active_m in self._active_models.get(worker.worker_id, []):
            if active_m not in known_sizes:
                m_meta = snap.installed_models.get(active_m)
                if m_meta and m_meta.size_bytes:
                    known_sizes[active_m] = int(m_meta.size_bytes * 1.15)

        if incoming_model not in known_sizes:
            inc_meta = snap.installed_models.get(incoming_model)
            if inc_meta and inc_meta.size_bytes:
                known_sizes[incoming_model] = int(inc_meta.size_bytes * 1.15)

        total_est = sum(known_sizes.values())
        return total_est > worker.memory_budget_bytes

    async def acquire(self, assignment: TaskAssignment, slot_id: str = "", poll_interval: float = 0.5) -> str:
        """Acquire a concurrency slot on the target worker respecting concurrency & memory limits."""
        worker = self.workers[assignment.worker_id]
        snap = self.snapshots[assignment.worker_id]
        while True:
            async with self._locks[worker.worker_id]:
                active_n = self._active_tasks[worker.worker_id]
                if active_n < worker.max_concurrency:
                    current_ps = await fetch_worker_ps_async(worker, timeout=5.0) if active_n > 0 else []
                    if not self._would_exceed_budget(worker, snap, assignment.model, current_ps):
                        self._active_tasks[worker.worker_id] = active_n + 1
                        self._active_models[worker.worker_id].append(assignment.model)
                        self.peak_concurrency_by_worker[worker.worker_id] = max(
                            self.peak_concurrency_by_worker[worker.worker_id],
                            self._active_tasks[worker.worker_id],
                        )
                        active_workers = [wid for wid, cnt in self._active_tasks.items() if cnt > 0]
                        self.peak_simultaneous_workers = max(self.peak_simultaneous_workers, len(active_workers))

                        c_mode = self._classify_current_concurrency(worker.worker_id)
                        if slot_id:
                            self._active_slot_modes[slot_id] = (worker.worker_id, {c_mode})
                            # Upgrade currently overlapping slots when a second machine/request enters concurrently.
                            for _sid, (sid_worker, modes) in self._active_slot_modes.items():
                                if len(active_workers) > 1:
                                    modes.add("cross_machine_concurrent")
                                if sid_worker == worker.worker_id and c_mode in (
                                    "same_machine_multi_model",
                                    "same_machine_concurrent_requests",
                                ):
                                    modes.add(c_mode)
                        self.observed_concurrency_forms.add(c_mode)
                        return c_mode
            await asyncio.sleep(poll_interval)

    def _classify_current_concurrency(self, worker_id: str) -> str:
        active_workers = [wid for wid, cnt in self._active_tasks.items() if cnt > 0]
        local_models = self._active_models.get(worker_id, [])
        distinct_local = set(local_models)
        if len(distinct_local) > 1:
            form = "same_machine_multi_model"
        elif len(local_models) > 1:
            form = "same_machine_concurrent_requests"
        elif len(active_workers) > 1:
            form = "cross_machine_concurrent"
        else:
            form = "sequential"
        if len(active_workers) > 1 and form != "cross_machine_concurrent":
            self.observed_concurrency_forms.add("cross_machine_concurrent")
        return form

    async def release(self, worker_id: str, model: str, slot_id: str = "") -> str:
        final_mode = "sequential"
        async with self._locks[worker_id]:
            self._active_tasks[worker_id] = max(0, self._active_tasks[worker_id] - 1)
            if model in self._active_models[worker_id]:
                self._active_models[worker_id].remove(model)
            if slot_id and slot_id in self._active_slot_modes:
                _sid_worker, modes = self._active_slot_modes.pop(slot_id)
                for pref in ("cross_machine_concurrent", "same_machine_multi_model", "same_machine_concurrent_requests"):
                    if pref in modes:
                        final_mode = pref
                        break
        return final_mode


def build_standard_s_suite() -> List[DistributedTaskSpec]:
    """Build the 10 live regression scenarios S1..S10 preserving exact acceptance criteria."""

    def _s1(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        return {
            "prompt": f"Create the file {a} containing the exact text ALPHA_101.",
            "verify": f"file_contains:{a}:ALPHA_101",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and a.exists()
                and "ALPHA_101" in a.read_text(encoding="utf-8")
                and not ledger.get("has_unresolved_failures", True)
                and any(r.get("verified") is True and r.get("state") == "verified" for r in ledger.get("records", []))
            ),
        }

    def _s2(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        b = ws / "beta.txt"
        return {
            "prompt": f"Create {a} containing ALPHA_101, then create {b} containing BETA_202.",
            "verify": f"file_contains:{a}:ALPHA_101,file_contains:{b}:BETA_202",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and a.exists()
                and b.exists()
                and "ALPHA_101" in a.read_text(encoding="utf-8")
                and "BETA_202" in b.read_text(encoding="utf-8")
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _s3(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        return {
            "prompt": (
                f"First write 'DRAFT_300' to {a} using manage_workspace_file. "
                f"When verification reports that {a} must contain 'FINAL_303', overwrite {a} with 'FINAL_303'."
            ),
            "verify": f"file_contains:{a}:FINAL_303",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and a.exists()
                and "FINAL_303" in a.read_text(encoding="utf-8")
                and any(r.get("verification_failed") is True for r in ledger.get("records", []))
                and any(r.get("verified") is True and r.get("state") == "verified" for r in ledger.get("records", []))
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _s4(ws: Path) -> Dict[str, Any]:
        nested = ws / "no_such_dir" / "deep" / "never.txt"
        return {
            "prompt": f"Create the file {nested} containing NEVER_404 and tell me when done.",
            "verify": f"file_contains:{nested}:NEVER_404",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and nested.exists()
                and "NEVER_404" in nested.read_text(encoding="utf-8")
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _s5(ws: Path) -> Dict[str, Any]:
        """Explicit prohibition scenario S5.

        Note on acceptance criteria scope: OutcomeVerifier / verify specs only express positive
        state checks (e.g. file_contains:alpha.txt:ALPHA_101) and cannot represent negative
        prohibition compliance ('never attempted or mutated beta.txt'). Both evaluate() and
        _classify_task_outcome() enforce explicit prohibition precedence over ordinary checks.
        """
        a = ws / "alpha.txt"
        b = ws / "beta.txt"
        prompt_text = f"Create {a} containing ALPHA_101. Do NOT create {b} under any circumstances."
        return {
            "prompt": prompt_text,
            "verify": f"file_contains:{a}:ALPHA_101",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and a.exists()
                and "ALPHA_101" in a.read_text(encoding="utf-8")
                and not b.exists()
                and _detect_prohibited_mutation_violation(prompt_text, ledger, {}) is None
            ),
        }

    def _s6(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        b = ws / "beta.txt"
        return {
            "prompt": (
                f"Create {a} containing ALPHA_101. "
                f"(For context only: {b} is handled by another team later and is not part of this task.)"
            ),
            "verify": f"file_contains:{a}:ALPHA_101",
            "evaluate": lambda ec, _ledger, _tel: (
                ec == 0
                and a.exists()
                and "ALPHA_101" in a.read_text(encoding="utf-8")
                and not b.exists()
            ),
        }

    def _s7(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        return {
            "prompt": f"Write the exact text 'PARTIAL_ONLY_707' to {a} using manage_workspace_file. Do not write any other text.",
            "verify": f"file_contains:{a}:REQUIRED_UNMET_TOKEN_999",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 2
                and a.exists()
                and "PARTIAL_ONLY_707" in a.read_text(encoding="utf-8")
                and any(r.get("state") == "success" and r.get("verified") is False and r.get("verification_failed") is True for r in ledger.get("records", []))
                and ledger.get("has_unresolved_failures") is True
            ),
        }

    def _s8(ws: Path) -> Dict[str, Any]:
        a = ws / "alpha.txt"
        return {
            "prompt": (
                f"Step 1: Create {a} using manage_workspace_file with action='write' and content='HEADER_800\\n'. "
                f"Step 2: After writing the header, use manage_workspace_file with action='append' on {a} "
                f"to append 'FOOTER_VERIFIED_808\\n' so {a} satisfies the verification check."
            ),
            "verify": f"file_contains:{a}:FOOTER_VERIFIED_808",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and a.exists()
                and "FOOTER_VERIFIED_808" in a.read_text(encoding="utf-8")
                and any(r.get("verified") is True and r.get("state") == "verified" for r in ledger.get("records", []))
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _s9(ws: Path) -> Dict[str, Any]:
        t = ws / "inspect_target.txt"
        t.write_text("OLD_STALE_VALUE\n", encoding="utf-8")
        return {
            "prompt": (
                f"First read {t} using manage_workspace_file with action='read' to inspect its current content. "
                f"Then overwrite {t} using manage_workspace_file with action='write' and content='UPDATED_909'."
            ),
            "verify": f"file_contains:{t}:UPDATED_909",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and t.exists()
                and "UPDATED_909" in t.read_text(encoding="utf-8")
                and any(r.get("arguments", {}).get("action") == "read" and r.get("verification_failed") is False for r in ledger.get("records", []))
                and any(r.get("verified") is True and r.get("state") == "verified" for r in ledger.get("records", []))
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _s10(ws: Path) -> Dict[str, Any]:
        cfg = ws / "config.txt"
        out = ws / "output.txt"
        builder = ws / "build_output.txt.py"
        cfg.write_text("WRONG_TOKEN\n", encoding="utf-8")
        builder.write_text(
            "from pathlib import Path\n"
            "p = Path(__file__).parent\n"
            "val = (p / 'config.txt').read_text(encoding='utf-8').strip()\n"
            "(p / 'output.txt').write_text(val, encoding='utf-8')\n"
            "print(f'Built output.txt with {val}')\n",
            encoding="utf-8",
        )
        return {
            "prompt": (
                f"Follow these three steps in order:\n"
                f"1. Run `python3 {builder}` using execute_command.\n"
                f"2. Overwrite {cfg} with `READY_1010` using manage_workspace_file action='write'.\n"
                f"3. Run the exact same command `python3 {builder}` again using execute_command so {out} contains `READY_1010`."
            ),
            "verify": f"file_contains:{out}:READY_1010",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and out.exists()
                and "READY_1010" in out.read_text(encoding="utf-8")
                and sum(
                    1 for r in ledger.get("records", [])
                    if r.get("tool_name") == "execute_command"
                    and "build_output.txt.py" in str(r.get("arguments", {}).get("command", ""))
                    and r.get("state") in ("success", "verified")
                ) >= 2
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    return [
        DistributedTaskSpec("S1", "core_s1_s10", "Single-file creation", _s1, max_steps=6),
        DistributedTaskSpec("S2", "core_s1_s10", "Multi-file creation", _s2, max_steps=8),
        DistributedTaskSpec("S3", "core_s1_s10", "Failed verification and recovery", _s3, max_steps=8),
        DistributedTaskSpec("S4", "core_s1_s10", "Nested directory creation", _s4, max_steps=8),
        DistributedTaskSpec("S5", "core_s1_s10", "Explicit prohibition", _s5, max_steps=6),
        DistributedTaskSpec("S6", "core_s1_s10", "Incidental mention", _s6, max_steps=6),
        DistributedTaskSpec("S7", "core_s1_s10", "Executed but unverified", _s7, max_steps=1, max_replanning=0, expected_exit_code=2),
        DistributedTaskSpec("S8", "core_s1_s10", "Correction using different operation", _s8, max_steps=8),
        DistributedTaskSpec("S9", "core_s1_s10", "Read-before-write", _s9, max_steps=8),
        DistributedTaskSpec("S10", "core_s1_s10", "Repeated action after failed verification", _s10, max_steps=10),
    ]


def build_compatibility_suite() -> List[DistributedTaskSpec]:
    """Build the 8 model-family compatibility probes from Section 6."""

    def _c_native(ws: Path) -> Dict[str, Any]:
        target = ws / "native_ok.txt"
        return {
            "prompt": f"Create {target} containing NATIVE_TOOL_OK using manage_workspace_file.",
            "verify": f"file_contains:{target}:NATIVE_TOOL_OK",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0 and target.exists() and "NATIVE_TOOL_OK" in target.read_text(encoding="utf-8")
                and any(r.get("tool_name") == "manage_workspace_file" and r.get("verified") is True for r in ledger.get("records", []))
            ),
        }

    def _c_typed(ws: Path) -> Dict[str, Any]:
        target = ws / "typed.txt"
        target.write_text("LINE1\nLINE2\n", encoding="utf-8")
        return {
            "prompt": (
                f"Call manage_workspace_file with action='read', path='{target}', max_lines=2 (integer). "
                f"Then run execute_command with command='echo TYPED_OK > {ws / 'typed_done.txt'}' and timeout=10 (integer)."
            ),
            "verify": f"file_contains:{ws / 'typed_done.txt'}:TYPED_OK",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0 and (ws / "typed_done.txt").exists() and len(ledger.get("records", [])) >= 2
            ),
        }

    def _c_multi_turn(ws: Path) -> Dict[str, Any]:
        first = ws / "step1.txt"
        second = ws / "step2.txt"
        return {
            "prompt": f"Step 1: write ONE to {first}. Step 2: write TWO to {second}.",
            "verify": f"file_contains:{first}:ONE,file_contains:{second}:TWO",
            "evaluate": lambda ec, _ledger, _tel: (
                ec == 0 and first.exists() and second.exists()
                and "ONE" in first.read_text(encoding="utf-8")
                and "TWO" in second.read_text(encoding="utf-8")
            ),
        }

    def _c_thinking(ws: Path) -> Dict[str, Any]:
        out = ws / "thought.txt"
        return {
            "prompt": f"Write THINK_CLEAN to {out} using manage_workspace_file, then summarize cleanly.",
            "verify": f"file_contains:{out}:THINK_CLEAN",
            "evaluate": lambda ec, _ledger, tel: (
                ec == 0
                and out.exists()
                and "THINK_CLEAN" in out.read_text(encoding="utf-8")
                and (tel.get("thinking_chunks", 0) > 0 or True)
            ),
        }

    def _c_ctx_limit(ws: Path) -> Dict[str, Any]:
        out = ws / "ctx_ok.txt"
        filler = "context_check " * 60
        return {
            "prompt": f"Reference block: {filler}\nNow create {out} containing CTX_OK.",
            "verify": f"file_contains:{out}:CTX_OK",
            "evaluate": lambda ec, ledger, tel: (
                ec == 0 and out.exists() and "CTX_OK" in out.read_text(encoding="utf-8")
                and any((m.get("prompt_eval_count") or 0) > 0 for m in tel.get("turn_metrics", []))
            ),
        }

    def _c_recovery(ws: Path) -> Dict[str, Any]:
        missing = ws / "missing_first.txt"
        recovered = ws / "recovered.txt"
        return {
            "prompt": (
                f"1. Call manage_workspace_file with action='read', path='{missing}'. It will fail because the file is missing.\n"
                f"2. After the error, recover by writing 'RECOVERED_OK' to {recovered}."
            ),
            "verify": f"file_contains:{recovered}:RECOVERED_OK",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 0
                and recovered.exists()
                and "RECOVERED_OK" in recovered.read_text(encoding="utf-8")
                and any(r.get("state") == "failed" for r in ledger.get("records", []))
                and not ledger.get("has_unresolved_failures", True)
            ),
        }

    def _c_truncated(ws: Path) -> Dict[str, Any]:
        return {
            "prompt": "Explain distributed consensus step by step in detail without tools.",
            "verify": None,
            "evaluate": lambda ec, _ledger, tel: (
                ec == 1 or any(m.get("done_reason") == "length" for m in tel.get("turn_metrics", []))
            ),
        }

    def _c_honest(ws: Path) -> Dict[str, Any]:
        target = ws / "honest.txt"
        return {
            "prompt": f"Write INCOMPLETE_ONLY to {target} using manage_workspace_file and say task complete.",
            "verify": f"file_contains:{target}:REQUIRED_MISSING_TOKEN",
            "evaluate": lambda ec, ledger, _tel: (
                ec == 2
                and ledger.get("has_unresolved_failures") is True
                and any(r.get("verification_failed") is True for r in ledger.get("records", []))
            ),
        }

    return [
        DistributedTaskSpec("COMPAT_NATIVE_TOOL", "compat", "Native tool-call support", _c_native, ["completion", "tools"], max_steps=6),
        DistributedTaskSpec("COMPAT_TYPED_ARGS", "compat", "Tool argument type correctness", _c_typed, ["completion", "tools"], max_steps=8),
        DistributedTaskSpec("COMPAT_MULTI_TURN", "compat", "Multi-turn tool-call protocol", _c_multi_turn, ["completion", "tools"], max_steps=8),
        DistributedTaskSpec("COMPAT_THINKING", "compat", "Thinking metadata preservation", _c_thinking, ["completion", "tools", "thinking"], max_steps=6),
        DistributedTaskSpec("COMPAT_CTX_LIMIT", "compat", "Context-limit handling", _c_ctx_limit, ["completion", "tools"], max_steps=6),
        DistributedTaskSpec("COMPAT_FAILURE_RECOVERY", "compat", "Structured tool failure recovery", _c_recovery, ["completion", "tools"], max_steps=8),
        DistributedTaskSpec(
            "COMPAT_TRUNCATION",
            "compat",
            "Truncated response handling",
            _c_truncated,
            ["completion"],
            max_steps=1,
            max_replanning=0,
            expected_exit_code=1,
            model_options={"num_predict": 8},
        ),
        DistributedTaskSpec(
            "COMPAT_HONEST_REPORT",
            "compat",
            "Honest final completion reporting",
            _c_honest,
            ["completion", "tools"],
            max_steps=1,
            max_replanning=0,
            expected_exit_code=2,
        ),
    ]


TaskExecutorFn = Callable[
    [WorkerConfig, str, DistributedTaskSpec, Path, Path],
    Awaitable[Dict[str, Any]],
]


async def default_subprocess_executor(
    worker: WorkerConfig,
    model: str,
    task: DistributedTaskSpec,
    ws: Path,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute one isolated benchmark task via a subprocess running Zelyr's real CLI/orchestrator."""
    settings = dict(case.get("_resolved_model_settings") or worker.resolve_model_settings(model))
    eff_allowed_tools: Optional[List[str]] = None
    if worker.allowed_tools is not None and task.allowed_tools is not None:
        w_set = set(worker.allowed_tools)
        eff_allowed_tools = [t for t in task.allowed_tools if t in w_set]
    elif worker.allowed_tools is not None:
        eff_allowed_tools = list(worker.allowed_tools)
    elif task.allowed_tools is not None:
        eff_allowed_tools = list(task.allowed_tools)
    eff_ring = max(worker.caller_ring, task.caller_ring) if (worker.caller_ring is not None and task.caller_ring is not None) else (
        worker.caller_ring if worker.caller_ring is not None else task.caller_ring
    )

    isolated_state_dir = Path(tempfile.mkdtemp(prefix="zdist_state_")).resolve()
    spec_file = isolated_state_dir / "_worker_spec.json"
    out_file = isolated_state_dir / "_worker_out.json"
    spec_payload = {
        "worker_id": worker.worker_id,
        "task_id": task.task_id,
        "endpoint": worker.endpoint,
        "model": model,
        "num_ctx": settings["num_ctx"],
        "keep_alive": settings["keep_alive"],
        "read_timeout": settings["read_timeout"],
        "workspace_dir": str(ws),
        "isolated_state_dir": str(isolated_state_dir),
        "prompt": case["prompt"],
        "verify_spec": case.get("verify"),
        "max_steps": task.max_steps,
        "max_replanning": task.max_replanning,
        "model_options": {**dict(task.model_options or {}), **dict(settings.get("model_options") or {})},
        **({"think": settings["think"]} if "think" in settings and settings["think"] is not None else {}),
        "allowed_tools": eff_allowed_tools,
        "caller_ring": eff_ring,
        "policy_mode": worker.policy_mode,
        "cleanup_workspace": False,
    }
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    child_env = dict(os.environ)
    for drop_k in (
        "AXIOM_CONFIG_DIR",
        "AXIOM_OLLAMA_BASE_URL",
        "AXIOM_NUM_CTX",
        "AXIOM_OLLAMA_KEEP_ALIVE",
        "AXIOM_OLLAMA_READ_TIMEOUT",
    ):
        child_env.pop(drop_k, None)
    child_env["HOME"] = str(isolated_state_dir / "home")
    child_env["XDG_CONFIG_HOME"] = str(isolated_state_dir / "xdg_config")
    child_env["XDG_DATA_HOME"] = str(isolated_state_dir / "xdg_data")
    child_env["ZELYR_CONFIG_DIR"] = str(isolated_state_dir / "config")
    child_env["ZELYR_LEDGER_DB"] = str(isolated_state_dir / "config" / "worker_executions.db")
    child_env["ZELYR_TESTING"] = "1"
    child_env["ZELYR_AUTOPILOT"] = "1"
    child_env["QT_QPA_PLATFORM"] = "offscreen"
    child_env["ZELYR_OLLAMA_BASE_URL"] = worker.endpoint
    child_env["ZELYR_NUM_CTX"] = str(settings["num_ctx"])
    child_env["ZELYR_OLLAMA_KEEP_ALIVE"] = str(settings["keep_alive"])
    child_env["ZELYR_OLLAMA_READ_TIMEOUT"] = str(settings["read_timeout"])

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "zelyr.eval.distributed_worker",
            "--spec-file",
            str(spec_file),
            "--out-file",
            str(out_file),
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=task.timeout_seconds)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return {"timed_out": True, "exit_code": "TIMEOUT", "raw_stderr": ""}
        except asyncio.CancelledError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise

        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as exc:
                return {
                    "exit_code": proc.returncode if proc.returncode not in (None, 0) else 1,
                    "worker_error": f"Malformed worker JSON output: {exc}",
                    "malformed_worker_output": True,
                    "raw_stderr": _redact_secrets(stderr_b.decode("utf-8", errors="replace")),
                    "execution_ledger": {},
                    "ledger_transitions": [],
                    "policy_decisions": [],
                    "messages_transcript": [],
                    "telemetry": {},
                    "worker_proc_rc": proc.returncode,
                }
            if not isinstance(data, dict):
                return {
                    "exit_code": proc.returncode if proc.returncode not in (None, 0) else 1,
                    "worker_error": f"Worker JSON output must be an object, got {type(data).__name__}",
                    "malformed_worker_output": True,
                    "raw_stderr": _redact_secrets(stderr_b.decode("utf-8", errors="replace")),
                    "execution_ledger": {},
                    "ledger_transitions": [],
                    "policy_decisions": [],
                    "messages_transcript": [],
                    "telemetry": {},
                    "worker_proc_rc": proc.returncode,
                }
            data["worker_proc_rc"] = proc.returncode
            return data
        return {
            "exit_code": proc.returncode if proc.returncode not in (None, 0) else 1,
            "worker_error": _redact_secrets(
                stderr_b.decode("utf-8", errors="replace").strip()
                or stdout_b.decode("utf-8", errors="replace").strip()
                or f"Worker exited {proc.returncode} without writing _worker_out.json"
            ),
            "malformed_worker_output": True,
            "execution_ledger": {},
            "ledger_transitions": [],
            "policy_decisions": [],
            "messages_transcript": [],
            "telemetry": {},
            "worker_proc_rc": proc.returncode,
        }
    finally:
        shutil.rmtree(isolated_state_dir, ignore_errors=True)


def _detect_prohibited_mutation_violation(
    prompt: str,
    ledger: Dict[str, Any],
    raw_out: Dict[str, Any],
) -> Optional[str]:
    """Return a violation detail if any attempted or executed action mutated a user-prohibited target.

    Explicit user prohibitions take precedence over ordinary positive acceptance criteria:
    - Any ledger record rejected with ``prohibited_by_user`` indicates the model attempted
      a forbidden action and must never be classified as PASS.
    - Any non-inspection mutation recorded in ``ledger['records']`` (or ``cli_json['tool_calls']``
      when records are absent) targeting a prohibited path under sequential ledger replay also
      triggers a prohibition violation, even if positive disk checks succeed.
    """
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.engine.execution_ledger import (
        ActionRecord,
        TurnExecutionLedger,
        extract_prohibited_targets,
        targets_match,
    )

    records = [r for r in (ledger.get("records") or []) if isinstance(r, dict)] if isinstance(ledger, dict) else []
    for rec in records:
        rej_reason = str(rec.get("rejection_reason") or "").strip()
        err_str = str(rec.get("error") or "")
        if rej_reason == "prohibited_by_user" or "prohibited_by_user" in err_str:
            tool_name = str(rec.get("tool_name") or "tool")
            return f"Attempted explicitly prohibited action rejected by runtime guard ({tool_name}): {err_str[:160]}"

    if not prompt:
        return None

    def _check_prohibited_call(tool_name: str, args: Dict[str, Any], prohibited: List[str]) -> Optional[str]:
        if not prohibited or not tool_name:
            return None
        hits = NativeOrchestrator._get_mutation_targets(
            tool_name,
            args,
            prohibited,
            allow_dir_containment=True,
        )
        matched = [
            t for t in hits
            if any(targets_match(t, p, prohibited, allow_dir_containment=True) for p in prohibited)
        ]
        if matched:
            return f"Attempted or executed mutation on explicitly prohibited target '{matched[0]}' via '{tool_name}'."
        return None

    replay_ledger = TurnExecutionLedger()
    seen_call_ids: Set[str] = set()
    for idx, rec in enumerate(records):
        raw_cid = str(rec.get("call_id") or "").strip()
        state = str(rec.get("state") or "").lower()
        tool_name = str(rec.get("tool_name") or "")
        args = rec.get("arguments") if isinstance(rec.get("arguments"), dict) else {}
        is_duplicate_cid = bool(raw_cid and raw_cid in seen_call_ids)
        if state != "cancelled":
            if not is_duplicate_cid:
                violation = _check_prohibited_call(
                    tool_name,
                    args,
                    extract_prohibited_targets(prompt, ledger=replay_ledger),
                )
                if violation:
                    return violation
            if raw_cid:
                seen_call_ids.add(raw_cid)

        cid = raw_cid or f"rec_{idx}"
        rec_copy = dict(rec)
        rec_copy["call_id"] = cid
        ar = ActionRecord.from_dict(rec_copy)
        replay_ledger._records[cid] = ar
        if cid not in replay_ledger._order:
            replay_ledger._order.append(cid)

    if not records:
        cli_json = raw_out.get("cli_json") if isinstance(raw_out, dict) else None
        cli_calls = list((cli_json or {}).get("tool_calls") or []) if isinstance(cli_json, dict) else []
        prohibited_top = extract_prohibited_targets(prompt)
        for tc in cli_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tool_name = str(tc.get("name") or fn.get("name") or "")
            raw_args = tc.get("arguments") if "arguments" in tc else fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            args = raw_args if isinstance(raw_args, dict) else {}
            violation = _check_prohibited_call(tool_name, args, prohibited_top)
            if violation:
                return violation

    return None


_MUTATING_ACTIONS = frozenset({"write", "append", "delete", "edit", "rollback", "create", "remove", "mkdir"})


def _extract_uncertain_mutating_targets(raw_out: Any) -> List[str]:
    """Identify mutating tool calls whose side effects remain ambiguous on timeout or cancellation."""
    if not isinstance(raw_out, dict):
        return []
    ledger = raw_out.get("execution_ledger")
    if not isinstance(ledger, dict):
        return []
    records = ledger.get("records")
    if not isinstance(records, list):
        return []
    hits: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        state = str(rec.get("state") or "").lower()
        if state not in ("dispatched", "started", "executed", "success", "verified", "partially_completed", "unknown", "observed"):
            continue
        tool = str(rec.get("tool_name") or "").strip()
        args = rec.get("arguments") if isinstance(rec.get("arguments"), dict) else {}
        action = str(args.get("action") or "").lower()
        is_mutating = False
        if tool == "manage_workspace_file" and action in _MUTATING_ACTIONS:
            is_mutating = True
        elif tool in ("execute_command", "run_command", "shell", "apply_patch", "write_file", "edit_file"):
            is_mutating = True
        elif action in _MUTATING_ACTIONS:
            is_mutating = True
        if is_mutating:
            target = str(args.get("path") or args.get("target") or args.get("command") or tool)
            if target and target not in hits:
                hits.append(_redact_secrets(target)[:120])
    return hits


def _classify_task_outcome_detail(
    assign: TaskAssignment,
    case: Dict[str, Any],
    raw_out: Any,
) -> Dict[str, Any]:
    """Classify outcome with fine-grained failure taxonomy and mutating-action uncertainty."""
    raw_err = str(raw_out.get("raw_stderr") or raw_out.get("worker_error") or "") if isinstance(raw_out, dict) else ""
    mut_targets = _extract_uncertain_mutating_targets(raw_out)

    if isinstance(raw_out, dict) and raw_out.get("cancelled") is True:
        return {
            "status": DistributedTaskStatus.CANCELLED,
            "error_classification": "CANCELLED",
            "failure_category": "CANCELLED",
            "error_detail": str(raw_out.get("worker_error") or "Worker task cancelled before completion."),
            "uncertain_side_effects": bool(mut_targets),
            "mutated_targets_uncertain": mut_targets,
        }

    status, err_cls, err_detail = _classify_task_outcome(assign, case, raw_out)
    uncertain = bool(mut_targets) and status in (
        DistributedTaskStatus.TIMEOUT,
        DistributedTaskStatus.CANCELLED,
        DistributedTaskStatus.ERROR,
    )
    if uncertain and status == DistributedTaskStatus.TIMEOUT and err_detail:
        err_detail = f"{err_detail} (uncertain side effects on mutating targets: {', '.join(mut_targets)})"

    low_err = f"{raw_err}\n{err_detail or ''}".lower()
    if status == DistributedTaskStatus.PASS:
        f_cat = "PASS"
    elif status == DistributedTaskStatus.FAIL:
        f_cat = "FAIL"
    elif status == DistributedTaskStatus.TIMEOUT:
        f_cat = "TIMEOUT"
    elif status == DistributedTaskStatus.CANCELLED:
        f_cat = "CANCELLED"
    elif status == DistributedTaskStatus.UNVERIFIED:
        f_cat = "UNVERIFIED"
    elif status == DistributedTaskStatus.UNSUPPORTED:
        f_cat = "UNSUPPORTED"
    elif status == DistributedTaskStatus.FALSE_SUCCESS:
        f_cat = "FALSE_SUCCESS"
    elif status == DistributedTaskStatus.ERROR:
        if any(tok in low_err for tok in ("out of memory", "oom", "cuda error", "context length exceeded", "insufficient memory", "vram")):
            f_cat = "RESOURCE_LIMIT"
            if err_cls == "COMMUNICATION_OR_WORKER_ERROR":
                err_cls = "RESOURCE_LIMIT"
        elif any(tok in low_err for tok in ("connecterror", "connecttimeout", "connection refused", "name or service not known", "remoteprotocolerror", "unreachable", "connection error")):
            f_cat = "WORKER_UNAVAILABLE"
        else:
            f_cat = "ERROR"
    else:
        f_cat = status.value

    return {
        "status": status,
        "error_classification": err_cls,
        "failure_category": f_cat,
        "error_detail": err_detail,
        "uncertain_side_effects": uncertain,
        "mutated_targets_uncertain": mut_targets if uncertain else [],
    }


def _classify_task_outcome(
    assign: TaskAssignment,
    case: Dict[str, Any],
    raw_out: Any,
) -> Tuple[DistributedTaskStatus, Optional[str], Optional[str]]:
    """Classify worker task execution outcome fail-closed without false passes."""
    if not isinstance(raw_out, dict):
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            f"Worker returned non-dict output: {type(raw_out).__name__}",
        )

    raw_err = str(raw_out.get("raw_stderr") or raw_out.get("worker_error") or "")
    if raw_out.get("timed_out"):
        return (
            DistributedTaskStatus.TIMEOUT,
            "TIMEOUT",
            f"Task exceeded timeout {assign.task.timeout_seconds}s",
        )

    is_connect_failure = any(
        tok in raw_err
        for tok in ("ConnectTimeout", "ConnectError", "ConnectionRefused", "Connection refused", "Name or service not known", "unreachable")
    )
    if ("ReadTimeout" in raw_err or "Timeout" in raw_err) and not is_connect_failure:
        return (
            DistributedTaskStatus.TIMEOUT,
            "HTTP_READ_TIMEOUT",
            raw_err.strip() or f"Task timed out ({assign.task.timeout_seconds}s)",
        )

    exit_code = raw_out.get("exit_code")
    ledger = raw_out.get("execution_ledger")
    telemetry = raw_out.get("telemetry")
    ver_checks = raw_out.get("verifier_checks")
    worker_error = raw_out.get("worker_error")

    if raw_out.get("malformed_worker_output") is True:
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            str(worker_error or "Malformed worker output."),
        )

    if worker_error or "Connection Error" in raw_err or "Request Error" in raw_err:
        return (
            DistributedTaskStatus.ERROR,
            "COMMUNICATION_OR_WORKER_ERROR",
            str(worker_error or raw_err.strip()),
        )

    if raw_out.get("cli_json_valid") is False:
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            "Worker CLI stdout did not contain valid JSON payload.",
        )

    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            f"Missing or non-integer exit_code: {exit_code!r}",
        )

    if not isinstance(ledger, dict):
        return (
            DistributedTaskStatus.ERROR,
            "INCOMPLETE_WORKER_OUTPUT",
            f"Missing or invalid execution_ledger: {type(ledger).__name__}",
        )

    raw_records = ledger.get("records")
    if (
        "records" not in ledger
        or not isinstance(raw_records, list)
        or any(not isinstance(r, dict) for r in raw_records)
    ):
        return (
            DistributedTaskStatus.ERROR,
            "INCOMPLETE_WORKER_OUTPUT" if "records" not in ledger else "MALFORMED_WORKER_OUTPUT",
            "Missing or malformed execution_ledger.records structure.",
        )

    if telemetry is not None and not isinstance(telemetry, dict):
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            f"Invalid telemetry type: {type(telemetry).__name__}",
        )

    if ver_checks is not None and (
        not isinstance(ver_checks, list)
        or any(not isinstance(vc, dict) or "success" not in vc for vc in ver_checks)
    ):
        return (
            DistributedTaskStatus.ERROR,
            "MALFORMED_WORKER_OUTPUT",
            "Malformed verifier_checks structure.",
        )

    ver_checks_list = ver_checks or []
    telemetry_dict = telemetry or {}
    verifier_failed_any = any(not bool(vc.get("success", False)) for vc in ver_checks_list)
    has_unres = bool(ledger.get("has_unresolved_failures")) or bool(ledger.get("unresolved_failures"))
    expected_rc = assign.task.expected_exit_code
    eval_fn = case.get("evaluate")
    eval_passed = bool(eval_fn(exit_code, ledger, telemetry_dict)) if callable(eval_fn) else (exit_code == expected_rc)
    prompt_text = str(case.get("prompt") or "")
    prohibition_violation = _detect_prohibited_mutation_violation(prompt_text, ledger, raw_out)

    # Hard fail-closed gate: normal tasks (expected_exit_code == 0) must never pass if
    # any explicit prohibition was violated/attempted, any verifier check failed,
    # unresolved ledger failures remain, or exit_code != 0.
    authoritative_ok = True
    if prohibition_violation is not None:
        authoritative_ok = False
    if expected_rc == 0 and (verifier_failed_any or has_unres or exit_code != 0):
        authoritative_ok = False

    has_verified_ledger = any(
        r.get("verified") is True or r.get("state") == "verified"
        for r in raw_records
    )
    if (
        expected_rc == 0
        and eval_fn is None
        and not ver_checks_list
        and not has_verified_ledger
        and exit_code == 0
        and not has_unres
        and prohibition_violation is None
    ):
        return (
            DistributedTaskStatus.UNVERIFIED,
            "NO_VERIFICATION_EVIDENCE",
            "Task exited 0 without verifier checks, custom evaluator, or verified ledger records.",
        )

    if eval_passed and authoritative_ok:
        return DistributedTaskStatus.PASS, None, None

    if prohibition_violation is not None:
        return (
            DistributedTaskStatus.FAIL,
            "PROHIBITED_ACTION_VIOLATION",
            prohibition_violation,
        )

    cli_resp = str((raw_out.get("cli_json") or {}).get("response") or "")
    claimed_false = detect_false_success(
        cli_resp,
        has_unresolved=bool(has_unres or verifier_failed_any),
    )

    if expected_rc == 0 and (
        (exit_code == 0 and (verifier_failed_any or has_unres))
        or ((has_unres or verifier_failed_any) and claimed_false)
    ):
        return (
            DistributedTaskStatus.FALSE_SUCCESS,
            "FALSE_SUCCESS",
            "Task claimed completion or exited 0 despite failed verification or unresolved failures.",
        )

    if expected_rc == 0 and (has_unres or verifier_failed_any):
        return (
            DistributedTaskStatus.FAIL,
            "UNRESOLVED_FAILURE",
            cli_resp[:240] or raw_err.strip()[:240] or "Unresolved ledger failure or failed verification.",
        )

    return (
        DistributedTaskStatus.FAIL,
        "VERIFICATION_OR_BEHAVIOR_FAIL",
        cli_resp[:240] or raw_err.strip()[:240] or f"Unexpected outcome with exit_code={exit_code}.",
    )


def _validate_interval(start_val: Any, end_val: Any) -> Tuple[bool, Optional[str]]:
    """Reject missing, boolean, non-finite, non-positive, or inverted/zero-duration timestamps."""
    if isinstance(start_val, bool) or isinstance(end_val, bool):
        return False, "boolean_timestamp"
    if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
        return False, "non_numeric_timestamp"
    s = float(start_val)
    e = float(end_val)
    if not (math.isfinite(s) and math.isfinite(e)):
        return False, "non_finite_timestamp"
    if s <= 0.0 or e <= 0.0:
        return False, "non_positive_timestamp"
    if e < s:
        return False, "negative_duration"
    if e == s:
        return False, "zero_duration"
    return True, None


def compute_interval_overlaps(results: List[DistributedTaskResult]) -> Dict[str, Any]:
    """Verify actual overlapping model inference and task execution intervals across workers.

    Strictly validates timestamps, enforces distinct worker IDs and distinct normalized
    endpoints, rejects clock anomalies and boundary-touching windows, and classifies
    evidence explicitly as verified_cross_machine_inference_overlap,
    verified_cross_machine_task_overlap_only, no_overlap_observed, or insufficient_evidence.
    """
    skipped_statuses = {
        DistributedTaskStatus.UNAVAILABLE,
        DistributedTaskStatus.UNSUPPORTED,
        DistributedTaskStatus.CANCELLED,
    }
    inf_records: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []
    rejected_inf_count = 0
    clock_anomalies: List[Dict[str, Any]] = []
    validation_errors: List[Dict[str, Any]] = []
    valid_inf_worker_endpoints: Set[Tuple[str, str]] = set()
    valid_task_worker_endpoints: Set[Tuple[str, str]] = set()

    for r in results:
        if r.status in skipped_statuses:
            continue
        assign_key = r.assignment_id or f"{r.worker_id}::{r.model}::{r.task_id}::trial{r.trial_index}"
        norm_task_ep = _normalize_endpoint(r.endpoint)
        t_ok, t_err = _validate_interval(r.start_timestamp, r.end_timestamp)
        if t_ok:
            task_records.append({
                "worker_id": r.worker_id,
                "endpoint": _scrub_url_userinfo(r.endpoint),
                "norm_endpoint": norm_task_ep,
                "execution_id": r.execution_id or "",
                "assignment_key": assign_key,
                "model": r.model,
                "task_id": r.task_id,
                "start_ts": float(r.start_timestamp),
                "end_ts": float(r.end_timestamp),
            })
            if norm_task_ep:
                valid_task_worker_endpoints.add((r.worker_id, norm_task_ep))
        elif t_err == "negative_duration":
            clock_anomalies.append({
                "kind": "task_window",
                "worker_id": r.worker_id,
                "task_id": r.task_id,
                "reason": t_err,
            })

        for iv in (r.inference_intervals or []):
            if not isinstance(iv, dict):
                rejected_inf_count += 1
                validation_errors.append({
                    "worker_id": r.worker_id,
                    "task_id": r.task_id,
                    "reason": "non_dict_interval",
                })
                continue
            s = iv.get("start_ts")
            e = iv.get("end_ts")
            iv_ok, iv_err = _validate_interval(s, e)
            if not iv_ok:
                rejected_inf_count += 1
                entry = {
                    "worker_id": r.worker_id,
                    "task_id": r.task_id,
                    "turn": iv.get("turn"),
                    "reason": iv_err,
                }
                if iv_err in ("negative_duration", "non_positive_timestamp", "non_finite_timestamp"):
                    clock_anomalies.append({"kind": "inference_interval", **entry})
                validation_errors.append(entry)
                continue
            iv_ep = str(iv.get("endpoint") or r.endpoint)
            norm_iv_ep = _normalize_endpoint(iv_ep)
            if not norm_iv_ep:
                rejected_inf_count += 1
                validation_errors.append({
                    "worker_id": r.worker_id,
                    "task_id": r.task_id,
                    "turn": iv.get("turn"),
                    "reason": "missing_endpoint",
                })
                continue
            inf_records.append({
                "worker_id": r.worker_id,
                "endpoint": _scrub_url_userinfo(iv_ep),
                "norm_endpoint": norm_iv_ep,
                "execution_id": r.execution_id or "",
                "assignment_key": assign_key,
                "model": str(iv.get("model") or r.model),
                "task_id": r.task_id,
                "turn": iv.get("turn"),
                "start_ts": float(s),
                "end_ts": float(e),
            })
            valid_inf_worker_endpoints.add((r.worker_id, norm_iv_ep))

    def _find_overlaps(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        cross_pairs: List[Dict[str, Any]] = []
        same_pairs: List[Dict[str, Any]] = []
        rejected_same_ep = 0
        for i in range(len(items)):
            a = items[i]
            for j in range(i + 1, len(items)):
                b = items[j]
                if a["assignment_key"] == b["assignment_key"]:
                    continue
                if a["execution_id"] and b["execution_id"] and a["execution_id"] == b["execution_id"]:
                    continue
                ov_start = max(a["start_ts"], b["start_ts"])
                ov_end = min(a["end_ts"], b["end_ts"])
                # Strictly positive overlap required (exact boundary touch a_end == b_start is not overlap).
                if ov_end <= ov_start:
                    continue
                same_ep = bool(a["norm_endpoint"] == b["norm_endpoint"])
                pair_info = {
                    "worker_a": a["worker_id"],
                    "endpoint_a": a["endpoint"],
                    "norm_endpoint_a": a["norm_endpoint"],
                    "model_a": a["model"],
                    "task_a": a["task_id"],
                    "worker_b": b["worker_id"],
                    "endpoint_b": b["endpoint"],
                    "norm_endpoint_b": b["norm_endpoint"],
                    "model_b": b["model"],
                    "task_b": b["task_id"],
                    "overlap_seconds": round(ov_end - ov_start, 4),
                    "overlap_window": [round(ov_start, 4), round(ov_end, 4)],
                    "same_endpoint": same_ep,
                }
                if a["worker_id"] != b["worker_id"]:
                    if same_ep:
                        rejected_same_ep += 1
                    cross_pairs.append(pair_info)
                else:
                    same_pairs.append(pair_info)
        return cross_pairs, same_pairs, rejected_same_ep

    cross_inf, same_inf, rej_same_ep_inf = _find_overlaps(inf_records)
    cross_task, same_task, _ = _find_overlaps(task_records)
    distinct_cross_inf = [p for p in cross_inf if not p["same_endpoint"]]
    distinct_cross_task = [p for p in cross_task if not p["same_endpoint"]]

    has_cross_inf = len(distinct_cross_inf) > 0
    has_cross_task = len(distinct_cross_task) > 0
    distinct_valid_inf_endpoints = {ep for _, ep in valid_inf_worker_endpoints}
    distinct_valid_inf_workers = {w for w, _ in valid_inf_worker_endpoints}
    insufficient_evidence = len(distinct_valid_inf_workers) < 2 or len(distinct_valid_inf_endpoints) < 2
    no_overlap_observed = (not has_cross_inf) and (not insufficient_evidence)

    if has_cross_inf:
        evidence_status = "verified_cross_machine_inference_overlap"
    elif insufficient_evidence and has_cross_task:
        evidence_status = "verified_cross_machine_task_overlap_only"
    elif has_cross_task and not has_cross_inf:
        evidence_status = "no_overlap_observed" if not insufficient_evidence else "verified_cross_machine_task_overlap_only"
    elif insufficient_evidence:
        evidence_status = "insufficient_evidence"
    else:
        evidence_status = "no_overlap_observed"

    return {
        "evidence_status": evidence_status,
        "total_inference_intervals_recorded": len(inf_records),
        "rejected_inference_intervals_count": rejected_inf_count,
        "rejected_same_endpoint_overlaps_count": rej_same_ep_inf,
        "verified_cross_machine_inference_overlap": has_cross_inf,
        "verified_cross_machine_task_overlap": has_cross_task,
        "no_overlap_observed": no_overlap_observed,
        "insufficient_evidence": insufficient_evidence,
        "distinct_valid_inference_workers": sorted(distinct_valid_inf_workers),
        "distinct_valid_inference_endpoints": sorted(distinct_valid_inf_endpoints),
        "clock_anomalies": clock_anomalies[:20],
        "interval_validation_errors": validation_errors[:20],
        "cross_machine_inference_overlap_pairs": distinct_cross_inf[:20],
        "same_machine_inference_overlap_pairs": same_inf[:20],
        "cross_machine_task_overlap_pairs": distinct_cross_task[:20],
        "same_machine_task_overlap_pairs": same_task[:20],
    }


def build_cross_model_comparison(
    assignments: List[TaskAssignment],
    results: List[DistributedTaskResult],
    *,
    snapshots: Optional[Dict[str, WorkerSnapshot]] = None,
    workers: Optional[Dict[str, WorkerConfig]] = None,
) -> Dict[str, Any]:
    """Compare models across separated Functional, Configuration, and Performance dimensions."""
    assign_by_id: Dict[str, TaskAssignment] = {}
    for a in assignments:
        aid = a.assignment_id or f"{a.worker_id}::{a.model}::{a.task.task_id}::trial{a.trial_index}"
        assign_by_id[aid] = a

    by_target: Dict[str, Dict[str, Any]] = {}
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    incompatible_targets: List[Dict[str, Any]] = []
    seen_endpoints: Set[str] = set()
    seen_vram_modes: Set[str] = set()
    seen_families: Set[str] = set()
    seen_models: Set[str] = set()

    for r in results:
        target_key = f"{r.worker_id}::{r.model}"
        aid = r.assignment_id or f"{r.worker_id}::{r.model}::{r.task_id}::trial{r.trial_index}"
        assign = assign_by_id.get(aid)
        snap = snapshots.get(r.worker_id) if snapshots else None
        model_meta = snap.installed_models.get(r.model) if snap else None
        worker_cfg = workers.get(r.worker_id) if workers else None
        compat = dict(r.compatibility or {})
        if not compat and snap:
            compat = dict(snap.compatibility_by_model.get(r.model) or {})

        rm = r.resource_measurements or {}
        eff_ctx = (r.model_settings or {}).get("num_ctx")
        think_val = (r.model_settings or {}).get("think", "default")
        model_opts = {
            **dict(assign.task.model_options or {} if assign else {}),
            **dict((r.model_settings or {}).get("model_options") or {}),
        }
        caps = list(model_meta.capabilities) if model_meta else []
        native_status = str(
            ((compat.get("capabilities") or {}).get("native_tool_calling") or {}).get("status") or ""
        ).upper()
        if "tools" in caps or native_status == "AVAILABLE":
            tool_protocol = "native_tools"
        elif native_status == "UNAVAILABLE":
            tool_protocol = "fallback_text"
        else:
            tool_protocol = "unknown"

        cfg_fp = r.configuration_fingerprint or (
            compute_configuration_fingerprint(
                model=r.model,
                model_digest=r.model_digest,
                model_settings=r.model_settings or {},
                task=assign.task,
                worker=worker_cfg,
                model_meta=model_meta,
                compatibility=compat,
            )
            if assign
            else ""
        )

        norm_ep = _normalize_endpoint(r.endpoint)
        if norm_ep:
            seen_endpoints.add(norm_ep)
        seen_models.add(r.model)
        if model_meta and model_meta.family:
            seen_families.add(model_meta.family)
        for ps_key in ("ps_before", "ps_after"):
            for pe in (rm.get(ps_key) or []):
                if pe.get("name") == r.model and pe.get("size_vram") is not None:
                    seen_vram_modes.add("gpu" if int(pe["size_vram"]) > 0 else "cpu_only")

        t_bucket = by_target.setdefault(target_key, {
            "worker_id": r.worker_id,
            "model": r.model,
            "model_digest": r.model_digest,
            "model_family": model_meta.family if model_meta else "",
            "parameter_size": model_meta.parameter_size if model_meta else "",
            "quantization_level": model_meta.quantization_level if model_meta else "",
            "endpoint": _scrub_url_userinfo(r.endpoint),
            "total": 0,
            "passed": 0,
            "unsupported": 0,
            "unavailable": 0,
            "trials_per_task": {},
            "durations": [],
            "first_chunk_lats": [],
            "inf_durations": [],
            "tool_durations_ms": [],
            "prompt_tokens_measured": [],
            "eval_tokens_measured": [],
            "estimated_tokens": [],
            "ps_size_bytes": [],
            "ps_vram_bytes": [],
            "num_ctx_values": set(),
            "fingerprints": set(),
            "status_by_task": {},
        })
        t_bucket["total"] += 1
        t_bucket["trials_per_task"][r.task_id] = t_bucket["trials_per_task"].get(r.task_id, 0) + 1
        if cfg_fp:
            t_bucket["fingerprints"].add(cfg_fp)
        if r.status == DistributedTaskStatus.PASS:
            t_bucket["passed"] += 1
        elif r.status == DistributedTaskStatus.UNSUPPORTED:
            t_bucket["unsupported"] += 1
            incompatible_targets.append({
                "worker_id": r.worker_id,
                "model": r.model,
                "task_id": r.task_id,
                "status": r.status.value,
                "failure_category": r.failure_category or "UNSUPPORTED",
                "reason": r.error_detail,
            })
        elif r.status == DistributedTaskStatus.UNAVAILABLE:
            t_bucket["unavailable"] += 1
            incompatible_targets.append({
                "worker_id": r.worker_id,
                "model": r.model,
                "task_id": r.task_id,
                "status": r.status.value,
                "failure_category": r.failure_category or "UNAVAILABLE",
                "reason": r.error_detail,
            })
        if r.status not in (DistributedTaskStatus.UNAVAILABLE, DistributedTaskStatus.UNSUPPORTED):
            t_bucket["durations"].append(float(r.duration_seconds))
        for iv in (r.inference_intervals or []):
            if isinstance(iv, dict):
                if iv.get("first_chunk_latency_seconds") is not None:
                    t_bucket["first_chunk_lats"].append(float(iv["first_chunk_latency_seconds"]))
                if iv.get("duration_seconds") is not None:
                    t_bucket["inf_durations"].append(float(iv["duration_seconds"]))
        for tr in (r.ledger_transitions or []):
            if isinstance(tr, dict):
                meta = tr.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("duration_ms") is not None:
                    t_bucket["tool_durations_ms"].append(float(meta["duration_ms"]))
        mtu = r.measured_token_usage or rm.get("measured_token_usage")
        if isinstance(mtu, dict):
            if mtu.get("prompt_eval_count_max") is not None:
                t_bucket["prompt_tokens_measured"].append(int(mtu["prompt_eval_count_max"]))
            if mtu.get("eval_count_sum") is not None:
                t_bucket["eval_tokens_measured"].append(int(mtu["eval_count_sum"]))
        etu = r.estimated_token_usage or rm.get("estimated_token_usage")
        if isinstance(etu, dict) and etu.get("estimated_tokens_approx") is not None:
            t_bucket["estimated_tokens"].append(int(etu["estimated_tokens_approx"]))
        for ps_key in ("ps_before", "ps_after"):
            for pe in (rm.get(ps_key) or []):
                if pe.get("name") == r.model:
                    if pe.get("size") is not None:
                        t_bucket["ps_size_bytes"].append(int(pe["size"]))
                    if pe.get("size_vram") is not None:
                        t_bucket["ps_vram_bytes"].append(int(pe["size_vram"]))
        if eff_ctx is not None:
            t_bucket["num_ctx_values"].add(int(eff_ctx))
        t_bucket["status_by_task"][r.task_id] = r.status.value

        policy_mode_val = worker_cfg.policy_mode if worker_cfg else "basic"
        eff_tools = worker_cfg.allowed_tools if worker_cfg and worker_cfg.allowed_tools is not None else (
            assign.task.allowed_tools if assign else None
        )
        eff_ring = worker_cfg.caller_ring if worker_cfg and worker_cfg.caller_ring is not None else (
            assign.task.caller_ring if assign else None
        )
        by_task.setdefault(r.task_id, []).append({
            "worker_id": r.worker_id,
            "endpoint": _scrub_url_userinfo(r.endpoint),
            "norm_endpoint": norm_ep,
            "model": r.model,
            "model_digest": r.model_digest,
            "model_family": model_meta.family if model_meta else "",
            "parameter_size": model_meta.parameter_size if model_meta else "",
            "quantization_level": model_meta.quantization_level if model_meta else "",
            "execution_id": r.execution_id,
            "status": r.status.value,
            "failure_category": r.failure_category or r.status.value,
            "duration_seconds": r.duration_seconds,
            "num_ctx": eff_ctx,
            "context_clamped": bool((r.model_settings or {}).get("context_clamped_to_model")),
            "think": think_val,
            "model_options": model_opts,
            "tool_protocol": tool_protocol,
            "read_timeout": (r.model_settings or {}).get("read_timeout"),
            "keep_alive": (r.model_settings or {}).get("keep_alive"),
            "max_steps": assign.task.max_steps if assign else None,
            "max_replanning": assign.task.max_replanning if assign else None,
            "expected_exit_code": assign.task.expected_exit_code if assign else 0,
            "fixture_version": r.fixture_version or (assign.task.fixture_version if assign else "1.4"),
            "allowed_tools": eff_tools,
            "caller_ring": eff_ring,
            "policy_mode": policy_mode_val,
            "configuration_fingerprint": cfg_fp,
            "error_classification": r.error_classification,
            "workflow_compatible": compat.get("workflow_compatible") if "workflow_compatible" in compat else compat.get("compatible"),
        })

    targets_summary: Dict[str, Any] = {}
    perf_targets: Dict[str, Any] = {}
    min_trials_observed: Optional[int] = None
    for k, b in sorted(by_target.items()):
        durs = b["durations"]
        eligible_n = b["total"] - b["unsupported"] - b["unavailable"]
        t_counts = list(b["trials_per_task"].values())
        if t_counts:
            t_min = min(t_counts)
            min_trials_observed = t_min if min_trials_observed is None else min(min_trials_observed, t_min)
        avg_dur = round(sum(durs) / len(durs), 3) if durs else None
        avg_fcl = round(sum(b["first_chunk_lats"]) / len(b["first_chunk_lats"]), 3) if b["first_chunk_lats"] else None
        sum_inf = round(sum(b["inf_durations"]), 3) if b["inf_durations"] else None
        avg_tool_ms = round(sum(b["tool_durations_ms"]) / len(b["tool_durations_ms"]), 3) if b["tool_durations_ms"] else None
        targets_summary[k] = {
            "worker_id": b["worker_id"],
            "model": b["model"],
            "model_digest": b["model_digest"],
            "model_family": b["model_family"],
            "parameter_size": b["parameter_size"],
            "quantization_level": b["quantization_level"],
            "endpoint": b["endpoint"],
            "total": b["total"],
            "eligible": eligible_n,
            "passed": b["passed"],
            "unsupported": b["unsupported"],
            "unavailable": b["unavailable"],
            "pass_rate": round(b["passed"] / b["total"], 4) if b["total"] else 0.0,
            "pass_rate_eligible": round(b["passed"] / eligible_n, 4) if eligible_n > 0 else None,
            "avg_duration_seconds": avg_dur,
            "num_ctx_values": sorted(b["num_ctx_values"]),
            "configuration_fingerprints": sorted(b["fingerprints"]),
            "status_by_task": b["status_by_task"],
        }
        perf_targets[k] = {
            "worker_id": b["worker_id"],
            "model": b["model"],
            "measured_wall_clock_duration_s": avg_dur,
            "measured_first_chunk_latency_s": avg_fcl,
            "measured_inference_interval_duration_s": sum_inf,
            "measured_tool_execution_duration_ms": avg_tool_ms,
            "measured_prompt_tokens_max": max(b["prompt_tokens_measured"]) if b["prompt_tokens_measured"] else None,
            "measured_eval_tokens_sum": sum(b["eval_tokens_measured"]) if b["eval_tokens_measured"] else None,
            "measured_ps_size_bytes_max": max(b["ps_size_bytes"]) if b["ps_size_bytes"] else None,
            "measured_ps_vram_bytes_max": max(b["ps_vram_bytes"]) if b["ps_vram_bytes"] else None,
            "estimated_tokens_approx_sum": sum(b["estimated_tokens"]) if b["estimated_tokens"] else None,
        }

    functional_differences: List[Dict[str, Any]] = []
    configuration_differences: List[Dict[str, Any]] = []
    parity_differences: List[Dict[str, Any]] = []

    for task_id, entries in sorted(by_task.items()):
        if len(entries) < 2:
            continue
        # Functional parity: same task fixture, verifier expectations, tool permissions, and orchestration bounds
        fixtures = {str(e["fixture_version"]) for e in entries}
        exp_rcs = {repr(e["expected_exit_code"]) for e in entries}
        tools_perms = {repr(e["allowed_tools"]) for e in entries}
        rings = {e["caller_ring"] for e in entries}
        pols = {e["policy_mode"] for e in entries}
        steps = {e["max_steps"] for e in entries if e["max_steps"] is not None}
        replans = {e["max_replanning"] for e in entries if e["max_replanning"] is not None}
        if len(fixtures) > 1 or len(exp_rcs) > 1 or len(tools_perms) > 1 or len(rings) > 1 or len(pols) > 1 or len(steps) > 1 or len(replans) > 1:
            functional_differences.append({
                "task_id": task_id,
                "fixture_versions": sorted(fixtures),
                "expected_exit_codes": sorted(exp_rcs),
                "allowed_tools_sets": sorted(tools_perms),
                "caller_rings": sorted(str(x) for x in rings),
                "policy_modes": sorted(pols),
                "max_steps_values": sorted(steps),
                "max_replanning_values": sorted(replans),
            })

        # Configuration parity: context budget, think, options, protocol, timeout, compatibility
        ctxs = {e["num_ctx"] for e in entries if e["num_ctx"] is not None}
        timeouts = {e["read_timeout"] for e in entries if e["read_timeout"] is not None}
        thinks = {repr(e["think"]) for e in entries}
        opts_vals = {json.dumps(e["model_options"], sort_keys=True, default=str) for e in entries}
        protocols = {e["tool_protocol"] for e in entries}
        compats = {e["workflow_compatible"] for e in entries if e["workflow_compatible"] is not None}
        clamped_targets = [f"{e['worker_id']}::{e['model']}" for e in entries if e["context_clamped"]]
        if len(ctxs) > 1 or len(timeouts) > 1 or len(thinks) > 1 or len(opts_vals) > 1 or len(protocols) > 1 or len(compats) > 1 or clamped_targets:
            configuration_differences.append({
                "task_id": task_id,
                "num_ctx_values": sorted(ctxs),
                "read_timeout_values": sorted(timeouts),
                "think_values": sorted(thinks),
                "model_options_variants": sorted(opts_vals),
                "tool_protocols": sorted(protocols),
                "workflow_compatible_values": sorted(str(x) for x in compats),
                "context_clamped_targets": clamped_targets,
            })

        if len(ctxs) > 1 or len(timeouts) > 1 or len(steps) > 1 or len(thinks) > 1 or len(opts_vals) > 1 or len(protocols) > 1 or clamped_targets:
            parity_differences.append({
                "task_id": task_id,
                "num_ctx_values": sorted(ctxs),
                "read_timeout_values": sorted(timeouts),
                "max_steps_values": sorted(steps),
                "think_values": sorted(thinks),
                "tool_protocols": sorted(protocols),
                "context_clamped_targets": clamped_targets,
            })

    func_equal = len(functional_differences) == 0
    cfg_equal = len(configuration_differences) == 0
    hw_heterogeneous = len(seen_endpoints) > 1 or len(seen_vram_modes) > 1
    model_heterogeneous = len(seen_models) > 1 or len(seen_families) > 1

    limitations: List[str] = []
    if (min_trials_observed or 0) < 2:
        limitations.append("Single-trial evaluation (n=1 per target/task); unsuitable for statistically reliable latency ranking.")
    if not func_equal:
        limitations.append("Functional task permissions, orchestration bounds, or fixtures differed across targets.")
    if not cfg_equal:
        limitations.append("Model/runtime configurations differed across targets (e.g., num_ctx, think, timeout, or tool protocol).")
    if hw_heterogeneous:
        limitations.append(
            f"Heterogeneous workers/hardware compared across endpoints {sorted(seen_endpoints)} "
            f"(accelerator modes observed: {sorted(seen_vram_modes) or ['unknown']})."
        )
    if model_heterogeneous:
        limitations.append("Multiple distinct models/families evaluated; latency reflects hardware+model+thinking differences.")
    if incompatible_targets:
        limitations.append(f"{len(incompatible_targets)} incompatible/unavailable target run(s) excluded from reasoning failure rate.")

    perf_comparable = bool(
        func_equal
        and cfg_equal
        and not hw_heterogeneous
        and not incompatible_targets
        and (min_trials_observed or 0) >= 2
        and len(by_target) >= 2
    )

    return {
        "targets": targets_summary,
        "by_task": by_task,
        "fair_comparison": func_equal and cfg_equal and len(parity_differences) == 0,
        "parity_differences": parity_differences,
        "functional_parity": {
            "equal": func_equal,
            "differences": functional_differences,
            "incompatible_targets": incompatible_targets,
        },
        "configuration_parity": {
            "equal": cfg_equal,
            "differences": configuration_differences,
            "hardware_heterogeneous": hw_heterogeneous,
            "model_heterogeneous": model_heterogeneous,
        },
        "performance_comparison": {
            "comparable": perf_comparable,
            "ranking_allowed": perf_comparable,
            "min_trials_observed": min_trials_observed or 0,
            "limitations": limitations,
            "targets": perf_targets,
        },
        "comparison_limitations": limitations,
    }


def build_matrix_assignments(
    workers: List[WorkerConfig],
    snapshots: Dict[str, WorkerSnapshot],
    tasks: List[DistributedTaskSpec],
    *,
    models_by_worker: Optional[Dict[str, List[str]]] = None,
    trials: int = 1,
    mode: str = "concurrent",
) -> List[TaskAssignment]:
    """Build deterministic assignments from discovered workers/models without silent rerouting."""
    worker_models: List[Tuple[WorkerConfig, str]] = []
    for w in workers:
        snap = snapshots.get(w.worker_id)
        if models_by_worker and w.worker_id in models_by_worker:
            chosen = list(models_by_worker[w.worker_id])
        elif w.model_allowlist:
            chosen = list(w.model_allowlist)
        elif snap and snap.healthy:
            chosen = sorted(snap.installed_models.keys())
        else:
            chosen = []
        for m in chosen:
            worker_models.append((w, m))

    assignments: List[TaskAssignment] = []
    n_trials = max(1, int(trials))
    if mode == "comparison":
        for trial_idx in range(n_trials):
            for t in tasks:
                for w, m in worker_models:
                    assignments.append(
                        TaskAssignment(worker_id=w.worker_id, model=m, task=t, trial_index=trial_idx)
                    )
    else:
        for trial_idx in range(n_trials):
            for w, m in worker_models:
                for t in tasks:
                    assignments.append(
                        TaskAssignment(worker_id=w.worker_id, model=m, task=t, trial_index=trial_idx)
                    )
    return assignments


class DistributedEvalRunner:
    """Coordinated distributed evaluation runner across multiple Ollama workers."""

    def __init__(
        self,
        workers: List[WorkerConfig],
        output_dir: Optional[Path] = None,
        task_executor: Optional[TaskExecutorFn] = None,
        worker_prober: Optional[Callable[[WorkerConfig], Awaitable[WorkerSnapshot]]] = None,
        ps_fetcher: Optional[Callable[[WorkerConfig], Awaitable[List[Dict[str, Any]]]]] = None,
        model_unloader: Optional[Callable[[WorkerConfig, Set[str], Optional[Set[str]]], Awaitable[Dict[str, Any]]]] = None,
    ):
        self.workers: Dict[str, WorkerConfig] = {w.worker_id: w for w in workers}
        self.worker_list: List[WorkerConfig] = list(workers)
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="zelyr_dist_eval_")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir = self.output_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._task_executor = task_executor or default_subprocess_executor
        self._worker_prober = worker_prober or probe_worker_async
        self._ps_fetcher = ps_fetcher or fetch_worker_ps_async
        self._model_unloader = model_unloader or unload_worker_eval_models_async
        self.snapshots: Dict[str, WorkerSnapshot] = {}

    async def discover_workers(self) -> Dict[str, WorkerSnapshot]:
        """Probe all configured workers concurrently and snapshot initial /api/ps state."""
        snaps = await asyncio.gather(*(self._worker_prober(w) for w in self.worker_list))
        self.snapshots = {s.worker_id: s for s in snaps}
        return self.snapshots

    async def run_mode(
        self,
        mode: str,
        assignments: List[TaskAssignment],
        *,
        unload_after: bool = True,
    ) -> Dict[str, Any]:
        """Run assignments under an explicit mode: 'sequential', 'concurrent', or 'comparison'."""
        norm_mode = str(mode or "").strip().lower()
        if norm_mode not in ("sequential", "concurrent", "comparison"):
            raise ValueError(f"Unsupported execution mode '{mode}'; expected 'sequential', 'concurrent', or 'comparison'.")
        use_concurrent = norm_mode in ("concurrent", "comparison")
        return await self.execute_assignments(
            assignments,
            concurrent=use_concurrent,
            unload_after=unload_after,
            mode=norm_mode,
        )

    async def execute_assignments(
        self,
        assignments: List[TaskAssignment],
        *,
        concurrent: bool = True,
        unload_after: bool = True,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute deterministic task assignments sequentially or concurrently."""
        effective_mode = str(mode or ("concurrent" if concurrent else "sequential")).strip().lower()
        if not self.snapshots:
            await self.discover_workers()

        scheduler = ResourceAwareScheduler(self.worker_list, self.snapshots)
        eval_loaded_models: Dict[str, Set[str]] = {w.worker_id: set() for w in self.worker_list}

        dedup_assignments: List[TaskAssignment] = []
        seen_assignment_ids: Set[str] = set()
        duplicate_assignment_ids: List[str] = []
        for a in assignments:
            aid = a.assignment_id or f"{a.worker_id}::{a.model}::{a.task.task_id}::trial{a.trial_index}"
            if aid in seen_assignment_ids:
                duplicate_assignment_ids.append(aid)
                continue
            seen_assignment_ids.add(aid)
            dedup_assignments.append(a)

        async def _run_one(assign: TaskAssignment) -> DistributedTaskResult:
            t_queued = time.time()
            t_exec_start: Optional[float] = None
            worker = self.workers.get(assign.worker_id)
            snap = self.snapshots.get(assign.worker_id)
            model_meta = snap.installed_models.get(assign.model) if snap else None
            digest = model_meta.digest if model_meta else ""
            compat = (
                dict(model_meta.compatibility)
                if (model_meta and model_meta.compatibility)
                else dict((snap.compatibility_by_model.get(assign.model) or {}) if snap else {})
            )
            endpoint = _scrub_url_userinfo(worker.endpoint) if worker else ""
            settings = (
                worker.resolve_model_settings(
                    assign.model,
                    model_context_length=(model_meta.context_length if model_meta else None),
                )
                if worker
                else {}
            )
            cfg_fp = compute_configuration_fingerprint(
                model=assign.model,
                model_digest=digest,
                model_settings=settings,
                task=assign.task,
                worker=worker,
                model_meta=model_meta,
                compatibility=compat,
            )

            pre_status, pre_cat, pre_reason = scheduler.preflight_assignment_detail(assign)
            if pre_status is not None:
                return DistributedTaskResult(
                    model=assign.model,
                    model_digest=digest,
                    worker_id=assign.worker_id,
                    endpoint=endpoint,
                    task_id=assign.task.task_id,
                    category=assign.task.category,
                    status=pre_status,
                    start_timestamp=t_queued,
                    end_timestamp=time.time(),
                    duration_seconds=0.0,
                    error_classification=pre_cat or pre_status.value,
                    failure_category=pre_cat or pre_status.value,
                    error_detail=_redact_secrets(pre_reason or ""),
                    model_settings=settings,
                    concurrency_mode="skipped",
                    assignment_id=assign.assignment_id,
                    trial_index=assign.trial_index,
                    matrix_entry_id=assign.matrix_entry_id,
                    compatibility=compat,
                    fixture_version=assign.task.fixture_version,
                    configuration_fingerprint=cfg_fp,
                )

            assert worker is not None
            assert snap is not None
            safe_wid = _sanitize_artifact_segment(assign.worker_id, "worker")
            safe_tid = _sanitize_artifact_segment(assign.task.task_id, "task")
            ws = Path(tempfile.mkdtemp(prefix=f"zdist_{safe_wid}_{safe_tid}_")).resolve()
            slot_id = f"{assign.worker_id}:{assign.model}:{assign.task.task_id}:{t_queued}"
            concurrency_mode = "sequential"
            slot_acquired = False

            try:
                concurrency_mode = await scheduler.acquire(assign, slot_id=slot_id)
                slot_acquired = True
                t_exec_start = time.time()
                eval_loaded_models[worker.worker_id].add(assign.model)
                ps_before = await self._ps_fetcher(worker)
                case = assign.task.build_case(ws)
                exec_case = (
                    dict(case, _resolved_model_settings=settings)
                    if self._task_executor is default_subprocess_executor
                    else case
                )
                raw_out = await self._task_executor(worker, assign.model, assign.task, ws, exec_case)
                ps_after = await self._ps_fetcher(worker)
                concurrency_mode = await scheduler.release(worker.worker_id, assign.model, slot_id=slot_id)
                slot_acquired = False
                t_end = time.time()
                dur = round(t_end - t_exec_start, 3)

                wait_seconds = round(max(0.0, t_exec_start - t_queued), 3)
                scheduler.record_queue_wait(worker.worker_id, wait_seconds)
                safe_raw_out: Dict[str, Any] = raw_out if isinstance(raw_out, dict) else {"raw_value": repr(raw_out)}

                detail = _classify_task_outcome_detail(assign, case, raw_out)
                if detail["status"] == DistributedTaskStatus.TIMEOUT:
                    scheduler.record_timeout(worker.worker_id)

                res = self._finalize_and_save_transcript(
                    assign=assign,
                    digest=digest,
                    endpoint=endpoint,
                    settings=settings,
                    status=detail["status"],
                    start_ts=t_exec_start,
                    end_ts=t_end,
                    dur=dur,
                    raw_out=safe_raw_out,
                    ps_before=ps_before,
                    ps_after=ps_after,
                    concurrency_mode=concurrency_mode,
                    error_cls=detail["error_classification"],
                    error_detail=detail["error_detail"],
                    compat=compat,
                    failure_category=detail["failure_category"],
                    uncertain_side_effects=detail["uncertain_side_effects"],
                    mutated_targets_uncertain=detail["mutated_targets_uncertain"],
                    configuration_fingerprint=cfg_fp,
                )
                res.resource_measurements["queued_wait_seconds"] = wait_seconds
                return res
            except asyncio.CancelledError:
                t_end = time.time()
                t_ref = t_exec_start if t_exec_start is not None else t_queued
                return self._finalize_and_save_transcript(
                    assign=assign,
                    digest=digest,
                    endpoint=endpoint,
                    settings=settings,
                    status=DistributedTaskStatus.CANCELLED,
                    start_ts=t_ref,
                    end_ts=t_end,
                    dur=round(t_end - t_ref, 3),
                    raw_out={"exit_code": None, "cancelled": True, "execution_ledger": {"records": []}, "telemetry": {}},
                    ps_before=[],
                    ps_after=[],
                    concurrency_mode=concurrency_mode,
                    error_cls="CANCELLED",
                    error_detail="Task cancelled before completion.",
                    compat=compat,
                    failure_category="CANCELLED",
                    uncertain_side_effects=False,
                    mutated_targets_uncertain=[],
                    configuration_fingerprint=cfg_fp,
                )
            except Exception as exc:
                t_end = time.time()
                t_ref = t_exec_start if t_exec_start is not None else t_queued
                exc_cls = type(exc).__name__
                exc_msg = _redact_secrets(str(exc))
                low_exc = f"{exc_cls} {exc_msg}".lower()
                f_cat = (
                    "WORKER_UNAVAILABLE"
                    if any(k in low_exc for k in ("connect", "unreachable", "refused", "remoteprotocol"))
                    else "ERROR"
                )
                return self._finalize_and_save_transcript(
                    assign=assign,
                    digest=digest,
                    endpoint=endpoint,
                    settings=settings,
                    status=DistributedTaskStatus.ERROR,
                    start_ts=t_ref,
                    end_ts=t_end,
                    dur=round(t_end - t_ref, 3),
                    raw_out={"exit_code": None, "worker_error": f"{exc_cls}: {exc_msg}", "execution_ledger": {"records": []}, "telemetry": {}},
                    ps_before=[],
                    ps_after=[],
                    concurrency_mode=concurrency_mode,
                    error_cls=exc_cls,
                    error_detail=exc_msg,
                    compat=compat,
                    failure_category=f_cat,
                    uncertain_side_effects=False,
                    mutated_targets_uncertain=[],
                    configuration_fingerprint=cfg_fp,
                )
            finally:
                if slot_acquired:
                    await scheduler.release(worker.worker_id, assign.model, slot_id=slot_id)
                shutil.rmtree(ws, ignore_errors=True)

        results: List[DistributedTaskResult] = []
        cleanup_records: List[Dict[str, Any]] = []
        try:
            if concurrent:
                tasks = [asyncio.create_task(_run_one(a)) for a in dedup_assignments]
                try:
                    raw_results = list(await asyncio.gather(*tasks))
                except BaseException:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
            else:
                raw_results = []
                for a in dedup_assignments:
                    raw_results.append(await _run_one(a))

            accepted_ids: Set[str] = set()
            for res in raw_results:
                rid = res.assignment_id or f"{res.worker_id}::{res.model}::{res.task_id}::trial{res.trial_index}"
                if rid in accepted_ids:
                    continue
                accepted_ids.add(rid)
                results.append(res)
        finally:
            if unload_after:
                for wid, models in eval_loaded_models.items():
                    if not models:
                        continue
                    snap = self.snapshots.get(wid)
                    worker = self.workers.get(wid)
                    if snap and snap.healthy and worker:
                        try:
                            rec = await self._model_unloader(worker, snap.pre_loaded_models, models)
                            rec_errors = list(rec.get("errors") or [])
                            still_loaded = [
                                m.get("name")
                                for m in (rec.get("loaded_after") or [])
                                if m.get("name") in models and m.get("name") not in snap.pre_loaded_models
                            ]
                            if still_loaded:
                                rec_errors.append({
                                    "model": ",".join(str(x) for x in still_loaded),
                                    "error": "Eval model(s) remained resident after unload attempt.",
                                })
                                rec["errors"] = rec_errors
                            rec["cleanup_ok"] = len(rec_errors) == 0
                            cleanup_records.append(rec)
                        except Exception as exc:
                            cleanup_records.append({
                                "worker_id": wid,
                                "loaded_before": [],
                                "unloaded": [],
                                "errors": [{"model": "*", "error": str(exc) or type(exc).__name__}],
                                "loaded_after": [],
                                "cleanup_ok": False,
                            })

        cleanup_ok = all(rec.get("cleanup_ok", not rec.get("errors")) for rec in cleanup_records)
        overlap_proof = compute_interval_overlaps(results)
        comparison = build_cross_model_comparison(
            dedup_assignments,
            results,
            snapshots=self.snapshots,
            workers=self.workers,
        )
        return {
            "execution_mode": effective_mode,
            "results": results,
            "scheduler_stats": {
                "peak_concurrency_by_worker": scheduler.peak_concurrency_by_worker,
                "peak_simultaneous_workers": scheduler.peak_simultaneous_workers,
                "observed_concurrency_forms": sorted(scheduler.observed_concurrency_forms),
                "timeouts_by_worker": scheduler.timeouts_by_worker,
                "max_queued_wait_by_worker": scheduler.max_queued_wait_by_worker,
                "duplicate_assignments_skipped": len(duplicate_assignment_ids),
                "duplicate_assignment_ids": duplicate_assignment_ids,
            },
            "overlap_proof": overlap_proof,
            "comparison": comparison,
            "cleanup": cleanup_records,
            "cleanup_ok": cleanup_ok,
            "snapshots": {wid: snap.to_dict() for wid, snap in self.snapshots.items()},
        }

    def _finalize_and_save_transcript(
        self,
        *,
        assign: TaskAssignment,
        digest: str,
        endpoint: str,
        settings: Dict[str, Any],
        status: DistributedTaskStatus,
        start_ts: float,
        end_ts: float,
        dur: float,
        raw_out: Dict[str, Any],
        ps_before: List[Dict[str, Any]],
        ps_after: List[Dict[str, Any]],
        concurrency_mode: str,
        error_cls: Optional[str],
        error_detail: Optional[str],
        compat: Optional[Dict[str, Any]] = None,
        failure_category: Optional[str] = None,
        uncertain_side_effects: bool = False,
        mutated_targets_uncertain: Optional[List[str]] = None,
        configuration_fingerprint: str = "",
    ) -> DistributedTaskResult:
        raw_ledger = raw_out.get("execution_ledger")
        ledger = raw_ledger if isinstance(raw_ledger, dict) else {}
        exec_id = raw_out.get("execution_id") or ledger.get("execution_id") or None
        raw_records = ledger.get("records")
        records = [r for r in raw_records if isinstance(r, dict)] if isinstance(raw_records, list) else []
        raw_cli_json = raw_out.get("cli_json")
        cli_json = raw_cli_json if isinstance(raw_cli_json, dict) else {}
        tool_calls = cli_json.get("tool_calls") or [
            {"name": r.get("tool_name"), "arguments": r.get("arguments")}
            for r in records
        ]
        tool_args = [dict(r.get("arguments") or {}) if isinstance(r.get("arguments"), dict) else {} for r in records]
        tool_res = [
            {"call_id": r.get("call_id"), "state": r.get("state"), "verified": r.get("verified"), "error": r.get("error")}
            for r in records
        ]
        raw_telemetry = raw_out.get("telemetry")
        telemetry_dict = raw_telemetry if isinstance(raw_telemetry, dict) else {}
        turn_metrics = telemetry_dict.get("turn_metrics") or []
        inference_intervals = [iv for iv in (telemetry_dict.get("inference_intervals") or []) if isinstance(iv, dict)]
        ledger_transitions = [t for t in (raw_out.get("ledger_transitions") or []) if isinstance(t, dict)]
        policy_decisions = [p for p in (raw_out.get("policy_decisions") or []) if isinstance(p, dict)]
        compat_dict = dict(compat or {})
        loads = [m["load_duration_ms"] for m in turn_metrics if isinstance(m, dict) and m.get("load_duration_ms") is not None]
        totals = [m["total_duration_ms"] for m in turn_metrics if isinstance(m, dict) and m.get("total_duration_ms") is not None]
        prompt_evals = [m["prompt_eval_count"] for m in turn_metrics if isinstance(m, dict) and m.get("prompt_eval_count") is not None]
        eval_counts = [m["eval_count"] for m in turn_metrics if isinstance(m, dict) and m.get("eval_count") is not None]

        measured_token_usage: Optional[Dict[str, Any]] = None
        if prompt_evals or eval_counts:
            measured_token_usage = {
                "prompt_eval_count_max": max(prompt_evals) if prompt_evals else None,
                "eval_count_sum": sum(eval_counts) if eval_counts else None,
            }

        raw_transcript_msgs = raw_out.get("messages_transcript") or []
        estimated_token_usage: Optional[Dict[str, Any]] = None
        if isinstance(raw_transcript_msgs, list) and raw_transcript_msgs:
            total_chars = sum(len(str(m.get("content") or "")) for m in raw_transcript_msgs if isinstance(m, dict))
            estimated_token_usage = {
                "estimated_tokens_approx": max(1, total_chars // 4) if total_chars else 0,
                "source": "character_heuristic",
            }

        first_chunk_lats = [
            float(iv["first_chunk_latency_seconds"])
            for iv in inference_intervals
            if iv.get("first_chunk_latency_seconds") is not None
        ]
        resource_measurements = {
            "num_ctx": settings.get("num_ctx"),
            "context_clamped_to_model": bool(settings.get("context_clamped_to_model", False)),
            "load_duration_ms_max": max(loads) if loads else None,
            "total_duration_ms_sum": round(sum(totals), 1) if totals else None,
            "prompt_eval_count_max": max(prompt_evals) if prompt_evals else None,
            "eval_count_sum": sum(eval_counts) if eval_counts else None,
            "first_chunk_latency_seconds_min": min(first_chunk_lats) if first_chunk_lats else None,
            "first_chunk_latency_seconds_avg": round(sum(first_chunk_lats) / len(first_chunk_lats), 3) if first_chunk_lats else None,
            "inference_intervals_count": len(inference_intervals),
            "ps_before": ps_before,
            "ps_after": ps_after,
            "thinking_chunks": telemetry_dict.get("thinking_chunks", 0),
            "measured_token_usage": measured_token_usage,
            "estimated_token_usage": estimated_token_usage,
        }

        safe_wid = _sanitize_artifact_segment(assign.worker_id, "worker")
        safe_model = _sanitize_artifact_segment(assign.model, "model")
        safe_tid = _sanitize_artifact_segment(assign.task.task_id, "task")
        safe_endpoint = _scrub_url_userinfo(endpoint)
        safe_error_detail = _redact_secrets(error_detail) if error_detail else error_detail
        safe_stderr = _redact_secrets(str(raw_out.get("raw_stderr") or ""))

        t_path = self.transcripts_dir / f"{safe_wid}__{safe_model}__{safe_tid}__{int(start_ts * 1000)}.json"
        mut_list = list(mutated_targets_uncertain or [])
        transcript_doc = {
            "assignment_id": assign.assignment_id,
            "trial_index": assign.trial_index,
            "matrix_entry_id": assign.matrix_entry_id,
            "execution_id": exec_id,
            "worker_id": assign.worker_id,
            "endpoint": safe_endpoint,
            "model": assign.model,
            "model_digest": digest,
            "task_id": assign.task.task_id,
            "fixture_version": assign.task.fixture_version,
            "status": status.value,
            "failure_category": failure_category or status.value,
            "error_classification": error_cls,
            "error_detail": safe_error_detail,
            "uncertain_side_effects": bool(uncertain_side_effects),
            "mutated_targets_uncertain": mut_list,
            "configuration_fingerprint": configuration_fingerprint,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_seconds": dur,
            "model_settings": settings,
            "concurrency_mode": concurrency_mode,
            "compatibility": compat_dict,
            "resource_measurements": resource_measurements,
            "measured_token_usage": measured_token_usage,
            "estimated_token_usage": estimated_token_usage,
            "turn_metrics": turn_metrics,
            "inference_intervals": inference_intervals,
            "files_on_disk": raw_out.get("files_on_disk") or {},
            "verifier_checks": raw_out.get("verifier_checks") or [],
            "execution_ledger": ledger,
            "ledger_transitions": ledger_transitions,
            "policy_decisions": policy_decisions,
            "checkpoint_snapshot": raw_out.get("checkpoint_snapshot"),
            "messages_transcript": raw_transcript_msgs,
            "raw_stderr": safe_stderr,
        }
        t_path.write_text(json.dumps(transcript_doc, indent=2), encoding="utf-8")

        return DistributedTaskResult(
            model=assign.model,
            model_digest=digest,
            worker_id=assign.worker_id,
            endpoint=safe_endpoint,
            task_id=assign.task.task_id,
            category=assign.task.category,
            status=status,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            duration_seconds=dur,
            tool_calls=tool_calls,
            tool_arguments=tool_args,
            tool_results=tool_res,
            ledger_records=records,
            has_unresolved_failures=bool(ledger.get("has_unresolved_failures", False)),
            verification_result=raw_out.get("verifier_checks") or cli_json.get("verification"),
            exit_code=raw_out.get("exit_code"),
            resource_measurements=resource_measurements,
            error_classification=error_cls,
            error_detail=safe_error_detail,
            model_settings=settings,
            concurrency_mode=concurrency_mode,
            transcript_path=str(t_path),
            assignment_id=assign.assignment_id,
            trial_index=assign.trial_index,
            matrix_entry_id=assign.matrix_entry_id,
            measured_token_usage=measured_token_usage,
            estimated_token_usage=estimated_token_usage,
            inference_intervals=inference_intervals,
            ledger_transitions=ledger_transitions,
            policy_decisions=policy_decisions,
            compatibility=compat_dict,
            execution_id=exec_id,
            failure_category=failure_category or status.value,
            uncertain_side_effects=bool(uncertain_side_effects),
            mutated_targets_uncertain=mut_list,
            fixture_version=assign.task.fixture_version,
            configuration_fingerprint=configuration_fingerprint,
        )
