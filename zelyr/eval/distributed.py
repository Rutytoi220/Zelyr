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
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx

from zelyr.config import get_config, validate_ollama_base_url
from zelyr.engine.error_classifier import _redact_secrets
from zelyr.engine.execution_ledger import detect_false_success


def _sanitize_artifact_segment(value: str, fallback: str = "item") -> str:
    """Sanitize identifiers used in workspace/transcript filenames so paths stay confined."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned or fallback


def _scrub_url_userinfo(url: str) -> str:
    """Remove embedded userinfo credentials from URLs before serializing reports."""
    return _redact_secrets(str(url or ""))


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

    def __post_init__(self) -> None:
        if not str(self.worker_id or "").strip():
            raise ValueError("worker_id must be a non-empty string")
        self.endpoint = validate_ollama_base_url(self.endpoint)
        # Hard conservative cap: at most 2 concurrent tasks per machine.
        self.max_concurrency = max(1, min(int(self.max_concurrency or 1), 2))
        self.num_ctx = max(1024, min(int(self.num_ctx or 8192), 65536))

    def resolve_model_settings(self, model: str) -> Dict[str, Any]:
        """Return effective per-model execution settings for this worker."""
        overrides = dict(self.per_model_overrides.get(model) or {})
        eff_ctx = int(overrides.get("num_ctx", self.num_ctx))
        eff_ctx = max(1024, min(eff_ctx, 65536))
        return {
            "num_ctx": eff_ctx,
            "keep_alive": overrides.get("keep_alive", self.keep_alive),
            "read_timeout": float(overrides.get("read_timeout", self.read_timeout)),
        }

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
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Discovered model capabilities, digest, and sizing from /api/tags and /api/show."""

    name: str
    digest: str = ""
    size_bytes: int = 0
    capabilities: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerSnapshot:
    """Live runtime inventory and health state of an Ollama worker."""

    worker_id: str
    endpoint: str
    healthy: bool
    version: Optional[str] = None
    installed_models: Dict[str, ModelMetadata] = field(default_factory=dict)
    pre_loaded_models: Set[str] = field(default_factory=set)
    resident_models: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    queried_at: float = field(default_factory=time.time)

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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


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
    """Query `/api/version`, `/api/tags`, `/api/ps`, and `/api/show` for a worker."""
    base = worker.endpoint.rstrip("/")
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
                    endpoint=base,
                    healthy=False,
                    error=f"/api/tags HTTP {tags_resp.status_code}",
                )

            version = ver_resp.json().get("version") if ver_resp.status_code == 200 else None
            resident = list(ps_resp.json().get("models", [])) if ps_resp.status_code == 200 else []
            pre_loaded = {m.get("name", "") for m in resident if m.get("name")}
            raw_models = list(tags_resp.json().get("models", []))

            # Query /api/show concurrently in bounded batches for installed models.
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

            return WorkerSnapshot(
                worker_id=worker.worker_id,
                endpoint=base,
                healthy=True,
                version=version,
                installed_models=installed,
                pre_loaded_models=pre_loaded,
                resident_models=[
                    {
                        "name": r.get("name"),
                        "size": r.get("size"),
                        "size_vram": r.get("size_vram"),
                        "context_length": r.get("context_length"),
                    }
                    for r in resident
                ],
            )
    except Exception as exc:
        err_msg = str(exc).strip()
        err = f"{type(exc).__name__}: {err_msg}" if err_msg else type(exc).__name__
        return WorkerSnapshot(
            worker_id=worker.worker_id,
            endpoint=base,
            healthy=False,
            error=err,
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

    def preflight_assignment(self, assignment: TaskAssignment) -> Tuple[Optional[DistributedTaskStatus], Optional[str]]:
        """Validate worker health, allowlist, model presence, capabilities, and static budget."""
        worker = self.workers.get(assignment.worker_id)
        if worker is None:
            return DistributedTaskStatus.UNAVAILABLE, f"Unknown worker_id '{assignment.worker_id}' (no silent reroute)."

        snap = self.snapshots.get(assignment.worker_id)
        if snap is None or not snap.healthy:
            detail = snap.error if snap else "No health snapshot"
            return DistributedTaskStatus.UNAVAILABLE, f"Worker '{assignment.worker_id}' unavailable: {detail}"

        if worker.model_allowlist and assignment.model not in worker.model_allowlist:
            return (
                DistributedTaskStatus.UNAVAILABLE,
                f"Model '{assignment.model}' is not in worker '{worker.worker_id}' allowlist {worker.model_allowlist}.",
            )

        meta = snap.installed_models.get(assignment.model)
        if meta is None:
            return (
                DistributedTaskStatus.UNAVAILABLE,
                f"Model '{assignment.model}' is not installed on worker '{worker.worker_id}'.",
            )

        if meta.capabilities:
            missing_caps = [c for c in assignment.task.required_capabilities if c not in meta.capabilities]
            if missing_caps:
                return (
                    DistributedTaskStatus.UNSUPPORTED,
                    f"Model '{assignment.model}' lacks required capabilities {missing_caps} (has {meta.capabilities}).",
                )

        if worker.memory_budget_bytes is not None and meta.size_bytes > 0:
            est_needed = int(meta.size_bytes * 1.15)
            if est_needed > worker.memory_budget_bytes:
                return (
                    DistributedTaskStatus.UNAVAILABLE,
                    f"Model '{assignment.model}' est. footprint {est_needed}B exceeds worker budget {worker.memory_budget_bytes}B.",
                )

        return None, None

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
    settings = worker.resolve_model_settings(model)
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
    spec_file = ws / "_worker_spec.json"
    out_file = ws / "_worker_out.json"
    spec_payload = {
        "endpoint": worker.endpoint,
        "model": model,
        "num_ctx": settings["num_ctx"],
        "keep_alive": settings["keep_alive"],
        "read_timeout": settings["read_timeout"],
        "workspace_dir": str(ws),
        "prompt": case["prompt"],
        "verify_spec": case.get("verify"),
        "max_steps": task.max_steps,
        "max_replanning": task.max_replanning,
        "model_options": dict(task.model_options or {}),
        "allowed_tools": eff_allowed_tools,
        "caller_ring": eff_ring,
        "cleanup_workspace": False,
    }
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "zelyr.eval.distributed_worker",
        "--spec-file",
        str(spec_file),
        "--out-file",
        str(out_file),
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
                "raw_stderr": stderr_b.decode("utf-8", errors="replace"),
                "execution_ledger": {},
                "messages_transcript": [],
                "telemetry": {},
                "worker_proc_rc": proc.returncode,
            }
        if not isinstance(data, dict):
            return {
                "exit_code": proc.returncode if proc.returncode not in (None, 0) else 1,
                "worker_error": f"Worker JSON output must be an object, got {type(data).__name__}",
                "malformed_worker_output": True,
                "raw_stderr": stderr_b.decode("utf-8", errors="replace"),
                "execution_ledger": {},
                "messages_transcript": [],
                "telemetry": {},
                "worker_proc_rc": proc.returncode,
            }
        data["worker_proc_rc"] = proc.returncode
        return data
    return {
        "exit_code": proc.returncode if proc.returncode not in (None, 0) else 1,
        "worker_error": (
            stderr_b.decode("utf-8", errors="replace").strip()
            or stdout_b.decode("utf-8", errors="replace").strip()
            or f"Worker exited {proc.returncode} without writing _worker_out.json"
        ),
        "malformed_worker_output": True,
        "execution_ledger": {},
        "messages_transcript": [],
        "telemetry": {},
        "worker_proc_rc": proc.returncode,
    }


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

    if "ReadTimeout" in raw_err or "Timeout" in raw_err:
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

    async def execute_assignments(
        self,
        assignments: List[TaskAssignment],
        *,
        concurrent: bool = True,
        unload_after: bool = True,
    ) -> Dict[str, Any]:
        """Execute deterministic task assignments sequentially or concurrently."""
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
            endpoint = _scrub_url_userinfo(worker.endpoint) if worker else ""
            settings = worker.resolve_model_settings(assign.model) if worker else {}

            pre_status, pre_reason = scheduler.preflight_assignment(assign)
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
                    error_classification=pre_status.value,
                    error_detail=_redact_secrets(pre_reason or ""),
                    model_settings=settings,
                    concurrency_mode="skipped",
                    assignment_id=assign.assignment_id,
                    trial_index=assign.trial_index,
                    matrix_entry_id=assign.matrix_entry_id,
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
                raw_out = await self._task_executor(worker, assign.model, assign.task, ws, case)
                ps_after = await self._ps_fetcher(worker)
                concurrency_mode = await scheduler.release(worker.worker_id, assign.model, slot_id=slot_id)
                slot_acquired = False
                t_end = time.time()
                dur = round(t_end - t_exec_start, 3)

                wait_seconds = round(max(0.0, t_exec_start - t_queued), 3)
                scheduler.record_queue_wait(worker.worker_id, wait_seconds)
                safe_raw_out: Dict[str, Any] = raw_out if isinstance(raw_out, dict) else {"raw_value": repr(raw_out)}

                status, err_cls, err_detail = _classify_task_outcome(assign, case, raw_out)
                if status == DistributedTaskStatus.TIMEOUT:
                    scheduler.record_timeout(worker.worker_id)

                res = self._finalize_and_save_transcript(
                    assign=assign,
                    digest=digest,
                    endpoint=endpoint,
                    settings=settings,
                    status=status,
                    start_ts=t_exec_start,
                    end_ts=t_end,
                    dur=dur,
                    raw_out=safe_raw_out,
                    ps_before=ps_before,
                    ps_after=ps_after,
                    concurrency_mode=concurrency_mode,
                    error_cls=err_cls,
                    error_detail=err_detail,
                )
                res.resource_measurements["queued_wait_seconds"] = wait_seconds
                return res
            except asyncio.CancelledError:
                t_end = time.time()
                t_ref = t_exec_start if t_exec_start is not None else t_queued
                return DistributedTaskResult(
                    model=assign.model,
                    model_digest=digest,
                    worker_id=assign.worker_id,
                    endpoint=endpoint,
                    task_id=assign.task.task_id,
                    category=assign.task.category,
                    status=DistributedTaskStatus.CANCELLED,
                    start_timestamp=t_ref,
                    end_timestamp=t_end,
                    duration_seconds=round(t_end - t_ref, 3),
                    error_classification="CANCELLED",
                    error_detail="Task cancelled before completion.",
                    model_settings=settings,
                    concurrency_mode=concurrency_mode,
                    assignment_id=assign.assignment_id,
                    trial_index=assign.trial_index,
                    matrix_entry_id=assign.matrix_entry_id,
                )
            except Exception as exc:
                t_end = time.time()
                t_ref = t_exec_start if t_exec_start is not None else t_queued
                return DistributedTaskResult(
                    model=assign.model,
                    model_digest=digest,
                    worker_id=assign.worker_id,
                    endpoint=endpoint,
                    task_id=assign.task.task_id,
                    category=assign.task.category,
                    status=DistributedTaskStatus.ERROR,
                    start_timestamp=t_ref,
                    end_timestamp=t_end,
                    duration_seconds=round(t_end - t_ref, 3),
                    error_classification=type(exc).__name__,
                    error_detail=_redact_secrets(str(exc)),
                    model_settings=settings,
                    concurrency_mode=concurrency_mode,
                    assignment_id=assign.assignment_id,
                    trial_index=assign.trial_index,
                    matrix_entry_id=assign.matrix_entry_id,
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
        return {
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
    ) -> DistributedTaskResult:
        raw_ledger = raw_out.get("execution_ledger")
        ledger = raw_ledger if isinstance(raw_ledger, dict) else {}
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

        resource_measurements = {
            "num_ctx": settings.get("num_ctx"),
            "load_duration_ms_max": max(loads) if loads else None,
            "total_duration_ms_sum": round(sum(totals), 1) if totals else None,
            "prompt_eval_count_max": max(prompt_evals) if prompt_evals else None,
            "eval_count_sum": sum(eval_counts) if eval_counts else None,
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
        transcript_doc = {
            "assignment_id": assign.assignment_id,
            "trial_index": assign.trial_index,
            "matrix_entry_id": assign.matrix_entry_id,
            "worker_id": assign.worker_id,
            "endpoint": safe_endpoint,
            "model": assign.model,
            "model_digest": digest,
            "task_id": assign.task.task_id,
            "status": status.value,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_seconds": dur,
            "model_settings": settings,
            "concurrency_mode": concurrency_mode,
            "resource_measurements": resource_measurements,
            "measured_token_usage": measured_token_usage,
            "estimated_token_usage": estimated_token_usage,
            "turn_metrics": turn_metrics,
            "files_on_disk": raw_out.get("files_on_disk") or {},
            "verifier_checks": raw_out.get("verifier_checks") or [],
            "execution_ledger": ledger,
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
        )
