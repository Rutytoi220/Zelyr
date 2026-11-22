from contextlib import contextmanager
import gc
import importlib
import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# Lock ordering policy: whenever both locks are required, always acquire
# _engine_lock BEFORE _vla_lock to prevent AB-BA deadlocks while preserving
# strict mutual exclusion between ZelyrGroundingEngine and SpotlessVLA.
_engine_instance = None
_engine_lock = threading.Lock()

_vla_instance = None
_vla_lock = threading.Lock()
_vla_cond = threading.Condition(_vla_lock)
_vla_active_count = 0

_SPOTLESS_CHECKPOINT_DIR = Path.home() / "Projects" / "good-llm" / "checkpoints"
_SPOTLESS_PACKAGE = "spotless_arms"
_SPOTLESS_DISABLED_VALUES = {"0", "false", "no", "off"}
_SPOTLESS_REQUIRED_MODULES = (
    "spotless_arms",
    "spotless_arms.models",
    "spotless_arms.models.latent_core",
    "spotless_arms.models.motor_head",
    "spotless_arms.actuators",
    "spotless_arms.actuators.uinput_arm",
    "spotless_arms.pipeline",
    "spotless_arms.pipeline.pipelined_runner",
)


def _is_spotless_arms_enabled_by_env() -> bool:
    """Return False when ZELYR_USE_SPOTLESS_ARMS is set to 0/false/no/off (case-insensitive)."""
    raw = os.environ.get("ZELYR_USE_SPOTLESS_ARMS", "")
    return raw.strip().lower() not in _SPOTLESS_DISABLED_VALUES


def _is_optional_spotless_module_error(exc: ModuleNotFoundError) -> bool:
    """Return True only when ModuleNotFoundError identifies spotless_arms or its submodules."""
    exc_name = getattr(exc, "name", None)
    if isinstance(exc_name, str) and exc_name:
        return exc_name == _SPOTLESS_PACKAGE or exc_name.startswith(f"{_SPOTLESS_PACKAGE}.")
    return False


def _import_spotless_components() -> Tuple[Any, Any, Any, Any]:
    """Import and return (SpotlessVLA, PipelinedVLARunner, AsyncScreenBuffer, LinuxVirtualArm)."""
    for mod_name in _SPOTLESS_REQUIRED_MODULES:
        if mod_name in sys.modules and sys.modules[mod_name] is None:
            raise ModuleNotFoundError(
                f"import of {mod_name} halted; None in sys.modules",
                name=mod_name,
            )

    motor_mod = importlib.import_module("spotless_arms.models.motor_head")
    runner_mod = importlib.import_module("spotless_arms.pipeline.pipelined_runner")
    arm_mod = importlib.import_module("spotless_arms.actuators.uinput_arm")

    spotless_vla = getattr(motor_mod, "SpotlessVLA", None)
    pipelined_runner = getattr(runner_mod, "PipelinedVLARunner", None)
    screen_buffer = getattr(runner_mod, "AsyncScreenBuffer", None)
    virtual_arm = getattr(arm_mod, "LinuxVirtualArm", None)
    if any(sym is None for sym in (spotless_vla, pipelined_runner, screen_buffer, virtual_arm)):
        raise ImportError("Required Spotless-Arms components are missing or None.")
    return spotless_vla, pipelined_runner, screen_buffer, virtual_arm


def _load_spotless_components_or_evict() -> Tuple[Any, Any, Any, Any]:
    """Validate env opt-out and import Spotless-Arms components, evicting cache if unavailable."""
    if not _is_spotless_arms_enabled_by_env():
        evict_vla_engine()
        raise RuntimeError(
            "Spotless-Arms VLA integration is disabled via ZELYR_USE_SPOTLESS_ARMS."
        )

    try:
        return _import_spotless_components()
    except ModuleNotFoundError as exc:
        if _is_optional_spotless_module_error(exc):
            evict_vla_engine()
        raise


def is_spotless_arms_available() -> bool:
    """Check whether Spotless-Arms is enabled and all required components are importable.

    - Returns False immediately (and evicts any cached VLA engine) if ZELYR_USE_SPOTLESS_ARMS
      is disabled or if spotless_arms is absent / masked to None in sys.modules.
    - Re-raises transitive ModuleNotFoundError (e.g. broken installation missing a sub-dependency)
      and unrelated import errors.
    """
    if not _is_spotless_arms_enabled_by_env():
        evict_vla_engine()
        return False

    for mod_name in _SPOTLESS_REQUIRED_MODULES:
        if mod_name in sys.modules and sys.modules[mod_name] is None:
            evict_vla_engine()
            return False

    if _SPOTLESS_PACKAGE not in sys.modules:
        try:
            spec = importlib.util.find_spec(_SPOTLESS_PACKAGE)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            evict_vla_engine()
            return False

    try:
        _import_spotless_components()
        return True
    except ModuleNotFoundError as exc:
        if _is_optional_spotless_module_error(exc):
            evict_vla_engine()
            return False
        raise


def _evict_ollama_vram() -> None:
    """Inspect Ollama `/api/ps` for resident models without unloading user sessions.

    Ollama's HTTP API (`/api/ps`, `/api/generate`) identifies loaded models globally by
    model name alone and exposes no per-client session or instance ownership metadata.
    Because Zelyr cannot prove that a resident Ollama model belongs exclusively to this
    process rather than a pre-existing or concurrent user session, no `keep_alive=0`
    unload request is sent. If resident models are detected, a warning is logged to
    report the 8GB VRAM coexistence constraint.
    """
    try:
        import httpx
        from zelyr.config import get_config

        config = get_config()
        base_url = getattr(config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")

        with httpx.Client(timeout=3.0) as client:
            ps_resp = client.get(f"{base_url}/api/ps", timeout=2.0)
            if ps_resp.status_code != 200:
                return

            models = ps_resp.json().get("models", []) or []
            resident = [
                (m.get("name") or m.get("model"))
                for m in models
                if isinstance(m, dict) and (m.get("name") or m.get("model"))
            ]
            if resident:
                logger.warning(
                    "Ollama models %s are currently resident in VRAM, but safe Zelyr session "
                    "ownership cannot be established from Ollama /api/ps. Skipping keep_alive=0 "
                    "unload to preserve the user's existing Ollama session (8GB VRAM ceiling "
                    "may be constrained).",
                    resident,
                )
    except Exception as exc:
        logger.debug("Ollama VRAM check skipped or failed: %s", exc)


def _evict_grounding_engine_unlocked() -> None:
    """Internal helper: evict resident ZelyrGroundingEngine. Caller MUST hold _engine_lock."""
    global _engine_instance
    if _engine_instance is not None:
        print(">>> [Zelyr] Evicting resident ZelyrGroundingEngine from CUDA memory...")
        instance = _engine_instance
        _engine_instance = None
        try:
            if hasattr(instance, "unload"):
                instance.unload()
        except Exception as e:
            print(f"Warning during vision engine eviction: {e}")
        try:
            from zelyr.core.vision_engine.zelyr_engine import ZelyrGroundingEngine
            ZelyrGroundingEngine._instance = None
        except Exception:
            pass
        del instance
        gc.collect()
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass


def _evict_vla_engine_unlocked() -> None:
    """Internal helper: evict resident SpotlessVLA after waiting for active operations.

    Caller MUST hold _vla_lock / _vla_cond.
    """
    global _vla_instance
    while _vla_active_count > 0 and _vla_instance is not None:
        _vla_cond.wait()

    if _vla_instance is not None:
        print(">>> [Zelyr] Evicting resident SpotlessVLA from CUDA memory...")
        instance = _vla_instance
        _vla_instance = None
        try:
            if hasattr(instance, "unload"):
                instance.unload()
        except Exception as exc:
            logger.debug("Warning during SpotlessVLA unload: %s", exc)
        finally:
            del instance
            gc.collect()
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                try:
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                except Exception:
                    pass


def get_grounding_engine():
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            with _vla_cond:
                _evict_vla_engine_unlocked()
            try:
                from zelyr.core.vision_engine.zelyr_engine import ZelyrGroundingEngine
            except ImportError as e:
                raise ImportError(
                    "'torch' and 'transformers' are required for local VLM features. "
                    "Install them with: pip install 'zelyr-core[ml]'"
                ) from e
            print(">>> [Zelyr] Bootstrapping resident ZelyrGroundingEngine into CUDA memory...")
            _engine_instance = ZelyrGroundingEngine()
            ZelyrGroundingEngine._instance = _engine_instance
    return _engine_instance


def evict_grounding_engine():
    with _engine_lock:
        _evict_grounding_engine_unlocked()


def get_vla_engine(
    adapter_path: Optional[str] = None,
    motor_head_path: Optional[str] = None,
    device: str = "cuda",
    _spotless_vla_cls: Optional[Any] = None,
    _acquire_lease: bool = False,
):
    """Return resident SpotlessVLA engine, unloading legacy VLM if needed."""
    global _vla_instance, _vla_active_count
    if _spotless_vla_cls is None:
        spotless_vla_cls, _, _, _ = _load_spotless_components_or_evict()
    else:
        if not _is_spotless_arms_enabled_by_env():
            evict_vla_engine()
            raise RuntimeError(
                "Spotless-Arms VLA integration is disabled via ZELYR_USE_SPOTLESS_ARMS."
            )
        spotless_vla_cls = _spotless_vla_cls

    # Canonical lock order: _engine_lock -> _vla_lock (_vla_cond)
    with _engine_lock, _vla_cond:
        if _vla_instance is None:
            _evict_grounding_engine_unlocked()
            _evict_ollama_vram()

            resolved_adapter = adapter_path
            if resolved_adapter is None:
                cand = _SPOTLESS_CHECKPOINT_DIR / "best_motor_adapter.pt"
                if cand.is_file():
                    resolved_adapter = str(cand)

            resolved_head = motor_head_path
            if resolved_head is None:
                cand_h = _SPOTLESS_CHECKPOINT_DIR / "best_motor_head.pt"
                if cand_h.is_file():
                    resolved_head = str(cand_h)

            print(">>> [Zelyr] Bootstrapping resident SpotlessVLA (Closed-Loop Arms) into CUDA memory...")
            _vla_instance = spotless_vla_cls.from_pretrained(
                adapter_path=resolved_adapter,
                motor_head_path=resolved_head,
                device=device,
                prefer_motor_adapter=True,
            )
        if _acquire_lease:
            _vla_active_count += 1
        return _vla_instance


@contextmanager
def _acquire_vla_engine_lease(
    _spotless_vla_cls: Optional[Any] = None,
) -> Iterator[Any]:
    """Acquire resident SpotlessVLA and prevent eviction until the active operation finishes."""
    global _vla_active_count
    vla = get_vla_engine(_spotless_vla_cls=_spotless_vla_cls, _acquire_lease=True)
    try:
        yield vla
    finally:
        with _vla_cond:
            _vla_active_count = max(0, _vla_active_count - 1)
            if _vla_active_count == 0:
                _vla_cond.notify_all()


def evict_vla_engine() -> None:
    """Evict resident SpotlessVLA from CUDA memory once active operations complete."""
    with _vla_cond:
        _evict_vla_engine_unlocked()


def run_reflex_action(
    instruction: str,
    max_ticks: int = 6,
    move_only: bool = False,
    dry_run_arm: bool = False,
    k_initial: int = 2,
    k_track: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a closed-loop Spotless-Arms VLA reflex action on Wayland via /dev/uinput.

    Preserves Ollama VRAM coexistence (<7.20 GB), uses Adaptive-K (k_initial=2 -> k_track=1),
    EMA damping, single-click auto-halt, and early spatial convergence detection.
    """
    (
        spotless_vla_cls,
        pipelined_runner_cls,
        screen_buffer_cls,
        virtual_arm_cls,
    ) = _load_spotless_components_or_evict()

    t0 = time.perf_counter()
    with _acquire_vla_engine_lease(_spotless_vla_cls=spotless_vla_cls) as vla:
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        with screen_buffer_cls(jpeg_quality=80, max_width=1280) as screen_buf, virtual_arm_cls(
            dry_run=dry_run_arm
        ) as arm:
            runner = pipelined_runner_cls(
                vla=vla,
                arm=arm,
                screen_buffer=screen_buf,
                k_initial=k_initial,
                k_track=k_track,
                auto_halt_on_click=True,
                halt_on_convergence=True,
            )
            telemetry = runner.run(
                prompt=instruction,
                max_ticks=max_ticks,
                move_only=move_only,
                verbose=verbose,
            )

    total_ms = (time.perf_counter() - t0) * 1000.0
    peak_vram_gb = 0.0
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                peak_vram_gb = torch_mod.cuda.max_memory_allocated() / (1024**3)
        except Exception:
            pass

    last_rec = telemetry[-1] if telemetry else {}
    last_act = last_rec.get("action", {})
    summary = getattr(runner, "last_summary", {}) or {}

    return {
        "status": "ok" if telemetry else "empty",
        "instruction": instruction,
        "ticks": len(telemetry),
        "k_schedule": [t["k_steps"] for t in telemetry],
        "final_xy": [last_act.get("px", 0), last_act.get("py", 0)],
        "final_xy_norm": [
            round(float(last_act.get("x_norm", 0.5)), 4),
            round(float(last_act.get("y_norm", 0.5)), 4),
        ],
        "clicked": last_act.get("clicked"),
        "converged": bool(summary.get("converged", False)),
        "converged_tick": summary.get("converged_tick"),
        "halted_reason": last_rec.get("halted_reason") or "max_ticks",
        "total_ms": round(total_ms, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "backend": last_act.get("backend", "unknown"),
        "move_only": move_only,
    }
