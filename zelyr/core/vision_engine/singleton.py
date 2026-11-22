import gc
import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_engine_instance = None
_engine_lock = threading.Lock()

_vla_instance = None
_vla_lock = threading.Lock()

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

    try:
        _import_spotless_components()
        return True
    except ModuleNotFoundError as exc:
        if _is_optional_spotless_module_error(exc):
            evict_vla_engine()
            return False
        raise


def _evict_ollama_vram() -> None:
    """Unload active Ollama models (keep_alive=0) so RTX 4060 8GB VRAM is clear for VLA."""
    try:
        import httpx
        from zelyr.config import get_config

        config = get_config()
        base_url = getattr(config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
        evicted = False
        with httpx.Client(timeout=3.0) as client:
            if getattr(config, "ollama_model", None):
                try:
                    client.post(
                        f"{base_url}/api/generate",
                        json={"model": config.ollama_model, "keep_alive": 0},
                    )
                    evicted = True
                except Exception:
                    pass
            try:
                ps_resp = client.get(f"{base_url}/api/ps", timeout=2.0)
                if ps_resp.status_code == 200:
                    for m in ps_resp.json().get("models", []):
                        m_name = m.get("name") or m.get("model")
                        if m_name and m_name != getattr(config, "ollama_model", None):
                            client.post(
                                f"{base_url}/api/generate",
                                json={"model": m_name, "keep_alive": 0},
                            )
                            evicted = True
            except Exception:
                pass
        if evicted:
            time.sleep(0.8)
    except Exception as exc:
        logger.debug("Ollama VRAM eviction skipped/failed: %s", exc)


def get_grounding_engine():
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            evict_vla_engine()
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
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            print(">>> [Zelyr] Evicting resident ZelyrGroundingEngine from CUDA memory...")
            try:
                if hasattr(_engine_instance, "unload"):
                    _engine_instance.unload()
            except Exception as e:
                print(f"Warning during vision engine eviction: {e}")
            _engine_instance = None
            try:
                from zelyr.core.vision_engine.zelyr_engine import ZelyrGroundingEngine
                ZelyrGroundingEngine._instance = None
            except Exception:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass


def get_vla_engine(
    adapter_path: Optional[str] = None,
    motor_head_path: Optional[str] = None,
    device: str = "cuda",
    _spotless_vla_cls: Optional[Any] = None,
):
    """Return resident SpotlessVLA engine, unloading Ollama & legacy VLM if needed."""
    global _vla_instance
    if _spotless_vla_cls is None:
        spotless_vla_cls, _, _, _ = _load_spotless_components_or_evict()
    else:
        if not _is_spotless_arms_enabled_by_env():
            evict_vla_engine()
            raise RuntimeError(
                "Spotless-Arms VLA integration is disabled via ZELYR_USE_SPOTLESS_ARMS."
            )
        spotless_vla_cls = _spotless_vla_cls

    with _vla_lock:
        if _vla_instance is None:
            evict_grounding_engine()
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
    return _vla_instance


def evict_vla_engine() -> None:
    """Evict resident SpotlessVLA from CUDA memory."""
    global _vla_instance
    with _vla_lock:
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
    vla = get_vla_engine(_spotless_vla_cls=spotless_vla_cls)

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
