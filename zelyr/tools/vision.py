from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import subprocess
import shutil
import sys
import logging
import io
import time
import json
import httpx
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from PIL import Image

from zelyr.tools.core import BaseTool, ToolResult, ToolParameter
from zelyr.core.vision_engine.singleton import (
    _is_optional_spotless_module_error,
    get_grounding_engine,
    is_spotless_arms_available,
    run_reflex_action,
)
from zelyr.config import get_config

logger = logging.getLogger(__name__)

# Standard Zelyr security ring: Tier 3 physical UI actuation requires Ring 0 privilege.
REQUIRED_RING = 0
TIER = 3


@dataclass(frozen=True)
class TrustedUIExecutionContext:
    """Trusted caller authorization metadata bound outside model-visible tool arguments."""

    caller_ring: Optional[int] = None
    allowed_tools: Optional[Any] = None


_ACTIVE_TRUSTED_UI_CONTEXT: ContextVar[Optional[TrustedUIExecutionContext]] = ContextVar(
    "_ACTIVE_TRUSTED_UI_CONTEXT",
    default=None,
)


@contextmanager
def trusted_ui_execution_context(
    *,
    caller_ring: Optional[int] = None,
    allowed_tools: Optional[Iterable[str]] = None,
) -> Iterator[TrustedUIExecutionContext]:
    """Establish a trusted Ring-0 execution context for direct InteractWithUITool calls."""
    ctx = TrustedUIExecutionContext(
        caller_ring=caller_ring,
        allowed_tools=allowed_tools,
    )
    token = _ACTIVE_TRUSTED_UI_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE_TRUSTED_UI_CONTEXT.reset(token)

async def get_hyprland_scale() -> float:
    """Gets the scale factor of the focused monitor in Hyprland."""
    import asyncio
    try:
        proc = await asyncio.create_subprocess_shell(
            "hyprctl monitors -j",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            monitors = json.loads(stdout.decode('utf-8'))
            for monitor in monitors:
                if monitor.get('focused', False):
                    return float(monitor.get('scale', 1.0))
    except Exception as e:
        logger.warning(f"Failed to get hyprland scale: {e}")
    return 1.0


class InteractWithUITool(BaseTool):
    """Interacts with the UI using the Zelyr Grounding Engine."""

    REQUIRED_RING = 0

    def __init__(self):
        super().__init__(
            tool_id="interact_with_ui",
            name="interact_with_ui",
            description=(
                "Autonomously finds and interacts with a UI element on the screen based on a text instruction. "
                "Used for clicking buttons, typing into fields, etc. "
                "STRICT RESTRICTION: DO NOT use this tool for web browsers or web pages. "
                "If interacting with a browser tab, you MUST use interact_with_browser with click_element or fill_element."
            ),
            tier=3,
            is_core=False,
            requires_bridge=False,
            requires_window=True,
            token_cost=600,
        )
        self.required_ring = 0
        self.requires_approval = True
        self.parameters = [
            ToolParameter(
                name="instruction",
                type="string",
                description="Precise text instruction of what to interact with (e.g., 'Click the submit button', 'Type \"hello\" into the search bar').",
                required=True,
            ),
            ToolParameter(
                name="move_only",
                type="boolean",
                description="Move cursor without clicking.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="dry_run_arm",
                type="boolean",
                description="Simulate virtual arm actuation without opening /dev/uinput.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_ticks",
                type="integer",
                description="Maximum closed-loop VLA ticks.",
                required=False,
                default=6,
            ),
            ToolParameter(
                name="use_legacy_json",
                type="boolean",
                description="Use legacy single-shot VLM grounding instead of Spotless-Arms VLA.",
                required=False,
                default=False,
            ),
        ]

    def _resolve_trusted_caller_context(
        self,
    ) -> Tuple[bool, Optional[Any], Optional[int], Optional[Any]]:
        """Inspect trusted ToolRegistry frame or ContextVar; never reads model tool arguments.

        Returns (has_trusted_source, registry_instance, caller_ring, allowed_tools).
        """
        from zelyr.tool_registry import ToolRegistry

        active_ctx = _ACTIVE_TRUSTED_UI_CONTEXT.get()
        frame = sys._getframe(1)
        depth = 0
        while frame is not None and depth < 12:
            code_name = frame.f_code.co_name
            if code_name in ("execute_async", "execute"):
                cand_self = frame.f_locals.get("self")
                cand_tool = frame.f_locals.get("tool")
                if (
                    isinstance(cand_self, ToolRegistry)
                    and cand_tool is self
                    and frame.f_locals.get("auth_error") is None
                    and frame.f_locals.get("schema_error") is None
                ):
                    f_ring = frame.f_locals.get("caller_ring")
                    f_allowed = frame.f_locals.get("allowed_tools")
                    if f_ring is None and active_ctx is not None:
                        f_ring = active_ctx.caller_ring
                    if f_allowed is None and active_ctx is not None:
                        f_allowed = active_ctx.allowed_tools
                    return True, cand_self, f_ring, f_allowed
            frame = frame.f_back
            depth += 1

        if active_ctx is not None:
            return True, None, active_ctx.caller_ring, active_ctx.allowed_tools

        return False, None, None, None

    async def _capture_wayland_grim(self, cmd: str) -> Optional[bytes]:
        logger.info(f"Triggering grim screenshot... ({cmd})")
        import asyncio
        try:
            proc = await asyncio.create_subprocess_shell(
                f"{cmd} -c -", 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode != 0:
                logger.warning(f"Failed to capture with {cmd}: {stderr.decode()}")
                return None
            return stdout
        except asyncio.TimeoutError:
            logger.error(f"TimeoutExpired: grim screenshot hung for command {cmd}")
            raise RuntimeError(f"grim screenshot timeout for command {cmd}")
        except Exception as e:
            logger.warning(f"Failed to capture with {cmd}: {e}")
            return None

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        import asyncio
        from zelyr.tool_registry import (
            authorize_and_validate_tool_call,
            get_active_task_local_scope,
        )

        if not isinstance(params, dict):
            return ToolResult(False, error="Invalid tool arguments.", error_type="SCHEMA_VALIDATION_ERROR")

        # Untrusted model-visible tool arguments must never supply caller identity or privilege.
        if "caller_ring" in params or "allowed_tools" in params or "task_local_scope" in params:
            return ToolResult(
                False,
                output="",
                error=(
                    "AuthorizationError: Security metadata ('caller_ring', 'allowed_tools') "
                    "cannot be supplied through tool arguments."
                ),
                error_type="AUTHORIZATION_DENIED",
            )

        # Never allow an isolated task-local test scope to reach real UI actuation
        if get_active_task_local_scope() is not None:
            return ToolResult(
                False,
                output="",
                error="AuthorizationError: Isolated task-local test executor cannot execute production UI actuation.",
                error_type="AUTHORIZATION_DENIED",
            )

        # Guard against Tier 3 regression on web browser elements
        if "element_id" in params or params.get("action") in ("click_element", "fill_element"):
            return ToolResult(
                False,
                error=(
                    "STRICT RESTRICTION: DO NOT use interact_with_ui for web elements. "
                    "'element_id' belongs to interact_with_browser. "
                    "You MUST use interact_with_browser with action='click_element' or 'fill_element'."
                ),
            )

        instruction = params.get("instruction")
        if not instruction:
            return ToolResult(False, error="instruction is required.")

        lower_inst = instruction.lower()
        if any(term in lower_inst for term in ["get_page_snapshot", "click_element", "fill_element"]) or \
           ("element #" in lower_inst or "element id" in lower_inst):
            return ToolResult(
                False,
                error=(
                    "STRICT RESTRICTION: DO NOT use interact_with_ui on web elements or browser pages. "
                    "You MUST use interact_with_browser with action='click_element' or 'fill_element'."
                ),
            )

        # Fail-closed authorization boundary: Ring 0 physical UI actuation requires a trusted
        # execution context (ToolRegistry or trusted_ui_execution_context) with explicit
        # caller_ring and allowed_tools metadata.
        has_trusted_ctx, reg_instance, ctx_ring, ctx_allowed = self._resolve_trusted_caller_context()
        if not has_trusted_ctx:
            return ToolResult(
                False,
                output="",
                error=(
                    "AuthorizationError: Direct execution of 'interact_with_ui' is denied without "
                    "a trusted ToolRegistry or trusted_ui_execution_context authorization boundary."
                ),
                error_type="AUTHORIZATION_DENIED",
            )

        if ctx_ring is None:
            return ToolResult(
                False,
                output="",
                error=(
                    "AuthorizationError: Physical UI actuation ('interact_with_ui') requires "
                    "explicit Ring 0 caller_ring=0 in the trusted execution context."
                ),
                error_type="AUTHORIZATION_DENIED",
            )

        if ctx_allowed is None:
            return ToolResult(
                False,
                output="",
                error=(
                    "AuthorizationError: Physical UI actuation ('interact_with_ui') requires "
                    "explicit allowed_tools in the trusted execution context."
                ),
                error_type="AUTHORIZATION_DENIED",
            )

        decision = authorize_and_validate_tool_call(
            self.name,
            {},
            caller_ring=ctx_ring,
            allowed_tools=ctx_allowed,
            tool=self,
            tool_registry=reg_instance,
            require_registered=bool(reg_instance is not None),
            normalize_args=False,
            validate_schema=False,
        )
        if not decision.allowed:
            return ToolResult(
                False,
                output="",
                error=decision.detail or "AuthorizationError: Tool execution denied.",
                error_type=decision.error_type or "AUTHORIZATION_DENIED",
            )

        # Follow existing GovernorService.instance() approval policy when not already checked by ToolRegistry
        if reg_instance is None and getattr(self, "requires_approval", False):
            try:
                from zelyr.services.governor import GovernorService

                governor = GovernorService.instance()
                if governor and governor.is_strict_mode():
                    approved = await governor.request_approval(self.name, params)
                    if not approved:
                        return ToolResult(
                            False,
                            output="",
                            error="User denied execution via The Governor.",
                            error_type="AUTHORIZATION_DENIED",
                        )
            except Exception as e:
                return ToolResult(False, error=f"Governor failed: {e}", error_type="AUTHORIZATION_DENIED")

        # Primary Tier 3 Path: Closed-loop Spotless-Arms VLA via persistent /dev/uinput (<0.1ms actuation)
        move_only = bool(params.get("move_only", False))
        dry_run_arm = bool(params.get("dry_run_arm", False))
        max_ticks = int(params.get("max_ticks", 6))
        use_legacy_json = bool(params.get("use_legacy_json", False))

        if not use_legacy_json and is_spotless_arms_available():
            try:
                reflex = await asyncio.to_thread(
                    run_reflex_action,
                    instruction,
                    max_ticks,
                    move_only,
                    dry_run_arm,
                )
                msg = (
                    f"Closed-loop VLA reflex '{instruction}' completed in {reflex['total_ms']:.1f}ms "
                    f"({reflex['ticks']} ticks, K={reflex['k_schedule']}, xy={reflex['final_xy']}, "
                    f"clicked={reflex['clicked']}, converged={reflex['converged']}, "
                    f"halted={reflex['halted_reason']}, backend={reflex['backend']}, "
                    f"peak_vram={reflex['peak_vram_gb']:.2f}GB)."
                )
                return ToolResult(True, output=msg, metadata=reflex)
            except ModuleNotFoundError as exc:
                if not _is_optional_spotless_module_error(exc):
                    raise
                logger.warning("spotless-arms unavailable (%s); falling back to legacy single-shot VLM.", exc)

        image_bytes = None
        
        # 1. Capture screen
        if shutil.which("distrobox-host-exec"):
            try:
                proc = await asyncio.create_subprocess_shell(
                    "distrobox-host-exec which grim", 
                    stdout=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                if proc.returncode == 0:
                    try:
                        image_bytes = await self._capture_wayland_grim("distrobox-host-exec grim")
                    except RuntimeError as e:
                        return ToolResult(False, error=str(e))
            except Exception:
                pass
                
        if image_bytes is None and shutil.which("grim"):
            try:
                image_bytes = await self._capture_wayland_grim("grim")
            except RuntimeError as e:
                return ToolResult(False, error=str(e))
            
        if not image_bytes:
            # Fallback to saving to a file via grim
            screenshot_path = "/tmp/screen.png"
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            try:
                proc = await asyncio.create_subprocess_exec("grim", screenshot_path)
                await proc.communicate()
                if proc.returncode == 0:
                    with open(screenshot_path, "rb") as f:
                        image_bytes = f.read()
                else:
                    return ToolResult(False, error="grim fallback returned non-zero exit code")
            except Exception as e:
                return ToolResult(False, error=f"Failed to capture screen: {e}")

        logger.info("Screenshot captured. Passing to Grounding Engine...")
        
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # VRAM Juggling: Evict Ollama models from VRAM before loading the 6GB grounding engine
            logger.info("Evicting Ollama model from VRAM...")
            try:
                config = get_config()
                base_url = getattr(config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{base_url}/api/generate",
                        json={"model": config.ollama_model, "keep_alive": 0},
                        timeout=5.0
                    )
                    # Check and evict any other running models
                    try:
                        ps_resp = await client.get(f"{base_url}/api/ps", timeout=2.0)
                        if ps_resp.status_code == 200:
                            for m in ps_resp.json().get("models", []):
                                m_name = m.get("name") or m.get("model")
                                if m_name and m_name != config.ollama_model:
                                    await client.post(
                                        f"{base_url}/api/generate",
                                        json={"model": m_name, "keep_alive": 0},
                                        timeout=3.0
                                    )
                    except Exception:
                        pass
                await asyncio.sleep(1.0) # Allow GPU memory to clear
            except Exception as e:
                logger.warning(f"Failed to evict Ollama model from VRAM: {e}")

            engine = get_grounding_engine()
            
            start_time = time.time()
            import asyncio
            result = await asyncio.to_thread(engine.predict_action, img, instruction, 4)
            latency = (time.time() - start_time) * 1000
            
            action = result.get("action")
            point = result.get("point")
            text = result.get("text")
            
            if action == "unknown" or not point:
                return ToolResult(False, error=f"Engine failed to parse action from raw output: {result.get('raw_output')}")
            
            x, y = point[0], point[1]
            logger.info(f"Engine returned action '{action}' at ({x}, {y}) in {latency:.2f}ms")
            
            # Scale coordinates for ydotool in Hyprland
            scale = await get_hyprland_scale()
            if scale != 1.0:
                logger.info(f"Applying Hyprland scale factor: {scale}")
            logical_x = int(x / scale)
            logical_y = int(y / scale)
            
            # Execute via ydotool
            if action == "click" or action == "hover":
                # Move mouse
                proc = await asyncio.create_subprocess_shell(f"ydotool mousemove --absolute {logical_x} {logical_y}")
                await proc.wait()
                await asyncio.sleep(0.1)
                
                if action == "click":
                    proc = await asyncio.create_subprocess_shell("ydotool click 0xC0")
                    await proc.wait()
                    
                msg = f"Action '{action}' executed at coordinates [{x}, {y}] (scaled to [{logical_x}, {logical_y}])."
                return ToolResult(True, output=msg)
                
            elif action == "type":
                proc = await asyncio.create_subprocess_shell(f"ydotool mousemove --absolute {logical_x} {logical_y}")
                await proc.wait()
                await asyncio.sleep(0.1)
                proc = await asyncio.create_subprocess_shell("ydotool click 0xC0") # Focus
                await proc.wait()
                await asyncio.sleep(0.1)
                
                # Type text
                if text:
                    import shlex
                    safe_text = shlex.quote(text)
                    proc = await asyncio.create_subprocess_shell(f"ydotool type {safe_text}")
                    await proc.wait()
                
                msg = f"Action 'type' executed at coordinates [{x}, {y}] (scaled to [{logical_x}, {logical_y}]) with text '{text}'."
                return ToolResult(True, output=msg)
                
            else:
                return ToolResult(False, error=f"Unknown action: {action}")
                
        except ImportError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool requires optional dependency: {e}",
            )
        except Exception as e:
            return ToolResult(False, error=f"UI interaction failed: {e}")
