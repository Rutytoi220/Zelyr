import os
import subprocess
import shutil
import logging
import io
import time
import json
import httpx
from typing import Any, Dict, Optional
from PIL import Image

from zelyr.tools.core import BaseTool, ToolResult, ToolParameter
from zelyr.core.vision_engine.singleton import get_grounding_engine
from zelyr.config import get_config

logger = logging.getLogger(__name__)

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
        self.parameters = [
            ToolParameter(
                name="instruction",
                type="string",
                description="Precise text instruction of what to interact with (e.g., 'Click the submit button', 'Type \"hello\" into the search bar').",
                required=True
            )
        ]

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
