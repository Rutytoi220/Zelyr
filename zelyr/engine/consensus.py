import logging
import time
import subprocess
from typing import Dict, Any, List, Optional
from zelyr.core.events import EventBus, Event

logger = logging.getLogger(__name__)

class SwarmConsensusEngine:
    """Manages the verification loop and self-correction for generated code."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.max_retries = 3

    def run_verification_loop(self, coder_agent, initial_prompt: str) -> str:
        """
        Orchestrate a generate -> test -> refine loop.
        coder_agent: A callable or object with a `generate(prompt)` method that returns code/text.
        """
        current_prompt = initial_prompt
        last_result = ""
        
        for attempt in range(1, self.max_retries + 1):
            if self.event_bus:
                self.event_bus.publish_sync(
                    "telemetry.update",
                    data={"message": f"[🔄 Swarm Consensus: Testing generated code (Attempt {attempt}/{self.max_retries})...]"}
                )
            logger.info(f"Swarm Consensus: Generation Attempt {attempt}/{self.max_retries}")
            
            # Step 1: Generate
            result = coder_agent.chat([{"role": "user", "content": current_prompt}])
            last_result = result
            
            # Extract code block if any (very basic extraction)
            code = self._extract_code(result)
            if not code:
                # If no code was generated, maybe it was a direct answer. Just return.
                return result
                
            # Step 2: Verify
            is_valid, error_trace = self._verify_syntax(code)
            
            # Step 3: Consensus/Refinement
            if is_valid:
                logger.info("Swarm Consensus: Verification PASSED.")
                if self.event_bus:
                    self.event_bus.publish_sync(
                        "swarm.consensus.passed",
                        data={"attempts": attempt, "max": self.max_retries}
                    )
                return result
            else:
                logger.warning(f"Swarm Consensus: Verification FAILED. {error_trace}")
                current_prompt = (
                    f"Your previous code failed with the following error:\n\n{error_trace}\n\n"
                    "Please correct the code and provide the fixed version."
                )
                
        # If we exhausted retries, return the last result anyway.
        logger.error("Swarm Consensus: Max retries exhausted.")
        if self.event_bus:
            self.event_bus.publish_sync(
                "swarm.consensus.failed",
                data={"attempts": self.max_retries, "max": self.max_retries}
            )
        return last_result

    def _extract_code(self, text: str) -> Optional[str]:
        """Extracts python code from markdown blocks."""
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return None

    def _verify_syntax(self, code: str) -> tuple[bool, str]:
        """Runs python -m py_compile on the code in a temp file to verify syntax."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
            f.write(code)
            temp_path = f.name
            
        try:
            # Run py_compile
            result = subprocess.run(
                ["python3", "-m", "py_compile", temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Verification timed out."
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
