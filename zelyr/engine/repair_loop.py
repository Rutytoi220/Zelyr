import logging
import re
import os
from zelyr.agents.healer_agent import HealerAgent
from zelyr.tools.plugin_loader import load_plugins
from zelyr.llm.universal_client import UniversalLLMClient

logger = logging.getLogger(__name__)

async def trigger_repair(traceback_str: str, registry, event_bus):
    logger.info("[RepairLoop] Repair sequence initiated.")
    
    # Extract the failing file from the traceback
    # Tracebacks usually look like: File "/home/<user>/.config/zelyr/plugins/foo.py" (or the legacy .config/axiom/plugins path), line X
    match = re.search(r'File "([^"]+zelyr/plugins/[^"]+\.py)"', traceback_str)
    if not match:
        logger.error("[RepairLoop] Could not isolate a specific plugin file from the traceback.")
        return
        
    file_path = match.group(1)
    
    if not os.path.exists(file_path):
        logger.error(f"[RepairLoop] Extracted file path does not exist: {file_path}")
        return
        
    with open(file_path, 'r') as f:
        source_code = f.read()
        
    # Instantiate LLM Client and Healer Agent
    llm = UniversalLLMClient()
    healer = HealerAgent(llm_client=llm)
    
    logger.info(f"[RepairLoop] Delegating to HealerAgent for {file_path}")
    success = await healer.run(traceback_str, source_code, file_path)
    
    if success:
        logger.info("[RepairLoop] HealerAgent successfully patched the file. Hot-reloading plugins...")
        # Re-trigger plugin loader to pull the patched code into the registry
        load_plugins(registry)
        logger.info("[RepairLoop] Repair complete. Plugin re-registered.")
        if event_bus:
            event_bus.publish("repair.completed", {"file": file_path, "success": True})
    else:
        logger.error("[RepairLoop] HealerAgent failed to repair the plugin.")
        if event_bus:
            event_bus.publish("repair.failed", {"file": file_path, "success": False})
