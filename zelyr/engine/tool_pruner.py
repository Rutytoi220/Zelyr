import re
from typing import List, Dict, Any

class ToolPruner:
    """Dynamically clusters and prunes tool schemas to optimize LLM context usage."""

    CLUSTER_MAPPINGS = {
        "filesystem": {
            "keywords": ["file", "document", "read", "write", "text", "search"],
            "tools": ["read_document_content", "file_read", "file_write", "file_search"]
        },
        "system_os": {
            "keywords": ["run", "command", "os", "app", "launch", "system"],
            "tools": ["execute_command", "system_info", "launch_app"]
        },
        "perception": {
            "keywords": ["screen", "see", "view", "controller", "game"],
            "tools": ["capture_screenshot", "nxbt_controller"]
        }
    }

    @staticmethod
    def prune_schemas(user_prompt: str, raw_schemas: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
        """Filters schemas based on the intent and semantic relevance."""
        if intent == "chat":
            return []
            
        prompt_lower = user_prompt.lower()
        
        # Always keep core foundational tools
        core_tools = {"shell", "file_read", "file_write", "safe_file_search", "echo"}
        
        active_tools = []
        for schema in raw_schemas:
            tool_name = schema.get("function", {}).get("name", "")
            
            # 1. Core tools are always injected
            if tool_name in core_tools:
                active_tools.append(schema)
                continue
                
            # 2. Check cluster mappings
            matched_cluster = False
            for cluster, data in ToolPruner.CLUSTER_MAPPINGS.items():
                if tool_name in data["tools"]:
                    if any(kw in prompt_lower for kw in data["keywords"]):
                        matched_cluster = True
                        break
            
            if matched_cluster:
                active_tools.append(schema)
                continue
                
            # 3. Fuzzy keyword check for other extraneous tools
            # If the tool name or description shares words with the prompt
            desc = schema.get("function", {}).get("description", "").lower()
            if any(kw in prompt_lower for kw in tool_name.split("_") if len(kw) > 3) or \
               any(kw in prompt_lower for kw in desc.split() if len(kw) > 5):
                active_tools.append(schema)
                continue
                
            # If nothing matches, we PRUNE it (do not append).
            
        # Fallback: if we somehow pruned everything (shouldn't happen due to core_tools), return raw
        return active_tools if active_tools else raw_schemas
