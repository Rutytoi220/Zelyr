"""Zelyr Autonomous Tool Crafter.

Empowers Zelyr to synthesize, validate, test, register, and hot-reload new dynamic
tools into ~/.config/zelyr/tools.d/. Enforces duplicate capability prevention,
contract validation, and privilege escalation guards (no generated tools at Ring 0).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zelyr.core.plugins import reload_plugin, unregister_plugin
from zelyr.db.memory import MemoryStore
from zelyr.tools.core import ToolResult
from zelyr.tools.validator import ToolValidator

logger = logging.getLogger(__name__)


def _extract_schema_metadata_from_code(code: str) -> Tuple[Optional[str], Optional[str]]:
    """Statically inspect tool candidate code to extract tool name and description if possible."""
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TOOL_SCHEMA":
                        if isinstance(node.value, ast.Dict):
                            for k, v in zip(node.value.keys, node.value.values):
                                if isinstance(k, ast.Constant) and k.value == "function" and isinstance(v, ast.Dict):
                                    name_val = None
                                    desc_val = None
                                    for fk, fv in zip(v.keys, v.values):
                                        if isinstance(fk, ast.Constant):
                                            if fk.value == "name" and isinstance(fv, ast.Constant):
                                                name_val = str(fv.value)
                                            elif fk.value == "description" and isinstance(fv, ast.Constant):
                                                desc_val = str(fv.value)
                                    return name_val, desc_val
    except Exception:
        pass
    return None, None


async def craft_tool(
    name: str,
    code: str,
    test_params: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    tools_dir: Optional[Path] = None,
) -> ToolResult:
    """Synthesize, validate, test, register, and hot-reload a new tool.

    Steps:
      1. Tool name validation format (^[a-zA-Z_][a-zA-Z0-9_]*$).
      2. Duplicate capability prevention via MemoryStore(category='capability').
      3. Complete contract & staging validation via ToolValidator.
      4. Privilege escalation check: candidate tools CANNOT declare REQUIRED_RING = 0.
      5. Atomic write to tools.d/{name}.py.
      6. Hot-reload via reload_plugin(name).
      7. MemoryStore capability indexing verification.
      8. Return structured ToolResult.
    """
    clean_name = (name or "").strip()
    if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
        return ToolResult(
            success=False,
            error=f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
        )

    store = MemoryStore(db_path=db_path)

    # 1. Duplicate Capability Prevention
    # Check exact tool key
    key = f"tool.{clean_name}"
    existing_by_key = store.get_by_key(key)
    if existing_by_key and existing_by_key.get("is_active"):
        return ToolResult(
            success=False,
            error=(
                f"Duplicate capability rejected: Tool '{clean_name}' already exists in active capabilities. "
                f"Existing description: {existing_by_key.get('content')}"
            ),
        )

    # Statically extract candidate description to check semantic duplicate
    _, cand_desc = _extract_schema_metadata_from_code(code)
    if cand_desc:
        cand_desc_clean = cand_desc.strip().lower()
        active_caps = store.search(cand_desc_clean, category="capability", limit=5)
        for cap in active_caps:
            cap_key = cap.get("key", "")
            cap_content = cap.get("content", "").lower()
            if cap.get("is_active") and cap_key != key:
                if f": {cand_desc_clean}" in cap_content or cap_content.endswith(cand_desc_clean):
                    existing_tool_name = cap_key.replace("tool.", "")
                    return ToolResult(
                        success=False,
                        error=(
                            f"Duplicate capability rejected: Active tool '{existing_tool_name}' already provides "
                            f"an identical capability ({cap.get('content')})."
                        ),
                    )

    # 2. Comprehensive Validation Gate
    validator = ToolValidator()
    val_res = await validator.validate_all(code, test_params=test_params)
    if not val_res.is_valid:
        return ToolResult(
            success=False,
            error=f"Validation failed: {val_res.error}",
        )

    # Tool name matching
    if val_res.tool_name and val_res.tool_name != clean_name:
        return ToolResult(
            success=False,
            error=(
                f"Tool name mismatch: Provided name '{clean_name}' does not match "
                f"TOOL_SCHEMA function name '{val_res.tool_name}'."
            ),
        )

    # 3. Privilege Escalation Guard
    if val_res.ring == 0:
        return ToolResult(
            success=False,
            error="Privilege escalation rejected: generated dynamic tools cannot claim REQUIRED_RING = 0.",
        )

    # 4. Atomic Installation & Hot-Reload
    from zelyr.config import get_config_dir
    target_dir = tools_dir or (get_config_dir() / "tools.d")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{clean_name}.py"

    # Atomic write via temporary file in target_dir
    tmp_file = target_dir / f".tmp_{clean_name}_{uuid.uuid4().hex}.py"
    try:
        tmp_file.write_text(code, encoding="utf-8")
        tmp_file.replace(target_file)
    except Exception as e:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        return ToolResult(
            success=False,
            error=f"Failed to write tool to {target_file}: {e}",
        )

    # Hot reload with Rollback
    try:
        reload_plugin(target_file)
    except Exception as err:
        logger.warning("reload_plugin failed for %s: %s. Rolling back installation.", target_file, err)
        # Rollback: remove target_file immediately
        if target_file.exists():
            try:
                target_file.unlink()
            except OSError:
                pass

        # Clean up any partial schemas or cached modules
        try:
            unregister_plugin(clean_name)
        except Exception:
            pass

        # Soft-delete capability record if created in MemoryStore
        try:
            cap_record = store.get_by_key(f"tool.{clean_name}")
            if cap_record:
                store.delete(cap_record["id"], hard_delete=False)
            with store._write_lock:
                with store._get_conn() as conn:
                    conn.execute(
                        "UPDATE memories SET is_active = 0, updated_at = ? WHERE key = ? AND is_active = 1",
                        (time.time(), f"tool.{clean_name}"),
                    )
                    conn.commit()
        except Exception:
            pass

        try:
            from zelyr.memory.capability_indexer import reconcile_tool_capabilities
            reconcile_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception:
            pass

        return ToolResult(
            success=False,
            error=f"Installation failed during reload: {err}. File and runtime state rolled back.",
        )

    # Confirm capability indexed into MemoryStore
    try:
        from zelyr.memory.capability_indexer import index_tool_capabilities
        index_tool_capabilities(store=store, tools_dir=target_dir)
    except Exception as e:
        logger.warning("Capability indexing failed for %s: %s", clean_name, e)

    return ToolResult(
        success=True,
        output={
            "tool_name": clean_name,
            "file_path": str(target_file),
            "status": "installed_and_reloaded",
            "tier": val_res.tier,
            "ring": val_res.ring,
        },
    )
