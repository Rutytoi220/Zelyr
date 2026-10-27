"""Zelyr Dynamic Tool Lifecycle Manager.

Provides programmatic and CLI-accessible lifecycle management for dynamic tools:
- Listing installed dynamic tools with metadata (name, tier, ring, description).
- Updating dynamic tools with safety validation, atomic write, hot-reload, and capability superseding.
- Deleting dynamic tools while strictly guarding built-in protected tools, removing from runtime registries,
  and soft-deleting capabilities in persistent MemoryStore.
"""

from __future__ import annotations

import ast
import datetime
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zelyr.config import get_config_dir
from zelyr.core.plugins import reload_plugin, unregister_plugin
from zelyr.db.memory import MemoryStore
from zelyr.tools.core import ToolResult
from zelyr.tools.validator import ToolValidator

logger = logging.getLogger(__name__)

PROTECTED_TOOLS = {
    "execute_command",
    "manage_desktop_window",
    "manage_system_process",
    "manage_system_clipboard",
    "query_system_journal",
    "manage_media_playback",
    "manage_workspace_file",
    "send_desktop_notification",
    "inspect_network",
    "remember_fact",
    "search_memory",
    "create_tool",
    "manage_tools",
}


def _extract_tool_metadata_from_ast(code: str) -> Dict[str, Any]:
    """Statically inspect tool source code via AST to extract metadata."""
    meta: Dict[str, Any] = {
        "name": None,
        "description": "",
        "tier": 1,
        "ring": 1,
        "tui_hint": "",
    }
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "TOOL_SCHEMA" and isinstance(node.value, ast.Dict):
                            for k, v in zip(node.value.keys, node.value.values):
                                if isinstance(k, ast.Constant) and k.value == "function" and isinstance(v, ast.Dict):
                                    for fk, fv in zip(v.keys, v.values):
                                        if isinstance(fk, ast.Constant):
                                            if fk.value == "name" and isinstance(fv, ast.Constant):
                                                meta["name"] = str(fv.value)
                                            elif fk.value == "description" and isinstance(fv, ast.Constant):
                                                meta["description"] = str(fv.value)
                        elif target.id == "TIER" and isinstance(node.value, ast.Constant):
                            try:
                                meta["tier"] = int(node.value.value)
                            except Exception:
                                pass
                        elif target.id == "REQUIRED_RING" and isinstance(node.value, ast.Constant):
                            try:
                                meta["ring"] = int(node.value.value)
                            except Exception:
                                pass
                        elif target.id == "TUI_HINT" and isinstance(node.value, ast.Constant):
                            meta["tui_hint"] = str(node.value.value)
    except Exception:
        pass
    return meta


class ToolManager:
    """Manager for dynamic tool inspection, validation, updating, and unregistration."""

    PROTECTED_TOOLS = PROTECTED_TOOLS

    def __init__(
        self,
        tools_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.tools_dir = tools_dir or (get_config_dir() / "tools.d")
        self.db_path = db_path

    @classmethod
    def list_tools(cls, tools_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all dynamic tools in tools.d with their metadata."""
        target_dir = tools_dir or (get_config_dir() / "tools.d")
        if not target_dir.exists():
            return []

        tools_list: List[Dict[str, Any]] = []
        for file_path in target_dir.glob("*.py"):
            if file_path.name.startswith((".", "_")):
                continue

            try:
                code = file_path.read_text(encoding="utf-8")
                meta = _extract_tool_metadata_from_ast(code)
            except Exception:
                meta = {
                    "name": file_path.stem,
                    "description": "",
                    "tier": 1,
                    "ring": 1,
                    "tui_hint": "",
                }

            name = meta["name"] or file_path.stem
            desc = meta["description"] or ""
            tier = meta["tier"]
            ring = meta["ring"]

            # If runtime metadata is available, supplement any missing AST info
            try:
                from zelyr.core.plugins import get_tool_schemas, _rings, get_tool_capability
                schemas = get_tool_schemas(0)
                for s in schemas:
                    fn = s.get("function", {})
                    s_name = fn.get("name") or s.get("name")
                    if s_name == name:
                        if not desc and fn.get("description"):
                            desc = fn.get("description")
                        if name in _rings:
                            ring = _rings[name]
                        cap = get_tool_capability(name)
                        if cap and getattr(cap, "tier", None) is not None:
                            tier = cap.tier
                        break
            except Exception:
                pass

            try:
                st = file_path.stat()
                created_at = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = "unknown"

            tools_list.append({
                "name": name,
                "description": desc,
                "tier": tier,
                "ring": ring,
                "created_at": created_at,
                "is_protected": name in cls.PROTECTED_TOOLS,
                "file_path": str(file_path),
            })

        tools_list.sort(key=lambda t: t["name"])
        return tools_list

    @classmethod
    async def update_tool(
        cls,
        name: str,
        code: str,
        test_params: Optional[Dict[str, Any]] = None,
        tools_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
    ) -> ToolResult:
        """Validate, overwrite, hot-reload, and supersede capability of an existing dynamic tool."""
        clean_name = (name or "").strip()
        if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
            return ToolResult(
                success=False,
                error=f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
            )

        if clean_name in cls.PROTECTED_TOOLS:
            return ToolResult(
                success=False,
                error=f"Permission denied: Protected tool '{clean_name}' cannot be modified or updated.",
            )

        target_dir = tools_dir or (get_config_dir() / "tools.d")
        target_file = target_dir / f"{clean_name}.py"

        if not target_file.exists():
            return ToolResult(
                success=False,
                error=f"Tool '{clean_name}' does not exist in {target_dir}. Cannot update non-existent tool.",
            )

        # Preserve existing working code prior to replacement
        try:
            backup_code = target_file.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read existing tool code for backup: {e}",
            )

        # 1. Comprehensive Validation Gate
        validator = ToolValidator()
        val_res = await validator.validate_all(code, test_params=test_params)
        if not val_res.is_valid:
            return ToolResult(
                success=False,
                error=f"Validation failed: {val_res.error}",
            )

        if val_res.tool_name and val_res.tool_name != clean_name:
            return ToolResult(
                success=False,
                error=(
                    f"Tool name mismatch: Provided name '{clean_name}' does not match "
                    f"TOOL_SCHEMA function name '{val_res.tool_name}'."
                ),
            )

        # 2. Privilege Escalation Guard
        if val_res.ring == 0:
            return ToolResult(
                success=False,
                error="Privilege escalation rejected: dynamic tools cannot claim REQUIRED_RING = 0.",
            )

        # 3. Atomic Overwrite
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
                error=f"Failed to write updated tool to {target_file}: {e}",
            )

        # 4. Hot Reload with Rollback
        try:
            reload_plugin(target_file)
        except Exception as err:
            logger.warning(
                "Hot-reload encountered an error for %s: %s. Rolling back to previous version.",
                target_file,
                err,
            )
            # Rollback: restore previous working code atomically
            restore_tmp = target_dir / f".tmp_restore_{clean_name}_{uuid.uuid4().hex}.py"
            try:
                restore_tmp.write_text(backup_code, encoding="utf-8")
                restore_tmp.replace(target_file)
            except Exception as restore_err:
                logger.error("Failed to restore backup code to %s: %s", target_file, restore_err)
                if restore_tmp.exists():
                    try:
                        restore_tmp.unlink()
                    except OSError:
                        pass

            # Re-run reload_plugin on restored file
            try:
                reload_plugin(target_file)
            except Exception as r_err:
                logger.warning("Failed to reload restored plugin for %s: %s", target_file, r_err)

            # Re-index capability to reflect restored version
            try:
                from zelyr.memory.capability_indexer import index_tool_capabilities
                store = MemoryStore(db_path=db_path)
                index_tool_capabilities(store=store, tools_dir=target_dir)
            except Exception:
                pass

            return ToolResult(
                success=False,
                error=f"Update failed during reload: {err}. Previous tool version was restored.",
            )

        # 5. Supersede capability in MemoryStore
        try:
            from zelyr.memory.capability_indexer import index_tool_capabilities
            store = MemoryStore(db_path=db_path)
            index_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception as e:
            logger.warning("Capability indexing encountered an error for %s: %s", clean_name, e)

        return ToolResult(
            success=True,
            output={
                "tool_name": clean_name,
                "status": "updated_and_reloaded",
                "ring": val_res.ring,
                "tier": getattr(val_res, "tier", 1),
            },
        )

    @classmethod
    def delete_tool(
        cls,
        name: str,
        tools_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
    ) -> ToolResult:
        """Protect built-ins, delete dynamic tool file, unregister runtime schemas, and soft-delete capability."""
        clean_name = (name or "").strip()
        if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
            return ToolResult(
                success=False,
                error=f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
            )

        if clean_name in cls.PROTECTED_TOOLS:
            return ToolResult(
                success=False,
                error=f"Permission denied: Protected tool '{clean_name}' cannot be deleted.",
            )

        target_dir = tools_dir or (get_config_dir() / "tools.d")
        target_file = target_dir / f"{clean_name}.py"

        file_exists = target_file.exists()

        from zelyr.core.plugins import get_tool_schemas, unregister_plugin
        schemas = get_tool_schemas()
        is_registered = any((s.get("function", {}).get("name") or s.get("name")) == clean_name for s in schemas)

        store = MemoryStore(db_path=db_path)
        cap_record = store.get_by_key(f"tool.{clean_name}")

        if not file_exists and not is_registered and not cap_record:
            return ToolResult(
                success=False,
                error=f"Dynamic tool '{clean_name}' not found.",
            )

        # Delete file
        if file_exists:
            try:
                target_file.unlink()
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to delete tool file {target_file}: {e}",
                )

        # Unregister from runtime
        unregister_plugin(clean_name)

        # Soft-delete capability in MemoryStore
        if cap_record:
            store.delete(cap_record["id"], hard_delete=False)

        with store._write_lock:
            with store._get_conn() as conn:
                conn.execute(
                    "UPDATE memories SET is_active = 0, updated_at = ? WHERE key = ? AND is_active = 1",
                    (time.time(), f"tool.{clean_name}"),
                )
                conn.commit()

        try:
            from zelyr.memory.capability_indexer import reconcile_tool_capabilities
            reconcile_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception:
            pass

        return ToolResult(
            success=True,
            output={
                "tool_name": clean_name,
                "status": "deleted_and_unregistered",
            },
        )


# Convenience module-level aliases
def list_dynamic_tools(tools_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    return ToolManager.list_tools(tools_dir=tools_dir)


async def update_dynamic_tool(
    name: str,
    code: str,
    test_params: Optional[Dict[str, Any]] = None,
    tools_dir: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> ToolResult:
    return await ToolManager.update_tool(
        name=name,
        code=code,
        test_params=test_params,
        tools_dir=tools_dir,
        db_path=db_path,
    )


def delete_dynamic_tool(
    name: str,
    tools_dir: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> ToolResult:
    return ToolManager.delete_tool(
        name=name,
        tools_dir=tools_dir,
        db_path=db_path,
    )


list_tools = list_dynamic_tools
update_tool = update_dynamic_tool
delete_tool = delete_dynamic_tool
