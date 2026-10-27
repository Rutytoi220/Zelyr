"""Zelyr Session Memory Promotion Engine.

Extracts high-signal user preferences, environment configurations, and workflow directives
from conversational session history and promotes them to persistent long-term memory.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from zelyr.db.memory import MemoryStore, get_session_messages

logger = logging.getLogger(__name__)

USER_FACT_PATTERNS = [
    # "remember that I prefer zsh", "remember: use python 3.12"
    r"(?:please\s+)?remember(?:\s+that|:\s*|\s+to\s+)?\s+([^\n\r.]+)",
    # "my preference is Tokyo Night", "my shell is zsh"
    r"(?:my\s+(?:preference|favorite|shell|editor|theme|browser|workflow)\s+is)\s+([^\n\r.]+)",
    # "always use git rebase", "never use rm -rf"
    r"\b((?:always|never)\s+use\s+[^\n\r.]+)",
    # "I prefer Neovim over VSCode"
    r"\b(i\s+(?:strongly\s+)?prefer\s+[^\n\r.]+)",
    r"\b(i\s+always\s+(?:use|prefer)\s+[^\n\r.]+)",
    # "note that server runs on port 8080"
    r"\bnote\s+that\s+([^\n\r.]+)",
]


def _categorize_fact(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ("git", "rebase", "merge", "commit", "workflow", "ci", "branch")):
        return "workflow"
    if any(k in lower for k in ("postgres", "mysql", "redis", "docker", "homelab", "nas", "proxmox")):
        return "homelab"
    if any(k in lower for k in ("server", "port", "ip", "ubuntu", "debian", "arch", "linux", "distro", "env")):
        return "environment"
    return "preference"


def promote_session_facts(
    session_id: str,
    store: Optional[MemoryStore] = None,
) -> List[int]:
    """Promote salient facts and directives from a session into persistent memory.

    Scans both user requests and assistant tool outputs.
    Automatically categorizes, avoids exact duplicates, and performs contradiction superseding.

    Returns list of newly created or superseded memory IDs.
    """
    if not session_id:
        return []

    mem_store = store or MemoryStore()
    messages = get_session_messages(session_id, db_path=mem_store.db_path)
    if not messages:
        return []

    candidates: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "user":
            for pat in USER_FACT_PATTERNS:
                for match in re.finditer(pat, content, re.IGNORECASE):
                    extracted = match.group(0).strip()
                    # Strip leading directive words if needed for cleaner storage
                    clean = re.sub(r"^(?:please\s+)?remember(?:\s+that|:\s*|\s+to\s+)?\s*", "", extracted, flags=re.IGNORECASE).strip()
                    if not clean:
                        clean = extracted

                    if len(clean) >= 6 and clean.lower() not in seen_texts:
                        seen_texts.add(clean.lower())
                        candidates.append({
                            "content": clean,
                            "category": _categorize_fact(clean),
                            "key": None,
                        })

        elif role == "assistant":
            # Check for tool call payloads or mentions of remember_fact
            if "remember_fact" in content:
                # Try finding JSON function call parameters
                for json_match in re.finditer(r'\{[^{}]*"fact"\s*:\s*"([^"]+)"[^{}]*\}', content):
                    try:
                        parsed = json.loads(json_match.group(0))
                        fact_val = parsed.get("fact") or parsed.get("content")
                        if fact_val and fact_val.lower() not in seen_texts:
                            seen_texts.add(fact_val.lower())
                            candidates.append({
                                "content": fact_val.strip(),
                                "category": parsed.get("category") or _categorize_fact(fact_val),
                                "key": parsed.get("key"),
                            })
                    except Exception:
                        pass

    promoted_ids: List[int] = []

    for item in candidates:
        text = item["content"]
        cat = item["category"]
        key = item.get("key")

        # Skip if exact text already exists actively in memory store
        try:
            with mem_store._get_conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND is_active = 1",
                    (text,),
                ).fetchone()
                if existing:
                    continue
        except Exception:
            pass

        try:
            new_id = mem_store.store_fact(content=text, category=cat, key=key)
            promoted_ids.append(new_id)
        except Exception as e:
            logger.warning("Failed to promote fact '%s': %s", text, e)

    return promoted_ids
