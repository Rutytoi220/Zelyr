"""Deterministic observation framing and semantic state anchors for tool execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


class EvidenceLevel(str, Enum):
    """Epistemic reliability classification for tool and execution observations."""

    CONFIRMED = "CONFIRMED"
    REPORTED = "REPORTED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    CONTRADICTED = "CONTRADICTED"


class ObservationExecutionStatus(str, Enum):
    """Distinguished execution lifecycle status exposed to planner and recovery."""

    SUCCEEDED = "succeeded"
    SUCCEEDED_VERIFIED = "succeeded_verified"
    SUCCEEDED_UNVERIFIED = "succeeded_unverified"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    EXECUTED_ERROR = "executed_error"
    FAILED_BEFORE_EXECUTION = "failed_before_execution"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    MALFORMED = "malformed"
    EMPTY = "empty"


_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"(?i)\[\s*(?:system\s*(?::|prompt|override|instruction|message|directive|notice|role)"
        r"|developer\s*(?::|message|override)|assistant\s*(?::|message)|admin\s+override|instruction\s*:)"
        r"[^\]]*\]"
    ),
    re.compile(r"(?i)<\|\s*(?:im_start|im_end|system|assistant|user)\s*\|>"),
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
    re.compile(r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
)

_ANCHOR_SPOOF_PATTERNS = (
    (re.compile(r"(?i)\[TOOL OBSERVATION:"), "[untrusted_output: TOOL OBSERVATION:"),
    (re.compile(r"(?i)\[Machine State Verified\]:"), "[untrusted_output: Machine State Verified]:"),
    (re.compile(r"(?i)\[Outcome Verification Failed\]:"), "[untrusted_output: Outcome Verification Failed]:"),
    (re.compile(r"(?i)\[Error Diagnostic:"), "[untrusted_output: Error Diagnostic:"),
    (re.compile(r"(?im)(^|\s)(-\s*)Verified:\s*(TRUE|FALSE)\b"), r"\1\2claimed Verified: \3"),
    (re.compile(r"(?im)(^|\s)(-\s*)Status:\s*(SUCCESS|ERROR)\b"), r"\1\2claimed Status: \3"),
)

_BULKY_OBSERVATION_KEYS = frozenset(
    {
        "image_b64",
        "_vision_b64",
        "screenshot_b64",
        "raw_bytes",
        "binary_data",
        "dom_snapshot",
        "raw_dom",
        "page_source",
    }
)

_SHELL_TOOLS = frozenset({"execute_command", "shell", "bash", "sh", "run_command", "cmd"})
_FILE_TOOLS = frozenset(
    {
        "manage_workspace_file",
        "file_write",
        "file_read",
        "write_file",
        "read_file",
        "replace_file_content",
        "multi_replace_file_content",
        "file_search",
        "safe_file_search",
        "file_opener",
        "read_document_content",
    }
)
_BROWSER_TOOLS = frozenset({"interact_with_browser", "playwright_web"})
_UI_VLA_TOOLS = frozenset({"interact_with_ui", "gui_actuate", "gui_inspect"})


def _redact_observation_text(text: str) -> Tuple[str, bool]:
    """Scrub credentials, API tokens, and secret env vars using shared ledger scrubber."""
    if not text:
        return "", False
    try:
        from zelyr.engine.execution_ledger import _redact_sensitive_text

        scrubbed = _redact_sensitive_text(text) or ""
    except Exception:
        scrubbed = re.sub(r"(?i)(https?://)([^:/@\s]+):([^/@\s]+)@", r"\1\2:[REDACTED]@", text)
        scrubbed = re.sub(r"(?i)\b(bearer\s+)[a-z0-9._\-+=]+", r"\1[REDACTED]", scrubbed)
        scrubbed = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", scrubbed)
        scrubbed = re.sub(
            r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization|cookie)\s*([=:])\s*['\"]?([^\s,'\"&;]+)['\"]?",
            r"\1\2[REDACTED]",
            scrubbed,
        )
    return scrubbed, (scrubbed != text)


def _sanitize_untrusted_text(
    text: Any,
    *,
    is_external: bool = False,
    escape_anchors: bool = True,
    max_chars: int = 8192,
) -> Tuple[str, bool, bool]:
    """Redact secrets, strip control/injection payloads, escape anchors, and bound text.

    Returns:
        ``(cleaned_text, redacted, truncated)``
    """
    if text is None:
        return "", False, False
    raw = text if isinstance(text, str) else str(text)
    if not raw:
        return "", False, False

    redacted = False
    truncated = False

    cleaned, was_secret_red = _redact_observation_text(raw)
    if was_secret_red:
        redacted = True

    stripped_ctrl = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", cleaned)
    stripped_ctrl = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", stripped_ctrl)
    if stripped_ctrl != cleaned:
        cleaned = stripped_ctrl
        redacted = True

    if is_external:
        try:
            from zelyr.tools.external import sanitize_external_text

            ext_clean = sanitize_external_text(cleaned, max_len=max(max_chars, 512))
            if ext_clean != cleaned:
                cleaned = ext_clean
                redacted = True
        except Exception:
            pass

    for pat in _PROMPT_INJECTION_PATTERNS:
        if pat.search(cleaned):
            cleaned = pat.sub("[FILTERED_UNTRUSTED_INSTRUCTION]", cleaned)
            redacted = True

    if escape_anchors:
        for anchor_pat, anchor_repl in _ANCHOR_SPOOF_PATTERNS:
            if anchor_pat.search(cleaned):
                cleaned = anchor_pat.sub(anchor_repl, cleaned)
                redacted = True

    if max_chars > 0 and len(cleaned) > max_chars:
        truncated = True
        half = max(64, (max_chars - 96) // 2)
        omitted = len(cleaned) - (2 * half)
        if omitted > 0:
            cleaned = f"{cleaned[:half]}\n\n[... {omitted} chars omitted to bound observation size ...]\n\n{cleaned[-half:]}"
        else:
            cleaned = cleaned[:max_chars]

    return cleaned, redacted, truncated


def _sanitize_structured_dict(
    data: Any,
    *,
    is_external: bool = False,
    depth: int = 0,
    max_str_chars: int = 1024,
) -> Tuple[Any, bool, bool]:
    """Sanitize arbitrary dict/list payloads for safe structured observation storage."""
    if data is None or isinstance(data, (bool, int, float)):
        return data, False, False
    if isinstance(data, (bytes, bytearray)):
        return {"omitted": True, "reason": "binary_payload", "size": len(data)}, False, True
    if isinstance(data, str):
        return _sanitize_untrusted_text(
            data,
            is_external=is_external,
            escape_anchors=True,
            max_chars=max_str_chars,
        )
    if depth >= 4:
        clipped, was_red, _ = _sanitize_untrusted_text(str(data), is_external=is_external, max_chars=256)
        return f"{clipped} [TRUNCATED]", was_red, True

    red = False
    trunc = False
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        items = list(data.items())
        if len(items) > 32:
            items = items[:32]
            trunc = True
        for k, v in items:
            key_str = str(k)
            low_k = key_str.lower()
            if any(tok in low_k for tok in ("api_key", "apikey", "token", "secret", "password", "passwd", "authorization", "cookie", "private_key", "credential")):
                out[key_str] = "[REDACTED]"
                red = True
                continue
            if low_k in _BULKY_OBSERVATION_KEYS:
                sz = len(v) if isinstance(v, (str, bytes, bytearray, list, dict)) else None
                out[key_str] = {"omitted": True, "reason": "large_payload", **({"size": sz} if sz is not None else {})}
                trunc = True
                continue
            clean_v, v_red, v_trunc = _sanitize_structured_dict(
                v,
                is_external=is_external,
                depth=depth + 1,
                max_str_chars=max_str_chars,
            )
            red = red or v_red
            trunc = trunc or v_trunc
            out[key_str] = clean_v
        if len(data) > 32:
            out["_omitted_keys"] = len(data) - 32
        return out, red, trunc

    if isinstance(data, (list, tuple, set)):
        seq = list(data)
        if len(seq) > 24:
            seq = seq[:24]
            trunc = True
        out_l: List[Any] = []
        for item in seq:
            clean_i, i_red, i_trunc = _sanitize_structured_dict(
                item,
                is_external=is_external,
                depth=depth + 1,
                max_str_chars=max_str_chars,
            )
            red = red or i_red
            trunc = trunc or i_trunc
            out_l.append(clean_i)
        if len(data) > 24:
            out_l.append({"_omitted_items": len(data) - 24})
        return out_l, red, trunc

    return _sanitize_untrusted_text(str(data), is_external=is_external, max_chars=max_str_chars)


@dataclass
class StructuredObservation:
    """Canonical structured, evidence-backed observation produced for planner and ledger."""

    observation_id: str
    source_tool_id: str
    execution_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    external_source: Optional[Dict[str, Any]] = None
    operation_attempted: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = ObservationExecutionStatus.SUCCEEDED_UNVERIFIED.value
    structured_result: Dict[str, Any] = field(default_factory=dict)
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    evidence_level: str = EvidenceLevel.REPORTED.value
    remaining_uncertainty: Optional[str] = None
    error_classification: Optional[Dict[str, Any]] = None
    verification_status: str = "not_verified"
    independent_verification: Optional[Dict[str, Any]] = None
    recovery_guidance: Dict[str, Any] = field(default_factory=dict)
    suggested_next_action: Optional[str] = None
    ledger_refs: Dict[str, Any] = field(default_factory=dict)
    redacted: bool = False
    truncated: bool = False

    def content_digest(self) -> str:
        """Stable content hash used by ledger deduplication when observation updates arrive."""
        payload = {
            "source_tool_id": self.source_tool_id,
            "execution_status": self.execution_status,
            "evidence_level": self.evidence_level,
            "verification_status": self.verification_status,
            "structured_result": self.structured_result,
            "state_changes": self.state_changes,
            "independent_verification": self.independent_verification,
            "remaining_uncertainty": self.remaining_uncertainty,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete structured observation to JSON-safe dictionary."""
        return asdict(self)

    def to_compact_dict(self, max_chars: int = 1200) -> Dict[str, Any]:
        """Produce compact planner dictionary preserving failures, evidence, uncertainty & refs."""
        res = dict(self.structured_result)
        preview = str(res.get("output_preview") or "")
        preview_budget = max(80, min(280, max_chars // 4))
        if len(preview) > preview_budget:
            preview = preview[:preview_budget] + "... [TRUNCATED]"
        compact: Dict[str, Any] = {
            "observation_id": self.observation_id,
            "execution_id": self.execution_id,
            "correlation_id": self.correlation_id,
            "tool": self.source_tool_id,
            "execution_status": self.execution_status,
            "evidence_level": self.evidence_level,
            "verification_status": self.verification_status,
            "attempted": {
                "requested_tool": self.operation_attempted.get("requested_tool") or self.source_tool_id,
                "executed_tool": self.operation_attempted.get("executed_tool") or self.source_tool_id,
                "action": self.operation_attempted.get("requested_action"),
                "dispatched": bool(self.operation_attempted.get("dispatched", True)),
            },
            "reported": {
                "success": res.get("tool_reported_success"),
                "exit_code": res.get("exit_code"),
                "summary": res.get("summary"),
                "preview": preview,
            },
            "state_changes": list(self.state_changes[:4]),
            "remaining_uncertainty": self.remaining_uncertainty,
            "suggested_next_action": self.suggested_next_action,
            "redacted": bool(self.redacted),
            "truncated": bool(self.truncated or len(str(res.get("output_preview") or "")) > preview_budget),
        }
        if self.external_source:
            compact["external_source"] = dict(self.external_source)
        if self.independent_verification:
            compact["verified"] = dict(self.independent_verification)
        if self.error_classification:
            compact["error_classification"] = dict(self.error_classification)
        if self.ledger_refs:
            compact["ledger_refs"] = dict(self.ledger_refs)
        return compact

    def to_compact_planner_context(self, max_chars: int = 1200) -> str:
        """Render a concise, failure-preserving structured summary for planner context."""
        res = self.structured_result
        req_t = self.operation_attempted.get("requested_tool") or self.source_tool_id
        exec_t = self.operation_attempted.get("executed_tool") or self.source_tool_id
        dispatched = bool(self.operation_attempted.get("dispatched", True))
        req_str = f"requested={req_t}" + (f" -> executed={exec_t}" if req_t != exec_t else "") + f" (dispatched={dispatched})"

        lines: List[str] = [
            f"[OBSERVATION {self.source_tool_id} | status={self.execution_status} | evidence={self.evidence_level} | verified={self.verification_status}]",
            f"- Operation: {req_str}",
        ]
        if self.external_source and self.external_source.get("is_external"):
            lines.append(
                f"- External Source: {self.external_source.get('source_id')} ({self.external_source.get('source_type')}: {self.external_source.get('external_tool_name')})"
            )
        if res.get("summary"):
            lines.append(f"- Reported: {str(res['summary'])[:240]}")
        if res.get("error"):
            err_t = (self.error_classification or {}).get("error_type") or "ERROR"
            lines.append(f"- Error ({err_t}): {str(res['error'])[:260]}")
        if self.independent_verification:
            lines.append(
                f"- Independent Verification: passed={self.independent_verification.get('passed')} "
                f"check={self.independent_verification.get('check')} target={self.independent_verification.get('target')} "
                f"observed={str(self.independent_verification.get('observed', ''))[:160]}"
            )
        if self.state_changes:
            sc_str = "; ".join(
                f"{sc.get('change', 'mutated')}:{sc.get('target', '?')}(verified={bool(sc.get('verified', False))})"
                for sc in self.state_changes[:3]
            )
            lines.append(f"- State Changes: {sc_str}")
        if self.remaining_uncertainty:
            lines.append(f"- Uncertainty: {str(self.remaining_uncertainty)[:220]}")
        if self.recovery_guidance.get("may_have_side_effect"):
            lines.append("- Side-Effect Caution: Prior execution was dispatched and may already have modified state.")
        if self.suggested_next_action:
            lines.append(f"- Next Action: {str(self.suggested_next_action)[:240]}")
        exec_ref = self.ledger_refs.get("execution_id") or self.execution_id
        seqs = self.ledger_refs.get("sequence_numbers") or []
        if exec_ref:
            seq_txt = f" seq={seqs[-1]}" if seqs else ""
            lines.append(f"- Ledger Ref: {exec_ref}{seq_txt} ({self.observation_id})")

        preview = str(res.get("output_preview") or "").strip()
        if preview and not res.get("error"):
            joined_len = sum(len(l) + 1 for l in lines)
            rem = max_chars - joined_len - 32
            if rem > 40:
                clipped_p = preview[:rem] + ("... [TRUNCATED]" if len(preview) > rem else "")
                lines.append(f"- Result Preview: {clipped_p}")

        rendered = "\n".join(lines)
        if len(rendered) > max_chars > 32:
            # Keep critical lines (header, error, uncertainty, next action) and clip middle/longest preview
            rendered = rendered[: max_chars - 17] + "\n... [TRUNCATED]"
        return rendered


class ObservationFramingEngine:
    """Formats, frames, verifies, and parses tool execution observations."""

    MAX_FRAMED_STREAM_CHARS = 65536

    @classmethod
    def _bound_stream_text(cls, text: str) -> str:
        """Bound oversized output/stderr streams without dropping headers or error diagnostics."""
        limit = cls.MAX_FRAMED_STREAM_CHARS
        if len(text) <= limit:
            return text
        half = (limit - 128) // 2
        omitted = len(text) - (2 * half)
        return f"{text[:half]}\n\n[... {omitted} chars omitted to bound observation size ...]\n\n{text[-half:]}"

    @classmethod
    def _extract_external_source(
        cls,
        tool_name: str,
        parsed: Optional[Dict[str, Any]],
        external_source: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Normalize external/MCP source provenance metadata if present."""
        cand_meta: Optional[Dict[str, Any]] = None
        if isinstance(external_source, dict):
            cand_meta = dict(external_source)
        elif isinstance(parsed, dict):
            res_block = parsed.get("result") if isinstance(parsed.get("result"), dict) else None
            if res_block and isinstance(res_block.get("metadata"), dict):
                cand_meta = res_block["metadata"]
            elif isinstance(parsed.get("metadata"), dict):
                cand_meta = parsed["metadata"]
            elif parsed.get("external") or parsed.get("is_external") or parsed.get("source_id"):
                cand_meta = parsed

        if not cand_meta:
            return None

        is_ext = bool(
            cand_meta.get("external") is True
            or cand_meta.get("is_external") is True
            or str(cand_meta.get("source_type") or "").lower().startswith(("mcp", "external"))
        )
        if not is_ext and not cand_meta.get("source_id"):
            return None

        return {
            "is_external": True,
            "source_id": str(cand_meta.get("source_id") or "external"),
            "source_type": str(cand_meta.get("source_type") or "external"),
            "external_tool_name": str(cand_meta.get("external_tool_name") or tool_name),
        }

    @classmethod
    def _extract_domain_details(
        cls,
        tool_name: str,
        parsed: Optional[Dict[str, Any]],
        output: str,
        stderr: str,
        is_error: bool,
        ext_source_meta: Optional[Dict[str, Any]],
        executed_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        """Extract tool-domain structured fields, state changes, evidence, and uncertainty notes."""
        fields: Dict[str, Any] = {}
        state_changes: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        uncertainty_note: Optional[str] = None
        args_dict = dict(executed_args) if isinstance(executed_args, dict) else {}

        res_inner = (
            parsed.get("result")
            if isinstance(parsed, dict) and isinstance(parsed.get("result"), dict)
            else None
        )
        out_dict = parsed.get("output") if isinstance(parsed, dict) and isinstance(parsed.get("output"), dict) else None
        if out_dict is None and res_inner and isinstance(res_inner.get("output"), dict):
            out_dict = res_inner["output"]
        if out_dict is None and res_inner and not isinstance(res_inner.get("output"), str):
            out_dict = res_inner
        payload_dict = out_dict if isinstance(out_dict, dict) else (parsed if isinstance(parsed, dict) else {})

        low_tool = (tool_name or "").lower()

        if low_tool in _SHELL_TOOLS or "command" in low_tool or "shell" in low_tool or "bash" in low_tool:
            fields["domain"] = "shell"
            if args_dict.get("command"):
                cmd_clean, _, _ = _sanitize_untrusted_text(str(args_dict["command"]), escape_anchors=False, max_chars=256)
                fields["command"] = cmd_clean
            if stderr:
                fields["has_stderr"] = True
            if not is_error:
                evidence.append({"source": "shell_exit", "status": "reported_exit_zero"})
                uncertainty_note = "Command reported exit code 0, but host side effects were not independently verified."
        elif low_tool in _FILE_TOOLS or low_tool.startswith("file_"):
            fields["domain"] = "file"
            action = str(
                payload_dict.get("action")
                or args_dict.get("action")
                or ("read" if "read" in low_tool else "write" if "write" in low_tool else "")
            ).strip()
            target_path = (
                payload_dict.get("restored_path")
                or payload_dict.get("path")
                or args_dict.get("path")
                or args_dict.get("file_path")
                or args_dict.get("target_file")
            )
            if action:
                fields["action"] = action
            if target_path:
                fields["path"] = str(target_path)
            for k in ("bytes", "bytes_written", "occurrences_replaced", "count", "backup", "backup_source", "restored_path"):
                if k in payload_dict and payload_dict[k] is not None:
                    fields[k] = payload_dict[k]
            mutating_actions = {"write", "create", "patch", "edit", "replace", "rollback", "delete", "remove", "mkdir"}
            is_file_mut = (
                action.lower() in mutating_actions
                or any(tok in low_tool for tok in ("write", "replace", "patch", "rollback", "delete"))
            )
            if not is_error and is_file_mut and target_path:
                occ = payload_dict.get("occurrences_replaced")
                if occ == 0:
                    uncertainty_note = f"Patch on '{target_path}' reported 0 replacements; file contents may be unchanged."
                else:
                    state_changes.append({"target": str(target_path), "change": action or "file_mutated", "verified": False})
                    evidence.append({"source": "tool_report", "target": str(target_path), "kind": "file_operation"})
                    uncertainty_note = f"Tool reported file operation on '{target_path}', pending independent verification."
        elif low_tool in _BROWSER_TOOLS or "browser" in low_tool:
            fields["domain"] = "browser"
            for k in ("action", "status", "url", "current_url", "title", "selector", "screenshot_path", "error_type"):
                val = payload_dict.get(k) or args_dict.get(k)
                if val is not None:
                    fields[k] = val
            if isinstance(payload_dict.get("tabs"), list):
                fields["tabs_count"] = len(payload_dict["tabs"])
            url_target = payload_dict.get("url") or payload_dict.get("current_url") or args_dict.get("url")
            act_b = str(fields.get("action") or "")
            if not is_error and url_target and act_b.lower() in ("navigate", "goto", "click", "type", "fill", "submit", "open_tab", "close_tab"):
                state_changes.append({"target": str(url_target), "change": act_b or "browser_action", "verified": False})
            if not is_error and fields.get("status") == "partial":
                uncertainty_note = "Browser reported partial completion; target page/DOM state remains uncertain."
        elif low_tool in _UI_VLA_TOOLS or "ui" == low_tool:
            fields["domain"] = "vla_ui"
            for k in (
                "action_taken",
                "action",
                "target_description",
                "target",
                "coordinates",
                "screen_changed",
                "post_action_verified",
                "confidence",
                "fallback_used",
                "error_type",
            ):
                val = payload_dict.get(k) or args_dict.get(k)
                if val is not None:
                    fields[k] = val
            ui_tgt = str(fields.get("target_description") or fields.get("target") or "desktop_ui")
            if not is_error:
                state_changes.append({"target": ui_tgt, "change": str(fields.get("action_taken") or fields.get("action") or "ui_actuation"), "verified": bool(payload_dict.get("post_action_verified") is True)})
                if payload_dict.get("screen_changed") is False or payload_dict.get("post_action_verified") is False:
                    uncertainty_note = f"VLA action on '{ui_tgt}' executed, but post-action UI state change was NOT confirmed."
                elif payload_dict.get("post_action_verified") is None:
                    uncertainty_note = f"VLA action on '{ui_tgt}' executed without independent post-action visual confirmation."
        elif ext_source_meta is not None:
            fields["domain"] = "external"
            evidence.append({
                "source": "external_tool",
                "source_id": ext_source_meta.get("source_id"),
                "source_type": ext_source_meta.get("source_type"),
                "trusted": False,
            })
            if not is_error:
                uncertainty_note = (
                    f"Output reported by external source '{ext_source_meta.get('source_id')}'; "
                    "treat as untrusted/unverified unless independently confirmed."
                )

        if not evidence and not is_error and (output or fields):
            evidence.append({"source": "tool_report", "tool": tool_name})

        return fields, state_changes, evidence, uncertainty_note

    @classmethod
    def _coerce_independent_verification(cls, raw_ver: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize independent verification payloads into a consistent dict."""
        if not isinstance(raw_ver, dict):
            return None
        passed_val = raw_ver.get("passed")
        if passed_val is None and "success" in raw_ver:
            passed_val = raw_ver.get("success")
        if passed_val is None and "verified" in raw_ver:
            passed_val = raw_ver.get("verified")
        if not isinstance(passed_val, bool):
            return None

        obs_clean, _, _ = _sanitize_untrusted_text(raw_ver.get("observed", ""), escape_anchors=True, max_chars=512)
        err_clean, _, _ = _sanitize_untrusted_text(raw_ver.get("error", ""), escape_anchors=True, max_chars=512)
        return {
            "passed": passed_val,
            "check": str(raw_ver.get("check") or "independent_check"),
            "target": str(raw_ver.get("target") or ""),
            "observed": obs_clean,
            "error": err_clean or None,
        }

    @classmethod
    def derive_recovery_guidance(
        cls,
        observation: StructuredObservation,
        classification: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute safe recovery guidance enforcing non-retryability and side-effect awareness."""
        exec_status = observation.execution_status
        tool_name = observation.source_tool_id
        dispatched = bool(observation.operation_attempted.get("dispatched", True))
        res = observation.structured_result
        exit_code = res.get("exit_code")
        err_msg = str(res.get("error") or "")
        out_msg = str(res.get("output_preview") or "")
        err_type = str((observation.error_classification or {}).get("error_type") or "")

        if exec_status in (ObservationExecutionStatus.SUCCEEDED.value, ObservationExecutionStatus.SUCCEEDED_VERIFIED.value):
            return {
                "is_recoverable": False,
                "should_auto_retry": False,
                "may_have_side_effect": False,
                "action": "proceed",
                "next_step": "Continue workflow; operation completed.",
            }

        if exec_status == ObservationExecutionStatus.SUCCEEDED_UNVERIFIED.value:
            return {
                "is_recoverable": False,
                "should_auto_retry": False,
                "may_have_side_effect": bool(observation.state_changes or tool_name in (_SHELL_TOOLS | _UI_VLA_TOOLS | _BROWSER_TOOLS)),
                "action": "verify_or_proceed",
                "next_step": "Do not repeat identical parameters; run independent verification if state confirmation is required.",
            }

        if exec_status == ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value:
            return {
                "is_recoverable": True,
                "should_auto_retry": False,
                "may_have_side_effect": dispatched,
                "action": "inspect_and_replan",
                "next_step": "Tool report contradicted verification or host evidence; inspect actual host state before re-acting.",
            }

        if classification is None and (err_msg or exit_code not in (None, 0) or err_type):
            try:
                from zelyr.engine.error_classifier import classify_tool_error

                classification = classify_tool_error(
                    tool_name,
                    exit_code,
                    out_msg,
                    err_msg,
                    error_type=err_type or None,
                )
            except Exception:
                classification = None

        eff_err_type = (
            getattr(classification, "error_type", None)
            or err_type
            or ("COMMAND_TIMEOUT" if exec_status == ObservationExecutionStatus.TIMED_OUT.value else "TOOL_EXECUTION_FAILED")
        )
        non_recoverable_types = {
            "TOOL_NOT_FOUND",
            "AUTHORIZATION_DENIED",
            "CANCELLED",
            "TOOL_UNAVAILABLE",
            "DESKTOP_UNAVAILABLE",
            "BROWSER_UNAVAILABLE",
            "PROHIBITED_BY_USER",
            "CYCLE_DETECTED",
            "PERMISSION_DENIED",
        }
        is_recoverable = bool(getattr(classification, "is_recoverable", True))
        if eff_err_type in non_recoverable_types or exec_status == ObservationExecutionStatus.CANCELLED.value:
            is_recoverable = False

        if not dispatched or exec_status == ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value:
            may_have_side_effect = False
            if eff_err_type in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR"):
                return {
                    "is_recoverable": True,
                    "should_auto_retry": True,
                    "may_have_side_effect": False,
                    "action": "fix_arguments_and_reinvoke",
                    "next_step": str(
                        (observation.error_classification or {}).get("remedy_hint")
                        or getattr(classification, "remedy_hint", None)
                        or "Fix tool arguments to match schema and retry."
                    ),
                }
            return {
                "is_recoverable": False,
                "should_auto_retry": False,
                "may_have_side_effect": False,
                "action": "stop_and_report",
                "next_step": f"Blocked before execution ({eff_err_type}); do not retry without resolving authorization/availability.",
            }

        low_tool = (tool_name or "").lower()
        act_name = str(observation.operation_attempted.get("requested_action") or "").lower()
        readonly_file_actions = {"read", "list", "list_tree", "search", "grep", "inspect", "status", "get_state"}
        is_readonly = (
            low_tool in {"file_read", "read_file", "file_search", "safe_file_search", "read_document_content", "verify_outcome"}
            or act_name in readonly_file_actions
        )
        may_have_side_effect = bool(
            (not is_readonly and (low_tool in _SHELL_TOOLS or low_tool in _FILE_TOOLS or low_tool in _BROWSER_TOOLS or low_tool in _UI_VLA_TOOLS or observation.external_source))
            or exec_status in (ObservationExecutionStatus.TIMED_OUT.value, ObservationExecutionStatus.PARTIALLY_SUCCEEDED.value)
            or observation.state_changes
        )

        if not is_recoverable:
            return {
                "is_recoverable": False,
                "should_auto_retry": False,
                "may_have_side_effect": may_have_side_effect,
                "action": "stop_and_report",
                "next_step": str(
                    (observation.error_classification or {}).get("remedy_hint")
                    or getattr(classification, "remedy_hint", None)
                    or f"Non-recoverable condition ({eff_err_type}); report limitation truthfully."
                ),
            }

        if may_have_side_effect:
            return {
                "is_recoverable": True,
                "should_auto_retry": False,
                "may_have_side_effect": True,
                "action": "inspect_state_before_retry",
                "next_step": "Inspect host state before retrying; prior execution was dispatched and may already have modified state.",
            }

        return {
            "is_recoverable": True,
            "should_auto_retry": True,
            "may_have_side_effect": False,
            "action": "retry_with_adjustment",
            "next_step": str(
                (observation.error_classification or {}).get("remedy_hint")
                or getattr(classification, "remedy_hint", None)
                or "Retry with adjusted parameters."
            ),
        }

    @classmethod
    def frame_structured(
        cls,
        tool_name: str,
        result: Any,
        *,
        requested_tool: Optional[str] = None,
        requested_args: Optional[Dict[str, Any]] = None,
        executed_tool: Optional[str] = None,
        executed_args: Optional[Dict[str, Any]] = None,
        dispatched: Optional[bool] = None,
        timed_out: bool = False,
        cancelled: bool = False,
        partial: bool = False,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        external_source: Optional[Dict[str, Any]] = None,
        independent_verification: Optional[Dict[str, Any]] = None,
        ledger_refs: Optional[Dict[str, Any]] = None,
        raw_text_limit: int = 8192,
    ) -> StructuredObservation:
        """Build a canonical StructuredObservation separating execution, evidence, and uncertainty."""
        orig_metadata = getattr(result, "metadata", None)
        orig_err_type = getattr(result, "error_type", None)

        unpacked = result
        if hasattr(result, "to_dict"):
            try:
                unpacked = result.to_dict(tool=tool_name)
            except TypeError:
                unpacked = result.to_dict()

        parsed: Optional[Dict[str, Any]] = None
        parsed_list: Optional[List[Any]] = None
        raw_str = ""
        malformed = False

        if isinstance(unpacked, dict):
            parsed = unpacked
            raw_str = json.dumps(unpacked, default=str)
        elif isinstance(unpacked, list):
            parsed_list = unpacked
            raw_str = json.dumps(unpacked, default=str)
        elif isinstance(unpacked, str):
            raw_str = unpacked.strip()
            if raw_str.startswith("{") or raw_str.startswith("["):
                try:
                    loaded = json.loads(raw_str)
                    if isinstance(loaded, dict):
                        parsed = loaded
                    elif isinstance(loaded, list):
                        parsed_list = loaded
                except Exception:
                    malformed = True
        elif unpacked is None:
            raw_str = ""
        else:
            raw_str = str(unpacked)

        ext_meta = cls._extract_external_source(tool_name, parsed, external_source=external_source)
        is_ext = bool(ext_meta and ext_meta.get("is_external"))

        exit_code: Optional[int] = None
        is_error = False
        output = ""
        stderr = ""
        error = ""
        error_type = str(orig_err_type or "")
        remedy_hint = ""
        allowed_actions: List[str] = []
        verified: Optional[bool] = None
        summary = ""
        is_partial_flag = bool(partial)
        is_timeout_flag = bool(timed_out)
        is_cancel_flag = bool(cancelled)
        has_contradiction = False

        if parsed is not None:
            if (
                parsed.get("error_type") in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR")
                or (isinstance(parsed.get("result"), dict) and parsed["result"].get("error_type") in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR"))
                or (isinstance(parsed.get("output"), dict) and parsed["output"].get("error_type") in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR"))
            ):
                env = (
                    parsed
                    if parsed.get("error_type") in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR")
                    else (
                        parsed["result"]
                        if isinstance(parsed.get("result"), dict) and parsed["result"].get("error_type") in ("SCHEMA_VALIDATION_ERROR", "ARGUMENT_NORMALIZATION_ERROR")
                        else parsed["output"]
                    )
                )
                invalid_param = env.get("invalid_parameter") or ""
                error_type = str(env.get("error_type") or "SCHEMA_VALIDATION_ERROR")
                error = str(env.get("error") or f"Validation failed for '{invalid_param}'")
                remedy_hint = str(env.get("remedy_hint") or "")
                is_error = True
                exit_code = 1
                dispatched = False if dispatched is None else dispatched
                summary = f"Pre-execution validation failed for '{invalid_param or tool_name}'."
            else:
                res_block = parsed.get("result") if isinstance(parsed.get("result"), dict) else None
                status_val = str(parsed.get("status") or (res_block.get("status") if res_block else "") or "").strip().lower()
                if status_val in ("timed_out", "timeout") or parsed.get("timed_out") is True or (isinstance(orig_metadata, dict) and orig_metadata.get("timed_out") is True):
                    is_timeout_flag = True
                if status_val in ("cancelled", "canceled") or str(parsed.get("reason") or "").lower() == "cancelled":
                    is_cancel_flag = True
                if status_val in ("partial", "partially_succeeded") or parsed.get("partial") is True or (res_block and res_block.get("partial") is True):
                    is_partial_flag = True
                if parsed.get("dispatched") is False or status_val == "rejected":
                    dispatched = False if dispatched is None else dispatched

                success_flag = parsed.get("success")
                err_candidate = parsed.get("error") or (res_block.get("error") if res_block else None)
                if "exit_code" in parsed and parsed["exit_code"] is not None:
                    try:
                        exit_code = int(parsed["exit_code"])
                    except Exception:
                        exit_code = 1
                        malformed = True
                elif "returncode" in parsed and parsed["returncode"] is not None:
                    try:
                        exit_code = int(parsed["returncode"])
                    except Exception:
                        exit_code = 1
                        malformed = True

                # Contradictory raw signals inside tool response itself
                if success_flag is True and (err_candidate or (exit_code is not None and exit_code != 0)):
                    has_contradiction = True

                is_error = (
                    success_flag is False
                    or status_val in ("error", "failed", "failure", "rejected", "tool_execution_failed", "timed_out", "timeout", "cancelled", "canceled")
                    or bool(err_candidate)
                    or (exit_code is not None and exit_code != 0)
                )

                if exit_code is None:
                    if is_error:
                        exit_code = 1
                    elif tool_name in _SHELL_TOOLS or "command" in tool_name or "exec" in tool_name or "bash" in tool_name:
                        exit_code = 0

                if "output" in parsed:
                    out_val = parsed["output"]
                elif "stdout" in parsed:
                    out_val = parsed["stdout"]
                elif res_block is not None and "output" in res_block:
                    out_val = res_block["output"]
                elif res_block is not None:
                    out_val = "" if (is_error and "error" in res_block) else res_block
                elif "result" in parsed:
                    out_val = parsed["result"]
                else:
                    out_val = "" if is_error else raw_str

                output = json.dumps(out_val, default=str) if isinstance(out_val, (dict, list)) else str(out_val if out_val is not None else "")
                stderr = str(parsed.get("stderr") or (res_block.get("stderr") if res_block else "") or "")
                if err_candidate:
                    error = str(err_candidate)
                elif is_error:
                    error = stderr or f"Tool '{tool_name}' failed"

                error_type = str(
                    parsed.get("error_type")
                    or (res_block.get("error_type") if res_block else "")
                    or error_type
                    or ""
                )
                if parsed.get("reason") and not error_type:
                    error_type = str(parsed["reason"]).upper()
                remedy_hint = str(parsed.get("remedy_hint") or (res_block.get("remedy_hint") if res_block else "") or "")
                if isinstance(parsed.get("allowed_actions"), list):
                    allowed_actions = list(parsed["allowed_actions"])
                elif res_block and isinstance(res_block.get("allowed_actions"), list):
                    allowed_actions = list(res_block["allowed_actions"])

                raw_ver = parsed.get("verified") if "verified" in parsed else (res_block.get("verified") if res_block else None)
                if isinstance(raw_ver, bool):
                    if raw_ver is True and not is_error:
                        verified = True
                    elif raw_ver is True and is_error:
                        has_contradiction = True
                        verified = False
                    elif raw_ver is False:
                        is_legacy_def = (
                            "state" in parsed
                            and parsed.get("verification_failed") is not True
                            and not any(k in parsed for k in ("failed_criterion", "verifier_reason", "required_correction"))
                        )
                        verified = None if (not is_error and is_legacy_def) else False
                if parsed.get("verification_failed") is True and verified is True:
                    has_contradiction = True
                    verified = False
                elif parsed.get("verification_failed") is True and not is_error:
                    verified = False
                if parsed.get("summary"):
                    summary = str(parsed["summary"])
        elif parsed_list is not None:
            output = json.dumps(parsed_list, default=str)
        elif raw_str:
            exit_code_match = re.match(r"^Exit code\s+(\d+)(?:\n([\s\S]*))?$", raw_str, re.IGNORECASE)
            if exit_code_match:
                exit_code = int(exit_code_match.group(1))
                body = (exit_code_match.group(2) or "").strip()
                if exit_code == 0:
                    output = body
                else:
                    is_error = True
                    error = body or f"Command exited with code {exit_code}"
            elif raw_str.startswith(("Error executing command:", "Tool execution failed:", "Error:")):
                is_error = True
                exit_code = 1
                error = raw_str
            else:
                output = raw_str
                if tool_name in _SHELL_TOOLS or "command" in tool_name or "exec" in tool_name:
                    exit_code = 0

        if tool_name == "verify_outcome" and raw_str.lstrip().startswith("[VERIFIED]") and not is_error:
            verified = True

        if not is_timeout_flag and ("timed out" in error.lower() or error_type == "COMMAND_TIMEOUT"):
            is_timeout_flag = True
        if not is_cancel_flag and (error_type in ("CANCELLED", "TASK_CANCELLED") or "cancelled by user" in error.lower()):
            is_cancel_flag = True

        # Pre-execution failure attribution
        pre_exec_error_types = {
            "SCHEMA_VALIDATION_ERROR",
            "ARGUMENT_NORMALIZATION_ERROR",
            "AUTHORIZATION_DENIED",
            "TOOL_NOT_FOUND",
            "TOOL_UNAVAILABLE",
            "CYCLE_DETECTED",
            "PROHIBITED_BY_USER",
        }
        if dispatched is None:
            dispatched = error_type not in pre_exec_error_types

        coerced_ver = cls._coerce_independent_verification(independent_verification)
        if coerced_ver is not None:
            if coerced_ver["passed"] is True:
                if is_error or has_contradiction:
                    has_contradiction = True
                else:
                    verified = True
            else:
                if not is_error or verified is True:
                    has_contradiction = True
                verified = False

        # Determine domain facts before sanitizing output streams
        domain_fields, state_changes, evidence, domain_uncertainty = cls._extract_domain_details(
            tool_name=tool_name,
            parsed=parsed,
            output=output,
            stderr=stderr,
            is_error=is_error,
            ext_source_meta=ext_meta,
            executed_args=executed_args or requested_args,
        )
        if coerced_ver is not None:
            evidence.append({
                "source": "independent_verification",
                "check": coerced_ver["check"],
                "target": coerced_ver["target"],
                "passed": coerced_ver["passed"],
            })
            for sc in state_changes:
                sc["verified"] = bool(coerced_ver["passed"])

        # Sanitize all untrusted strings and structured details
        safe_out, out_red, out_trunc = _sanitize_untrusted_text(output, is_external=is_ext, escape_anchors=True, max_chars=raw_text_limit)
        safe_stderr, err_s_red, err_s_trunc = _sanitize_untrusted_text(stderr, is_external=is_ext, escape_anchors=True, max_chars=min(raw_text_limit, 4096))
        safe_err, err_red, err_trunc = _sanitize_untrusted_text(error, is_external=is_ext, escape_anchors=True, max_chars=2048)
        safe_remedy, rem_red, _ = _sanitize_untrusted_text(remedy_hint, is_external=is_ext, escape_anchors=True, max_chars=512)
        safe_domain, dom_red, dom_trunc = _sanitize_structured_dict(domain_fields, is_external=is_ext)
        safe_req_args, req_red, _ = _sanitize_structured_dict(requested_args or {}, is_external=False, max_str_chars=256)
        safe_exec_args, ex_red, _ = _sanitize_structured_dict(executed_args if executed_args is not None else (requested_args or {}), is_external=False, max_str_chars=256)

        any_redacted = bool(out_red or err_s_red or err_red or rem_red or dom_red or req_red or ex_red)
        any_truncated = bool(out_trunc or err_s_trunc or err_trunc or dom_trunc)

        # Error classification on failures
        err_classification_dict: Optional[Dict[str, Any]] = None
        classification_obj: Any = None
        if is_error or error_type:
            try:
                from zelyr.engine.error_classifier import classify_tool_error

                classification_obj = classify_tool_error(
                    tool_name,
                    exit_code,
                    safe_out,
                    safe_err or safe_stderr,
                    error_type=error_type or None,
                )
                error_type = error_type or classification_obj.error_type
                if not safe_remedy:
                    safe_remedy = classification_obj.remedy_hint
                if not allowed_actions:
                    allowed_actions = list(classification_obj.allowed_actions)
                err_classification_dict = {
                    "error_type": error_type,
                    "is_recoverable": bool(getattr(classification_obj, "is_recoverable", True)),
                    "remedy_hint": safe_remedy,
                    "allowed_actions": list(allowed_actions),
                }
            except Exception:
                err_classification_dict = {
                    "error_type": error_type or "TOOL_EXECUTION_FAILED",
                    "is_recoverable": False,
                    "remedy_hint": safe_remedy,
                    "allowed_actions": list(allowed_actions),
                }

        # Determine canonical execution_status, evidence_level, verification_status, remaining_uncertainty
        is_empty_result = (
            not is_error
            and not malformed
            and not is_timeout_flag
            and not is_cancel_flag
            and not is_partial_flag
            and not safe_out.strip()
            and not safe_stderr.strip()
            and (parsed is None or parsed == {} or not any(v not in (None, "", [], {}) for k, v in parsed.items() if k not in ("status", "success", "exit_code", "tool", "arguments")))
        )

        remaining_uncertainty: Optional[str] = domain_uncertainty
        if is_cancel_flag:
            exec_status = ObservationExecutionStatus.CANCELLED.value
            ev_level = EvidenceLevel.UNKNOWN.value
            ver_status = "not_verified"
            remaining_uncertainty = "Execution was cancelled; final host state may be incomplete." if dispatched else "Cancelled before execution; no tool action occurred."
        elif is_timeout_flag:
            exec_status = ObservationExecutionStatus.TIMED_OUT.value
            ev_level = EvidenceLevel.PARTIAL.value
            ver_status = "not_verified"
            remaining_uncertainty = "Operation timed out before completing cleanly; state may be partially modified."
        elif not dispatched:
            exec_status = ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value
            ev_level = EvidenceLevel.CONFIRMED.value
            ver_status = "not_verified"
            remaining_uncertainty = None
        elif malformed:
            exec_status = ObservationExecutionStatus.MALFORMED.value
            ev_level = EvidenceLevel.UNKNOWN.value
            ver_status = "not_verified"
            remaining_uncertainty = "Raw tool output was malformed JSON; outcome cannot be reliably interpreted."
        elif has_contradiction or (coerced_ver is not None and coerced_ver["passed"] is False and not is_error):
            exec_status = ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value
            ev_level = EvidenceLevel.CONTRADICTED.value
            ver_status = "verification_failed" if (coerced_ver and coerced_ver["passed"] is False) else "contradicted"
            remaining_uncertainty = (
                (coerced_ver.get("error") if coerced_ver else None)
                or "Observed signals contradict the tool's claimed outcome."
            )
        elif is_error:
            exec_status = ObservationExecutionStatus.EXECUTED_ERROR.value
            ev_level = EvidenceLevel.REPORTED.value
            ver_status = "verification_failed" if (verified is False or (coerced_ver and not coerced_ver["passed"])) else "not_verified"
            remaining_uncertainty = domain_uncertainty or "Execution failed after dispatch; verify whether partial changes remain on host."
        elif is_partial_flag:
            exec_status = ObservationExecutionStatus.PARTIALLY_SUCCEEDED.value
            ev_level = EvidenceLevel.PARTIAL.value
            ver_status = "not_verified"
            remaining_uncertainty = domain_uncertainty or "Tool reported only partial completion; remaining steps/state remain unverified."
        elif is_empty_result:
            exec_status = ObservationExecutionStatus.EMPTY.value
            ev_level = EvidenceLevel.UNKNOWN.value
            ver_status = "not_verified"
            remaining_uncertainty = f"Tool '{tool_name}' returned an empty payload; outcome is unconfirmed."
        elif verified is True and (coerced_ver is None or coerced_ver["passed"] is True) and not is_ext:
            exec_status = ObservationExecutionStatus.SUCCEEDED_VERIFIED.value
            ev_level = EvidenceLevel.CONFIRMED.value
            ver_status = "verified"
            remaining_uncertainty = None
        elif verified is True and coerced_ver is not None and coerced_ver["passed"] is True:
            exec_status = ObservationExecutionStatus.SUCCEEDED_VERIFIED.value
            ev_level = EvidenceLevel.CONFIRMED.value
            ver_status = "verified"
            remaining_uncertainty = None
        elif verified is False:
            exec_status = ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value
            ev_level = EvidenceLevel.CONTRADICTED.value
            ver_status = "verification_failed"
            remaining_uncertainty = domain_uncertainty or "Execution succeeded per tool output, but verification failed."
        else:
            exec_status = ObservationExecutionStatus.SUCCEEDED_UNVERIFIED.value
            ev_level = EvidenceLevel.REPORTED.value
            ver_status = "not_verified"
            if not remaining_uncertainty:
                remaining_uncertainty = f"Tool '{tool_name}' reported completion without independent verification."

        if not summary:
            if exec_status == ObservationExecutionStatus.SUCCEEDED_VERIFIED.value:
                summary = "Action completed and outcome independently verified."
            elif exec_status == ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value:
                summary = f"Pre-execution check rejected '{tool_name}' ({error_type or 'BLOCKED'})."
            elif exec_status == ObservationExecutionStatus.TIMED_OUT.value:
                summary = f"Tool '{tool_name}' timed out before completion."
            elif exec_status == ObservationExecutionStatus.CANCELLED.value:
                summary = f"Tool '{tool_name}' execution was cancelled."
            elif exec_status == ObservationExecutionStatus.MALFORMED.value:
                summary = f"Tool '{tool_name}' returned malformed structured output."
            elif exec_status == ObservationExecutionStatus.EMPTY.value:
                summary = f"Tool '{tool_name}' returned no output."
            elif exec_status == ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value:
                summary = f"Contradictory outcome evidence detected for '{tool_name}'."
            elif is_error:
                summary = f"Tool '{tool_name}' failed ({error_type or 'ERROR'})."
            else:
                summary = f"Tool '{tool_name}' reported completion (unverified)."

        safe_summary, sum_red, _ = _sanitize_untrusted_text(summary, is_external=is_ext, escape_anchors=True, max_chars=320)
        any_redacted = any_redacted or sum_red

        obs_id = f"obs_{uuid.uuid4().hex[:12]}"
        req_action = (
            (requested_args or {}).get("action")
            if isinstance(requested_args, dict)
            else ((executed_args or {}).get("action") if isinstance(executed_args, dict) else None)
        )
        operation_attempted: Dict[str, Any] = {
            "requested_tool": str(requested_tool or tool_name),
            "executed_tool": str(executed_tool or tool_name),
            "dispatched": bool(dispatched),
            "requested_action": str(req_action) if req_action else None,
            "requested_args": safe_req_args,
            "executed_args": safe_exec_args,
        }

        structured_result: Dict[str, Any] = {
            "tool_reported_success": not is_error,
            "exit_code": exit_code,
            "summary": safe_summary,
            "output_preview": safe_out[:1200] if safe_out else "",
            "stderr_preview": safe_stderr[:600] if safe_stderr else "",
            "error": safe_err,
            "domain_fields": safe_domain if isinstance(safe_domain, dict) else {},
        }

        obs_obj = StructuredObservation(
            observation_id=obs_id,
            source_tool_id=str(tool_name),
            execution_id=execution_id,
            correlation_id=correlation_id,
            external_source=ext_meta,
            operation_attempted=operation_attempted,
            execution_status=exec_status,
            structured_result=structured_result,
            state_changes=state_changes,
            evidence=evidence,
            evidence_level=ev_level,
            remaining_uncertainty=remaining_uncertainty,
            error_classification=err_classification_dict,
            verification_status=ver_status,
            independent_verification=coerced_ver,
            ledger_refs=dict(ledger_refs or {}),
            redacted=any_redacted,
            truncated=any_truncated,
        )
        guidance = cls.derive_recovery_guidance(obs_obj, classification=classification_obj)
        obs_obj.recovery_guidance = guidance
        obs_obj.suggested_next_action = guidance.get("next_step")
        return obs_obj

    @classmethod
    def verify_and_frame(
        cls,
        observation: StructuredObservation,
        *,
        check: str,
        target: str,
        expected: Optional[str] = None,
        verifier_fn: Optional[Any] = None,
        **kwargs: Any,
    ) -> StructuredObservation:
        """Attach an independent OutcomeVerifier result onto a StructuredObservation."""
        if verifier_fn is not None:
            vr = verifier_fn(check=check, target=target, expected=expected, **kwargs)
        else:
            from zelyr.engine.verifier import OutcomeVerifier

            vr = OutcomeVerifier.verify(check=check, target=target, expected=expected, **kwargs)

        vr_dict = {
            "passed": bool(getattr(vr, "success", False)),
            "check": str(getattr(vr, "check", check)),
            "target": str(getattr(vr, "target", target)),
            "observed": getattr(vr, "observed", {}),
            "error": getattr(vr, "error", None),
        }
        coerced = cls._coerce_independent_verification(vr_dict)
        updated = StructuredObservation(
            observation_id=observation.observation_id,
            source_tool_id=observation.source_tool_id,
            execution_id=observation.execution_id,
            correlation_id=observation.correlation_id,
            timestamp=time.time(),
            external_source=dict(observation.external_source) if observation.external_source else None,
            operation_attempted=dict(observation.operation_attempted),
            execution_status=observation.execution_status,
            structured_result=dict(observation.structured_result),
            state_changes=[dict(sc) for sc in observation.state_changes],
            evidence=[dict(ev) for ev in observation.evidence],
            evidence_level=observation.evidence_level,
            remaining_uncertainty=observation.remaining_uncertainty,
            error_classification=dict(observation.error_classification) if observation.error_classification else None,
            verification_status=observation.verification_status,
            independent_verification=coerced,
            ledger_refs=dict(observation.ledger_refs),
            redacted=observation.redacted,
            truncated=observation.truncated,
        )
        if coerced is not None:
            updated.evidence.append({
                "source": "independent_verification",
                "check": coerced["check"],
                "target": coerced["target"],
                "passed": coerced["passed"],
            })
            for sc in updated.state_changes:
                sc["verified"] = bool(coerced["passed"])
            tool_reported_ok = bool(updated.structured_result.get("tool_reported_success", True))
            if coerced["passed"]:
                if not tool_reported_ok or observation.execution_status in (
                    ObservationExecutionStatus.EXECUTED_ERROR.value,
                    ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value,
                    ObservationExecutionStatus.TIMED_OUT.value,
                ):
                    updated.evidence_level = EvidenceLevel.CONTRADICTED.value
                    updated.execution_status = ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value
                    updated.verification_status = "contradicted"
                    updated.remaining_uncertainty = (
                        "Independent verifier passed, contradicting earlier failed tool report."
                    )
                else:
                    updated.evidence_level = EvidenceLevel.CONFIRMED.value
                    updated.execution_status = ObservationExecutionStatus.SUCCEEDED_VERIFIED.value
                    updated.verification_status = "verified"
                    updated.remaining_uncertainty = None
            else:
                updated.evidence_level = EvidenceLevel.CONTRADICTED.value
                updated.execution_status = ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value
                updated.verification_status = "verification_failed"
                updated.remaining_uncertainty = (
                    coerced.get("error")
                    or "Independent verification failed against expected criteria."
                )
        updated.recovery_guidance = cls.derive_recovery_guidance(updated)
        updated.suggested_next_action = updated.recovery_guidance.get("next_step")
        return updated

    @classmethod
    def reconcile_model_claim(
        cls,
        claim_text: str,
        observations: List[StructuredObservation],
        *,
        require_independent_verification: bool = False,
    ) -> Dict[str, Any]:
        """Reconcile a model completion claim against structured observation evidence."""
        text = str(claim_text or "").strip()
        if not text:
            return {
                "consistent": True,
                "claim_status": "no_claim",
                "contradictions": [],
                "unverified_observations": [],
                "reason": "Empty model text.",
            }

        contradictions: List[Dict[str, Any]] = []
        unverified_obs: List[str] = []
        has_failed_or_contradicted = False

        for ob in observations:
            if ob.execution_status in (
                ObservationExecutionStatus.EXECUTED_ERROR.value,
                ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value,
                ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value,
                ObservationExecutionStatus.TIMED_OUT.value,
                ObservationExecutionStatus.CANCELLED.value,
                ObservationExecutionStatus.MALFORMED.value,
            ) or ob.evidence_level == EvidenceLevel.CONTRADICTED.value:
                has_failed_or_contradicted = True
                contradictions.append({
                    "observation_id": ob.observation_id,
                    "tool": ob.source_tool_id,
                    "execution_status": ob.execution_status,
                    "evidence_level": ob.evidence_level,
                    "detail": ob.remaining_uncertainty or ob.structured_result.get("error") or ob.structured_result.get("summary"),
                })
            elif ob.evidence_level != EvidenceLevel.CONFIRMED.value:
                unverified_obs.append(ob.observation_id)

        from zelyr.engine.execution_ledger import detect_false_success, FALSE_SUCCESS_PATTERNS, is_truthful_failure_report

        claims_completion = False
        if not is_truthful_failure_report(text):
            low = text.lower()
            claims_completion = any(re.search(pat, low) for pat in FALSE_SUCCESS_PATTERNS)

        if has_failed_or_contradicted and detect_false_success(text, has_unresolved=True):
            return {
                "consistent": False,
                "claim_status": "contradicted",
                "contradictions": contradictions,
                "unverified_observations": unverified_obs,
                "reason": "Model claimed completion despite failed, timed-out, cancelled, or contradicted observations.",
            }

        if claims_completion and not observations and require_independent_verification:
            return {
                "consistent": False,
                "claim_status": "unverified",
                "contradictions": [],
                "unverified_observations": [],
                "reason": "Model claimed completion without any execution observations or independent verification.",
            }

        if claims_completion and unverified_obs and (require_independent_verification or not any(o.evidence_level == EvidenceLevel.CONFIRMED.value for o in observations)):
            return {
                "consistent": False,
                "claim_status": "unverified",
                "contradictions": [],
                "unverified_observations": unverified_obs,
                "reason": "Model claimed completion backed only by unverified/reported evidence.",
            }

        return {
            "consistent": True,
            "claim_status": "verified" if any(o.evidence_level == EvidenceLevel.CONFIRMED.value for o in observations) else "consistent_with_reported",
            "contradictions": contradictions,
            "unverified_observations": unverified_obs,
            "reason": "Model statement aligns with observed execution evidence.",
        }

    @classmethod
    def compact_for_planner(
        cls,
        observation: Any,
        *,
        max_chars: int = 1200,
        as_dict: bool = False,
    ) -> Any:
        """Render a bounded planner representation without raw bulky payload dumps."""
        if isinstance(observation, StructuredObservation):
            return observation.to_compact_dict(max_chars=max_chars) if as_dict else observation.to_compact_planner_context(max_chars=max_chars)
        if isinstance(observation, str):
            parsed = cls.parse_observation(observation)
            so = cls.frame_structured(parsed.get("tool") or "unknown", parsed)
            return so.to_compact_dict(max_chars=max_chars) if as_dict else so.to_compact_planner_context(max_chars=max_chars)
        if isinstance(observation, dict):
            so = cls.frame_structured(str(observation.get("source_tool_id") or observation.get("tool") or "unknown"), observation)
            return so.to_compact_dict(max_chars=max_chars) if as_dict else so.to_compact_planner_context(max_chars=max_chars)
        so = cls.frame_structured("unknown", str(observation))
        return so.to_compact_dict(max_chars=max_chars) if as_dict else so.to_compact_planner_context(max_chars=max_chars)

    @classmethod
    def frame_observation(
        cls,
        tool_name: str,
        result: Any,
        as_json: bool = False,
        *,
        structured: Optional[StructuredObservation] = None,
        requested_tool: Optional[str] = None,
        requested_args: Optional[Dict[str, Any]] = None,
        executed_args: Optional[Dict[str, Any]] = None,
        dispatched: Optional[bool] = None,
        timed_out: bool = False,
        cancelled: bool = False,
        partial: bool = False,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        external_source: Optional[Dict[str, Any]] = None,
        independent_verification: Optional[Dict[str, Any]] = None,
        ledger_refs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Frame a tool execution result with deterministic semantic anchors and sanitization.

        Returns:
            Clean human/model-readable framed string or structured JSON string if as_json=True.
        """
        orig_result = result
        # 1. Unpack result if it has a to_dict method
        if hasattr(result, "to_dict"):
            try:
                result = result.to_dict(tool=tool_name)
            except TypeError:
                result = result.to_dict()

        parsed: Optional[Dict[str, Any]] = None
        raw_str = ""

        if isinstance(result, dict):
            parsed = result
            raw_str = json.dumps(result, default=str)
        elif isinstance(result, str):
            raw_str = result.strip()
            try:
                data = json.loads(raw_str)
                if isinstance(data, dict):
                    parsed = data
            except Exception:
                pass
        else:
            raw_str = str(result)

        # 2. Extract fields
        exit_code: Optional[int] = None
        status = "SUCCESS"
        output = ""
        stderr = ""
        error = ""
        error_type = ""
        remedy_hint = ""
        allowed_actions: List[str] = []
        verified: Optional[bool] = None
        summary = ""
        verification_detail = ""

        if parsed is not None:
            # Handle SCHEMA_VALIDATION_ERROR envelopes
            if (
                parsed.get("error_type") == "SCHEMA_VALIDATION_ERROR"
                or (isinstance(parsed.get("result"), dict) and parsed["result"].get("error_type") == "SCHEMA_VALIDATION_ERROR")
                or (isinstance(parsed.get("output"), dict) and parsed["output"].get("error_type") == "SCHEMA_VALIDATION_ERROR")
            ):
                envelope = (
                    parsed
                    if parsed.get("error_type") == "SCHEMA_VALIDATION_ERROR"
                    else (
                        parsed["result"]
                        if isinstance(parsed.get("result"), dict) and parsed["result"].get("error_type") == "SCHEMA_VALIDATION_ERROR"
                        else parsed["output"]
                    )
                )
                invalid_param = envelope.get("invalid_parameter") or ""
                expected_type = envelope.get("expected_type") or ""
                err_msg = envelope.get("error") or f"Schema validation failed for '{invalid_param}'"
                remedy = envelope.get("remedy_hint") or f"Provide parameter '{invalid_param}' as a valid {expected_type} and retry."

                status = "ERROR"
                exit_code = 1
                error_type = "SCHEMA_VALIDATION_ERROR"
                error = err_msg
                remedy_hint = remedy
                allowed_actions = [tool_name]
                summary = f"Schema validation failed for parameter '{invalid_param}'."

            else:
                # Normal dictionary extraction
                if "allowed_actions" in parsed and isinstance(parsed["allowed_actions"], list):
                    allowed_actions = list(parsed["allowed_actions"])
                elif isinstance(parsed.get("result"), dict) and "allowed_actions" in parsed["result"]:
                    allowed_actions = list(parsed["result"]["allowed_actions"] or [])
                elif isinstance(parsed.get("result"), dict) and isinstance(parsed["result"].get("metadata"), dict) and "allowed_actions" in parsed["result"]["metadata"]:
                    allowed_actions = list(parsed["result"]["metadata"]["allowed_actions"] or [])

                # Success / error status detection
                is_error = False
                if parsed.get("success") is False:
                    is_error = True
                elif parsed.get("status") in ("error", "failed", "failure", "rejected", "tool_execution_failed", "timed_out", "timeout", "cancelled", "canceled"):
                    is_error = True
                elif parsed.get("error"):
                    is_error = True
                elif isinstance(parsed.get("result"), dict) and parsed["result"].get("error"):
                    is_error = True

                # Exit code extraction
                if "exit_code" in parsed and parsed["exit_code"] is not None:
                    try:
                        exit_code = int(parsed["exit_code"])
                    except Exception:
                        exit_code = 1 if is_error else 0
                elif "returncode" in parsed and parsed["returncode"] is not None:
                    try:
                        exit_code = int(parsed["returncode"])
                    except Exception:
                        exit_code = 1 if is_error else 0
                elif is_error:
                    exit_code = 1
                elif "command" in tool_name or "exec" in tool_name or "bash" in tool_name or "shell" in tool_name:
                    exit_code = 0

                status = "ERROR" if (is_error or (exit_code is not None and exit_code != 0)) else "SUCCESS"

                # Output extraction
                if "output" in parsed:
                    out_val = parsed["output"]
                elif "stdout" in parsed:
                    out_val = parsed["stdout"]
                elif "result" in parsed:
                    res_val = parsed["result"]
                    if isinstance(res_val, dict) and "output" in res_val:
                        out_val = res_val["output"]
                    elif isinstance(res_val, dict) and "error" in res_val and is_error:
                        out_val = ""
                    else:
                        out_val = res_val
                else:
                    out_val = raw_str if status == "SUCCESS" else ""

                output = json.dumps(out_val, default=str) if isinstance(out_val, dict) else str(out_val if out_val is not None else "")

                # Stderr extraction
                if "stderr" in parsed and parsed["stderr"]:
                    stderr = str(parsed["stderr"])
                elif isinstance(parsed.get("result"), dict) and parsed["result"].get("stderr"):
                    stderr = str(parsed["result"]["stderr"])
                elif isinstance(parsed.get("result"), dict) and isinstance(parsed["result"].get("metadata"), dict) and parsed["result"]["metadata"].get("stderr"):
                    stderr = str(parsed["result"]["metadata"]["stderr"])
                elif isinstance(parsed.get("metadata"), dict) and parsed["metadata"].get("stderr"):
                    stderr = str(parsed["metadata"]["stderr"])

                # Error extraction
                if status == "ERROR":
                    err_val = (
                        parsed.get("error")
                        or parsed.get("stderr")
                        or (parsed.get("result", {}).get("error") if isinstance(parsed.get("result"), dict) else None)
                        or "Tool execution failed"
                    )
                    error = str(err_val)
                    remedy_hint = (
                        parsed.get("remedy_hint")
                        or (parsed.get("result", {}).get("remedy_hint") if isinstance(parsed.get("result"), dict) else None)
                        or (parsed.get("result", {}).get("metadata", {}).get("remedy_hint") if isinstance(parsed.get("result"), dict) else None)
                        or ""
                    )
                    if parsed.get("error_type"):
                        error_type = str(parsed["error_type"])

                # Verification flags (strict boolean validation — never coerce strings/ints via bool())
                raw_ver = None
                ver_source = None
                if "verified" in parsed and parsed["verified"] is not None:
                    raw_ver = parsed["verified"]
                    ver_source = parsed
                elif isinstance(parsed.get("result"), dict) and "verified" in parsed["result"] and parsed["result"]["verified"] is not None:
                    raw_ver = parsed["result"]["verified"]
                    ver_source = parsed["result"]

                if isinstance(raw_ver, bool):
                    if raw_ver is True and status == "SUCCESS":
                        verified = True
                    elif raw_ver is False:
                        is_legacy_record_default = (
                            "state" in parsed
                            and parsed.get("verification_failed") is not True
                            and not any(k in parsed for k in ("failed_criterion", "verifier_reason", "required_correction"))
                        )
                        verified = None if (status == "SUCCESS" and is_legacy_record_default) else False

                if parsed.get("verification_failed") is True and status == "SUCCESS":
                    verified = False

                if verified is not None and ver_source is not None:
                    if ver_source.get("verification_detail"):
                        verification_detail = str(ver_source["verification_detail"])
                    elif "observed" in ver_source:
                        verification_detail = str(ver_source["observed"])
                    elif "check" in ver_source:
                        verification_detail = str(ver_source["check"])

                if parsed.get("summary"):
                    summary = str(parsed["summary"])

        else:
            # Raw string parsing
            exit_code_match = re.match(r"^Exit code\s+(\d+)(?:\n([\s\S]*))?$", raw_str, re.IGNORECASE)
            if exit_code_match:
                exit_code = int(exit_code_match.group(1))
                body = (exit_code_match.group(2) or "").strip()
                if exit_code == 0:
                    status = "SUCCESS"
                    output = body
                    error = ""
                else:
                    status = "ERROR"
                    output = ""
                    error = body or f"Command exited with code {exit_code}"
            elif raw_str.startswith(("Error executing command:", "Tool execution failed:", "Error:")):
                status = "ERROR"
                exit_code = 1
                output = ""
                error = raw_str
            else:
                status = "SUCCESS"
                exit_code = 0 if ("command" in tool_name or "exec" in tool_name or "shell" in tool_name) else None
                output = raw_str
                error = ""

        # Check for inline [VERIFIED] marker only from dedicated verification tool (before output anchor escaping)
        if tool_name == "verify_outcome" and raw_str.lstrip().startswith("[VERIFIED]") and status == "SUCCESS":
            verified = True

        # If tool is verify_outcome, enforce verified boolean
        if tool_name == "verify_outcome":
            if verified is None:
                verified = (status == "SUCCESS")
            if verified is False:
                status = "ERROR"
                if not error:
                    error = "Observed state did not match expected criteria"
                if not error_type:
                    error_type = "VERIFICATION_FAILED"
        elif status != "SUCCESS" and verified is True:
            verified = False

        # Remediation classification on failure
        if status == "ERROR":
            from zelyr.engine.error_classifier import classify_tool_error

            classification = classify_tool_error(
                tool_name,
                exit_code,
                output,
                error,
                error_type=error_type or None,
            )
            if not error_type:
                error_type = classification.error_type
            if not remedy_hint:
                remedy_hint = classification.remedy_hint
            if not allowed_actions:
                allowed_actions = list(classification.allowed_actions)

            err_clean = error.rstrip(".")
            if "[Tool Error]:" in err_clean:
                err_clean = err_clean.split("[Tool Error]:", 1)[1].strip()
            if "Recovery Hint:" in err_clean:
                err_clean = err_clean.split("Recovery Hint:", 1)[0].strip().rstrip(".")

            error = err_clean

        ext_source_meta = cls._extract_external_source(tool_name, parsed, external_source=external_source)
        is_ext = bool(ext_source_meta and ext_source_meta.get("is_external"))

        structured_obs = structured or cls.frame_structured(
            tool_name,
            orig_result,
            requested_tool=requested_tool,
            requested_args=requested_args,
            executed_args=executed_args,
            dispatched=dispatched,
            timed_out=timed_out,
            cancelled=cancelled,
            partial=partial,
            execution_id=execution_id,
            correlation_id=correlation_id,
            external_source=external_source or ext_source_meta,
            independent_verification=independent_verification,
            ledger_refs=ledger_refs,
        )

        # Sanitize untrusted output streams, secrets, and spoofed observation anchors
        output, _, _ = _sanitize_untrusted_text(output, is_external=is_ext, escape_anchors=True, max_chars=0)
        stderr, _, _ = _sanitize_untrusted_text(stderr, is_external=is_ext, escape_anchors=True, max_chars=0)
        error, _, _ = _sanitize_untrusted_text(error, is_external=is_ext, escape_anchors=True, max_chars=4096)
        remedy_hint, _, _ = _sanitize_untrusted_text(remedy_hint, is_external=is_ext, escape_anchors=True, max_chars=1024)
        verification_detail, _, _ = _sanitize_untrusted_text(verification_detail, is_external=is_ext, escape_anchors=True, max_chars=512)

        # Summary generation
        if not summary:
            if verified is True:
                summary = "Outcome matched expected criteria and machine state verified."
            elif verified is False and status == "SUCCESS":
                summary = f"Tool '{tool_name}' executed successfully, but outcome is NOT verified (acceptance criteria failed)."
            elif verified is False:
                summary = "Verification failed: observed state did not match expected criteria."
            elif status == "SUCCESS":
                if exit_code is not None:
                    summary = f"Command executed successfully with exit code {exit_code}."
                else:
                    summary = f"Tool '{tool_name}' executed successfully."
            else:
                if exit_code is not None:
                    summary = f"Command failed with exit code {exit_code} ({error_type or 'ERROR'})."
                else:
                    summary = f"Tool '{tool_name}' failed ({error_type or 'ERROR'})."
        else:
            summary, _, _ = _sanitize_untrusted_text(summary, is_external=is_ext, escape_anchors=True, max_chars=320)

        # Format as JSON if requested
        if as_json:
            formatted_error = ""
            if status == "ERROR":
                formatted_error = f"[Tool Error]: {error}. Recovery Hint: {remedy_hint}"

            data_payload: Dict[str, Any] = {
                "tool": tool_name,
                "status": "success" if status == "SUCCESS" else "error",
                "exit_code": exit_code if exit_code is not None else (0 if status == "SUCCESS" else 1),
                "verified": bool(status == "SUCCESS" and verified is True),
                "verification_failed": bool(status == "SUCCESS" and verified is False),
                "summary": summary,
                "output": output,
                "error": formatted_error if status == "ERROR" else "",
                "evidence_level": structured_obs.evidence_level,
                "execution_status": structured_obs.execution_status,
            }
            if structured_obs.remaining_uncertainty:
                data_payload["remaining_uncertainty"] = structured_obs.remaining_uncertainty
            if structured_obs.ledger_refs or execution_id:
                data_payload["ledger_refs"] = dict(structured_obs.ledger_refs)
            if structured_obs.recovery_guidance:
                data_payload["recovery_guidance"] = dict(structured_obs.recovery_guidance)
            if ext_source_meta is not None:
                data_payload["external"] = True
                data_payload["source_id"] = ext_source_meta["source_id"]
                data_payload["source_type"] = ext_source_meta["source_type"]
                data_payload["external_tool_name"] = ext_source_meta["external_tool_name"]
            if status == "SUCCESS":
                if verified is True:
                    data_payload["guidance"] = "Action completed and machine state verified."
                elif verified is False:
                    data_payload["guidance"] = (
                        str(parsed.get("required_correction"))
                        if (parsed and parsed.get("required_correction"))
                        else "Tool executed, but outcome is NOT verified. Emit a fresh tool call with corrected parameters to satisfy the acceptance criteria."
                    )
                else:
                    data_payload["guidance"] = (
                        "Action executed successfully. Do not repeat identical parameters; proceed to next step or provide final response."
                    )
            if status == "ERROR":
                data_payload["error_type"] = error_type or "TOOL_EXECUTION_FAILED"
                data_payload["remedy_hint"] = remedy_hint
                data_payload["allowed_actions"] = allowed_actions
            if stderr:
                data_payload["stderr"] = cls._bound_stream_text(stderr)
            if verification_detail:
                data_payload["verification_detail"] = verification_detail
            if parsed is not None:
                for ver_key in ("target", "check", "failed_criterion", "verifier_reason"):
                    if parsed.get(ver_key):
                        cleaned_vk, _, _ = _sanitize_untrusted_text(str(parsed[ver_key]), is_external=is_ext, escape_anchors=True, max_chars=512)
                        data_payload[ver_key] = cleaned_vk
            data_payload["output"] = cls._bound_stream_text(output) if output else output

            return json.dumps(data_payload, indent=2)

        # Build clean framed text observation (preserving lines 0..4 semantic anchor layout)
        lines: List[str] = [
            f"[TOOL OBSERVATION: {tool_name}]",
            f"- Status: {status}",
            f"- Exit Code: {exit_code if exit_code is not None else 'N/A'}",
            f"- Verified: {'TRUE' if verified is True else ('FALSE' if verified is False else 'N/A')}",
            f"- Summary: {summary}",
            f"- Evidence Level: {structured_obs.evidence_level}",
            f"- Execution Status: {structured_obs.execution_status}",
        ]
        if structured_obs.remaining_uncertainty:
            unc_line = " ".join(str(structured_obs.remaining_uncertainty).splitlines())[:240]
            lines.append(f"- Uncertainty: {unc_line}")
        if ext_source_meta is not None:
            lines.append(
                f"- External Source: {ext_source_meta['source_id']} ({ext_source_meta['source_type']}:{ext_source_meta['external_tool_name']})"
            )
        exec_ref = structured_obs.ledger_refs.get("execution_id") or structured_obs.execution_id
        if exec_ref:
            corr_part = f" corr={structured_obs.correlation_id}" if structured_obs.correlation_id else ""
            lines.append(f"- Ledger Ref: {exec_ref}{corr_part}")
        if ext_source_meta is not None:
            lines.append(
                f"- External Source: {ext_source_meta['source_id']} ({ext_source_meta['external_tool_name']})"
            )

        if status == "SUCCESS":
            if verified is True:
                lines.append("- Action Authority: COMPLETED & VERIFIED. Machine state verified.")
            elif verified is False:
                lines.append("- Action Authority: EXECUTED BUT UNVERIFIED. Tool executed, but requested outcome is NOT verified.")
            else:
                lines.append("- Action Authority: EXECUTED (UNVERIFIED). Action executed; independent verification not performed (do not repeat identical parameters).")

            if output:
                lines.append("")
                lines.append("[Output]:")
                lines.append(cls._bound_stream_text(output.strip()))

            if stderr and stderr.strip():
                lines.append("")
                lines.append("[Informational Diagnostic Stream (stderr)]:")
                lines.append(cls._bound_stream_text(stderr.strip()))
                lines.append("(Note: Command executed successfully with exit code 0; this stream contains non-error progress info.)")

            if verified is True:
                lines.append("")
                target_note = f" (target: {parsed['target']})" if (parsed and parsed.get("target")) else ""
                if verification_detail:
                    lines.append(f"[Machine State Verified]: {verification_detail}{target_note}")
                else:
                    lines.append(f"[Machine State Verified]: Outcome matched expected criteria{target_note}.")
            elif verified is False:
                lines.append("")
                lines.append("[Outcome Verification Failed]:")
                if parsed and parsed.get("target"):
                    lines.append(f"- Target: {parsed['target']}")
                if parsed and parsed.get("failed_criterion"):
                    lines.append(f"- Failed Criterion: {parsed['failed_criterion']}")
                if parsed and parsed.get("verifier_reason"):
                    lines.append(f"- Verifier Reason: {parsed['verifier_reason']}")
                elif verification_detail:
                    lines.append(f"- Verifier Reason: {verification_detail}")
                if parsed and parsed.get("required_correction"):
                    lines.append(f"- Required Correction: {parsed['required_correction']}")
                else:
                    lines.append("- Required Correction: Emit a fresh tool call with corrected parameters to satisfy the acceptance criteria.")

        else:
            lines.append("")
            lines.append(f"[Error Diagnostic: {error_type or 'TOOL_EXECUTION_FAILED'}]")
            lines.append(f"- Failure Reason: {error or 'Execution failed'}")
            lines.append(f"- Recovery Hint: {remedy_hint}")
            lines.append(f"- Allowed Actions: {json.dumps(allowed_actions) if allowed_actions else '[]'}")

            if stderr and stderr.strip():
                lines.append("")
                lines.append("[Error Output (stderr)]:")
                lines.append(cls._bound_stream_text(stderr.strip()))

            if output and output.strip():
                lines.append("")
                lines.append("[Output Trace]:")
                lines.append(cls._bound_stream_text(output.strip()))

            lines.append("")
            lines.append(f"[Tool Error]: {error or 'Command failed'}. Recovery Hint: {remedy_hint}")

        return "\n".join(lines)

    @classmethod
    def _normalize_parsed_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce fail-closed verification and status invariants on parsed observation dicts."""
        out = dict(data)

        raw_ver = out.get("verified")
        if not isinstance(raw_ver, bool):
            out["verified"] = None

        raw_status = out.get("status")
        status_str = str(raw_status).strip().lower() if raw_status is not None else ""
        exit_code = out.get("exit_code")
        if exit_code is not None and not isinstance(exit_code, bool):
            try:
                exit_code = int(exit_code)
                out["exit_code"] = exit_code
            except Exception:
                exit_code = 1
                out["exit_code"] = 1

        is_error = (
            status_str in ("error", "failed", "failure", "rejected", "tool_execution_failed")
            or out.get("success") is False
            or (exit_code is not None and exit_code != 0)
            or bool(out.get("error") and status_str != "success")
        )
        raw_state = str(out.get("state", "")).strip().lower()
        has_explicit_success = (
            status_str == "success"
            or out.get("success") is True
            or raw_state in ("success", "verified", "observed", "unverified")
        )

        if is_error:
            if status_str not in ("error", "rejected"):
                out["status"] = "error"
            if out.get("verified") is True:
                out["verified"] = False
            out["verification_failed"] = False
            return out

        if not has_explicit_success:
            if out.get("verified") is True:
                out["verified"] = None
            out["verification_failed"] = False
            return out

        if "status" not in out:
            out["status"] = "success"

        if "verification_failed" in out:
            ver_failed = isinstance(out["verification_failed"], bool) and out["verification_failed"] is True
        elif "state" in out:
            ver_failed = False
        else:
            ver_failed = bool(
                out.get("verified") is False
                and (
                    any(k in out for k in ("failed_criterion", "verifier_reason", "required_correction"))
                    or "not verified" in str(out.get("summary", "")).lower()
                    or str(out.get("verification_detail", "")).startswith("Verification failed")
                )
            )

        out["verification_failed"] = ver_failed
        if ver_failed:
            out["verified"] = False
        return out

    @classmethod
    def parse_observation(cls, obs: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse a framed observation string, JSON string, or dict into a structured dictionary."""
        if isinstance(obs, dict):
            return cls._normalize_parsed_dict(obs)

        if not isinstance(obs, str):
            return {"raw": str(obs), "verified": None, "verification_failed": False}

        stripped = obs.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    return cls._normalize_parsed_dict(data)
            except Exception:
                pass
            return {
                "tool": "",
                "status": "error",
                "exit_code": 1,
                "verified": False,
                "verification_failed": False,
                "summary": "Malformed JSON observation",
                "output": "",
                "stderr": "",
                "error": "Malformed JSON observation",
                "error_type": "MALFORMED_OBSERVATION",
                "remedy_hint": "",
                "allowed_actions": [],
            }

        parsed: Dict[str, Any] = {
            "tool": "",
            "status": "success",
            "exit_code": None,
            "verified": None,
            "verification_failed": False,
            "summary": "",
            "output": "",
            "stderr": "",
            "error": "",
            "error_type": "",
            "remedy_hint": "",
            "allowed_actions": [],
        }

        # Unframed raw strings must never be parsed for [TOOL OBSERVATION] semantic anchors
        if not stripped.startswith("[TOOL OBSERVATION:"):
            exit_code_match = re.match(r"^Exit code\s+(\d+)(?:\n([\s\S]*))?$", stripped, re.IGNORECASE)
            if exit_code_match:
                ec = int(exit_code_match.group(1))
                body = (exit_code_match.group(2) or "").strip()
                parsed["exit_code"] = ec
                if ec == 0:
                    parsed["status"] = "success"
                    parsed["output"] = body
                else:
                    parsed["status"] = "error"
                    parsed["error"] = body or f"Command exited with code {ec}"
            elif stripped.startswith(("Error executing command:", "Tool execution failed:", "Error:")):
                parsed["status"] = "error"
                parsed["exit_code"] = 1
                parsed["error"] = stripped
            else:
                parsed["output"] = stripped
            return parsed

        # 1. Header parsing (strictly confined to the top header block before any [Output] body)
        header_block = stripped.split("\n\n", 1)[0]
        tool_match = re.search(r"^\[TOOL OBSERVATION:\s*([^\]]+)\]", header_block, re.MULTILINE)
        if tool_match:
            parsed["tool"] = tool_match.group(1).strip()

        status_match = re.search(r"^- Status:\s*(\w+)", header_block, re.MULTILINE)
        has_valid_status = False
        if status_match:
            st = status_match.group(1).strip().upper()
            if st in ("SUCCESS", "ERROR"):
                has_valid_status = True
                parsed["status"] = "error" if st == "ERROR" else "success"
            else:
                parsed["status"] = "error"

        ec_match = re.search(r"^- Exit Code:\s*([^\n]+)", header_block, re.MULTILINE)
        if ec_match:
            ec_str = ec_match.group(1).strip()
            if ec_str != "N/A":
                try:
                    parsed["exit_code"] = int(ec_str)
                except Exception:
                    parsed["exit_code"] = 1
                    parsed["status"] = "error"

        if parsed["exit_code"] is not None and parsed["exit_code"] != 0:
            parsed["status"] = "error"

        ver_match = re.search(r"^- Verified:\s*([^\n]+)", header_block, re.MULTILINE)
        if ver_match:
            ver_str = ver_match.group(1).strip().upper()
            if ver_str == "TRUE":
                parsed["verified"] = True
            elif ver_str == "FALSE":
                parsed["verified"] = False
            else:
                parsed["verified"] = None

        # Fail-closed guard on framed observations with missing status, error diagnostics, or verification failure blocks
        if (
            not has_valid_status
            or parsed["status"] != "success"
            or parsed["verified"] is False
            or "\n\n[Outcome Verification Failed]:" in stripped
            or "\n\n[Error Diagnostic:" in stripped
        ):
            if parsed["verified"] is True:
                parsed["verified"] = False if ("\n\n[Outcome Verification Failed]:" in stripped or parsed["status"] == "error") else None

        sum_match = re.search(r"^- Summary:\s*([^\n]+)", header_block, re.MULTILINE)
        if sum_match:
            parsed["summary"] = sum_match.group(1).strip()

        ev_match = re.search(r"^- Evidence Level:\s*([^\n]+)", header_block, re.MULTILINE)
        if ev_match:
            parsed["evidence_level"] = ev_match.group(1).strip()

        es_match = re.search(r"^- Execution Status:\s*([^\n]+)", header_block, re.MULTILINE)
        if es_match:
            parsed["execution_status"] = es_match.group(1).strip()

        unc_match = re.search(r"^- Uncertainty:\s*([^\n]+)", header_block, re.MULTILINE)
        if unc_match:
            parsed["remaining_uncertainty"] = unc_match.group(1).strip()

        ext_match = re.search(r"^- External Source:\s*([^\n]+)", header_block, re.MULTILINE)
        if ext_match:
            parsed["external_source_raw"] = ext_match.group(1).strip()

        ref_match = re.search(r"^- Ledger Ref:\s*([^\n]+)", header_block, re.MULTILINE)
        if ref_match:
            parsed["ledger_ref"] = ref_match.group(1).strip()

        # 2. Output parsing
        out_match = re.search(
            r"\[Output\]:\n([\s\S]*?)(?=\n\n\[(?:Error Diagnostic|Informational Diagnostic Stream|Machine State Verified|Outcome Verification Failed|Error Output|Output Trace|Tool Error)|$)",
            stripped,
        )
        if out_match:
            parsed["output"] = out_match.group(1).strip()

        ver_detail_match = re.search(r"\[Machine State Verified\]:\s*([^\n]+)", stripped)
        if ver_detail_match and parsed.get("verified") is True:
            parsed["verification_detail"] = ver_detail_match.group(1).strip()
        else:
            ver_fail_reason = re.search(r"^- Verifier Reason:\s*([^\n]+)", stripped, re.MULTILINE)
            if ver_fail_reason:
                parsed["verification_detail"] = ver_fail_reason.group(1).strip()
                parsed["verifier_reason"] = ver_fail_reason.group(1).strip()

        if parsed.get("status") == "success" and ("[Outcome Verification Failed]:" in stripped or parsed.get("verified") is False):
            parsed["verification_failed"] = True
            parsed["verified"] = False
            tgt_match = re.search(r"^- Target:\s*([^\n]+)", stripped, re.MULTILINE)
            if tgt_match:
                parsed["target"] = tgt_match.group(1).strip()
            crit_match = re.search(r"^- Failed Criterion:\s*([^\n]+)", stripped, re.MULTILINE)
            if crit_match:
                parsed["failed_criterion"] = crit_match.group(1).strip()
            corr_match = re.search(r"^- Required Correction:\s*([^\n]+)", stripped, re.MULTILINE)
            if corr_match:
                parsed["required_correction"] = corr_match.group(1).strip()
        else:
            parsed["verification_failed"] = False

        # 3. Informational stderr parsing
        info_stderr_match = re.search(
            r"\[Informational Diagnostic Stream \(stderr\)\]:\n([\s\S]*?)(?=\n\(Note:|\n\n\[|$)",
            stripped,
        )
        if info_stderr_match:
            parsed["stderr"] = info_stderr_match.group(1).strip()

        # 4. Error Diagnostic parsing
        diag_match = re.search(r"\[Error Diagnostic:\s*([^\]]+)\]", stripped)
        if diag_match:
            parsed["error_type"] = diag_match.group(1).strip()

        reason_match = re.search(r"^- Failure Reason:\s*([^\n]+)", stripped, re.MULTILINE)
        if reason_match:
            parsed["error"] = reason_match.group(1).strip()

        hint_match = re.search(r"^- Recovery Hint:\s*([^\n]+)", stripped, re.MULTILINE)
        if hint_match:
            parsed["remedy_hint"] = hint_match.group(1).strip()

        actions_match = re.search(r"^- Allowed Actions:\s*([^\n]+)", stripped, re.MULTILINE)
        if actions_match:
            actions_str = actions_match.group(1).strip()
            try:
                parsed["allowed_actions"] = json.loads(actions_str)
            except Exception:
                parsed["allowed_actions"] = [a.strip() for a in actions_str.split(",") if a.strip()]

        # 5. Error stderr parsing
        err_stderr_match = re.search(
            r"\[Error Output \(stderr\)\]:\n([\s\S]*?)(?=\n\n\[|$)",
            stripped,
        )
        if err_stderr_match:
            parsed["stderr"] = err_stderr_match.group(1).strip()

        # 6. Fallback Tool Error marker parsing
        tool_err_match = re.search(r"\[Tool Error\]:\s*([^\n]+)", stripped)
        if tool_err_match and not parsed.get("error"):
            full_err = tool_err_match.group(1).strip()
            if "Recovery Hint:" in full_err:
                parts = full_err.split("Recovery Hint:", 1)
                parsed["error"] = parts[0].strip().rstrip(".")
                if not parsed.get("remedy_hint"):
                    parsed["remedy_hint"] = parts[1].strip()
            else:
                parsed["error"] = full_err

        if not parsed.get("error") and parsed.get("status") == "error":
            parsed["error"] = parsed.get("summary") or "Tool execution failed"

        return parsed

    @classmethod
    def to_dict(cls, tool_name: str, result: Any) -> Dict[str, Any]:
        """Format result as structured dictionary."""
        framed_json = cls.frame_observation(tool_name, result, as_json=True)
        return json.loads(framed_json)
