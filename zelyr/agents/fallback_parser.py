"""Robust fallback tool call parser extracting tool calls from Markdown, XML, and embedded JSON."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedToolCall:
    """Represents a validated tool call extracted from degraded model text."""

    name: str
    arguments: Dict[str, Any]
    tool_call_id: str
    cleaned_text: str  # Original text with the parsed tool block stripped


class FallbackToolParser:
    r"""Robust fallback tool call parser extracting tool calls from Markdown, XML, and embedded JSON.

    Supports:
    1. Markdown code fences: ```(?:json)?\s*({[\s\S]*?})\s*```
    2. XML pseudo-tags: <tool_call>{...}</tool_call> or <function_call>{...}</function_call>
    3. Raw embedded or trailing JSON objects matching known tool schemas.
    """

    @classmethod
    def get_alias_map(cls) -> Dict[str, str]:
        """Retrieve canonical alias mappings."""
        try:
            from zelyr.agents.native_orchestrator import NativeOrchestrator

            return getattr(NativeOrchestrator, "TOOL_ALIAS_MAP", {})
        except (ImportError, AttributeError):
            return {}

    @classmethod
    def get_known_tool_names(cls, active_schemas: Optional[List[Dict[str, Any]]] = None) -> Set[str]:
        """Retrieve known tool names from active schemas or ToolRegistry."""
        try:
            from zelyr.agents.native_orchestrator import NativeOrchestrator

            return NativeOrchestrator.get_known_tool_names(active_schemas=active_schemas)
        except Exception:
            try:
                from zelyr.tool_registry import BUILTIN_TOOLS, ToolRegistry

                tools = set(BUILTIN_TOOLS.keys())
                reg = ToolRegistry()
                tools.update(reg.list_tools().keys())
                return tools
            except Exception:
                return {
                    "execute_command",
                    "manage_workspace_file",
                    "verify_outcome",
                    "manage_desktop_window",
                    "interact_with_browser",
                    "manage_memory",
                    "craft_tool",
                    "manage_routine",
                }

    @classmethod
    def _strip_tool_prefixes(cls, name: Any) -> str:
        """Strip prefixes such as 'tool.', 'tools.', 'function.', 'functions.'."""
        if not name or not isinstance(name, str):
            return ""
        name = name.strip()
        for pfx in ("tool.", "tools.", "functions.", "function."):
            if name.startswith(pfx):
                name = name[len(pfx) :]
        return name

    @classmethod
    def _normalize_name_and_args(
        cls,
        raw_name: str,
        raw_args: Any,
        known_tools: Optional[Set[str]] = None,
        active_schemas: Optional[List[Dict[str, Any]]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Validate candidate tool name and arguments against known tools and ring permissions."""
        clean_name = cls._strip_tool_prefixes(raw_name)
        if not clean_name:
            return None

        alias_map = cls.get_alias_map()
        canonical_name = alias_map.get(clean_name, clean_name)

        effective_known = known_tools if known_tools is not None else cls.get_known_tool_names(active_schemas)

        # Discard if tool does not resolve to a known registered tool
        if effective_known and canonical_name not in effective_known and clean_name not in effective_known:
            return None

        target_name = canonical_name if canonical_name in effective_known else clean_name

        # Parse stringified arguments if necessary (reject non-object strings cleanly)
        args_dict: Dict[str, Any] = {}
        if isinstance(raw_args, dict):
            args_dict = dict(raw_args)
        elif isinstance(raw_args, str):
            s = raw_args.strip()
            if s:
                parsed: Any = None
                try:
                    parsed = json.loads(s)
                except Exception:
                    try:
                        parsed = ast.literal_eval(s)
                    except Exception:
                        return None
                if not isinstance(parsed, dict):
                    return None
                args_dict = parsed
        elif raw_args is not None:
            return None

        # Canonicalize alias rewrites (e.g., interact_with_ui with browser DOM params -> interact_with_browser)
        # BEFORE allowlist, ring, and schema normalization checks.
        from zelyr.tool_registry import authorize_and_validate_tool_call, canonicalize_tool_invocation

        target_name, args_dict = canonicalize_tool_invocation(target_name, args_dict)

        # Normalize arguments via NativeOrchestrator if available
        try:
            from zelyr.agents.native_orchestrator import NativeOrchestrator

            args_dict = NativeOrchestrator.normalize_tool_arguments(
                target_name,
                args_dict,
                original_name=raw_name,
                active_schemas=active_schemas,
            )
        except Exception:
            pass

        # Authoritative shared permission check (canonical target, allowlist, fail-closed ring)
        decision = authorize_and_validate_tool_call(
            target_name,
            args_dict,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            active_schemas=active_schemas,
            known_tools=effective_known if effective_known else None,
            require_registered=(known_tools is None),
            normalize_args=False,
            validate_schema=False,
        )
        if not decision.allowed:
            logger.warning(
                "[FallbackToolParser] Rejected invocation of %s: %s",
                decision.tool_name or target_name,
                decision.detail,
            )
            return None

        return decision.tool_name, decision.normalized_arguments

    @classmethod
    def _extract_name_and_args_from_dict(cls, obj: Dict[str, Any]) -> Tuple[str, Any]:
        """Extract candidate tool name and arguments from a JSON dictionary."""
        fn_obj = obj.get("function") if isinstance(obj.get("function"), dict) else obj

        # Candidate tool name
        name = ""
        for k in ("name", "tool", "action"):
            if k in fn_obj and isinstance(fn_obj[k], str) and fn_obj[k].strip():
                name = fn_obj[k].strip()
                break
        if not name and isinstance(fn_obj.get("function"), str):
            name = fn_obj["function"].strip()

        # Candidate arguments
        args = None
        for k in ("arguments", "args", "parameters", "params"):
            if k in fn_obj:
                args = fn_obj[k]
                break

        # Flat arguments fallback
        if args is None:
            args = {
                k: v
                for k, v in fn_obj.items()
                if k not in ("name", "tool", "function", "action", "type", "id", "index")
            }

        return name, args

    @classmethod
    def _is_quoted_or_explanatory(cls, text: str, span: Tuple[int, int]) -> bool:
        """Check if candidate tool call span is inside a markdown blockquote or preceded by an example indicator."""
        start_pos, _ = span
        # 1. Check for markdown blockquote on current line or preceding line
        line_start = text.rfind("\n", 0, start_pos) + 1
        line_prefix = text[line_start:start_pos].strip()
        if line_prefix.startswith(">"):
            return True
        if line_start > 1:
            prev_line_end = line_start - 1
            prev_line_start = text.rfind("\n", 0, prev_line_end) + 1
            prev_line = text[prev_line_start:prev_line_end].strip()
            if prev_line.startswith(">") and not line_prefix:
                return True

        # 2. Check preceding context (up to 150 chars) for example / illustrative markers
        preceding = text[max(0, start_pos - 150):start_pos]
        example_pattern = re.compile(
            r"(?i)\b("
            r"for\s+example|"
            r"for\s+instance|"
            r"example(?:\s+invocation|\s+call|\s+syntax|\s+usage|\s+format|\s+payload)?|"
            r"sample(?:\s+invocation|\s+call|\s+payload)?|"
            r"e\.g\.|"
            r"such\s+as|"
            r"here\s+is\s+an\s+example|"
            r"like\s+this|"
            r"as\s+an\s+example|"
            r"syntax\s+example"
            r")\b\s*[:\-]?",
            re.IGNORECASE,
        )
        if example_pattern.search(preceding):
            return True

        return False

    @classmethod
    def _are_candidates_ambiguous(
        cls,
        text: str,
        spans: List[Tuple[int, int]],
    ) -> bool:
        """Check if candidate tool calls are presented as ambiguous or competing alternatives."""
        if len(spans) < 2:
            return False

        ambiguity_pattern = re.compile(
            r"(?i)\b("
            r"or(?:\s+alternatively|\s+else)?|"
            r"either\b|"
            r"alternatively|"
            r"versus|"
            r"vs\.?|"
            r"instead\s+of|"
            r"choose\s+(?:between|one)|"
            r"which\s+(?:one|of\s+these)|"
            r"option\s+[12ab]"
            r")\b",
            re.IGNORECASE,
        )

        # Check the introductory text before the first candidate
        intro = text[max(0, spans[0][0] - 120):spans[0][0]]
        if ambiguity_pattern.search(intro) and re.search(r"\b(either|choose|options?)\b", intro, re.IGNORECASE):
            return True

        # Check interstitial text between spans
        for i in range(len(spans) - 1):
            interstitial = text[spans[i][1]:spans[i + 1][0]]
            if ambiguity_pattern.search(interstitial):
                return True

        return False

    @classmethod
    def parse_tool_calls(
        cls,
        text: str,
        known_tools: Optional[Set[str]] = None,
        active_schemas: Optional[List[Dict[str, Any]]] = None,
        max_calls: Optional[int] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
    ) -> List[ExtractedToolCall]:
        """Extract up to max_calls valid tool calls from degraded model text.

        Strips parsed tool blocks from cleaned_text while preserving preceding reasoning.
        """
        if not text or not text.strip():
            return []

        candidates: List[Tuple[Tuple[int, int], Any]] = []
        covered_spans: List[Tuple[int, int]] = []

        # 1. XML Pseudo-Tags (<tool_call>, <function_call>, <tool>, <action>)
        xml_pattern = re.compile(
            r"<(tool_call|function_call|tool|action)>\s*([\{\[][\s\S]*?[\}\]])\s*</\1>",
            re.IGNORECASE,
        )
        for match in xml_pattern.finditer(text):
            span = match.span()
            raw_content = match.group(2).strip()
            try:
                parsed_json = json.loads(raw_content)
                candidates.append((span, parsed_json))
                covered_spans.append(span)
            except Exception:
                try:
                    parsed_ast = ast.literal_eval(raw_content)
                    if isinstance(parsed_ast, (dict, list)):
                        candidates.append((span, parsed_ast))
                        covered_spans.append(span)
                except Exception:
                    pass

        # 2. Markdown Code Fences (```json ... ``` or bare ``` ... ```)
        fence_pattern = re.compile(
            r"```[a-zA-Z0-9_-]*\s*([\{\[][\s\S]*?[\}\]])\s*```",
            re.IGNORECASE,
        )
        for match in fence_pattern.finditer(text):
            span = match.span()
            if any(s[0] <= span[0] and span[1] <= s[1] for s in covered_spans):
                continue
            raw_content = match.group(1).strip()
            try:
                parsed_json = json.loads(raw_content)
                candidates.append((span, parsed_json))
                covered_spans.append(span)
            except Exception:
                try:
                    parsed_ast = ast.literal_eval(raw_content)
                    if isinstance(parsed_ast, (dict, list)):
                        candidates.append((span, parsed_ast))
                        covered_spans.append(span)
                except Exception:
                    pass

        # 3. Raw Embedded or Trailing JSON Objects
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            # Find next brace or bracket outside covered spans
            brace_pos = text.find("{", idx)
            bracket_pos = text.find("[", idx)
            if brace_pos == -1 and bracket_pos == -1:
                break
            elif brace_pos != -1 and bracket_pos != -1:
                start_pos = min(brace_pos, bracket_pos)
            elif brace_pos != -1:
                start_pos = brace_pos
            else:
                start_pos = bracket_pos

            # Skip if inside an already-covered span
            in_covered = False
            for c_start, c_end in covered_spans:
                if c_start <= start_pos < c_end:
                    idx = c_end
                    in_covered = True
                    break
            if in_covered:
                continue

            try:
                obj, end_pos = decoder.raw_decode(text, start_pos)
                if isinstance(obj, (dict, list)):
                    span = (start_pos, end_pos)
                    candidates.append((span, obj))
                    covered_spans.append(span)
                    idx = end_pos
                else:
                    idx = start_pos + 1
            except json.JSONDecodeError:
                idx = start_pos + 1

        # Sort candidate matches sequentially by start offset
        candidates.sort(key=lambda c: c[0][0])

        # Filter out quoted examples, markdown blockquotes, and explanatory blocks
        filtered_candidates: List[Tuple[Tuple[int, int], Any]] = []
        for span, item in candidates:
            if cls._is_quoted_or_explanatory(text, span):
                logger.debug("[FallbackToolParser] Skipped quoted/explanatory candidate at %s", span)
                continue
            filtered_candidates.append((span, item))

        if not filtered_candidates:
            return []

        # Check if remaining candidates present ambiguous or competing alternatives
        candidate_spans = [span for span, _ in filtered_candidates]
        if cls._are_candidates_ambiguous(text, candidate_spans):
            logger.warning("[FallbackToolParser] Rejected ambiguous/competing candidate tool calls")
            return []

        valid_calls: List[Tuple[Tuple[int, int], str, Dict[str, Any], str]] = []
        seen_signatures: Set[Tuple[str, str]] = set()

        for span, item in filtered_candidates:
            if max_calls is not None and len(valid_calls) >= max_calls:
                break

            items_to_check = item if isinstance(item, list) else [item]
            block_had_valid_tool = False

            for sub_item in items_to_check:
                if max_calls is not None and len(valid_calls) >= max_calls:
                    break
                if not isinstance(sub_item, dict):
                    continue

                raw_name, raw_args = cls._extract_name_and_args_from_dict(sub_item)
                norm_res = cls._normalize_name_and_args(
                    raw_name,
                    raw_args,
                    known_tools=known_tools,
                    active_schemas=active_schemas,
                    caller_ring=caller_ring,
                    allowed_tools=allowed_tools,
                )
                if norm_res is not None:
                    canonical_name, normalized_args = norm_res
                    args_key = json.dumps(normalized_args, sort_keys=True)
                    sig = (canonical_name, args_key)
                    if sig in seen_signatures:
                        # Avoid duplicate execution of identical call in same turn
                        continue
                    seen_signatures.add(sig)

                    # Synthesize deterministic tool call ID
                    payload_bytes = f"{canonical_name}:{args_key}".encode("utf-8")
                    synth_id = f"synth_{hashlib.sha256(payload_bytes).hexdigest()[:12]}"

                    valid_calls.append((span, canonical_name, normalized_args, synth_id))
                    block_had_valid_tool = True

            # If this span produced a valid tool call, keep it for stripping
            if not block_had_valid_tool:
                # Do not strip non-matching candidate blocks
                pass

        if not valid_calls:
            return []

        # Remove stripped spans from text to construct cleaned_text
        spans_to_remove = sorted({call[0] for call in valid_calls}, key=lambda s: s[0])
        cleaned_parts = []
        last_idx = 0
        for s_start, s_end in spans_to_remove:
            cleaned_parts.append(text[last_idx:s_start])
            last_idx = s_end
        cleaned_parts.append(text[last_idx:])

        cleaned_text = "".join(cleaned_parts).strip()

        return [
            ExtractedToolCall(
                name=call[1],
                arguments=call[2],
                tool_call_id=call[3],
                cleaned_text=cleaned_text,
            )
            for call in valid_calls
        ]
