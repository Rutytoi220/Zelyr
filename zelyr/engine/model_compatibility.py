"""Model and provider compatibility diagnostics distinguishing detected protocol facts from unknowns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

from zelyr.config import ZelyrConfig, get_config, validate_ollama_base_url
from zelyr.engine.environment import CapabilityStatus


_SUPPORTED_WORKFLOWS = frozenset({"chat", "react_tools", "planned_graph", "vision"})


def _scrub_url(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"(?i)(https?://)([^/@\s]+@)", r"\1[REDACTED]@", str(url)).strip()


def _scrub_err(val: Any) -> str:
    txt = str(val or "")
    txt = re.sub(r"(?i)(https?://)([^/@\s]+@)", r"\1[REDACTED]@", txt)
    txt = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", txt)
    txt = re.sub(r"(?i)\b(bearer\s+)[a-z0-9._\-+=]+", r"\1[REDACTED]", txt)
    return txt


def _default_http_json(
    method: str,
    url: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 2.0,
) -> tuple[int, Dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        code = int(getattr(resp, "status", 200))
        raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip() else {}
        return code, parsed if isinstance(parsed, dict) else {}


def _cap_entry(
    status: CapabilityStatus,
    *,
    source: str,
    confidence: str,
    detail: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": status.value,
        "verified": status in (CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED, CapabilityStatus.UNAVAILABLE)
        and source in ("detected", "protocol_probe"),
        "source": source,
        "confidence": confidence,
        "detail": _scrub_err(detail),
    }
    if extra:
        out.update(extra)
    return out


@dataclass(frozen=True)
class ModelCompatibilityReport:
    """Honest compatibility assessment for the selected model, endpoint provider, and workflow."""

    model: str
    endpoint: str
    provider_type: str
    workflow: str
    overall_status: str
    model_capability_confidence: str
    model_installed: Optional[bool]
    installed_models: List[str] = field(default_factory=list)
    capabilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context_info: Dict[str, Any] = field(default_factory=dict)
    workflow_compatible: Optional[bool] = None
    blocking_gaps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelCompatibilityDiagnostics:
    """Evaluates provider availability and model capabilities strictly from protocol/runtime evidence."""

    @staticmethod
    def diagnose(
        *,
        config: Optional[ZelyrConfig] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        workflow: str = "planned_graph",
        probe: bool = True,
        timeout: float = 2.0,
        http_request_fn: Optional[Callable[..., tuple[int, Dict[str, Any]]]] = None,
        protocol_observations: Optional[Mapping[str, Any]] = None,
    ) -> ModelCompatibilityReport:
        cfg = config or get_config()
        eff_workflow = str(workflow or "planned_graph").strip().lower()
        if eff_workflow not in _SUPPORTED_WORKFLOWS:
            eff_workflow = "planned_graph"

        eff_model = str(
            model
            or (cfg.vision_model if eff_workflow == "vision" and cfg.vision_model else None)
            or getattr(cfg, "ollama_model", "qwen3:8b")
        ).strip()
        eff_url = validate_ollama_base_url(
            base_url
            or (cfg.get_effective_ollama_base_url() if hasattr(cfg, "get_effective_ollama_base_url") else getattr(cfg, "ollama_base_url", ""))
        )
        if eff_url:
            parsed_scheme = urllib.parse.urlparse(eff_url).scheme.lower()
            if parsed_scheme not in ("http", "https"):
                raise ValueError(
                    f"Unsupported provider URL scheme '{parsed_scheme}' (expected http or https)."
                )
        scrubbed_url = _scrub_url(eff_url)
        configured_ctx = (
            cfg.get_effective_num_ctx()
            if hasattr(cfg, "get_effective_num_ctx")
            else int(getattr(cfg, "ollama_num_ctx", 8192))
        )

        # Provider inference from explicit model prefix / endpoint shape (protocol type only, NEVER model capability!)
        low_url = eff_url.lower()
        if eff_model.startswith("litellm:") or "/" in eff_model and not eff_model.startswith("ollama/"):
            provider_type = "litellm_external"
        elif low_url.endswith("/v1") or "/v1" in low_url:
            provider_type = "openai_compatible"
        else:
            provider_type = "ollama"
        eff_req_model = eff_model[len("ollama/"):] if eff_model.startswith("ollama/") else eff_model

        requester = http_request_fn or _default_http_json
        bounded_timeout = max(0.05, min(float(timeout or 2.0), 5.0))
        blocking_gaps: List[str] = []
        warnings: List[str] = []

        installed_models: List[str] = []
        model_installed: Optional[bool] = None
        provider_status = CapabilityStatus.UNKNOWN
        provider_detail = "Live endpoint probing skipped (--no-probe); provider status is unknown."
        provider_source = "unknown"

        detected_caps: Optional[List[str]] = None
        detected_ctx_len: Optional[int] = None
        model_family: Optional[str] = None
        parameter_size: Optional[str] = None
        show_probed = False
        show_ok = False
        chat_endpoint_reachable: Optional[bool] = None

        if probe and eff_url:
            tags_path = "/models" if eff_url.rstrip("/").endswith("/v1") else "/api/tags"
            tags_url = eff_url.rstrip("/") + tags_path
            try:
                status_code, tags_data = requester("GET", tags_url, timeout=bounded_timeout)
                provider_source = "detected"
                if status_code != 200:
                    provider_status = CapabilityStatus.DEGRADED
                    provider_detail = f"Provider endpoint returned HTTP {status_code}."
                    blocking_gaps.append(provider_detail)
                else:
                    raw_models = tags_data.get("models") if isinstance(tags_data, dict) else None
                    if isinstance(raw_models, list):
                        for item in raw_models:
                            if isinstance(item, dict):
                                mid = item.get("name") or item.get("model") or item.get("id")
                                if isinstance(mid, str) and mid.strip():
                                    installed_models.append(mid.strip())
                    elif isinstance(tags_data, dict) and isinstance(tags_data.get("data"), list):
                        for item in tags_data["data"]:
                            if isinstance(item, dict) and isinstance(item.get("id"), str):
                                installed_models.append(item["id"].strip())

                    if installed_models or (isinstance(tags_data, dict) and ("models" in tags_data or "data" in tags_data)):
                        base_req = eff_req_model.split(":")[0].lower()
                        matched_exact = any(m.lower() == eff_req_model.lower() for m in installed_models)
                        matched_family = any(m.split(":")[0].lower() == base_req for m in installed_models)
                        model_installed = bool(matched_exact or matched_family)
                        if model_installed:
                            provider_status = CapabilityStatus.AVAILABLE
                            provider_detail = f"Provider reachable; model '{eff_req_model}' found in catalog."
                        else:
                            provider_status = CapabilityStatus.DEGRADED
                            provider_detail = f"Provider reachable, but model '{eff_req_model}' not found in catalog."
                            blocking_gaps.append(
                                f"Configured model '{eff_req_model}' is not listed by provider endpoint."
                            )
                    else:
                        provider_status = CapabilityStatus.AVAILABLE
                        provider_detail = "Provider endpoint reachable."
            except (TimeoutError, socket.timeout):
                provider_status = CapabilityStatus.UNAVAILABLE
                provider_source = "detected"
                provider_detail = f"Provider health probe timed out after {bounded_timeout:.2f}s."
                blocking_gaps.append(provider_detail)
            except urllib.error.HTTPError as exc:
                provider_source = "detected"
                provider_status = (
                    CapabilityStatus.DEGRADED if exc.code in (401, 403, 404) else CapabilityStatus.UNAVAILABLE
                )
                provider_detail = f"Provider endpoint returned HTTP {exc.code}."
                blocking_gaps.append(provider_detail)
            except Exception as exc:
                provider_source = "detected"
                provider_status = CapabilityStatus.UNAVAILABLE
                provider_detail = f"Provider endpoint unreachable ({_scrub_err(exc)})."
                blocking_gaps.append(provider_detail)

            if provider_status == CapabilityStatus.AVAILABLE:
                chat_endpoint_reachable = True

            # Query Ollama /api/show metadata without triggering model weight load when provider is Ollama and reachable
            if provider_type == "ollama" and provider_status in (CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED):
                show_url = eff_url.rstrip("/") + "/api/show"
                show_target = eff_req_model
                if not any(m == eff_req_model for m in installed_models):
                    latest = f"{eff_req_model}:latest"
                    if latest in installed_models:
                        show_target = latest
                if model_installed is not False:
                    try:
                        show_probed = True
                        s_code, s_data = requester(
                            "POST",
                            show_url,
                            body={"model": show_target},
                            timeout=bounded_timeout,
                        )
                        if s_code == 200 and isinstance(s_data, dict):
                            show_ok = True
                            raw_caps = s_data.get("capabilities")
                            if isinstance(raw_caps, list):
                                detected_caps = [str(c).strip().lower() for c in raw_caps if str(c).strip()]
                            det = s_data.get("details")
                            if isinstance(det, dict):
                                model_family = det.get("family")
                                parameter_size = det.get("parameter_size")
                            mi = s_data.get("model_info")
                            if isinstance(mi, dict):
                                for k, v in mi.items():
                                    if str(k).endswith(".context_length") and isinstance(v, (int, float)) and not isinstance(v, bool):
                                        detected_ctx_len = int(v)
                                        break
                    except Exception:
                        show_ok = False

        proto = dict(protocol_observations or {})

        # 1. Provider availability
        caps_out: Dict[str, Dict[str, Any]] = {
            "provider_availability": _cap_entry(
                provider_status,
                source=provider_source,
                confidence="high" if provider_source == "detected" else "unknown",
                detail=provider_detail,
            )
        }

        # 2. Native tool calling (strictly detected from /api/show capabilities or explicit protocol_observations)
        if "native_tool_calling" in proto:
            nt_val = bool(proto["native_tool_calling"])
            caps_out["native_tool_calling"] = _cap_entry(
                CapabilityStatus.AVAILABLE if nt_val else CapabilityStatus.UNAVAILABLE,
                source="protocol_probe",
                confidence="high",
                detail="Verified via protocol test fixture." if nt_val else "Protocol fixture failed native tool call check.",
            )
        elif detected_caps is not None:
            has_tools = "tools" in detected_caps
            caps_out["native_tool_calling"] = _cap_entry(
                CapabilityStatus.AVAILABLE if has_tools else CapabilityStatus.UNAVAILABLE,
                source="detected",
                confidence="high",
                detail=(
                    "Provider /api/show explicitly declares 'tools' capability."
                    if has_tools
                    else f"Provider /api/show capabilities {detected_caps} omit 'tools'."
                ),
            )
            if not has_tools and eff_workflow in ("react_tools", "planned_graph"):
                warnings.append("Native tool calling unavailable in /api/show; Zelyr relies on FallbackToolParser (DEGRADED).")
        else:
            caps_out["native_tool_calling"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Native tool calling cannot be confirmed without /api/show capability metadata or protocol probe (never assumed from model name).",
            )

        # 3. Tool argument formatting
        if "tool_argument_formatting" in proto:
            taf_ok = bool(proto["tool_argument_formatting"])
            caps_out["tool_argument_formatting"] = _cap_entry(
                CapabilityStatus.AVAILABLE if taf_ok else CapabilityStatus.DEGRADED,
                source="protocol_probe",
                confidence="high",
                detail="Structured argument normalization & formatting verified via protocol probe." if taf_ok else "Malformed/stringified tool arguments required repair.",
            )
        elif caps_out["native_tool_calling"]["status"] == CapabilityStatus.AVAILABLE.value:
            caps_out["tool_argument_formatting"] = _cap_entry(
                CapabilityStatus.AVAILABLE,
                source="detected",
                confidence="medium",
                detail="Native /api/chat tool argument serializer + SafeArgumentNormalizer active.",
            )
        elif caps_out["native_tool_calling"]["status"] == CapabilityStatus.UNAVAILABLE.value:
            caps_out["tool_argument_formatting"] = _cap_entry(
                CapabilityStatus.DEGRADED,
                source="detected",
                confidence="medium",
                detail="FallbackToolParser handles text tool calls; native tool_calls schema unavailable.",
            )
        else:
            caps_out["tool_argument_formatting"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Tool argument reliability is unknown until native capability or protocol trace is observed.",
            )

        # 4. Multi-turn tool calling
        if "multi_turn_tool_call" in proto:
            mt_ok = bool(proto["multi_turn_tool_call"])
            caps_out["multi_turn_tool_calling"] = _cap_entry(
                CapabilityStatus.AVAILABLE if mt_ok else CapabilityStatus.UNAVAILABLE,
                source="protocol_probe",
                confidence="high",
                detail="Multi-turn tool roundtrip verified." if mt_ok else "Multi-turn tool roundtrip failed protocol check.",
            )
        else:
            caps_out["multi_turn_tool_calling"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Multi-turn tool-call adherence requires multi-step execution evidence and cannot be inferred safely offline.",
            )

        # 5. Thinking metadata
        if "thinking_metadata" in proto:
            th_ok = bool(proto["thinking_metadata"])
            caps_out["thinking_metadata"] = _cap_entry(
                CapabilityStatus.AVAILABLE if th_ok else CapabilityStatus.UNAVAILABLE,
                source="protocol_probe",
                confidence="high",
                detail="Model emitted structured thinking/reasoning tokens." if th_ok else "No thinking metadata emitted.",
            )
        elif detected_caps is not None:
            has_think = "thinking" in detected_caps or "reasoning" in detected_caps
            caps_out["thinking_metadata"] = _cap_entry(
                CapabilityStatus.AVAILABLE if has_think else CapabilityStatus.UNKNOWN,
                source="detected" if has_think else "unknown",
                confidence="high" if has_think else "low",
                detail=(
                    "Provider /api/show declares 'thinking' capability (supported alongside <think> & reasoning_content)."
                    if has_think
                    else "Provider /api/show did not declare 'thinking'; model may still emit <think> tags."
                ),
            )
        else:
            caps_out["thinking_metadata"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Adapter supports reasoning_content & <think> tags, but model emission is unknown without probe.",
            )

        # 6. Context limits
        eff_ctx_len = detected_ctx_len
        ctx_source = "detected"
        if eff_ctx_len is None and isinstance(proto.get("context_length"), (int, float)) and not isinstance(proto.get("context_length"), bool):
            eff_ctx_len = int(proto["context_length"])
            ctx_source = "protocol_probe"
        if eff_ctx_len is not None:
            if configured_ctx > eff_ctx_len:
                ctx_st = CapabilityStatus.DEGRADED
                ctx_detail = f"Configured num_ctx={configured_ctx} exceeds detected model context_length={eff_ctx_len}; risk of silent truncation."
                warnings.append(ctx_detail)
            else:
                ctx_st = CapabilityStatus.AVAILABLE
                ctx_detail = f"Configured num_ctx={configured_ctx} fits detected model context_length={eff_ctx_len}."
            caps_out["context_limits"] = _cap_entry(
                ctx_st,
                source=ctx_source,
                confidence="high",
                detail=ctx_detail,
                extra={"configured_num_ctx": configured_ctx, "detected_context_length": eff_ctx_len},
            )
        else:
            caps_out["context_limits"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail=f"Configured num_ctx={configured_ctx}, but model max context_length is unknown.",
                extra={"configured_num_ctx": configured_ctx, "detected_context_length": None},
            )

        # 7. Streaming behavior
        if "streaming" in proto:
            st_ok = bool(proto["streaming"])
            caps_out["streaming_behavior"] = _cap_entry(
                CapabilityStatus.AVAILABLE if st_ok else CapabilityStatus.UNAVAILABLE,
                source="protocol_probe",
                confidence="high",
                detail="Streaming NDJSON/SSE chunk parsing verified." if st_ok else "Streaming probe failed.",
            )
        elif chat_endpoint_reachable is True:
            caps_out["streaming_behavior"] = _cap_entry(
                CapabilityStatus.AVAILABLE,
                source="detected",
                confidence="medium",
                detail="Provider reachable; NDJSON & SSE stream adapters available.",
            )
        elif provider_status == CapabilityStatus.UNAVAILABLE:
            caps_out["streaming_behavior"] = _cap_entry(
                CapabilityStatus.UNAVAILABLE,
                source="detected",
                confidence="high",
                detail="Streaming unavailable because provider endpoint is unreachable.",
            )
        else:
            caps_out["streaming_behavior"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Streaming readiness unprobed.",
            )

        # 8. Structured output support
        if "structured_output" in proto:
            so_ok = bool(proto["structured_output"])
            caps_out["structured_output_support"] = _cap_entry(
                CapabilityStatus.AVAILABLE if so_ok else CapabilityStatus.UNAVAILABLE,
                source="protocol_probe",
                confidence="high",
                detail="Valid JSON / PlanCompiler structured response verified." if so_ok else "Structured JSON probe failed.",
            )
        elif detected_caps is not None and ("structured_output" in detected_caps or "json" in detected_caps):
            caps_out["structured_output_support"] = _cap_entry(
                CapabilityStatus.AVAILABLE,
                source="detected",
                confidence="high",
                detail="Provider explicitly declares structured JSON output capability.",
            )
        else:
            caps_out["structured_output_support"] = _cap_entry(
                CapabilityStatus.UNKNOWN,
                source="unknown",
                confidence="unknown",
                detail="Structured JSON adherence cannot be asserted reliably without protocol test (never assumed from model name).",
            )

        # Overall status, capability confidence & workflow compatibility
        has_degraded_cap = any(
            c["status"] == CapabilityStatus.DEGRADED.value for c in caps_out.values()
        )
        has_unavailable_non_provider_cap = any(
            name != "provider_availability" and c["status"] == CapabilityStatus.UNAVAILABLE.value
            for name, c in caps_out.items()
        )
        structured_unavailable = (
            caps_out["structured_output_support"]["status"] == CapabilityStatus.UNAVAILABLE.value
        )
        multi_turn_unavailable = (
            caps_out["multi_turn_tool_calling"]["status"] == CapabilityStatus.UNAVAILABLE.value
        )

        if provider_status == CapabilityStatus.UNAVAILABLE:
            overall = CapabilityStatus.UNAVAILABLE.value
            wf_compat: Optional[bool] = False
        elif model_installed is False:
            overall = CapabilityStatus.DEGRADED.value
            wf_compat = False
        elif eff_workflow == "vision" and (
            (detected_caps is not None and "vision" not in detected_caps) or proto.get("vision") is False
        ):
            overall = CapabilityStatus.DEGRADED.value
            wf_compat = False
            blocking_gaps.append(f"Selected vision workflow requires 'vision' capability missing in {detected_caps}.")
        elif eff_workflow in ("react_tools", "planned_graph") and multi_turn_unavailable:
            overall = CapabilityStatus.DEGRADED.value
            wf_compat = False
            blocking_gaps.append(
                f"Selected '{eff_workflow}' workflow failed multi-turn tool calling verification."
            )
        elif eff_workflow == "planned_graph" and structured_unavailable:
            overall = CapabilityStatus.DEGRADED.value
            wf_compat = False
            blocking_gaps.append(
                "Selected 'planned_graph' workflow requires structured JSON output, which failed protocol verification."
            )
        elif provider_status == CapabilityStatus.DEGRADED or has_degraded_cap or has_unavailable_non_provider_cap:
            overall = CapabilityStatus.DEGRADED.value
            wf_compat = False if blocking_gaps else True
        elif provider_status == CapabilityStatus.UNKNOWN:
            overall = CapabilityStatus.UNKNOWN.value
            wf_compat = None
        else:
            overall = CapabilityStatus.AVAILABLE.value
            if eff_workflow in ("react_tools", "planned_graph") and caps_out["native_tool_calling"]["status"] == CapabilityStatus.UNKNOWN.value:
                wf_compat = None  # Honestly unknown when tool capability unverified
            else:
                wf_compat = True

        detected_count = sum(1 for c in caps_out.values() if c["source"] in ("detected", "protocol_probe"))
        unknown_count = sum(1 for c in caps_out.values() if c["status"] == CapabilityStatus.UNKNOWN.value)
        if detected_count >= 6 and unknown_count == 0:
            confidence = "high"
        elif detected_count >= 3:
            confidence = "medium"
        elif detected_count >= 1:
            confidence = "low"
        else:
            confidence = "unknown"

        return ModelCompatibilityReport(
            model=eff_model,
            endpoint=scrubbed_url,
            provider_type=provider_type,
            workflow=eff_workflow,
            overall_status=overall,
            model_capability_confidence=confidence,
            model_installed=model_installed,
            installed_models=installed_models,
            capabilities=caps_out,
            context_info={
                "configured_num_ctx": configured_ctx,
                "detected_context_length": eff_ctx_len,
                "family": model_family,
                "parameter_size": parameter_size,
                "show_probed": show_probed,
                "show_ok": show_ok,
            },
            workflow_compatible=wf_compat,
            blocking_gaps=blocking_gaps,
            warnings=warnings,
        )


def check_model_compatibility(
    *,
    config: Optional[ZelyrConfig] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    workflow: str = "planned_graph",
    probe: bool = True,
    timeout: float = 2.0,
    http_request_fn: Optional[Callable[..., tuple[int, Dict[str, Any]]]] = None,
    protocol_observations: Optional[Mapping[str, Any]] = None,
) -> ModelCompatibilityReport:
    """Assess provider and model compatibility for a target Zelyr workflow."""
    return ModelCompatibilityDiagnostics.diagnose(
        config=config,
        model=model,
        base_url=base_url,
        workflow=workflow,
        probe=probe,
        timeout=timeout,
        http_request_fn=http_request_fn,
        protocol_observations=protocol_observations,
    )


def format_model_compatibility_report(report: ModelCompatibilityReport) -> str:
    """Format a ModelCompatibilityReport into readable terminal output."""
    lines = [
        f"Model & Provider Compatibility ({report.model} via {report.provider_type}):",
        f"  endpoint:              {report.endpoint}",
        f"  workflow:              {report.workflow} (compatible={report.workflow_compatible})",
        f"  overall_status:        {report.overall_status}  (confidence={report.model_capability_confidence})",
        f"  model_installed:       {report.model_installed} ({len(report.installed_models)} model(s) cataloged)",
        "  capabilities:",
    ]
    for cap_name, c_info in report.capabilities.items():
        lines.append(
            f"    {cap_name:<26} {c_info['status']:<12} [source={c_info['source']}, conf={c_info['confidence']}] - {c_info['detail']}"
        )
    if report.blocking_gaps:
        lines.append("  blocking_gaps:")
        for gap in report.blocking_gaps:
            lines.append(f"    - {gap}")
    if report.warnings:
        lines.append("  warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)


__all__ = [
    "ModelCompatibilityDiagnostics",
    "ModelCompatibilityReport",
    "check_model_compatibility",
    "format_model_compatibility_report",
]
