"""Deterministic observation framing and semantic state anchors for tool execution."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union


class ObservationFramingEngine:
    """Formats and parses tool execution observations with deterministic semantic anchors."""

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
    def frame_observation(
        cls,
        tool_name: str,
        result: Any,
        as_json: bool = False,
    ) -> str:
        """Frame a tool execution result with deterministic semantic anchors.

        Returns:
            Clean human/model-readable framed string or structured JSON string if as_json=True.
        """
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
            raw_str = json.dumps(result)
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
                received_val = envelope.get("received_value") or ""
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
                elif parsed.get("status") in ("error", "failed", "failure", "tool_execution_failed"):
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

                output = json.dumps(out_val) if isinstance(out_val, dict) else str(out_val if out_val is not None else "")

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
                        # Distinguish an explicit verification failure from a legacy ActionRecord dict
                        # where verified=False is merely the default unverified state.
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

        # Check for inline [VERIFIED] marker only from dedicated verification tool
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

            classification = classify_tool_error(tool_name, exit_code, output, error)
            if not error_type:
                error_type = classification.error_type
            if not remedy_hint:
                remedy_hint = classification.remedy_hint
            if not allowed_actions:
                allowed_actions = list(classification.allowed_actions)

            err_clean = error.rstrip(".")
            if "[Tool Error]:" in err_clean:
                # Remove duplicated prefix
                err_clean = err_clean.split("[Tool Error]:", 1)[1].strip()
            if "Recovery Hint:" in err_clean:
                err_clean = err_clean.split("Recovery Hint:", 1)[0].strip().rstrip(".")

            error = err_clean

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
            }
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
                        data_payload[ver_key] = parsed[ver_key]
            data_payload["output"] = cls._bound_stream_text(output) if output else output

            return json.dumps(data_payload, indent=2)

        # Build clean framed text observation
        lines: List[str] = [
            f"[TOOL OBSERVATION: {tool_name}]",
            f"- Status: {status}",
            f"- Exit Code: {exit_code if exit_code is not None else 'N/A'}",
            f"- Verified: {'TRUE' if verified is True else ('FALSE' if verified is False else 'N/A')}",
            f"- Summary: {summary}",
        ]

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
