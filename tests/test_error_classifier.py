"""Unit tests for ErrorClassifier engine and classified observation formatting."""

import json
from zelyr.engine.error_classifier import (
    ErrorClassification,
    ErrorClassifier,
    classify_tool_error,
)


class TestErrorClassifier:
    """Test suite for domain-specific error classification and remedy hints."""

    def test_classify_command_not_found_by_exit_code(self):
        res = classify_tool_error("execute_command", exit_code=127, output="", error="bash: foo: not found")
        assert res.error_type == "COMMAND_NOT_FOUND"
        assert "Command not found in PATH" in res.remedy_hint
        assert "execute_command" in res.allowed_actions
        assert "manage_workspace_file" in res.allowed_actions

    def test_classify_command_not_found_by_string(self):
        res = classify_tool_error("execute_command", exit_code=1, output="", error="zsh: command not found: mytool")
        assert res.error_type == "COMMAND_NOT_FOUND"
        assert "Command not found in PATH" in res.remedy_hint
        assert "execute_command" in res.allowed_actions

    def test_classify_permission_denied_by_exit_code(self):
        res = classify_tool_error("execute_command", exit_code=126, output="", error="cannot execute: Permission denied")
        assert res.error_type == "PERMISSION_DENIED"
        assert "Permission denied on target resource" in res.remedy_hint
        assert "Do NOT invoke interactive sudo" in res.remedy_hint
        assert "execute_command" in res.allowed_actions

    def test_classify_permission_denied_by_string(self):
        res = classify_tool_error("manage_workspace_file", exit_code=1, output="", error="Operation not permitted on /root/secret")
        assert res.error_type == "PERMISSION_DENIED"
        assert "Permission denied on target resource" in res.remedy_hint

    def test_classify_timeout_by_exit_code(self):
        res = classify_tool_error("execute_command", exit_code=124, output="", error="Job timed out after 60s")
        assert res.error_type == "TIMEOUT"
        assert "Command timed out" in res.remedy_hint
        assert res.allowed_actions == ["execute_command"]

    def test_classify_timeout_by_string(self):
        res = classify_tool_error("execute_command", exit_code=None, output="", error="Command 'sleep 100' timed out after 10.0 seconds")
        assert res.error_type == "TIMEOUT"
        assert "Command timed out" in res.remedy_hint

    def test_classify_path_not_found(self):
        res = classify_tool_error("execute_command", exit_code=2, output="", error="ls: cannot access '/foo/bar': No such file or directory")
        assert res.error_type == "PATH_NOT_FOUND"
        assert "Target file or directory does not exist" in res.remedy_hint
        assert "verify_outcome" in res.allowed_actions

    def test_classify_path_not_found_workspace_file(self):
        res = classify_tool_error("manage_workspace_file", exit_code=None, output="", error="File not found: /tmp/missing.txt")
        assert res.error_type == "PATH_NOT_FOUND"
        assert "Target file or directory does not exist" in res.remedy_hint

    def test_classify_port_in_use_by_exit_code(self):
        res = classify_tool_error("execute_command", exit_code=98, output="", error="bind: Address already in use")
        assert res.error_type == "PORT_IN_USE"
        assert "Network port is occupied" in res.remedy_hint
        assert "ss -tulpn" in res.remedy_hint
        assert "inspect_network" in res.allowed_actions

    def test_classify_port_in_use_by_string(self):
        res = classify_tool_error("execute_command", exit_code=1, output="", error="listen tcp 127.0.0.1:8080: bind: address already in use")
        assert res.error_type == "PORT_IN_USE"
        assert "Network port is occupied" in res.remedy_hint

    def test_classify_verification_failed_by_tool_name(self):
        res = classify_tool_error("verify_outcome", exit_code=0, output="", error="File /tmp/foo.txt does not exist")
        assert res.error_type == "VERIFICATION_FAILED"
        assert "Observed state did not match expected criteria" in res.remedy_hint
        assert "verify_outcome" in res.allowed_actions

    def test_classify_verification_failed_by_string(self):
        res = classify_tool_error("custom_checker", exit_code=1, output="", error="Checksum mismatch: expected abc, got def. Verification failed.")
        assert res.error_type == "VERIFICATION_FAILED"
        assert "Observed state did not match expected criteria" in res.remedy_hint

    def test_classify_general_error_fallback(self):
        res = classify_tool_error("execute_command", exit_code=1, output="", error="Unknown fatal runtime corruption 0x88472")
        assert res.error_type == "GENERAL_ERROR"
        assert "Action failed" in res.remedy_hint
        assert "execute_command" in res.allowed_actions

    def test_classification_to_dict_and_facade(self):
        res = ErrorClassifier.classify("execute_command", exit_code=127, output="", error="not found")
        assert isinstance(res, ErrorClassification)
        d = res.to_dict()
        assert d["error_type"] == "COMMAND_NOT_FOUND"
        assert "remedy_hint" in d
        assert "allowed_actions" in d
