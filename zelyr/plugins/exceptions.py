"""Custom exceptions for the Zelyr Plugin Sandbox Engine.

These exceptions form the public security API — callers can catch
`SandboxSecurityViolation` to detect any policy breach regardless of the
specific violation type.
"""

class PluginError(Exception):
    """Base class for all plugin-related errors."""

class PluginManifestError(PluginError):
    """Raised when a plugin.toml is malformed or fails schema validation."""

class PluginVersionError(PluginError):
    """Raised when a plugin's required Zelyr version is incompatible."""

class PluginPermissionError(PluginError):
    """Raised when a plugin requests a permission the user has not granted."""

class SandboxSecurityViolation(PluginError):
    """Raised when a sandboxed plugin attempts a forbidden operation.

    This is the critical security exception.  It indicates the plugin code
    tried to perform an action that exceeds its declared permission scope
    (e.g., opening a socket without `network` permission, reading files
    outside its workspace without `filesystem` permission, or invoking
    a shell without `shell` permission).

    Attributes:
        plugin_id: The plugin that violated the policy.
        violation_type: A short machine-readable key (``"network"``,
            ``"filesystem"``, ``"shell"``, ``"import"``).
        detail: Human-readable explanation of the exact violation.
    """

    def __init__(self, plugin_id: str, violation_type: str, detail: str) -> None:
        """Auto-generated docstring.

Args:
    plugin_id: Argument.
    violation_type: Argument.
    detail: Argument.

Returns:
    Return value.
"""
        self.plugin_id = plugin_id  # pragma: no cover
        self.violation_type = violation_type  # pragma: no cover
        self.detail = detail  # pragma: no cover
        super().__init__(f'[{plugin_id}] Security violation ({violation_type}): {detail}')  # pragma: no cover
