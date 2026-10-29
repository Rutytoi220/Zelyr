"""Isolated Process Execution and Resource Sandboxing for Plugins.

This module provides an isolated worker process architecture for executing untrusted
or dynamic third-party plugins. Plugins execute inside a dedicated child process with:
1. POSIX virtual memory limits (RLIMIT_AS via resource.setrlimit).
2. Strict execution timeouts with bounded SIGTERM and SIGKILL escalation.
3. Crash containment catching abnormal exits (SIGSEGV, SIGABRT, unhandled native exits)
   and returning structured ToolResult failure envelopes without destabilizing the host daemon.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import os
import signal
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB default RAM ceiling
DEFAULT_TIMEOUT_SECONDS = 30.0
SIGTERM_GRACE_PERIOD_SECONDS = 1.0


class PluginExecutionError(Exception):
    """Raised when an isolated plugin process fails or terminates unexpectedly."""


class PluginTimeoutError(PluginExecutionError):
    """Raised when an isolated plugin exceeds its execution timeout budget."""


class PluginMemoryLimitError(PluginExecutionError):
    """Raised when an isolated plugin exceeds its configured RAM ceiling."""


def _sandbox_worker(
    target: Any,
    args: tuple,
    kwargs: dict,
    max_memory_bytes: Optional[int],
    conn: Any,
    parent_conn: Optional[Any] = None,
) -> None:
    """Pre-exec child worker entrypoint running in an isolated process.

    Enforces POSIX resource limits and executes the plugin target, transmitting
    the structured result or failure envelope across the IPC pipe before exiting.
    """
    if parent_conn is not None:
        try:
            parent_conn.close()
        except Exception:
            pass

    try:
        # Disable faulthandler in the isolated child to prevent stderr noise on deliberate crashes
        try:
            import faulthandler

            faulthandler.disable()
        except Exception:
            pass

        # 1. Enforce POSIX virtual memory limits (RLIMIT_AS)
        if max_memory_bytes is not None and max_memory_bytes > 0:
            try:
                import resource

                resource.setrlimit(
                    resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes)
                )
            except Exception as rlimit_exc:
                logger.warning("Failed to enforce RLIMIT_AS in sandbox: %s", rlimit_exc)

        # 2. Resolve target callable or tool
        if inspect.isclass(target):
            instance = target()
            target_fn = getattr(instance, "execute", instance)
        elif hasattr(target, "execute") and callable(target.execute):
            target_fn = target.execute
        elif callable(target):
            target_fn = target
        else:
            raise TypeError(f"Target '{target}' is not callable or a tool instance")

        # 3. Adapt calling convention (single dict parameter vs explicit kwargs)
        sig = inspect.signature(target_fn)
        params_list = list(sig.parameters.values())
        single_dict_param = len(params_list) == 1 and (
            params_list[0].annotation == Dict[str, Any]
            or params_list[0].name in ("params", "arguments", "kwargs")
        )

        if single_dict_param and kwargs and not args:
            invoke_args = (kwargs,)
            invoke_kwargs = {}
        else:
            invoke_args = args
            invoke_kwargs = kwargs

        # 4. Execute synchronously or asynchronously
        if inspect.iscoroutinefunction(target_fn):
            result = asyncio.run(target_fn(*invoke_args, **invoke_kwargs))
        else:
            result = target_fn(*invoke_args, **invoke_kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

        # 5. Transmit result envelope across IPC pipe
        if isinstance(result, ToolResult):
            conn.send(
                (
                    "tool_result",
                    result.success,
                    result.output,
                    result.error,
                    result.metadata,
                    result.remedy_hint,
                    result.allowed_actions,
                )
            )
        else:
            conn.send(("ok", result))

    except MemoryError as mem_exc:
        conn.send(("memory_error", str(mem_exc) or "Memory allocation ceiling exceeded"))
    except Exception as exc:
        conn.send(("error", str(exc), traceback.format_exc()))
    except BaseException as base_exc:
        if isinstance(base_exc, SystemExit) and base_exc.code not in (0, None):
            conn.send(("exit", base_exc.code))
        else:
            conn.send(("error", str(base_exc), traceback.format_exc()))
    finally:
        try:
            conn.close()
        except Exception:
            pass
        os._exit(0)


class SandboxedPluginRunner(BaseTool):
    """Supervised runner executing plugin tools inside an isolated child process.

    Provides complete process isolation, POSIX memory ceiling enforcement,
    guaranteed timeout cleanup, and crash containment to ensure host stability.
    """

    def __init__(
        self,
        target: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Any] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_memory_mb: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> None:
        self._target = target
        self.timeout = float(timeout)

        if max_memory_bytes is not None:
            self.max_memory_bytes = int(max_memory_bytes)
        elif max_memory_mb is not None:
            self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        else:
            self.max_memory_bytes = DEFAULT_MAX_MEMORY_BYTES

        # Extract tool metadata from target if available
        resolved_name = (
            name
            or getattr(target, "name", None)
            or getattr(target, "tool_id", None)
            or getattr(target, "__name__", "sandboxed_plugin")
        )
        resolved_description = (
            description
            or getattr(target, "description", None)
            or getattr(target, "__doc__", "Sandboxed plugin tool execution")
        )

        super().__init__(
            tool_id=str(resolved_name),
            name=str(resolved_name),
            description=str(resolved_description).strip(),
        )

        # Propagate parameters
        target_params = parameters or getattr(target, "parameters", None)
        if isinstance(target_params, list):
            for param in target_params:
                if isinstance(param, ToolParameter):
                    self.add_parameter(param)
                elif isinstance(param, dict):
                    self.add_parameter(
                        ToolParameter(
                            name=param.get("name", ""),
                            type=param.get("type", "string"),
                            description=param.get("description", ""),
                            required=param.get("required", True),
                            default=param.get("default"),
                        )
                    )
        elif isinstance(target_params, dict) and "properties" in target_params:
            props = target_params.get("properties", {})
            required_fields = set(target_params.get("required", []))
            for p_name, p_info in props.items():
                self.add_parameter(
                    ToolParameter(
                        name=p_name,
                        type=p_info.get("type", "string"),
                        description=p_info.get("description", ""),
                        required=p_name in required_fields,
                        default=p_info.get("default"),
                    )
                )

    @property
    def target(self) -> Any:
        """Return the wrapped target plugin or callable."""
        return self._target

    def reload(self, new_target: Any = None) -> SandboxedPluginRunner:
        """Hot-reload the plugin target without leaving zombie processes."""
        if new_target is not None:
            self._target = new_target
            if hasattr(new_target, "name"):
                self.name = new_target.name
                self.tool_id = new_target.name
            if hasattr(new_target, "description"):
                self.description = new_target.description
        return self

    def close(self) -> None:
        """Lifecycle teardown hook."""
        pass

    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute target in an isolated child process synchronously."""
        # Use 'fork' context on POSIX systems for lightweight and complete memory preservation
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context()

        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_sandbox_worker,
            args=(self._target, args, kwargs, self.max_memory_bytes, child_conn, parent_conn),
            daemon=True,
        )

        try:
            proc.start()
            child_conn.close()

            # Poll for worker response up to timeout threshold with process liveness checks
            import time
            deadline = time.monotonic() + self.timeout
            has_data = False
            while time.monotonic() < deadline:
                if parent_conn.poll(0.02):
                    has_data = True
                    break
                if not proc.is_alive():
                    # Process exited or was killed by signal
                    has_data = parent_conn.poll(0.0)
                    break

            if not has_data and proc.is_alive():
                # Execution timed out: escalate termination gracefully then forcefully
                logger.warning(
                    "Plugin '%s' exceeded timeout %.1fs. Escalating termination...",
                    self.name,
                    self.timeout,
                )
                proc.terminate()
                proc.join(timeout=SIGTERM_GRACE_PERIOD_SECONDS)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"PluginExecutionError: Plugin execution timed out after {self.timeout}s.",
                )

            # Read response envelope from child worker
            msg = None
            if has_data:
                try:
                    msg = parent_conn.recv()
                except EOFError:
                    msg = None

            proc.join(timeout=1.0)

            if msg is None:
                # Worker terminated prematurely (e.g. SIGSEGV, SIGABRT, OOM kill)
                exitcode = proc.exitcode if proc.exitcode is not None else -1
                sig_str = self._format_signal_info(exitcode)
                error_msg = (
                    f"PluginExecutionError: Plugin process terminated unexpectedly "
                    f"(signal={sig_str}, exit_code={exitcode})"
                )
                logger.error("Plugin crash contained for '%s': %s", self.name, error_msg)
                return ToolResult(success=False, output="", error=error_msg)

            # Map response messages to ToolResult
            status = msg[0]
            if status == "ok":
                payload = msg[1]
                if isinstance(payload, ToolResult):
                    return payload
                return ToolResult(success=True, output=payload)

            if status == "tool_result":
                _, success, output, error, meta, remedy, actions = msg
                return ToolResult(
                    success=success,
                    output=output,
                    error=error,
                    metadata=meta,
                    remedy_hint=remedy,
                    allowed_actions=actions,
                )

            if status == "memory_error":
                mem_detail = msg[1]
                error_msg = (
                    f"PluginExecutionError: Memory ceiling exceeded ({mem_detail})"
                )
                logger.warning("Memory limit exceeded for '%s': %s", self.name, error_msg)
                return ToolResult(success=False, output="", error=error_msg)

            if status == "error":
                err_detail = msg[1]
                return ToolResult(
                    success=False, output="", error=f"PluginError: {err_detail}"
                )

            if status == "exit":
                code = msg[1]
                sig_str = self._format_signal_info(code)
                error_msg = (
                    f"PluginExecutionError: Plugin process terminated unexpectedly "
                    f"(signal={sig_str}, exit_code={code})"
                )
                return ToolResult(success=False, output="", error=error_msg)

            return ToolResult(
                success=False,
                output="",
                error=f"PluginExecutionError: Unknown response status '{status}'",
            )

        finally:
            # Guarantee child process cleanup to prevent zombie processes
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=0.5)
                elif proc.exitcode is None:
                    proc.join(timeout=0.5)
            except Exception:
                pass
            try:
                parent_conn.close()
            except Exception:
                pass

    async def run_async(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute target in an isolated child process asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.run, *args, **kwargs)

    def execute(
        self, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> ToolResult:
        """Synchronous execution entrypoint compatible with BaseTool."""
        merged = dict(params or {})
        merged.update(kwargs)
        return self.run(**merged)

    async def execute_async(
        self, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> ToolResult:
        """Asynchronous execution entrypoint compatible with ToolRegistry."""
        merged = dict(params or {})
        merged.update(kwargs)
        return await self.run_async(**merged)

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Callable invocation proxy."""
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self.run(**args[0])
        return self.run(*args, **kwargs)

    @staticmethod
    def _format_signal_info(exitcode: int) -> str:
        """Format signal description from process exitcode."""
        sig_num: Optional[int] = None
        if exitcode < 0:
            sig_num = -exitcode
        elif exitcode > 128:
            sig_num = exitcode - 128

        if sig_num is not None:
            try:
                sig_name = signal.Signals(sig_num).name
                return f"{sig_name} ({sig_num})"
            except (ValueError, AttributeError):
                return str(sig_num)
        return "None"
