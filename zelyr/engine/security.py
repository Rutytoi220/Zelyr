import re
import subprocess
import logging
import os
from typing import Dict, Any, Tuple
from zelyr.engine.audit_ledger import AuditLedger
from zelyr.engine.container_sandbox import ContainerSandboxManager
from zelyr.engine.vm_orchestrator import MicroVMManager

logger = logging.getLogger(__name__)

class SecuritySandbox:
    """Security sandbox for evaluating and auditing tool executions."""
    
    HIGH_RISK_PATTERNS = [
        r'rm\s+-rf',
        r'sudo\s+',
        r'mkfs',
        r'dd\s+if=',
        r'/etc/',
        r'wget\s+',
        r'curl\s+',
        r'ssh\s+',
        r'scp\s+'
    ]

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecuritySandbox, cls).__new__(cls)
            cls._instance.ledger = AuditLedger()
            cls._instance._compiled_patterns = [re.compile(p) for p in cls.HIGH_RISK_PATTERNS]
            
            from zelyr.engine.snapshot_engine import SnapshotManager
            cls._instance.snapshot_mgr = SnapshotManager()
            cls._instance.container_mgr = ContainerSandboxManager()
            cls._instance.vm_mgr = MicroVMManager()
        return cls._instance

    def evaluate_command(self, agent_name: str, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a command for risk, log it, and return (is_allowed, reason)."""
        command = arguments.get('command', '')
        if not command and tool_name in ('file_write', 'file_opener'):
            command = arguments.get('path', '')
            
        if tool_name == 'ssh_teleport':
            is_high_risk = True
        else:
            is_high_risk = False
        
        # Check patterns
        for pattern in self._compiled_patterns:
            if pattern.search(command):
                is_high_risk = True
                break
                
        risk_level = "HIGH" if is_high_risk else "LOW"
        
        if is_high_risk:
            # Elevate to KVM Micro-VM mode
            status = "KVM_ELEVATED"
            logger.warning(f"Security Sandbox: High Risk Command detected. Elevating to KVM Micro-VM: {command}")
            vm_id = self.vm_mgr.create_disposable_vm()
            
            # Execute in VM and destroy
            try:
                output = self.vm_mgr.exec_in_vm(vm_id, command)
            finally:
                self.vm_mgr.destroy_vm(vm_id)
                
            self.ledger.log_execution(agent_name, tool_name, arguments, risk_level, status)
            
            if tool_name in ('shell_command', 'run_command', 'shell'):
                arguments['command'] = f"echo 'KVM Execution Intercepted: {output}'"
                
            return True, f"Command executed safely in KVM Micro-VM: {command}"
            
        # Allowed
        status = "ALLOWED"
        
        # If it's a modifying command but allowed (e.g. strict mode approved), create a snapshot
        if tool_name in ('shell_command', 'run_command', 'shell', 'file_write'):
            self.snapshot_mgr.create_checkpoint(f"Pre-execution of {tool_name} by {agent_name}")
            
            if tool_name in ('shell_command', 'run_command', 'shell'):
                # Dynamically inject the sandbox wrapper into the command string
                # We overwrite the arguments dict which will be returned to the executor
                workspace_dir = os.getcwd()
                arguments['command'] = self.container_mgr.wrap_command(command, workspace_dir)
            
        self.ledger.log_execution(agent_name, tool_name, arguments, risk_level, status)
        return True, "Command allowed"

    def _emit_dbus_notification(self, agent_name: str, command: str) -> None:
        """Emit a native desktop notification."""
        try:
            summary = "⚠️ Zelyr Security Alert"
            body = f"Intercepted High Risk Command from {agent_name}:\n{command}"
            # Use notify-send as it's the standard for Linux desktop notifications
            subprocess.run(['notify-send', '-u', 'critical', summary, body], check=False)
        except Exception as e:
            logger.error(f"Failed to emit DBus notification: {e}")
