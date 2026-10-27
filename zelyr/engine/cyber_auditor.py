import logging
import asyncio
import subprocess
import os
from typing import Dict, Any, List
from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.engine.audit_ledger import AuditLedger

logger = logging.getLogger(__name__)

class SecurityAuditorAgent(BaseSubagent):
    """Proactive Security scanning agent for vulnerability analysis."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "SecurityAuditorAgent")
        kwargs.setdefault("description", "Scans the local host for security vulnerabilities.")
        kwargs.setdefault("topic", "swarm.auditor")
        super().__init__(**kwargs)
        self.ledger = AuditLedger()
        
    async def run_audit(self) -> Dict[str, Any]:
        """Perform a complete cyber security audit of the local node."""
        logger.info("Starting Cyber-Security Audit...")
        
        findings = []
        score = 100
        
        # 1. Network Audit (Listening ports)
        try:
            res = subprocess.run(["ss", "-tulpn"], capture_output=True, text=True)
            if "0.0.0.0:23 " in res.stdout:
                findings.append({"level": "Critical", "rule": "Telnet exposed", "detail": "Port 23 is open."})
                score -= 40
            if "0.0.0.0:21 " in res.stdout:
                findings.append({"level": "Critical", "rule": "FTP exposed", "detail": "Port 21 is open."})
                score -= 30
            # Mock generic unencrypted port logic
            if "0.0.0.0:80 " in res.stdout:
                findings.append({"level": "Warning", "rule": "HTTP exposed", "detail": "Port 80 is open without TLS."})
                score -= 10
        except Exception as e:
            logger.warning(f"Network audit failed: {e}")
            
        # 2. Container Socket Audit
        try:
            res = subprocess.run(["ls", "-l", "/var/run/docker.sock"], capture_output=True, text=True)
            if "srwxrwxrwx" in res.stdout:
                findings.append({"level": "Critical", "rule": "Docker socket overly permissive", "detail": "777 permissions on docker.sock."})
                score -= 30
        except Exception:
            pass
            
        # 3. Secret Scanner
        # Mock grep for plaintext secrets in ~/.config
        try:
            res = subprocess.run(["grep", "-r", "-l", "BEGIN RSA PRIVATE KEY", os.path.expanduser("~/.config/")], capture_output=True, text=True)
            if res.stdout.strip():
                findings.append({"level": "Critical", "rule": "Unencrypted Private Key", "detail": f"Keys found in {res.stdout.strip()}"})
                score -= 40
        except Exception:
            pass
            
        score = max(0, score)
        
        report = {
            "posture_score": score,
            "findings": findings,
            "status": "SECURE" if score >= 90 else "VULNERABLE"
        }
        
        self.ledger.log_execution(self.name, "cyber_audit", {}, "INFO", f"Audit complete. Score: {score}")
        return report
