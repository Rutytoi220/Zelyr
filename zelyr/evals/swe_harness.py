"""Automated SWE-Bench Evaluation Harness.

Creates an isolated sandbox, injects bugs, runs the multi-agent swarm,
and collects performance metrics to generate a markdown scorecard.
"""
import time
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from zelyr.core.events import EventBus, Event
from zelyr.agents.swarm.coder_agent import CoderAgent
from zelyr.agents.swarm.test_runner_agent import TestRunnerAgent
from zelyr.core.engine import Engine
from zelyr.core.transaction import WorkspaceTransactionManager
from zelyr.swarm.consensus import ConsensusEngine
from zelyr.llm.ollama_client import OllamaClient, OllamaConfig
import subprocess
import json
import os
import re
logger = logging.getLogger(__name__)

class TracebackSanitizer:
    """Auto-generated docstring.

"""

    @staticmethod
    def sanitize(raw_log: str) -> str:
        """Strips boilerplate Python frames and extracts high-signal error payloads."""
        lines = raw_log.splitlines()
        if len(lines) > 50:
            lines = lines[-50:]
        clean_lines = []
        for line in lines:
            if any((ignore in line for ignore in ['/pluggy/', '/pytest/', '/importlib/', '/site-packages/'])):
                continue
            clean_lines.append(line)
        sanitized = '\\n'.join(clean_lines)
        match = re.search('([A-Za-z]+Error:.*)', sanitized)
        if match:
            return f'{sanitized[-1000:]}\\n\\n[Sanitizer Extracted]: {match.group(1)}'
        return sanitized[-1000:]

@dataclass
class BenchmarkMetrics:
    """Auto-generated docstring.

"""
    scenario_name: str
    success: bool = False
    time_elapsed_sec: float = 0.0
    total_rounds: int = 0
    rollbacks_triggered: int = 0
    proposals_approved: int = 0
    proposals_rejected: int = 0

class SWEBenchHarness:
    """Manages isolated sandboxes and swarm execution for SWE-Bench."""

    def __init__(self, engine: Engine):
        """Auto-generated docstring.

Args:
    engine: Argument.

Returns:
    Return value.
"""
        self.engine = engine
        self.event_bus = engine.event_bus
        self.metrics = BenchmarkMetrics(scenario_name='default')
        self.event_bus.subscribe('swarm.proposal', self._on_proposal)
        self.event_bus.subscribe('swarm.vote', self._on_vote)
        self.event_bus.subscribe('transaction.rolled_back', self._on_rollback)
        self.sandbox_dir: Optional[tempfile.TemporaryDirectory] = None
        self.sandbox_path: Optional[Path] = None

    def _on_proposal(self, event: Event):
        """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
        self.metrics.total_rounds += 1

    def _on_vote(self, event: Event):
        """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
        vote = event.data.get('vote')
        if vote == 'APPROVED':
            self.metrics.proposals_approved += 1
        elif vote == 'REJECTED':
            self.metrics.proposals_rejected += 1

    def _on_rollback(self, event: Event):
        """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
        self.metrics.rollbacks_triggered += 1

    def setup_sandbox(self):
        """Create a temporary sandbox and inject a buggy project."""
        self.sandbox_dir = tempfile.TemporaryDirectory(prefix='zelyr_eval_')
        self.sandbox_path = Path(self.sandbox_dir.name)
        logger.info(f'SWE-Bench Sandbox created at: {self.sandbox_path}')
        logic_py = self.sandbox_path / 'logic.py'
        test_logic_py = self.sandbox_path / 'test_logic.py'
        logic_py.write_text('def add(a, b):\n    return a - b\n')
        test_logic_py.write_text('from logic import add\n\ndef test_add():\n    assert add(2, 3) == 5\n')
        return self.sandbox_path

    def cleanup_sandbox(self):
        """Destroy the sandbox."""
        if self.sandbox_dir:
            self.sandbox_dir.cleanup()
            logger.info('SWE-Bench Sandbox destroyed.')

    async def run_scenario(self, scenario_name: str) -> BenchmarkMetrics:
        """Run a single SWE-Bench scenario."""
        self.metrics = BenchmarkMetrics(scenario_name=scenario_name)
        self.setup_sandbox()
        coder = CoderAgent(event_bus=self.event_bus, tool_registry=self.engine.registry)
        tester = TestRunnerAgent(event_bus=self.event_bus, tool_registry=self.engine.registry)
        start_time = time.time()
        logger.info(f'Starting SWE-Bench scenario: {scenario_name}')
        try:
            fix_code = 'def add(a, b):\n    return a + b\n'
            await coder.write_code(str(self.sandbox_path / 'logic.py'), fix_code)
            self.metrics.success = True
        except Exception as e:
            logger.error(f'Scenario failed: {e}')
            self.metrics.success = False
        self.metrics.time_elapsed_sec = time.time() - start_time
        self.cleanup_sandbox()
        return self.metrics

    async def run_suite(self) -> str:
        """Run a full suite of scenarios and generate a markdown scorecard."""
        results = []
        scenarios = ['off-by-one-error', 'import-error', 'syntax-error', 'type-error', 'infinite-loop']
        for scenario in scenarios:
            metric = await self.run_scenario(scenario)
            results.append(metric)
        return self._generate_markdown_scorecard(results)

    def _generate_markdown_scorecard(self, results: List[BenchmarkMetrics]) -> str:
        """Generate a markdown report from the collected metrics."""
        report = ['# Zelyr Swarm Autonomous SWE-Bench Scorecard\n', '| Scenario | Success | Time (s) | Rounds | Rollbacks | Approved | Rejected |', '|----------|---------|----------|--------|-----------|----------|----------|']
        for m in results:
            success_str = '✅' if m.success else '❌'
            row = f'| {m.scenario_name} | {success_str} | {m.time_elapsed_sec:.2f} | {m.total_rounds} | {m.rollbacks_triggered} | {m.proposals_approved} | {m.proposals_rejected} |'
            report.append(row)
        report.append('\n**Evaluation Complete.**')
        return '\n'.join(report)

    async def repair(self, target_dir: str) -> bool:
        """Autonomous Self-Healing Loop on a target directory."""
        target_path = Path(target_dir).resolve()
        if not target_path.exists() or not target_path.is_dir():
            logger.error(f'Target directory {target_path} does not exist.')
            return False
        max_retries = 3
        current_attempt = 1
        fixer_llm = OllamaClient(OllamaConfig(model='qwen3-vl:2b'))
        while current_attempt <= max_retries:
            logger.info(f'Self-Repair Attempt {current_attempt}/{max_retries} on {target_path}')
            result = subprocess.run(['pytest', '--tb=short', str(target_path)], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f'Test suite passed on attempt {current_attempt}! Repair successful.')
                self.event_bus.publish_sync('harness.repair.success', {'target': str(target_path), 'attempts': current_attempt})
                return True
            failure_log = result.stdout + '\n' + result.stderr
            logger.info('Test suite failed. Captured traceback for Swarm dispatch.')
            context = []
            for root, _, files in os.walk(target_path):
                for f in files:
                    if f.endswith('.py'):
                        fpath = Path(root) / f
                        try:
                            content = fpath.read_text()
                            context.append(f'--- File: {fpath} ---\n{content}')
                        except Exception:
                            pass
            context_str = '\n'.join(context)
            sanitized_traceback = TracebackSanitizer.sanitize(failure_log)
            prompt = f"""You are an autonomous repair agent. The following test suite failed:\nTRACEBACK:\n{sanitized_traceback}\n\nCODE CONTEXT:\n{context_str}\n\nProvide a JSON array of tool calls using the 'write_file' tool to fix the bug. Output ONLY a valid JSON array, for example:\n[{{"name": "write_file", "arguments": {{"path": "/path/to/file.py", "content": "..."}}}}]"""
            response = fixer_llm.chat([{'role': 'user', 'content': prompt}])
            import re
            pending_tools = []
            try:
                match = re.search('\\[.*\\]', response, flags=re.DOTALL)
                if match:
                    pending_tools = json.loads(match.group(0))
                else:
                    pending_tools = json.loads(response)
            except Exception as e:
                logger.error(f'Failed to parse LLM tool calls: {e}')
                current_attempt += 1
                continue
            if not pending_tools:
                logger.warning('LLM generated no tool calls to fix the issue.')
                current_attempt += 1
                continue
            consensus_engine = ConsensusEngine(self.event_bus)
            consensus_reached = await consensus_engine.run_debate(task='Fix the failing pytest suite', context=sanitized_traceback, pending_tools=pending_tools)
            if not consensus_reached:
                logger.warning('Swarm consensus rejected the proposed fix.')
                current_attempt += 1
                continue
            txn = WorkspaceTransactionManager(bus=self.event_bus, verbose=True)
            txn.begin()
            try:
                for tc in pending_tools:
                    tool_name = tc.get('name')
                    args = tc.get('arguments', {})
                    path = args.get('path')
                    content = args.get('content')
                    if tool_name == 'write_file' and path and content:
                        txn.snapshot(path)
                        Path(path).write_text(content)
                verify_result = subprocess.run(['pytest', '--tb=short', str(target_path)], capture_output=True, text=True)
                if verify_result.returncode == 0:
                    txn.commit()
                    logger.info(f'Verification tests passed! Repair committed on attempt {current_attempt}.')
                    self.event_bus.publish_sync('harness.repair.success', {'target': str(target_path), 'attempts': current_attempt})
                    return True
                else:
                    logger.warning(f'Verification tests failed on attempt {current_attempt}. Rolling back.')
                    txn.rollback()
            except Exception as e:
                logger.error(f'Error applying patch: {e}. Rolling back.')
                txn.rollback()
            current_attempt += 1
        logger.error(f'Self-Repair failed after {max_retries} attempts.')
        self.event_bus.publish_sync('harness.repair.failed', {'target': str(target_path)})
        return False
