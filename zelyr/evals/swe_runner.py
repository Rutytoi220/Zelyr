from zelyr.compat import state_dir
import json
import logging
import subprocess
import time
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from zelyr.llm.ollama_client import OllamaClient, OllamaConfig
from zelyr.evals.swe_harness import TracebackSanitizer
from zelyr.swarm.consensus import ConsensusEngine
from zelyr.agents.swarm.coder_agent import CoderAgent
from zelyr.core.transaction import WorkspaceTransactionManager
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkProblem:
    """Auto-generated docstring.

"""
    problem_statement: str
    repo_path: str
    test_command: str

class SWERunner:
    """Auto-generated docstring.

"""

    def __init__(self, engine):
        """Auto-generated docstring.

Args:
    engine: Argument.

Returns:
    Return value.
"""
        self.engine = engine
        self.event_bus = getattr(engine, 'event_bus', None)
        self.registry = getattr(engine, 'registry', None)
        self.report_path = state_dir() / 'benchmarks' / 'latest_report.json'
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = OllamaClient(OllamaConfig(model='qwen3-vl:2b'))

    def ingest_problem(self, json_path: str) -> BenchmarkProblem:
        """Auto-generated docstring.

Args:
    json_path: Argument.

Returns:
    Return value.
"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return BenchmarkProblem(problem_statement=data.get('problem_statement', ''), repo_path=data.get('repo_path', ''), test_command=data.get('test_command', 'pytest'))

    async def run_evaluation(self, problem: BenchmarkProblem) -> Dict[str, Any]:
        """Auto-generated docstring.

Args:
    problem: Argument.

Returns:
    Return value.
"""
        target_path = Path(problem.repo_path).resolve()
        if not target_path.exists():
            logger.error(f'Repository path {target_path} not found.')
            return {'success': False, 'error': 'Repo not found'}
        max_retries = 3
        current_attempt = 1
        start_time = time.time()
        context_files = []
        for root, _, files in os.walk(target_path):
            for f in files:
                if f.endswith('.py'):
                    fpath = Path(root) / f
                    try:
                        content = fpath.read_text()
                        context_files.append(f'--- File: {fpath} ---\n{content}')
                    except Exception:
                        pass
        context_str = '\n'.join(context_files)
        while current_attempt <= max_retries:
            logger.info(f'SWERunner Attempt {current_attempt}/{max_retries}')
            prompt = f"""You are an autonomous SWE. The problem statement is:\n{problem.problem_statement}\n\nCODE CONTEXT:\n{context_str}\n\nProvide a JSON array of tool calls using the 'write_file' tool to fix the bug. Output ONLY a valid JSON array. Example:\n[{{"name": "write_file", "arguments": {{"path": "/path/to/file.py", "content": "..."}}}}]"""
            response = self.llm.chat([{'role': 'user', 'content': prompt}])
            pending_tools = []
            try:
                match = re.search('\\[.*\\]', response, flags=re.DOTALL)
                if match:
                    pending_tools = json.loads(match.group(0))
                else:
                    pending_tools = json.loads(response)
            except Exception as e:
                logger.error(f'Coder failed to generate valid AST edits JSON: {e}')
                current_attempt += 1
                continue
            if not pending_tools:
                logger.warning('No edits proposed by Coder.')
                current_attempt += 1
                continue
            txn = None
            if self.event_bus:
                txn = WorkspaceTransactionManager(bus=self.event_bus, verbose=False)
                txn.begin()
            try:
                for tc in pending_tools:
                    tool_name = tc.get('name')
                    args = tc.get('arguments', {})
                    path = args.get('path')
                    content = args.get('content')
                    if tool_name == 'write_file' and path and content:
                        if txn:
                            txn.snapshot(path)
                        Path(path).write_text(content)
            except Exception as e:
                logger.error(f'Failed to apply patch: {e}')
                if txn:
                    txn.rollback()
                current_attempt += 1
                continue
            logger.info('Running test suite...')
            cmd = problem.test_command.split()
            cmd.append(str(target_path))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info('Test suite PASSED.')
                if txn:
                    txn.commit()
                break
            if txn:
                txn.rollback()
            failure_log = result.stdout + '\n' + result.stderr
            sanitized_traceback = TracebackSanitizer.sanitize(failure_log)
            logger.warning(f'Tests failed. Traceback extracted. Invoking Swarm Critic.')
            if self.event_bus:
                consensus_engine = ConsensusEngine(self.event_bus)
                await consensus_engine.run_debate(task='Fix the failing tests', context=sanitized_traceback, pending_tools=pending_tools)
            current_attempt += 1
        success = current_attempt <= max_retries
        elapsed = time.time() - start_time
        telemetry = {'success': success, 'latency': elapsed, 'retry_count': min(current_attempt, max_retries), 'model': 'qwen3-vl:2b'}
        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(telemetry, f, indent=2)
        return telemetry
