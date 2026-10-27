import asyncio
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import croniter
logger = logging.getLogger(__name__)

class RoutineEngine:
    """Auto-generated docstring.

"""

    def __init__(self, cli=None):
        """Auto-generated docstring.

Args:
    cli: Argument.

Returns:
    Return value.
"""
        self.cli = cli
        if cli:
            self.memory = cli.memory
            self.event_bus = cli.engine.event_bus
            self.orchestrator = cli.orchestrator
        self._is_running = False
        self._task = None

    async def parse_schedule_to_cron(self, prompt: str) -> str:
        """Use LLM to translate natural language into a cron expression."""
        if not self.cli or not self.cli.ollama:
            return '0 0 * * *'
        sys_prompt = "You are a cron expression generator. Translate the user's natural language schedule into a standard 5-field cron expression. Output ONLY the cron expression, nothing else."
        try:
            response = await asyncio.to_thread(self.cli.ollama.generate, prompt, system_prompt=sys_prompt)
            cron_expr = response.strip()
            cron_expr = cron_expr.replace('`', '').strip()
            if croniter.croniter.is_valid(cron_expr):
                return cron_expr
        except Exception as e:
            logger.error(f'Failed to parse cron: {e}')
        return '0 0 * * *'

    def add_routine(self, prompt: str, cron_expr: str) -> str:
        """Auto-generated docstring.

Args:
    prompt: Argument.
    cron_expr: Argument.

Returns:
    Return value.
"""
        routine_id = str(uuid.uuid4())
        routine = {'id': routine_id, 'cron_expression': cron_expr, 'prompt': prompt, 'is_active': True, 'last_run': datetime.now().timestamp()}
        if self.cli:
            self.memory.set(f'routine:{routine_id}', routine, tags=['routine'])
        return routine_id

    def list_routines(self) -> List[Dict[str, Any]]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        routines: List[Dict[str, Any]] = []
        if not self.cli:
            return routines
        keys = self.memory.list_keys()
        for k in keys:
            if k.startswith('routine:'):
                r = self.memory.get(k)
                if r:
                    routines.append(r)
        return routines

    def delete_routine(self, routine_id: str) -> bool:
        """Auto-generated docstring.

Args:
    routine_id: Argument.

Returns:
    Return value.
"""
        if not self.cli:
            return False
        return self.memory.delete(f'routine:{routine_id}')

    def start(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if self._is_running:
            return
        self._is_running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._loop())
        except RuntimeError:
            pass
        logger.info('RoutineEngine started')

    async def stop(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info('RoutineEngine stopped')

    async def _loop(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        while self._is_running:
            try:
                routines = self.list_routines()
                now = datetime.now()
                now_ts = now.timestamp()
                for r in routines:
                    if not r.get('is_active'):
                        continue
                    last_run = r.get('last_run', 0)
                    cron_expr = r.get('cron_expression')
                    if not croniter.croniter.is_valid(cron_expr):
                        continue
                    cron = croniter.croniter(cron_expr, datetime.fromtimestamp(last_run))
                    next_run = cron.get_next(datetime)
                    if next_run <= now:
                        await self._execute_routine(r)
                        r['last_run'] = now_ts
                        if self.cli:
                            self.memory.set(f"routine:{r['id']}", r, tags=['routine'])
            except Exception as e:
                logger.error(f'RoutineEngine loop error: {e}')
            await asyncio.sleep(60)

    async def _execute_routine(self, routine: Dict[str, Any]):
        """Auto-generated docstring.

Args:
    routine: Argument.

Returns:
    Return value.
"""
        if not self.cli:
            return
        routine_id = routine['id']
        prompt = routine['prompt']
        self.event_bus.emit('routine.started', {'routine_id': routine_id, 'prompt': prompt})
        try:

            def _run():
                """Auto-generated docstring.


Returns:
    Return value.
"""
                import builtins
                original_input = builtins.input

                def mock_input(*args, **kwargs):
                    """Auto-generated docstring.


Returns:
    Return value.
"""
                    raise RuntimeError('Routine requires interactive attention')
                builtins.input = mock_input
                try:
                    res = self.orchestrator.run(prompt, use_tools=True, session_id=f'routine_{routine_id}')
                    if res.success:
                        return (True, res.output)
                    else:
                        return (False, res.error)
                finally:
                    builtins.input = original_input
            success, output = await asyncio.to_thread(_run)
            if success:
                self.event_bus.emit('routine.completed', {'routine_id': routine_id, 'output': output})
            else:
                self.event_bus.emit('routine.failed', {'routine_id': routine_id, 'error': output})
        except Exception as e:
            if 'Routine requires interactive attention' in str(e):
                self.event_bus.emit('routine.requires_attention', {'routine_id': routine_id, 'prompt': prompt})
            else:
                self.event_bus.emit('routine.failed', {'routine_id': routine_id, 'error': str(e)})
