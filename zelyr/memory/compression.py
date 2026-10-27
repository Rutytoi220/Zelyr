import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ObservationCompressor:
    """Middleware to compress or paginate large tool outputs before hitting the ReAct buffer."""

    MAX_CHARS = 10000  # Approx 2500 tokens

    @classmethod
    def compress(cls, result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Compress tool output if it exceeds the token/character threshold.
        
        Args:
            result: The structured tool result dict.
            tool_name: The name of the tool executed.
        """
        if not isinstance(result, dict) or 'result' not in result:
            return result
            
        res_inner = result['result']
        if not isinstance(res_inner, dict) or 'output' not in res_inner:
            return result

        out_data = res_inner['output']
        
        # We only compress string outputs for now
        if not isinstance(out_data, str):
            if isinstance(out_data, (list, dict)):
                import json
                try:
                    str_data = json.dumps(out_data)
                    if len(str_data) > cls.MAX_CHARS:
                        out_data = str_data
                    else:
                        return result
                except Exception:
                    return result
            else:
                return result

        if len(out_data) <= cls.MAX_CHARS:
            return result

        warning_msg = (
            "\n\n[SYSTEM WARNING: Output truncated because it exceeded 2500 tokens. "
            "To find specific information, use a search/grep tool, or read the file in smaller chunks.]"
        )
        
        is_shell = tool_name in ('shell', 'bash', 'execute_command', 'run_command', 'cmd')
        
        if is_shell:
            truncated_out = (
                "[SYSTEM WARNING: Output truncated (showing tail only). "
                "Exceeded 2500 tokens. Output may be incomplete.]\n\n...\n" + 
                out_data[-cls.MAX_CHARS:]
            )
        else:
            truncated_out = out_data[:cls.MAX_CHARS] + warning_msg

        result['result']['output'] = truncated_out
        logger.info(f"ObservationCompressor: Truncated {tool_name} output from {len(out_data)} to {len(truncated_out)} characters.")
        
        return result

    @classmethod
    def compress_observations_buffer(cls, observations: list, max_total_chars: int = 30000, iterations: int = 0) -> list:
        """Guard the cumulative observation buffer before context assembly.

        Applies a semantic filter to strip heavy metadata from older logs,
        and if still over budget, hard-prunes the oldest entries.
        """
        import json as _json

        # 1. Semantic Filter Pass: Strip metadata and compress older observations
        # We preserve the absolute latest observation fully intact.
        if iterations > 1:
            for idx in range(len(observations) - 1):
                obs = observations[idx]
                if not isinstance(obs, dict):
                    continue
                
                if 'result' in obs and isinstance(obs['result'], dict):
                    # Strip heavy metadata while preserving core facts
                    obs['result'].pop('metadata', None)
                    obs['result'].pop('debug_info', None)
                    obs['result'].pop('trace', None)
                        
                    # Heavy compression on output for historical steps
                    out = obs['result'].get('output')
                    if isinstance(out, str) and len(out) > 500:
                        obs['result']['output'] = out[:500] + "...[TRUNCATED FOR CONTEXT]"
                        
                if 'arguments' in obs and isinstance(obs['arguments'], dict):
                    for k, v in list(obs['arguments'].items()):
                        if isinstance(v, str) and len(v) > 200:
                            obs['arguments'][k] = v[:200] + "..."

        # 2. Check total budget
        total = 0
        budget_exceeded_at = -1

        for idx in range(len(observations) - 1, -1, -1):
            try:
                entry_len = len(_json.dumps(observations[idx], default=str))
            except Exception:
                entry_len = 500
            total += entry_len
            if total > max_total_chars:
                budget_exceeded_at = idx
                break

        if budget_exceeded_at < 0:
            return observations

        # 3. Fallback to hard pruning if we're STILL over budget
        pruned_count = budget_exceeded_at + 1
        pruned_tools = []
        for obs in observations[:pruned_count]:
            if isinstance(obs, dict):
                pruned_tools.append(obs.get('tool', 'unknown'))

        summary = {
            'tool': 'system',
            'success': True,
            'result': {
                'output': (
                    f"[SYSTEM: {pruned_count} earlier observation(s) hard-pruned to protect "
                    f"context window. Pruned tools: {', '.join(pruned_tools)}. "
                    f"Use targeted search/grep if you need data from those earlier steps.]"
                )
            }
        }
        compressed = [summary] + observations[pruned_count:]
        logger.info(
            "ObservationCompressor: Hard-Pruned %d old observations (%d chars over budget).",
            pruned_count, total - max_total_chars
        )
        return compressed
