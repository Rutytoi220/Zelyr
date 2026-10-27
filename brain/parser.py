"""Instruction parser for Zelyr brain.

Accepts multiple LLM tag formats and returns structured instructions.
Supports:
- Legacy format: [INSTRUCTION]action param[/INSTRUCTION]
- JSON format: [INSTRUCTION_JSON]{...}[/INSTRUCTION_JSON]
- Multiple instructions
"""

import re
import json
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def parse(text: str) -> Dict:
    """Parse LLM response and return instructions.
    
    Supports:
    - [INSTRUCTION]action params[/INSTRUCTION]
    - [INSTRUCTION_JSON]{...}[/INSTRUCTION_JSON]
    - [MESSAGE]text[/MESSAGE]
    
    Returns:
        Dict with 'type' and instruction details
    """
    if text is None:
        return {'type': 'no_action'}
    s = text.strip()
    
    # Try JSON format first
    json_result = _parse_json_instructions(s)
    if json_result:
        return json_result
    
    # Fall back to legacy format
    instructions = []
    for m in re.finditer(r"\[INSTRUCTION\](.*?)\[/INSTRUCTION\]", s, re.IGNORECASE | re.DOTALL):
        inner = m.group(1).strip()
        if inner:
            parts = inner.split(None, 1)
            action = parts[0].strip()
            params = parts[1].strip() if len(parts) > 1 else ''
            instructions.append({'action': action, 'params': params})
    
    # If we found instructions, return them
    if instructions:
        logger.debug("Parsed %d legacy instruction(s)", len(instructions))
        if len(instructions) == 1:
            # Single instruction - backward compatible format
            instr = instructions[0]
            return {'type': 'instruction', 'action': instr['action'], 'params': instr['params']}
        else:
            # Multiple instructions
            return {'type': 'instructions', 'instructions': instructions}
    
    # Check for message
    m = re.search(r"\[MESSAGE\](.*?)\[/MESSAGE\]", s, re.IGNORECASE | re.DOTALL)
    if m:
        return {'type': 'message', 'message': m.group(1).strip()}
    
    if re.search(r"\[NO_ACTION\]", s, re.IGNORECASE):
        return {'type': 'no_action'}
    
    return {'type': 'message', 'message': s}


def _parse_json_instructions(text: str) -> Optional[Dict]:
    """Parse JSON-formatted instructions.
    
    Format:
    [INSTRUCTION_JSON]
    {
      "actions": [
        {"name": "open_app", "params": {"name": "opera"}},
        {"name": "search", "params": {"query": "python"}}
      ]
    }
    [/INSTRUCTION_JSON]
    
    Args:
        text: Text potentially containing JSON instructions
    
    Returns:
        Parsed instruction dict or None
    """
    m = re.search(r"\[INSTRUCTION_JSON\](.*?)\[/INSTRUCTION_JSON\]", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    
    try:
        json_str = m.group(1).strip()
        payload = json.loads(json_str)
        
        # Validate structure
        if not isinstance(payload, dict):
            logger.warning("Invalid JSON structure: expected dict")
            return None
        
        actions = payload.get('actions', [])
        if not isinstance(actions, list):
            logger.warning("Invalid actions: expected list")
            return None
        
        if not actions:
            return {'type': 'no_action'}
        
        # Validate and normalize actions
        normalized_actions = []
        for action in actions:
            if not isinstance(action, dict) or 'name' not in action:
                logger.warning(f"Skipping invalid action: {action}")
                continue
            
            normalized_actions.append({
                'action': action['name'],
                'params': _serialize_action_params(action.get('params', {}))
            })
        
        if not normalized_actions:
            return {'type': 'no_action'}
        
        logger.debug("Parsed JSON instructions with %d action(s)", len(normalized_actions))
        
        if len(normalized_actions) == 1:
            # Single action - backward compatible
            instr = normalized_actions[0]
            return {'type': 'instruction', 'action': instr['action'], 'params': instr['params']}
        else:
            # Multiple actions - new format
            return {'type': 'instructions', 'instructions': normalized_actions}
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON instructions: {e}")
        return None


def _serialize_action_params(params) -> str:
    """Convert action params to string format for executor.
    
    Args:
        params: Params from JSON (dict or other)
    
    Returns:
        Serialized param string
    """
    if not params:
        return ''
    
    if isinstance(params, str):
        return params
    
    if isinstance(params, dict):
        parts = []
        for key, value in params.items():
            if isinstance(value, str):
                # Escape quotes in value
                value = value.replace('"', '\\"')
                parts.append(f'{key}="{value}"')
            else:
                parts.append(f'{key}="{value}"')
        return ' '.join(parts)
    
    return str(params)

