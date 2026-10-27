"""Dynamic Inference Router for Zelyr v2.

Routes incoming prompts to the most efficient Ollama model based on task
complexity using an LLM-driven micro-model classification.
"""
import logging
from typing import Dict, Any, List, Optional
import time
import asyncio
import httpx
from enum import Enum
from pydantic import BaseModel
import litellm
from zelyr.llm.universal_client import UniversalLLMClient
from zelyr.engine.telemetry import HardwareTelemetryDaemon
from zelyr.engine.cloud_adapter import CloudAdapter
from zelyr.config import get_config
logger = logging.getLogger(__name__)

class IntentCategory(str, Enum):
    """Auto-generated docstring.

"""
    CODE = 'CODE'
    CHAT = 'CHAT'
    VISION = 'VISION'
    SYSTEM = 'SYSTEM'
    REASONING = 'REASONING'

class RouterDecision(BaseModel):
    """Auto-generated docstring.

"""
    category: IntentCategory

class SmartRouter:
    """Smart wrapper around UniversalLLMClient that dynamically routes prompts."""

    def __init__(self, llm_client: UniversalLLMClient, telemetry_daemon: Optional[HardwareTelemetryDaemon]=None, event_bus=None):
        """Auto-generated docstring.

Args:
    llm_client: Argument.
    telemetry_daemon: Argument.
    event_bus: Argument.

Returns:
    Return value.
"""
        self.llm_client = llm_client
        self.telemetry = telemetry_daemon
        self.event_bus = event_bus
        self.cloud_adapter = CloudAdapter()
        self.model_tiers = self._resolve_dynamic_tiers()
        self._current_active_model: Optional[str] = None

    async def _async_get_installed_models(self) -> List[str]:
        """Fetch current installed models from Ollama asynchronously."""
        if getattr(self, '_ollama_offline', False):
            return []
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get('http://127.0.0.1:11434/api/tags', timeout=3.0)
                if r.status_code == 200:
                    return [m['name'] for m in r.json().get('models', [])]
        except Exception:
            self._ollama_offline = True
            logger.error("ERROR: Ollama daemon not found at 127.0.0.1:11434")
        return []

    def _get_installed_models(self) -> List[str]:
        """Synchronous wrapper for installed models (fallback for init)."""
        try:
            return asyncio.run(self._async_get_installed_models())
        except RuntimeError:
            # If there's an event loop running, we just return empty as fallback.
            return []

    async def _async_fetch_all_metadata(self, models: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch deep introspection data for multiple models asynchronously."""
        results = {}
        async with httpx.AsyncClient() as client:
            for model_name in models:
                try:
                    r = await client.post('http://127.0.0.1:11434/api/show', json={"name": model_name}, timeout=2.0)
                    if r.status_code == 200:
                        results[model_name] = r.json()
                except Exception as e:
                    logger.warning(f"[SmartRouter] Failed to fetch metadata for {model_name} (corrupted/timeout): {e}")
        return results

    def _score_for_tools(self, details: Dict[str, Any]) -> float:
        """Score a model dynamically based purely on its capabilities and parameter metadata."""
        if not details:
            return 0.0

        score = 0.0
        
        # 1. Tool Competency (+100)
        template = details.get("template", "")
        system = details.get("system", "")
        if template or system:
            combined = template + "\n" + system
            if '{{ .Tools }}' in combined or '{{- if .Tools }}' in combined or '{{.Tools}}' in combined:
                score += 100.0

        # 2. Parameter Weighting (+X)
        param_size_str = details.get("details", {}).get("parameter_size", "0B")
        param_val = 0.0
        if param_size_str and param_size_str.endswith("B"):
            try:
                param_val = float(param_size_str[:-1])
            except ValueError:
                param_val = 0.0
        score += param_val

        # 3. VRAM Safety Cap (-200 if > 40B)
        if param_val > 40.0:
            score -= 200.0
            
        return score

    def _score_for_vision(self, details: Dict[str, Any], model_name: str) -> float:
        """Score a model dynamically based on vision capabilities."""
        if not details:
            return 0.0

        score = 0.0
        
        # 1. Explicit Vision Metadata
        modelfile = details.get("modelfile", "").lower()
        families = [f.lower() for f in details.get("details", {}).get("families", [])]
        
        if "clip" in modelfile or "vision" in modelfile or "clip" in families:
            score += 100.0

        # 2. Gemma 3/4 Priority (+150 points for Tier 1 Multimodal)
        if "gemma4" in model_name.lower() or "gemma3" in model_name.lower():
            score += 150.0

        # 3. Parameter Weighting (+X)
        param_size_str = details.get("details", {}).get("parameter_size", "0B")
        param_val = 0.0
        if param_size_str and param_size_str.endswith("B"):
            try:
                param_val = float(param_size_str[:-1])
            except ValueError:
                param_val = 0.0
        score += param_val

        # 4. Fallback Name Heuristics (if metadata lacks explicit signs)
        if score < 50.0:
            if "-vl" in model_name.lower() or "vision" in model_name.lower() or "llava" in model_name.lower():
                score += 50.0

        return score

    def _resolve_dynamic_tiers(self) -> Dict[IntentCategory, str]:
        """Dynamically select the best installed models for each tier."""
        models = self._get_installed_models()
        if not models:
            return {}

        def pick_best(keywords: List[str]) -> str:
            for kw in keywords:
                for m in models:
                    if kw.lower() in m.lower():
                        return f"ollama/{m}"
            return f"ollama/{models[0]}" if models else ""

        # Router/Chat: Smallest/fastest model (often has '1.5b', '2b', '0.5b', '3b' in name)
        # Orchestrator (System/Reasoning/Code): Largest/most capable
        
        # We will parse parameter sizes if possible, but fallback to heuristics
        small_kws = ['0.5b', '1.5b', '2b', '3b', 'functiongemma', 'qwen3:0.6b', 'smollm']
        large_kws = ['32b', '70b', 'r1', 'deepseek', 'laguna', 'coder', 'qwen2.5', 'llama3']
        
        return {
            IntentCategory.CODE: pick_best(large_kws + ['coder', 'qwen']),
            IntentCategory.CHAT: pick_best(small_kws + ['llama', 'gemma', 'qwen']),
            IntentCategory.VISION: pick_best(['vl', 'llava', 'vision', 'pixtral']),
            IntentCategory.REASONING: pick_best(large_kws + ['reason', 'math']),
            IntentCategory.SYSTEM: pick_best(large_kws + ['coder', 'qwen'])
        }

    def _has_image(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if any message contains an image payload."""
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        return True
        return False

    def _classify_task(self, messages: List[Dict[str, Any]], tool_schemas: Optional[List[Dict[str, Any]]]=None) -> IntentCategory:
        """Classify the prompt complexity into a Task Intent using a micro-model."""
        if not messages:
            return IntentCategory.CHAT
        user_messages = [m.get('content', '') for m in messages if m.get('role') == 'user']
        user_texts = []
        for content in user_messages:
            if isinstance(content, str):
                user_texts.append(content)
            elif isinstance(content, list):
                user_texts.extend([p.get('text', '') for p in content if p.get('type') == 'text'])
        user_text = '\n'.join(user_texts)
        if not user_text:
            return IntentCategory.SYSTEM
            
        if self._has_image(messages):
            return IntentCategory.VISION
            
        system_prompt = "You are a semantic intent router. Classify the user's request into one of the following categories:\nCODE: For programming, refactoring, debugging, or execution.\nCHAT: For general conversation, greetings, or short chat.\nSYSTEM: For orchestrating tools, reading files, or complex multi-step OS tasks.\nREASONING: For deep logical puzzles, math, or complex analysis requiring advanced chain-of-thought.\nOutput strictly valid JSON conforming to the requested schema."
        try:
            response = litellm.completion(model='ollama/qwen3:0.6b', api_base='http://127.0.0.1:11434', messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'User Request: {user_text}'}], temperature=0.0, max_tokens=60, response_format=RouterDecision)
            content = response.choices[0].message.content
            import json
            decision = RouterDecision.model_validate_json(content)
            return decision.category
        except Exception as e:
            logger.warning(f'SmartRouter classification failed: {e}. Falling back to SYSTEM intent.')
            return IntentCategory.SYSTEM

    def _route_request(self, messages: List[Dict[str, Any]], tool_schemas: Optional[List[Dict[str, Any]]]=None) -> str:
        """Determine which model to use."""
        config = get_config()
        if getattr(config, 'model_selection_mode', 'auto') == 'manual':
            selected_model = getattr(config, 'ollama_model', 'ollama/qwen3:8b')
            if self._current_active_model and self._current_active_model != selected_model:
                logger.info(f'SmartRouter (Manual Override) Swapping Model: {self._current_active_model} -> {selected_model}')
            elif not self._current_active_model:
                logger.info(f'SmartRouter (Manual Override) initializing with Model: {selected_model}')
            self._current_active_model = selected_model
            if self.event_bus:
                from zelyr.core.events import Event
                self.event_bus.publish(Event('model.routed', 'SmartRouter', data={'target': selected_model, 'intent': 'MANUAL'}))
            return selected_model

        if tool_schemas:
            target_intent = IntentCategory.SYSTEM
        else:
            target_intent = self._classify_task(messages, tool_schemas)
        
        # Check for explicitly requested cloud tier
        is_cloud_requested = False
        if messages:
            last_user_msg = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
            if last_user_msg:
                content = last_user_msg.get('content', '')
                if isinstance(content, str) and (content.strip().startswith('/claude') or content.strip().startswith('/cloud')):
                    is_cloud_requested = True
                    
        if is_cloud_requested:
            from zelyr.engine.budget_mgr import TokenBudgetManager
            budget_mgr = TokenBudgetManager()
            can_afford, reason, _ = budget_mgr.can_afford_cloud_call(4000)
            if can_afford:
                selected_model = 'claude-3-5-sonnet-latest'
                self._current_active_model = selected_model
                if self.event_bus:
                    from zelyr.core.events import Event
                    self.event_bus.publish(Event('model.routed', 'SmartRouter', data={'target': selected_model, 'intent': 'CLOUD'}))
                return selected_model
            else:
                logger.warning(reason)
                if self.event_bus:
                    from zelyr.core.events import Event
                    self.event_bus.publish(Event('telemetry.warning', 'SmartRouter', data={'message': reason}))
                # Fallback to normal routing
                pass
        if target_intent == IntentCategory.CODE:
            if self.telemetry and self.telemetry.latest_state.get('warning'):
                ram_avail = self.telemetry.latest_state.get('ram_available_percent', 0)
                config = get_config()
                if getattr(config, 'allow_cloud_fallback', False) and self.cloud_adapter.is_configured:
                    logger.warning(f'Local VRAM exhausted ({ram_avail:.1f}%). Bursting to Cloud Fallback.')
                    return 'cloud'
                else:
                    logger.warning(f'Emergency Downgrade: Code task requested, but RAM is critically low ({ram_avail:.1f}%). Downgrading to Orchestration to prevent OOM crash.')
                    target_intent = IntentCategory.SYSTEM
        if target_intent in (IntentCategory.SYSTEM, IntentCategory.VISION):
            # Dynamic probing for tool scoring
            try:
                models = asyncio.run(self._async_get_installed_models())
            except RuntimeError:
                # Fallback if loop is running
                models = self._get_installed_models()
                
            if not models:
                raise RuntimeError("No models installed in Ollama. Cannot execute tool calls.")
            
            # Fetch all metadata asynchronously
            try:
                metadata_map = asyncio.run(self._async_fetch_all_metadata(models))
            except RuntimeError:
                metadata_map = {}
                logger.error("[SmartRouter] Could not run async metadata fetch due to existing event loop.")
            
            # Score all installed models
            scored_models = []
            for m in models:
                details = metadata_map.get(m, {})
                if target_intent == IntentCategory.VISION:
                    score = self._score_for_vision(details, m)
                else:
                    score = self._score_for_tools(details)
                scored_models.append((m, score))
                
            scored_models.sort(key=lambda x: x[1], reverse=True)
            best_model = scored_models[0][0]
            best_score = scored_models[0][1]
            
            # Graceful Degradation for Vision
            if target_intent == IntentCategory.VISION and best_score < 10.0:
                raise RuntimeError("Error: A visual task was requested, but no Vision-Language Model (VLM) is installed. Please run 'ollama pull qwen3-vl:2b' (or similar).")
                
            target_model_name = f"ollama/{best_model}"
            
            # Re-cache the tier mapping for consistency
            self.model_tiers[target_intent] = target_model_name
            logger.info(f"[SmartRouter] Selected {target_model_name} with score {best_score}")
        else:
            target_model_name = self.model_tiers.get(target_intent)
            
        selected_model: str = str(target_model_name or getattr(getattr(self.llm_client, 'config', None), 'model', 'ollama/qwen3:8b'))
        if self._current_active_model and self._current_active_model != selected_model:
            logger.info(f'SmartRouter Swapping Model: {self._current_active_model} -> {selected_model} (Intent: {target_intent.value})')
        elif not self._current_active_model:
            logger.info(f'SmartRouter initializing with Model: {selected_model} (Intent: {target_intent.value})')
        self._current_active_model = selected_model
        if self.event_bus:
            from zelyr.core.events import Event
            self.event_bus.publish(Event('model.routed', 'SmartRouter', data={'target': selected_model, 'intent': target_intent.value}))
        return selected_model

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Route and execute a standard chat completion."""
        target_model = self._route_request(messages)
        if target_model == 'cloud':
            return self.cloud_adapter.chat(messages, **kwargs)
        has_config = hasattr(self.llm_client, 'config') and hasattr(self.llm_client.config, 'model')
        original_model = self.llm_client.config.model if has_config else None
        try:
            if has_config:
                self.llm_client.config.model = target_model
            return self.llm_client.chat(messages, **kwargs)
        finally:
            if has_config and original_model is not None:
                self.llm_client.config.model = original_model

    def chat_with_tools(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]], **kwargs) -> Any:
        """Route and execute a tool-enabled chat completion."""
        target_model = self._route_request(messages, tool_schemas)
        if target_model == 'cloud':
            return self.cloud_adapter.chat_with_tools(messages, tool_schemas, **kwargs)
        has_config = hasattr(self.llm_client, 'config') and hasattr(self.llm_client.config, 'model')
        original_model = self.llm_client.config.model if has_config else None
        try:
            if has_config:
                self.llm_client.config.model = target_model
            return self.llm_client.chat_with_tools(messages, tool_schemas, **kwargs)
        finally:
            if has_config and original_model is not None:
                self.llm_client.config.model = original_model

    def is_available(self) -> bool:
        """Check availability."""
        return self.llm_client.is_available()

    def __getattr__(self, name: str) -> Any:
        """Transparently forward all un-implemented methods to the underlying LLM client."""
        return getattr(self.llm_client, name)
