from zelyr.compat import config_dir, get_env
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

try:
    import torch
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, PreTrainedModel
    import transformers.modeling_utils
    from zelyr.core.vision_engine.models.spotless_vlm import SpotlessLatentAdapter, SpotlessMiniCPMV
except ImportError as e:
    raise ImportError(
        "'torch' and 'transformers' are required for local VLM features. "
        "Install them with: pip install 'zelyr-core[ml]'"
    ) from e

class _TiedKeysDescriptor:
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj.__dict__.get("_all_tied_weights_keys", None)
        if val is None:
            val = getattr(obj, "_tied_weights_keys", None)
        return val if val is not None else {}

    def __set__(self, obj, value):
        obj.__dict__["_all_tied_weights_keys"] = value if value is not None else {}

PreTrainedModel.all_tied_weights_keys = _TiedKeysDescriptor()

# Disable caching_allocator_warmup to prevent false-positive CUDA OOM on shared GPUs
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

class ZelyrGroundingEngine:
    _instance: Any = None

    def __init__(self, model_id: str = "openbmb/MiniCPM-V-2_6", adapter_path: Optional[str] = None, device: str = "cuda"):
        if adapter_path is None:
            adapter_path = get_env("VLM_ADAPTER_PATH") or str(config_dir() / "models" / "best_vlm_adapter.pt")
        self.device = device
        
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print(f">>> Initializing base model {model_id} in 4-bit...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        base_vlm = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            quantization_config=bnb,
            device_map=device,
            trust_remote_code=True
        )

        # Freeze base VLM
        for param in base_vlm.parameters():
            param.requires_grad = False

        print(f">>> Loading adapter from {adapter_path}...")
        adapter = SpotlessLatentAdapter(d=3584, bottleneck=448).to(device=device, dtype=torch.bfloat16)
        if adapter_path:
            try:
                adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            except Exception as e:
                print(f"Warning: Could not load adapter from {adapter_path}: {e}")
        
        for param in adapter.parameters():
            param.requires_grad = False

        self.model = SpotlessMiniCPMV(base_vlm, adapter, processor=self.processor)
        self.model.eval()

        ZelyrGroundingEngine._instance = self
        self._warmup()

    def unload(self):
        """Unload vision model tensors from CUDA memory."""
        import gc
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        ZelyrGroundingEngine._instance = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warmup(self):
        print(">>> Warming up engine...")
        dummy_img = Image.new('RGB', (224, 224), color='white')
        self.predict_action(dummy_img, "Locate the element.")
        print(">>> Warmup complete.")

    def predict_action(self, image: Image.Image, instruction: str, k: int = 4, max_tokens: int = 32) -> Dict[str, Any]:
        start_time = time.time()
        
        data = self.model.prepare_multimodal_data(image, instruction)
        logits = None
        pkv = None
        _ = None
        token_id = None
        generated_ids = []

        try:
            with torch.no_grad():
                logits, pkv, _ = self.model.forward_latent(data, K=k)
                
                token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = [token_id.item()]
                
                for _ in range(max_tokens - 1):
                    if token_id.item() in [self.processor.tokenizer.eos_token_id, 151645]: # <|im_end|>
                        break
                    
                    token_embeds = self.model.vlm.llm.model.embed_tokens(token_id)
                    out = self.model.vlm.llm.model(
                        inputs_embeds=token_embeds,
                        past_key_values=pkv,
                        use_cache=True
                    )
                    pkv = out.past_key_values
                    
                    normed = self.model.vlm.llm.model.norm(out.last_hidden_state[:, -1:, :])
                    next_logits = self.model.vlm.llm.lm_head(normed)
                    token_id = torch.argmax(next_logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids.append(token_id.item())
                    
                    del token_embeds, out, normed, next_logits
        finally:
            del logits, pkv, data, _, token_id
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        latency_ms = (time.time() - start_time) * 1000
        
        response = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        result = {
            "action": "unknown",
            "point": [0, 0],
            "text": None,
            "raw_output": response,
            "latency_ms": latency_ms
        }
        
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                if "action" in parsed:
                    result["action"] = parsed["action"]
                if "point" in parsed and isinstance(parsed["point"], list) and len(parsed["point"]) == 2:
                    result["point"] = parsed["point"]
                if "text" in parsed:
                    result["text"] = parsed["text"]
        except json.JSONDecodeError:
            pass
            
        return result

# Legacy AXIOM API compatibility aliases
AxiomGroundingEngine = ZelyrGroundingEngine
