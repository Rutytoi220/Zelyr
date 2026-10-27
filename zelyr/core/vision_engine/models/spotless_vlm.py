import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

class SpotlessLatentAdapter(nn.Module):
    def __init__(self, d: int = 3584, bottleneck: int = 448):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d)
        )
        self.verbalizer = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d)
        )
        for module in [self.refiner, self.verbalizer]:
            nn.init.normal_(module[0].weight, std=0.01)
            nn.init.zeros_(module[0].bias)
            nn.init.normal_(module[2].weight, std=0.01)
            nn.init.zeros_(module[2].bias)

    def refine(self, h: torch.Tensor) -> torch.Tensor:
        target_dtype = self.refiner[0].weight.dtype
        h_cast = h.to(target_dtype)
        return h_cast + self.refiner(h_cast)

    def verbalize(self, h: torch.Tensor) -> torch.Tensor:
        target_dtype = self.verbalizer[0].weight.dtype
        h_cast = h.to(target_dtype)
        return h_cast + self.verbalizer(h_cast)


def _to_device(obj, device):
    """Recursively pushes nested tensors (lists of lists of tensors) to CUDA."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


class SpotlessMiniCPMV(nn.Module):
    def __init__(self, base_vlm, adapter: SpotlessLatentAdapter, processor: AutoProcessor = None):
        super().__init__()
        self.vlm = base_vlm
        self.adapter = adapter
        self.processor = processor
        
        embed_weights = self.vlm.llm.model.embed_tokens.weight.data
        self.register_buffer("embed_norm", embed_weights.norm(p=2, dim=-1).mean())

    def prepare_multimodal_data(self, image: Image.Image, prompt: str, max_slice_nums: int = 1):
        device = next(self.adapter.parameters()).device
        copy_msgs = [{'role': 'user', 'content': [image, prompt]}]

        images = []
        for msg in copy_msgs:
            cur_msgs = []
            for c in msg['content']:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append('(<image>./</image>)')
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg['content'] = '\n'.join(cur_msgs)

        formatted_prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            [formatted_prompt],
            [[image]],
            max_slice_nums=max_slice_nums,
            return_tensors='pt'
        )

        return _to_device(dict(inputs), device)

    def forward_latent(self, data: dict, K: int = 4):
        # 1. Prefill
        inputs_embeds, _ = self.vlm.get_vllm_embedding(data)
        model_dtype = self.vlm.llm.model.embed_tokens.weight.dtype

        out = self.vlm.llm.model(
            inputs_embeds=inputs_embeds,
            use_cache=True
        )
        pkv = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]

        # 2. Recurrent Deliberation (K=4)
        for _ in range(K):
            h_ref = self.adapter.refine(h)
            # Hypersphere projection aligned to embedding norm and base model dtype
            norm_val = h_ref.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            e_lat = ((h_ref / norm_val) * self.embed_norm).to(model_dtype)
            
            out = self.vlm.llm.model(
                inputs_embeds=e_lat,
                past_key_values=pkv,
                use_cache=True
            )
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]
            del h_ref, norm_val, e_lat, out

        # 3. Verbalizer Projection
        h_verb = self.adapter.verbalize(h)
        norm_dtype = self.vlm.llm.model.norm.weight.dtype
        normed_h = self.vlm.llm.model.norm(h_verb.to(norm_dtype))
        logits_0 = self.vlm.llm.lm_head(normed_h)

        return logits_0, pkv, h
