import os
import sys
import math
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)


# Force HF caches to the project directory and offline-only behavior.
# Repo-relative so it works on any cluster (mirrors policy_robosuite/models/smolvla.py).
_HF_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hf_weights'))
try:
    os.makedirs(_HF_CACHE, exist_ok=True)
except PermissionError:
    # Import-time guard: don't crash the whole train.py just because the
    # SmolVLA cache dir isn't writable. SmolVLA-using code paths will fail
    # later when they actually need the cache; other policies don't.
    pass
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Minimal constants (avoid lerobot dependency)
ACTION = "action"
OBS_STATE = "observation.state"


def _resolved_model_path(model_id: str) -> str:
    local_dir = os.path.join(_HF_CACHE, "models", "HuggingFaceTB__SmolVLM2-500M-Video-Instruct")
    return local_dir if os.path.isdir(local_dir) else model_id



def _make_prompt(dataset_path: str) -> str:
    name = str(dataset_path).lower()
    if "lift" in name:
        return "Lift the peg upright"
    if "push" in name:
        return "Push the cube to the target"
    if "roll" in name:
        return "Roll the ball to the target"
    raise ValueError("Could not infer task prompt from dataset_path; expected one of ['lift','push','roll'] in the name")


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_tensor_2d(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    b, l, d = x.shape
    if l >= target_len:
        return x
    out = torch.full((b, target_len, d), pad_value, dtype=x.dtype, device=x.device)
    out[:, :l] = x
    return out


def pad_tensor_1d(x: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    b, l = x.shape
    if l >= target_len:
        return x
    out = torch.full((b, target_len), pad_value, dtype=x.dtype, device=x.device)
    out[:, :l] = x
    return out


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10_000) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: float = 4.0, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


from models.backbone import FrozenBatchNorm2d


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        use_plucker: bool = False,
    ):
        super().__init__()
        resolved_id = _resolved_model_path(model_id)
        if load_vlm_weights:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                resolved_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=_HF_CACHE,
                local_files_only=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(resolved_id, cache_dir=_HF_CACHE, local_files_only=True)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(resolved_id, cache_dir=_HF_CACHE, local_files_only=True)
        if num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # derive expert config from loaded VLM config (avoid extra hub hits)
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.use_plucker = bool(use_plucker)
        self.set_requires_grad()

        # Plucker fusion modules (before connector)
        if self.use_plucker:
            # Encode 6-channel Plücker map to a 512-d feature grid
            self.plucker_encoder = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
                FrozenBatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            # Project to vision hidden size and fuse with SigLIP tokens
            vision_hidden = int(self.vlm.config.vision_config.hidden_size)
            self.plucker_out_proj = nn.Conv2d(512, vision_hidden, kernel_size=1)
            self.vision_fusion_proj = nn.Linear(vision_hidden * 2, vision_hidden)
            # Normalize streams and align dtypes with vision encoder
            self.vision_ln = nn.LayerNorm(vision_hidden, elementwise_affine=False)
            self.plucker_ln = nn.LayerNorm(vision_hidden, elementwise_affine=False)
            vf_dtype = self.get_vlm_model().vision_model.dtype
            self.plucker_out_proj = self.plucker_out_proj.to(dtype=vf_dtype)
            self.vision_fusion_proj = self.vision_fusion_proj.to(dtype=vf_dtype)

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for p in self.get_vlm_model().vision_model.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False
        else:
            # Match LeRobot semantics: freeze specific VLM parts even when training the VLM
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)

            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, p in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    p.requires_grad = False

        # Avoid unused params issue with distributed training (if present)
        for name, p in self.lm_expert.named_parameters():
            if "lm_head" in name:
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep frozen parts in eval mode during training, as in LeRobot
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        # Split RGB and (optional) Plücker channels
        if image.shape[1] >= 9 and self.use_plucker:
            rgb = image[:, :3]
            plucker = image[:, 3:9]
        else:
            rgb = image[:, :3]
            plucker = None

        # Resize RGB to 512x512 before SigLIP; keep Plücker unchanged
        rgb_512 = F.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=False)

        # Vision tokens from SigLIP vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(pixel_values=rgb_512.to(dtype=self.get_vlm_model().vision_model.dtype), patch_attention_mask=None)
            .last_hidden_state
        )  # [B, L, Dv]

        # If enabled, fuse Plücker tokens before the connector
        if self.use_plucker and plucker is not None:
            # Determine grid size from number of tokens L = s*s
            b, l, dv = image_hidden_states.shape
            s = int(l ** 0.5)
            if s * s != l:
                raise ValueError("Non-square token grid from vision encoder; cannot align Plücker features")

            # Encode Plücker and pool to s x s grid
            p_feat = self.plucker_encoder(plucker)
            p_feat = F.adaptive_avg_pool2d(p_feat, output_size=(s, s))  # [B, 512, s, s]
            p_feat = self.plucker_out_proj(p_feat)  # [B, Dv, s, s]
            p_tok = p_feat.flatten(2).transpose(1, 2)  # [B, L, Dv]

            # Concatenate and fuse back to Dv (normalize per token first)
            if p_tok.dtype != image_hidden_states.dtype:
                p_tok = p_tok.to(dtype=image_hidden_states.dtype)
            image_hidden_states = self.vision_ln(image_hidden_states)
            p_tok = self.plucker_ln(p_tok)
            fused = torch.cat([image_hidden_states, p_tok], dim=-1)  # [B, L, 2*Dv]
            image_hidden_states = self.vision_fusion_proj(fused)  # [B, L, Dv]

        # Connector to text hidden space
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        num_att_heads = self.config.text_config.num_attention_heads
        num_key_value_heads = self.config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        sequence_length = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
        value_states = value_states[:, :, :, None, :].expand(batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked, dim=-1).to(dtype=value_states.dtype)
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return att_output

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        inputs_embeds: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = [models[0].layers, models[1].layers]
        batch_size = 0
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = hidden_states.shape[0]
                break
        num_layers = len(model_layers[0])
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            # Self-attn or cross-attn selection simplified: use cross-attn by default
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[0][layer_idx] if i == 0 else model_layers[1][layer_idx]
                if hidden_states is None or layer is None:
                    continue
                hs = layer.input_layernorm(hidden_states)
                input_shape = hs.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                hs = hs.to(dtype=layer.self_attn.q_proj.weight.dtype)
                q = layer.self_attn.q_proj(hs).view(hidden_shape)
                k = layer.self_attn.k_proj(hs).view(hidden_shape)
                v = layer.self_attn.v_proj(hs).view(hidden_shape)
                query_states.append(q)
                key_states.append(k)
                value_states.append(v)
            qs = torch.cat(query_states, dim=1)
            ks = torch.cat(key_states, dim=1)
            vs = torch.cat(value_states, dim=1)
            seq_len = qs.shape[1]
            pos_ids = position_ids if seq_len >= position_ids.shape[1] else position_ids[:, :seq_len]
            att_mask = attention_mask if seq_len >= attention_mask.shape[-1] else attention_mask[:, :seq_len, :seq_len]
            qs = apply_rope(qs, pos_ids)
            ks = apply_rope(ks, pos_ids)
            att_out = self.eager_attention_forward(att_mask, batch_size, head_dim, qs, ks, vs)
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[0][layer_idx] if i == 0 else model_layers[1][layer_idx]
                if hidden_states is None or layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]
                if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_sl = att_out[:, start:end].to(layer.self_attn.o_proj.weight.dtype)
                else:
                    att_sl = att_out[:, start:end]
                out_emb = layer.self_attn.o_proj(att_sl)
                out_emb += hidden_states
                after_first = out_emb.clone()
                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb += after_first
                outputs_embeds.append(out_emb)
                start = end
            inputs_embeds = outputs_embeds
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values or {}


@dataclass
class SmolVLAConfig:
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: Tuple[int, int] = (256, 256)
    num_steps: int = 10
    use_cache: bool = True
    # Finetuning
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    # Backbone
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    min_period: float = 4e-3
    max_period: float = 4.0
    # Plücker fusion
    use_plucker: bool = False


class VLAFlowMatching(nn.Module):
    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            use_plucker=self.config.use_plucker,
        )
        self.state_proj = nn.Linear(self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size)
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None):
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks):
            if self.add_image_special_tokens:
                image_start_token = self.vlm_with_expert.embed_language_tokens(self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)).unsqueeze(0).expand(img.shape[0], -1, -1)
                image_start_mask = torch.ones_like(image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device)
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
            img_emb = self.vlm_with_expert.embed_image(img)
            # Use connector outputs directly without extra scaling to keep magnitudes consistent
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs
            if self.add_image_special_tokens:
                image_end_token = self.vlm_with_expert.embed_language_tokens(self.image_end_token.to(device=self.vlm_with_expert.vlm.device)).unsqueeze(0).expand(img.shape[0], -1, -1)
                image_end_mask = torch.ones_like(image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device)
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Avoid extra scaling of language embeddings; rely on model's internal norms
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * (state_emb.shape[1])
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
        if self.prefix_length > 0 and pad_masks.shape[1] < self.prefix_length:
            embs = pad_tensor_2d(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor_1d(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor_2d(att_masks, self.prefix_length, pad_value=0)
        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        embs = []
        pad_masks = []
        att_masks = []
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # sine-cosine time embedding
        half = self.vlm_with_expert.expert_hidden_size // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=device)
        period = self.config.min_period * (self.config.max_period / self.config.min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        time = timestep[:, None]  # [B,1]
        sin_in = scaling_factor[None, :] * time
        time_emb = torch.cat([torch.sin(sin_in), torch.cos(sin_in)], dim=1).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        embs.append(action_time_emb)
        if pad_mask is None:
            action_time_mask = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)
        else:
            action_time_mask = pad_mask.to(device=device)
        pad_masks.append(action_time_mask)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.ones((bsize, embs.shape[1]), dtype=torch.bool, device=device)
        return embs, pad_masks, att_masks

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, actions_is_pad: torch.Tensor) -> torch.Tensor:
        bsize, total_len, _ = actions.shape
        chunk = int(self.config.chunk_size)
        actions_chunk = actions[:, : chunk, :]
        mask_chunk = (~actions_is_pad).to(actions.device)[:, : chunk]
        noise = self.sample_noise(actions_chunk.shape, actions_chunk.device)
        time = self.sample_time(bsize, actions_chunk.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions_chunk
        u_t = noise - actions_chunk
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time, pad_mask=mask_chunk)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        causal = torch.tril(torch.ones((pad_masks.shape[1], pad_masks.shape[1]), dtype=torch.bool, device=pad_masks.device))
        att_2d_masks = causal[None, :, :].expand(bsize, -1, -1)
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -chunk :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        losses = losses * mask_chunk.unsqueeze(-1).float()
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> torch.Tensor:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        bsize, pre_len = prefix_pad_masks.shape
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, expanded_time)
            # Build combined square causal mask so suffix can attend to prefix
            total_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)  # (B, Lp+Ls)
            total_len = total_pad_masks.shape[1]
            causal_full = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool, device=total_pad_masks.device))
            att_2d_masks_full = causal_full[None, :, :].expand(bsize, -1, -1)
            pad_2d_masks_full = total_pad_masks[:, None, :] & total_pad_masks[:, :, None]
            att_2d_masks_full = att_2d_masks_full & pad_2d_masks_full
            position_ids_full = torch.cumsum(total_pad_masks, dim=1) - 1
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks_full,
                position_ids=position_ids_full,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t


class SmolVLAPolicy(nn.Module):
    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        resolved_id = _resolved_model_path(self.config.vlm_model_name)
        self.language_tokenizer = AutoProcessor.from_pretrained(
            resolved_id, cache_dir=_HF_CACHE, local_files_only=True
        ).tokenizer
        self.model = VLAFlowMatching(config)

    def prepare_images(self, batch):
        images = []
        img_masks = []
        present_img_keys = [k for k in batch if k.startswith("observation.images.")]
        if len(present_img_keys) == 0:
            raise ValueError("No image features found in batch")
        for key in present_img_keys:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch):
        device = batch[OBS_STATE].device
        tasks = batch.get("task", "")
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]
        tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
        tokenized = self.language_tokenizer.__call__(tasks, padding="longest", padding_side="right", max_length=48, return_tensors="pt")
        lang_tokens = tokenized["input_ids"].to(device=device)
        lang_masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        if state.shape[-1] < self.config.max_state_dim:
            pad = torch.zeros(state.shape[0], self.config.max_state_dim - state.shape[-1], device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        return state

    def prepare_action(self, batch):
        actions = batch[ACTION]
        if actions.shape[-1] < self.config.max_action_dim:
            pad = torch.zeros(actions.shape[0], actions.shape[1], self.config.max_action_dim - actions.shape[-1], device=actions.device, dtype=actions.dtype)
            actions = torch.cat([actions, pad], dim=-1)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        # Trim back to original action dim
        return actions[:, :, : self.config.max_action_dim]

    def forward(self, batch: Dict[str, torch.Tensor]):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch["actions_id_pad"].to(dtype=torch.bool)
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, actions_is_pad)
        losses = losses[:, :, : self.config.max_action_dim]
        valid_mask = (~actions_is_pad).to(device=losses.device, dtype=losses.dtype)[:, : self.config.chunk_size]
        valid_count = valid_mask.sum()
        denom = (valid_count * self.config.max_action_dim).clamp_min(1.0)
        loss = losses.sum() / denom
        return loss, {"loss": float(loss.item())}


class SmolVLAPolicyWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_cameras = int(args.num_side_cam)
        self.chunk_size = int(args.chunk_size)
        self.obs_dim = int(args.obs_dim)
        self.act_dim = int(args.action_dim)
        self.prompt = _make_prompt(args.dataset_path)
        cfg = SmolVLAConfig(
            n_obs_steps=1,
            chunk_size=self.chunk_size,
            n_action_steps=self.chunk_size,
            max_action_dim=self.act_dim,
            max_state_dim=max(32, self.obs_dim),
            resize_imgs_with_padding=(256, 256),
            freeze_vision_encoder=args.freeze_vision_encoder,
            train_expert_only=args.train_expert_only,
            use_plucker=args.use_plucker,
        )
        self.policy = SmolVLAPolicy(cfg)
        # Optimizer hyperparameters aligned with robosuite_policy
        self._optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=1e-4, 
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=1e-3
        )

    def configure_optimizers(self):
        return self._optimizer

    def _images_to_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, c, h, w = images.shape
        out = {}
        use_plucker = self.args.use_plucker
        for i in range(n):
            out[f"observation.images.cam{i}"] = images[:, i, :9] if use_plucker else images[:, i, :3]
        return out

    def _make_batch(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        imgs = data["image"]
        batch.update(self._images_to_batch(imgs))
        batch[OBS_STATE] = data["qpos"].float()
        batch["task"] = self.prompt
        if "actions" in data:
            batch[ACTION] = data["actions"].float()
            if "is_pad" in data:
                batch["actions_id_pad"] = data["is_pad"].bool()
        return batch

    def __call__(self, data_dict: Dict[str, torch.Tensor]):
        if "actions" in data_dict:
            batch = self._make_batch(data_dict)
            loss, _ = self.policy.forward(batch)
            return {"loss": loss}
        batch = self._make_batch(data_dict)
        with torch.no_grad():
            actions_norm = self.policy.predict_action_chunk(batch)
        return actions_norm[:, : self.chunk_size, : int(self.act_dim)]


