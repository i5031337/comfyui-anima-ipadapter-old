"""Anima IP-Adapter apply node for ComfyUI."""

import math
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import transforms as T

from .common import (
    _make_linear,
    _make_shared_proj,
    _make_shared_proj_ref,
    _LoRALinear,
)
from .patcher import patch_dit_blocks


class IPAdapterHandler:
    """IP token injection with optional independent IP CFG scaling."""

    def __init__(self, ip_tokens, null_tokens, ip_cfg_scale, ip_cfg_separate, ip_dtype):
        self.ip_tokens = ip_tokens
        self.null_tokens = null_tokens
        self.ip_cfg_scale = ip_cfg_scale
        self.ip_cfg_separate = ip_cfg_separate
        self.ip_dtype = ip_dtype
        self._printed = False
        self._cond_wo_ip = None
        self._enabled = (not ip_cfg_separate) or (ip_cfg_scale > 1.0)

    def __call__(self, apply_model_fn, args_dict):
        model_input = args_dict["input"]
        timestep = args_dict["timestep"]
        c = dict(args_dict["c"])
        cond_or_uncond = args_dict.get("cond_or_uncond", None)

        all_uncond = (cond_or_uncond is not None and all(u == 1 for u in cond_or_uncond))

        if not all_uncond:
            ip_B = model_input.shape[0]
            target_device = model_input.device

            ip_tok = self.ip_tokens.to(dtype=self.ip_dtype, device=target_device)
            ip_tok_batch = ip_tok.expand(ip_B, -1, -1).clone()

            if self._enabled and cond_or_uncond is not None:
                null_tok = self.null_tokens.to(dtype=self.ip_dtype, device=target_device)
                for idx, is_uncond in enumerate(cond_or_uncond):
                    if idx < ip_B and is_uncond:
                        ip_tok_batch[idx] = null_tok

                c["transformer_options"] = dict(c.get("transformer_options", {}))
                c["transformer_options"]["anima_ip_tokens"] = ip_tok_batch

                if self.ip_cfg_separate:
                    out = apply_model_fn(model_input, timestep, **c)
                    null_batch = null_tok.expand(ip_B, -1, -1)
                    c_null = dict(c)
                    to_null = dict(c["transformer_options"])
                    to_null["anima_ip_tokens"] = null_batch
                    c_null["transformer_options"] = to_null
                    out_null = apply_model_fn(model_input, timestep, **c_null)

                    batch_chunks = len(cond_or_uncond)
                    chunks_wo = out_null.chunk(batch_chunks)
                    for idx, is_uncond in enumerate(cond_or_uncond):
                        if not is_uncond and idx < len(chunks_wo):
                            self._cond_wo_ip = chunks_wo[idx]
                            break

                    if not self._printed:
                        print(f"[AnimaIPAdapter] Independent IP CFG (2-pass), batch={ip_B}, ip_cfg_scale={self.ip_cfg_scale}")
                        self._printed = True
                    return out
                else:
                    if not self._printed:
                        print(f"[AnimaIPAdapter] IP CFG bound to text CFG (1-pass), batch={ip_B}, ip_cfg_scale={self.ip_cfg_scale}")
                        self._printed = True
                    return apply_model_fn(model_input, timestep, **c)
            else:
                if not self._printed:
                    print(f"[AnimaIPAdapter] IP CFG disabled (batch={ip_B})")
                    self._printed = True

            c["transformer_options"] = dict(c.get("transformer_options", {}))
            c["transformer_options"]["anima_ip_tokens"] = ip_tok_batch

        return apply_model_fn(model_input, timestep, **c)

    def post_cfg(self, args):
        if not self.ip_cfg_separate or not self._enabled or self._cond_wo_ip is None:
            return args["denoised"]

        cond_w_ip = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        cond_scale = args["cond_scale"]
        cond_wo_ip = self._cond_wo_ip.to(dtype=cond_w_ip.dtype, device=cond_w_ip.device)

        ip_eff = self.ip_cfg_scale - 1.0
        correction = (ip_eff - cond_scale) * (cond_w_ip - cond_wo_ip)
        return args["denoised"] + correction

    def to(self, *args, **kwargs):
        return self


class AnimaIPAdapterApply:
    """Encodes ref image through SigLIP2, injects IP tokens via transformer_options."""

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ip_adapter": ("ANIMA_IP_ADAPTER",),
                "ref_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_image_size": ("INT", {"default": 512, "min": 224, "max": 2048, "step": 16}),
                "siglip_layer": ("INT", {"default": -1, "min": -1, "max": 24}),
                "ip_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.05}),
                "ip_cfg_separate": ("BOOLEAN", {"default": False}),
                "gray_null": ("BOOLEAN", {"default": False}),
                "use_lora": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    def apply(self, model, ip_adapter, ref_image, strength, ref_image_size=512,
              siglip_layer=-1, ip_cfg_scale=4.0, ip_cfg_separate=False,
              gray_null=False, use_lora=True):
        patched_model = model.clone()
        siglip_encoder = ip_adapter["siglip_encoder"]
        ip_weights = ip_adapter["ip_weights"]
        num_blocks = ip_adapter["num_blocks"]
        ip_embed_dim = ip_adapter["ip_embed_dim"]
        inner_dim = ip_adapter["inner_dim"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(patched_model.model, "get_dtype"):
            model_dtype = patched_model.model.get_dtype()
        else:
            model_dtype = next(patched_model.model.diffusion_model.parameters()).dtype

        if not model_dtype.is_floating_point:
            model_dtype = torch.bfloat16
        use_shared = ip_adapter.get("use_shared_projection", False)
        ip_norm_keys = ip_adapter.get("ip_norm_keys", False)
        ip_inject_before_mlp = ip_adapter.get("ip_inject_before_mlp", False)

        dit = patched_model.model.diffusion_model
        if use_shared:
            if not hasattr(dit, "shared_ip_k_proj"):
                dit.shared_ip_k_proj = _make_shared_proj(ip_embed_dim, inner_dim, device, model_dtype)
                dit.shared_ip_v_proj = _make_shared_proj(ip_embed_dim, inner_dim, device, model_dtype)

            for proj_name in ["shared_ip_k_proj", "shared_ip_v_proj"]:
                mlp = getattr(dit, proj_name)
                for pname, p in mlp.named_parameters():
                    key = f"{proj_name}.{pname}"
                    if key in ip_weights:
                        p.data.copy_(ip_weights[key].to(dtype=p.dtype, device=p.device))

            has_shared_q = any(k.startswith("shared_ip_q_proj") for k in ip_weights.keys())
            if has_shared_q and not hasattr(dit, "shared_ip_q_proj"):
                x_dim = ip_weights.get("blocks.0.adaln_ip.1.weight", None)
                x_dim = x_dim.shape[1] if x_dim is not None else inner_dim
                dit.shared_ip_q_proj = _make_linear(x_dim, inner_dim, bias=False).to(device=device, dtype=model_dtype)
            if has_shared_q:
                mlp = dit.shared_ip_q_proj
                for pname, p in mlp.named_parameters():
                    key = f"shared_ip_q_proj.{pname}"
                    if key in ip_weights:
                        p.data.copy_(ip_weights[key].to(dtype=p.dtype, device=p.device))

            for i, block in enumerate(dit.blocks):
                if i >= num_blocks:
                    break
                block_device = next(block.parameters()).device
                if not hasattr(block, "ip_k_proj"):
                    block.ip_k_proj = _make_shared_proj_ref(dit.shared_ip_k_proj)
                    block.ip_v_proj = _make_shared_proj_ref(dit.shared_ip_v_proj)
                    block.adaln_ip = nn.Sequential(
                        nn.SiLU(),
                        _make_linear(inner_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device),
                    )
                    nn.init.zeros_(block.adaln_ip[1].weight)
                    nn.init.constant_(block.adaln_ip[1].bias, 0.1)
                    block.use_ip_adapter = True
                    if not hasattr(block, "ip_norm_keys"):
                        block.ip_norm_keys = ip_norm_keys
                        block.ip_inject_before_mlp = ip_inject_before_mlp
                if ip_norm_keys and not hasattr(block, "ip_k_norm"):
                    h_d = block.cross_attn.head_dim
                    block.ip_k_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(device=block_device, dtype=model_dtype)
                if has_shared_q and not hasattr(block, "ip_q_proj"):
                    block.ip_q_proj = _make_shared_proj_ref(dit.shared_ip_q_proj)
                    h_d = block.cross_attn.head_dim
                    block.ip_q_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(device=block_device, dtype=model_dtype)
                bp = dict(block.named_parameters())
                for pname in ["adaln_ip.1.weight", "adaln_ip.1.bias"]:
                    key = f"blocks.{i}.{pname}"
                    if key in ip_weights and pname in bp:
                        bp[pname].data.copy_(ip_weights[key].to(dtype=bp[pname].dtype, device=bp[pname].device))
        else:
            for i, block in enumerate(dit.blocks):
                if i >= num_blocks:
                    break
                block_device = next(block.parameters()).device
                if not hasattr(block, "ip_k_proj"):
                    block.ip_k_proj = _make_linear(ip_embed_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device)
                    block.ip_v_proj = _make_linear(ip_embed_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device)
                    nn.init.normal_(block.ip_k_proj.weight, std=1.0 / math.sqrt(ip_embed_dim))
                    nn.init.zeros_(block.ip_k_proj.bias)
                    nn.init.normal_(block.ip_v_proj.weight, std=1.0 / math.sqrt(ip_embed_dim))
                    nn.init.zeros_(block.ip_v_proj.bias)
                    block.adaln_ip = nn.Sequential(
                        nn.SiLU(),
                        _make_linear(inner_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device),
                    )
                    nn.init.zeros_(block.adaln_ip[1].weight)
                    nn.init.constant_(block.adaln_ip[1].bias, 0.1)
                    block.use_ip_adapter = True
                    if not hasattr(block, "ip_norm_keys"):
                        block.ip_norm_keys = ip_norm_keys
                        block.ip_inject_before_mlp = ip_inject_before_mlp
                if ip_norm_keys and not hasattr(block, "ip_k_norm"):
                    h_d = block.cross_attn.head_dim
                    block.ip_k_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(device=block_device, dtype=model_dtype)
                bp = dict(block.named_parameters())
                for pname in ["ip_k_proj.weight", "ip_k_proj.bias", "ip_v_proj.weight", "ip_v_proj.bias", "adaln_ip.1.weight", "adaln_ip.1.bias"]:
                    key = f"blocks.{i}.{pname}"
                    if key in ip_weights and pname in bp:
                        bp[pname].data.copy_(ip_weights[key].to(dtype=bp[pname].dtype, device=bp[pname].device))

        lora_weights = ip_adapter.get("lora_weights", {})
        lora_rank = ip_adapter.get("lora_rank", 0)
        _lora_attrs = ("q_proj", "k_proj", "v_proj", "output_proj")
        if use_lora and lora_weights and lora_rank:
            alpha = lora_rank
            scale = alpha / lora_rank
            for bi, layers in lora_weights.items():
                if bi >= len(dit.blocks):
                    continue
                block = dit.blocks[bi]
                for ln, ab_dict in layers.items():
                    if "A" not in ab_dict or "B" not in ab_dict:
                        continue
                    original = getattr(block.cross_attn, ln, None)
                    if original is None or isinstance(original, _LoRALinear):
                        continue
                    setattr(block.cross_attn, ln, _LoRALinear(original, ab_dict["A"], ab_dict["B"], scale))
            print(f"[AnimaIPAdapter] LoRA installed on {len(lora_weights)} blocks")
        else:
            _unwrapped = 0
            for _blk in dit.blocks:
                for _ln in _lora_attrs:
                    _mod = getattr(_blk.cross_attn, _ln, None)
                    if isinstance(_mod, _LoRALinear):
                        setattr(_blk.cross_attn, _ln, _mod.base)
                        _unwrapped += 1
            if _unwrapped:
                print(f"[AnimaIPAdapter] LoRA unwrapped on {_unwrapped} layers")

        patch_dit_blocks(dit, num_blocks)

        norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        def resize_pad_tensor(pil_img, target=512):
            w, h = pil_img.size
            ratio = target / max(w, h)
            new_w = max(1, round(w * ratio))
            new_h = max(1, round(h * ratio))
            img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
            square = PILImage.new("RGB", (target, target), (0, 0, 0))
            px = (target - new_w) // 2
            py = (target - new_h) // 2
            square.paste(img, (px, py))
            return norm(T.ToTensor()(square))

        pil_images = []
        for b in range(ref_image.shape[0]):
            arr = (ref_image[b].cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(PILImage.fromarray(arr, mode="RGB"))

        tensors = [resize_pad_tensor(img, ref_image_size) for img in pil_images]
        img_tensor = torch.stack(tensors).to(device=device, dtype=torch.float32)

        siglip_norm = ip_adapter.get("siglip_norm", None)
        siglip_encoder.to(device)
        with torch.no_grad():
            if siglip_layer == -1:
                ip_tokens = siglip_encoder(img_tensor, interpolate_pos_encoding=True).last_hidden_state
            else:
                outputs = siglip_encoder(img_tensor, interpolate_pos_encoding=True, output_hidden_states=True)
                ip_tokens = outputs.hidden_states[siglip_layer]
                if siglip_norm is not None:
                    siglip_norm.to(device)
                    ip_tokens = siglip_norm(ip_tokens)
            ip_tokens = ip_tokens * strength
        siglip_encoder.to("cpu")

        siglip_compressor = ip_adapter.get("siglip_compressor", None)
        if siglip_compressor is not None:
            siglip_compressor.to(device)
            ip_tokens = siglip_compressor(ip_tokens)

        ip_self_attn = ip_adapter.get("ip_self_attn", None)
        if ip_self_attn is not None:
            ip_self_attn.to(device)
            ip_tokens = ip_self_attn(ip_tokens)

        print(f"[AnimaIPAdapter] ip_tokens shape={list(ip_tokens.shape)}, norm={ip_tokens.norm().item():.4f}, strength={strength}")

        ip_tokens_stored = ip_tokens.detach().to(dtype=model_dtype)
        null_tokens_stored = ip_adapter.get("null_tokens", None)
        if gray_null:
            gray_img = PILImage.new("RGB", (ref_image_size, ref_image_size), color=(128, 128, 128))
            gray_tensor = resize_pad_tensor(gray_img, ref_image_size).unsqueeze(0).to(device=device, dtype=torch.float32)
            siglip_encoder.to(device)
            with torch.no_grad():
                if siglip_layer == -1:
                    null_tokens_stored = siglip_encoder(gray_tensor, interpolate_pos_encoding=True).last_hidden_state
                else:
                    outputs = siglip_encoder(gray_tensor, interpolate_pos_encoding=True, output_hidden_states=True)
                    null_tokens_stored = outputs.hidden_states[siglip_layer]
                    if siglip_norm is not None:
                        siglip_norm.to(device)
                        null_tokens_stored = siglip_norm(null_tokens_stored)
            siglip_encoder.to("cpu")
            if siglip_compressor is not None:
                siglip_compressor.to(device)
                null_tokens_stored = siglip_compressor(null_tokens_stored)
            if ip_self_attn is not None:
                ip_self_attn.to(device)
                null_tokens_stored = ip_self_attn(null_tokens_stored)
            null_tokens_stored = null_tokens_stored.detach().to(dtype=model_dtype)
        elif null_tokens_stored is not None:
            null_tokens_stored = null_tokens_stored.to(dtype=model_dtype, device=device)
            if null_tokens_stored.shape != ip_tokens_stored.shape:
                null_tokens_stored = null_tokens_stored.expand_as(ip_tokens_stored)
        else:
            null_tokens_stored = torch.zeros_like(ip_tokens_stored)

        handler = IPAdapterHandler(ip_tokens_stored, null_tokens_stored, ip_cfg_scale, ip_cfg_separate, model_dtype)
        patched_model.set_model_unet_function_wrapper(handler)
        patched_model.set_model_sampler_post_cfg_function(handler.post_cfg)
        return (patched_model,)
