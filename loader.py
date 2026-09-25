"""Anima IP-Adapter loader node for ComfyUI."""

import os
import re
import folder_paths
import torch
import torch.nn as nn
from transformers import SiglipVisionModel
import safetensors
import safetensors.torch

from .common import (
    SIGLIP_HIDDEN_SIZE,
    IPSelfAttn,
    SigLIPCompressor,
)
from .config import SIGLIP2_DIR


class AnimaIPAdapterLoader:
    """Loads SigLIP2 encoder and IP-Adapter K/V weights."""

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"),),
                "auto_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-download SigLIP2 encoder to models/siglip2/",
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_IP_ADAPTER",)
    RETURN_NAMES = ("ip_adapter",)
    FUNCTION = "load"

    def load(self, ip_adapter_name, auto_download):
        siglip_id = "google/siglip2-base-patch16-512"
        if auto_download:
            os.makedirs(SIGLIP2_DIR, exist_ok=True)
            if not os.path.isfile(os.path.join(SIGLIP2_DIR, "config.json")):
                print(f"[AnimaIPAdapter] Downloading {siglip_id} ...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=siglip_id, local_dir=SIGLIP2_DIR)
                print(f"[AnimaIPAdapter] SigLIP2 downloaded to {SIGLIP2_DIR}")

        if os.path.isdir(SIGLIP2_DIR):
            siglip_encoder = SiglipVisionModel.from_pretrained(SIGLIP2_DIR)
        else:
            siglip_encoder = SiglipVisionModel.from_pretrained(siglip_id)
        siglip_encoder.requires_grad_(False)
        siglip_encoder.eval()

        ip_adapter_path = folder_paths.get_full_path_or_raise("ipadapter", ip_adapter_name)
        ip_sd = {}
        metadata = {}
        with safetensors.safe_open(ip_adapter_path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                ip_sd[k] = f.get_tensor(k)
        ip_norm_keys = metadata.get("ip_norm_keys", "False").lower() == "true"
        ip_inject_before_mlp = metadata.get("ip_inject_before_mlp", "False").lower() == "true"
        print(f"[AnimaIPAdapter] Loaded {len(ip_sd)} IP-Adapter keys from {ip_adapter_path}"
              f"{' (ip_norm_keys)' if ip_norm_keys else ''}"
              f"{' (ip_inject_before_mlp)' if ip_inject_before_mlp else ''}")

        use_shared = any(k.startswith("shared_ip_k_proj") for k in ip_sd.keys())

        block_indices = set()
        for k in ip_sd.keys():
            m = re.match(r"blocks\.(\d+)\.(ip_k_proj|ip_v_proj|adaln_ip)", k)
            if m:
                block_indices.add(int(m.group(1)))
        num_blocks = max(block_indices) + 1 if block_indices else 28
        print(f"[AnimaIPAdapter] {num_blocks} blocks detected"
              f"{' (shared projection)' if use_shared else ''}")

        if use_shared:
            k0 = ip_sd.get("shared_ip_k_proj.expand.weight")
            if k0 is None:
                k0 = ip_sd.get("shared_ip_k_proj.0.weight")
            inner_dim, ip_embed_dim = k0.shape if k0 is not None else (SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE)
        else:
            sample_weight = None
            for probe in [f"blocks.{i}.ip_k_proj.weight" for i in range(num_blocks)]:
                if probe in ip_sd:
                    sample_weight = ip_sd[probe]
                    break
            if sample_weight is not None:
                inner_dim, ip_embed_dim = sample_weight.shape
            else:
                ip_embed_dim = SIGLIP_HIDDEN_SIZE
                inner_dim = ip_embed_dim
        print(f"[AnimaIPAdapter] ip_embed_dim={ip_embed_dim}, inner_dim={inner_dim}")

        lora_weights = {}
        lora_rank = 0
        for k, v in ip_sd.items():
            if k.startswith("lora."):
                inner = k[len("lora."):]
                m = re.match(
                    r"base_model\.model\.blocks\.(\d+)\.cross_attn\.(\w+)\.lora_([AB])(?:\.default)?\.weight",
                    inner)
                if m:
                    bi = int(m.group(1))
                    ln = m.group(2)
                    ab = m.group(3)
                    lora_weights.setdefault(bi, {}).setdefault(ln, {})[ab] = v
                    if ab == "A":
                        lora_rank = max(lora_rank, v.shape[0])

        _lora_keys = [k for k in ip_sd if k.startswith("lora.")]
        if _lora_keys and not lora_weights:
            print(f"[AnimaIPAdapter] {len(_lora_keys)} lora keys unmatched, examples:")
            for k in _lora_keys[:3]:
                inner = k[len("lora."):]
                print(f"  {inner[:100]}")

        if lora_weights:
            print(f"[AnimaIPAdapter] Found LoRA: rank={lora_rank}, blocks={sorted(lora_weights.keys())}")

        self_attn_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("ip_self_attn."):
                self_attn_sd[k[len("ip_self_attn."):]] = v
        ip_self_attn = None
        if self_attn_sd:
            module = IPSelfAttn(dim=ip_embed_dim)
            module.load_state_dict(self_attn_sd, strict=False)
            ip_self_attn = module
            print(f"[AnimaIPAdapter] Found IP-SelfAttn: {len(self_attn_sd)} keys")

        siglip_norm_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("siglip_norm."):
                siglip_norm_sd[k[len("siglip_norm."):]] = v
        siglip_norm = None
        if siglip_norm_sd:
            siglip_norm = nn.LayerNorm(ip_embed_dim, elementwise_affine=True)
            siglip_norm.load_state_dict(siglip_norm_sd)
            print(f"[AnimaIPAdapter] Found SigLIP LayerNorm (siglip_norm): {len(siglip_norm_sd)} keys")

        compressor_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("siglip_compressor."):
                compressor_sd[k[len("siglip_compressor."):]] = v
        siglip_compressor = None
        if compressor_sd:
            queries_shape = compressor_sd.get("queries")
            num_queries = queries_shape.shape[0] if queries_shape is not None else 64
            num_comp_layers = max(
                (int(k.split(".")[1]) + 1 for k in compressor_sd if k.startswith("layers.")),
                default=2,
            )
            siglip_compressor = SigLIPCompressor(
                dim=ip_embed_dim, num_queries=num_queries, num_layers=num_comp_layers
            )
            siglip_compressor.load_state_dict(compressor_sd, strict=False)
            print(f"[AnimaIPAdapter] Found SigLIPCompressor: {num_queries} tokens, {len(compressor_sd)} keys")

        null_tokens = ip_sd.pop("null_tokens", None)
        if null_tokens is not None:
            print(f"[AnimaIPAdapter] Found learned null_tokens: shape={list(null_tokens.shape)}")

        return ({
            "ip_weights": ip_sd,
            "siglip_encoder": siglip_encoder,
            "num_blocks": num_blocks,
            "ip_embed_dim": ip_embed_dim,
            "inner_dim": inner_dim,
            "use_shared_projection": use_shared,
            "lora_weights": lora_weights,
            "lora_rank": lora_rank,
            "ip_self_attn": ip_self_attn,
            "siglip_norm": siglip_norm,
            "siglip_compressor": siglip_compressor,
            "ip_norm_keys": ip_norm_keys,
            "ip_inject_before_mlp": ip_inject_before_mlp,
            "null_tokens": null_tokens,
        },)
