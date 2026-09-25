"""Anima IP-Adapter visualization node for ComfyUI."""

import math
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torchvision import transforms as T


class AnimaIPAdapterVisualize:
    """Visualize where the IP-Adapter is looking in the reference image."""

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter": ("ANIMA_IP_ADAPTER",),
                "ref_image": ("IMAGE",),
                "mode": (["composite", "token_norm", "key_norm", "key_unique"],
                         {"default": "composite"}),
                "ref_image_size": ("INT", {"default": 512, "min": 224, "max": 2048, "step": 16}),
                "opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap",)
    FUNCTION = "visualize"

    def visualize(self, ip_adapter, ref_image, mode="composite",
                  ref_image_size=512, opacity=0.6):
        siglip_encoder = ip_adapter["siglip_encoder"]
        ip_weights = ip_adapter["ip_weights"]
        num_blocks = ip_adapter["num_blocks"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        arr = (ref_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr, mode="RGB")
        w, h = pil_img.size
        ratio = ref_image_size / max(w, h)
        nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
        img_resized = pil_img.resize((nw, nh), PILImage.BILINEAR)
        square = PILImage.new("RGB", (ref_image_size, ref_image_size), (0, 0, 0))
        square.paste(img_resized, ((ref_image_size - nw) // 2, (ref_image_size - nh) // 2))
        img_tensor = norm(T.ToTensor()(square)).unsqueeze(0).to(device)

        siglip_encoder.to(device)
        with torch.no_grad():
            tokens = siglip_encoder(img_tensor, interpolate_pos_encoding=True).last_hidden_state
        siglip_encoder.to("cpu")

        tok_norms = tokens[0].norm(dim=-1)
        token_norm_map = (tok_norms - tok_norms.min()) / (tok_norms.max() - tok_norms.min() + 1e-8)

        key_norm_map = torch.zeros(tokens.shape[1], device=tokens.device)
        key_unique_map = torch.zeros(tokens.shape[1], device=tokens.device)
        blocks_with_weights = 0

        use_shared = ip_adapter.get("use_shared_projection", False)
        if use_shared:
            pass
        else:
            for i in range(num_blocks):
                k_key = f"blocks.{i}.ip_k_proj.weight"
                if k_key not in ip_weights:
                    continue
                w_k = ip_weights[k_key].to(device)
                keys_proj = F.linear(tokens[0], w_k)
                knorms = keys_proj.norm(dim=-1)
                knorms = (knorms - knorms.min()) / (knorms.max() - knorms.min() + 1e-8)
                key_norm_map += knorms

                mean_key = keys_proj.mean(dim=0, keepdim=True)
                deviation = (keys_proj - mean_key).norm(dim=-1)
                deviation = (deviation - deviation.min()) / (deviation.max() - deviation.min() + 1e-8)
                key_unique_map += deviation
                blocks_with_weights += 1

        if blocks_with_weights > 0:
            key_norm_map /= blocks_with_weights
            key_unique_map /= blocks_with_weights

        if mode == "token_norm":
            score_map = token_norm_map
        elif mode == "key_norm":
            score_map = key_norm_map
        elif mode == "key_unique":
            score_map = key_unique_map
        else:
            def minmax(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            score_map = (minmax(token_norm_map) + minmax(key_norm_map) + minmax(key_unique_map)) / 3.0

        grid_dim = int(round(math.sqrt(score_map.shape[0])))
        score_2d = score_map[:grid_dim * grid_dim].cpu().numpy().reshape(grid_dim, grid_dim)
        score_tensor = torch.from_numpy(score_2d).float().unsqueeze(0).unsqueeze(0)
        score_up = F.interpolate(score_tensor, size=(512, 512), mode="bilinear", align_corners=False).squeeze().numpy()

        smin, smax = score_up.min().item(), score_up.max().item()
        if smax - smin > 1e-8:
            score_up = (score_up - smin) / (smax - smin)

        heatmap_rgb = np.zeros((512, 512, 3), dtype=np.float32)
        for c in range(3):
            lo = [0.0, 0.0, 0.8][c]
            mid = [0.0, 1.0, 0.0][c]
            hi = [1.0, 0.0, 0.0][c]
            heatmap_rgb[:, :, c] = np.where(
                score_up < 0.5,
                lo + (mid - lo) * (score_up / 0.5),
                mid + (hi - mid) * ((score_up - 0.5) / 0.5)
            )

        ref_rgb = T.ToTensor()(square).permute(1, 2, 0).cpu().numpy()
        overlay = np.clip(heatmap_rgb * opacity + ref_rgb * (1.0 - opacity), 0.0, 1.0)
        return (torch.from_numpy(overlay).float().unsqueeze(0),)
