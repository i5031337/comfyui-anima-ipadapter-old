"""Anima IP-Adapter ComfyUI Custom Nodes (SigLIP2)."""

from .config import SIGLIP2_DIR
from .loader import AnimaIPAdapterLoader
from .apply import AnimaIPAdapterApply
from .visualize import AnimaIPAdapterVisualize

NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterLoader": AnimaIPAdapterLoader,
    "AnimaIPAdapterApply": AnimaIPAdapterApply,
    "AnimaIPAdapterVisualize": AnimaIPAdapterVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterLoader": "Anima IP-Adapter Loader (SigLIP2)",
    "AnimaIPAdapterApply": "Anima IP-Adapter Apply (SigLIP2)",
    "AnimaIPAdapterVisualize": "Anima IP Attn Heatmap",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SIGLIP2_DIR"]
