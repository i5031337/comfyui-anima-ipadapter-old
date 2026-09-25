# ComfyUI Anima IP-Adapter

Custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) providing IP-Adapter (Image Prompt Adapter) support for the **Anima** and **Anima Turbo** models.

Enables reference-image-driven character consistency, expression sprite generation, and style steering via decoupled cross-attention injection into Anima's DiT blocks.

---

## Features

- **Character Consistency**: Steer Anima generations using a single reference portrait image.
- **Full GGUF Support**: Compatible with quantized Anima Turbo GGUF models (`UnetLoaderGGUFAdvanced`) for low VRAM (4GB+) as well as standard `.safetensors`.
- **SigLIP2 Vision Encoding**: Uses `google/siglip2-base-patch16-512` for rich visual feature extraction.
- **Dynamic Expressions & Body Language**: Optimized for generating character sprite sheets and contrasting emotions across all 28 expressions in `expressions.txt`.
- **Modular & Lightweight**: Clean architecture adhering to strict code modularity standards.

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:i5031337/comfyui-anima-ipadapter.git
pip install -r comfyui-anima-ipadapter/requirements.txt
```

---

## Model Download Commands

You can download all required models automatically using the included script:

```bash
python comfyui-anima-ipadapter/download_models.py
```

To also download the base Anima Turbo GGUF, Qwen Text Encoder, and VAE:
```bash
python comfyui-anima-ipadapter/download_models.py --include-base-models
```

### Manual Download Commands (CLI & Python)

#### 1. IP-Adapter Weights (`ip_adapter-Character_Reference-10.safetensors`)
Destination: `ComfyUI/models/ipadapter/`

```bash
# Using huggingface-cli:
huggingface-cli download LuciferTC/Anima-IP-Adapter ip_adapter-Character_Reference-10.safetensors --local-dir ComfyUI/models/ipadapter
```

```python
# Using Python:
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="LuciferTC/Anima-IP-Adapter",
    filename="ip_adapter-Character_Reference-10.safetensors",
    local_dir="ComfyUI/models/ipadapter",
)
```

#### 2. SigLIP2 Vision Encoder (`siglip2-base-patch16-512`)
Destination: `ComfyUI/models/siglip2/siglip2-base-patch16-512/`

```bash
# Using huggingface-cli:
huggingface-cli download google/siglip2-base-patch16-512 --local-dir ComfyUI/models/siglip2/siglip2-base-patch16-512
```

```python
# Using Python:
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/siglip2-base-patch16-512",
    local_dir="ComfyUI/models/siglip2/siglip2-base-patch16-512",
)
```

#### 3. Base Anima Turbo Model (GGUF, Text Encoder, VAE)
```bash
# Anima Turbo Q4_K_M GGUF (into ComfyUI/models/diffusion_models/):
huggingface-cli download city96/Anima-GGUF anima-turbo-v1.1-Q4_K_M.gguf --local-dir ComfyUI/models/diffusion_models

# Qwen 3.06B Text Encoder (into ComfyUI/models/text_encoders/split_files/text_encoders/):
huggingface-cli download circlestone-labs/Anima split_files/text_encoders/qwen_3_06b_base.safetensors --local-dir ComfyUI/models/text_encoders

# Qwen Image VAE (into ComfyUI/models/vae/split_files/vae/):
huggingface-cli download circlestone-labs/Anima split_files/vae/qwen_image_vae.safetensors --local-dir ComfyUI/models/vae
```

---

## Nodes

### 1. `AnimaIPAdapterLoader`
- **Inputs**:
  - `ip_adapter_name`: Select `.safetensors` file from `models/ipadapter/`.
  - `auto_download`: Automatically fetch SigLIP2 encoder if missing.
- **Outputs**: `ANIMA_IP_ADAPTER`

### 2. `AnimaIPAdapterApply`
- **Inputs**:
  - `model`: Anima model (safetensors or GGUF).
  - `ip_adapter`: From loader.
  - `ref_image`: Reference character `IMAGE`.
  - `strength`: Influence weight (default: `0.72` for expressive sprites, `1.0` for rigid likeness).
  - `ref_image_size`: Resolution to resize reference image (default: 512).
  - `ip_cfg_scale`: Independent CFG scale.
  - `use_lora`: Enable bundled cross-attention LoRA.
- **Outputs**: `MODEL` (connect to KSampler).

### 3. `AnimaIPAdapterVisualize`
- Renders an attention heatmap overlay on the reference image to inspect where the adapter is attending.

---

## Workflow

A complete, pre-configured workflow is included:
- [`workflows/Anima_Turbo_IPAdapter.json`](workflows/Anima_Turbo_IPAdapter.json)

Drag and drop this JSON file directly into the ComfyUI web interface to start generating!

---

## Generating Expression Sprites

The included [`generate_expression_sprites.py`](generate_expression_sprites.py) script automates generating full emotion sprite sheets for all 28 expressions in [`expressions.txt`](expressions.txt).

```bash
# Generate all 28 expressions:
python generate_expression_sprites.py --ref-image character_reference.png

# Custom options:
python generate_expression_sprites.py \
  --ref-image my_character.png \
  --output-dir ./sprites_output \
  --strength 0.72 \
  --seed 42096
```

### Tip for Vivid Expressions & Body Language
- **Strength 0.70 – 0.75**: Using `strength=0.72` prevents the reference image from locking down the pose and facial expression, allowing full body language (raised arms, fists, shock postures) and dramatic facial expressions while preserving 100% of character identity.

---

## License

Apache-2.0
