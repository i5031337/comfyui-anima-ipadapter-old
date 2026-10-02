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
- [`workflows/Expression_Sprites_API.json`](workflows/Expression_Sprites_API.json)

Drag and drop this JSON file directly into the ComfyUI web interface to start generating!

---

## Generating Expression Sprites

The included [`generate_expression_sprites.py`](generate_expression_sprites.py) script generates individual transparent PNG sprites for all 28 expressions in [`expressions.txt`](expressions.txt), a custom expression list, or a single expression.

### Setup

Start ComfyUI with this extension loaded and run the commands below from this repository's directory. All default filesystem paths are relative to the current working directory. `--ref-image` is required for generation and accepts a local relative or absolute image path, or a filename already in the server's `ComfyUI/input/` directory. If the value names an existing local file, the script uploads it to ComfyUI once per run and uses the returned filename. Otherwise, it passes the value through as a server input filename. Help and list commands do not require a reference image.

The script loads [`workflows/Expression_Sprites_API.json`](workflows/Expression_Sprites_API.json). This template requires these models to be available under the exact loader names:

- Diffusion model: `anima-turbo-v1.1.safetensors` (the default API template uses `UNETLoader`).
- IP-Adapter: `ip_adapter-Character_Reference-10.safetensors`, with the SigLIP2 encoder already installed (`auto_download` is disabled).
- Text encoder: `qwen_3_06b_base.safetensors`.
- VAE: `qwen_image_vae.safetensors`.
- Background removal: `birefnet.safetensors`, with the `LoadBackgroundRemovalModel` and `RemoveBackground` nodes installed.

To use your own models or a GGUF loader, configure and test your workflow in ComfyUI, then export it in **API format** and save it under `workflows/`. Use `--workflow ./workflows/my_sprites_api.json` to select it, or replace the default API template. Enable ComfyUI's developer options if the API export command is hidden. The editor workflow `Anima_Turbo_IPAdapter.json` is a different format and cannot be passed directly to this script.

Your API workflow must contain exactly one `KSampler`, `AnimaIPAdapterApply`, and `SaveImage`. The sampler's positive and negative inputs must connect directly to separate `CLIPTextEncode` nodes, its latent input to `EmptyLatentImage`, and the adapter's reference input to `LoadImage`. Node IDs can differ from the template. The script updates those prompts, the reference filename, seed, steps, CFG, strength, width, height, and save prefix; other settings, models, and connections come from your JSON. Keep batch size at one for one sprite per expression. Include background-removal and alpha nodes if you want transparent PNGs.

### Examples

```bash
# Generate all 28 expressions with the default 90s style:
python generate_expression_sprites.py --ref-image character_reference.png

# Discover available styles and expressions without contacting ComfyUI:
python generate_expression_sprites.py --list-styles --list-expressions

# Resume a batch using your own API workflow:
python generate_expression_sprites.py \
  --ref-image character_reference.png \
  --workflow ./workflows/my_sprites_api.json \
  --skip-existing

# Quick test with a character description and modern style:
python generate_expression_sprites.py \
  --ref-image character_reference.png \
  --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \
  --style modern \
  --limit 2

# Generate a single expression using the visual-novel preset:
python generate_expression_sprites.py \
  --ref-image character_reference.png \
  --expression amusement \
  --style vn

# Custom expression list, style prompt, and sampling settings:
python generate_expression_sprites.py \
  --ref-image my_character.png \
  --expressions ./my_expressions.txt \
  --style "watercolor anime illustration, soft pastel colors" \
  --extra-prompt "waist-up portrait, looking at viewer" \
  --negative "blurry, artifacts" \
  --output-dir ./sprites_output \
  --server 127.0.0.1:8188 \
  --strength 0.65 \
  --seed 42096 \
  --steps 8 \
  --cfg 1.0 \
  --width 832 \
  --height 1216
```

### Command-Line Options

| Option | Default | Behavior |
| --- | --- | --- |
| `--help`, `-h` | — | Show help and exit. |
| `--workflow` PATH | `./workflows/Expression_Sprites_API.json` | Load a ComfyUI API-format workflow; model choices come from this JSON. |
| `--list-styles` | Off | List preset descriptions, aliases, and recommended strengths, then exit. |
| `--list-expressions` | Off | List all expressions from `--expressions`, then exit. Can be combined with `--list-styles`; does not apply `--limit` or `--expression`. |
| `--skip-existing` | Off | Skip generation when the destination PNG exists; file numbering is preserved. Checks filenames only, not whether settings match or the image is valid. |
| `--character`, `-c` TEXT | Empty | Character description and franchise keywords prepended to the positive prompt. |
| `--style`, `-s` TEXT | `90s` | Style preset, alias, or custom style prompt text (see below). |
| `--expression`, `-e` NAME | Unset | Generate one expression; overrides `--expressions` and ignores `--limit`. |
| `--extra-prompt` TEXT | Empty | Additional keywords appended after character, style, and expression cues. |
| `--negative`, `-n` TEXT | Style-dependent | Override the negative prompt; use `--negative ""` for an empty prompt. |
| `--expressions` PATH | `./expressions.txt` | Expression file, one expression per line; surrounding quotes and commas are stripped. |
| `--ref-image` PATH/NAME | Required for generation | Local image path to upload (relative, absolute, or `~/…`), or an existing server input filename. Local files take precedence; always overrides the workflow's reference image. |
| `--output-dir` PATH | `../../output/expression_sprites` | Destination for copied or downloaded PNGs; created automatically. Resolves to `ComfyUI/output/expression_sprites` when run from this repository in `ComfyUI/custom_nodes/`. |
| `--server` HOST:PORT | `127.0.0.1:8188` | ComfyUI server address, without `http://`; the script uses HTTP. |
| `--seed` INT | `42096` | Random seed reused for every expression. |
| `--steps` INT | `8` | Sampling steps. |
| `--cfg` FLOAT | `1.0` | Diffusion CFG scale. |
| `--strength` FLOAT | Style-dependent | Override the preset's IP-Adapter conditioning weight. |
| `--width` INT | `832` | Output canvas width in pixels. |
| `--height` INT | `1216` | Output canvas height in pixels. |
| `--limit` INT | Unset (all) | Generate the first N expressions from the file. Currently, `0` leaves the list unlimited and negative values use Python slicing. |

### Style Presets

Preset names and aliases are case-insensitive. Any other string becomes custom style prompt text.

| Preset | Aliases | Style | Default strength | Default negative prompt |
| --- | --- | --- | --- | --- |
| `90s` | `retro`, `vintage`, `classic` | Retro cel-shaded anime | `0.55` | Empty |
| `modern` | — | Detailed digital anime with vibrant coloring | `0.72` | `low quality, blurry, worst quality, artifacts` |
| `visual-novel` | `vn` | Polished dialogue sprite with clean linework | `0.65` | `low quality, blurry, worst quality, artifacts` |
| `cinematic` | — | Atmospheric anime lighting and shading | `0.60` | `low quality, blurry, worst quality, artifacts` |
| `raw` | `none` | No added style tokens | `0.70` | Empty |
| Custom text | — | Your supplied style prompt | `0.65` | `low quality, blurry, worst quality, artifacts` |

`--strength` and `--negative` override these defaults independently. Start with the preset's strength and adjust to balance character likeness with expression and pose freedom; results vary with the reference image and prompt.

### Expressions and Saved Files

The bundled expressions include tailored facial and body-language cues. Custom expression names are also accepted and receive generic expression and posture cues. The script adds a flat dark-grey background prompt; the default API template removes the background to produce transparency.

List mode saves `sprite_01_admiration.png`, `sprite_02_amusement.png`, and so on in file order. Single-expression mode saves `sprite_amusement.png`. Reusing the same destination filenames overwrites earlier sprites unless `--skip-existing` is set. ComfyUI also saves the original generated files under `output/sprites/`.

For a remote server, required models must be installed on that server. Local reference images are uploaded through `/upload/image`; existing server input filenames can also be used. The script can download results through ComfyUI's `/view` endpoint when it cannot find them locally. Each expression has a fixed 180-second completion timeout.

---

## License

Apache-2.0
