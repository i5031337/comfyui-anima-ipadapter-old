"""Generate character expression sprites using Anima Turbo + IP-Adapter.

Reads expressions from expressions.txt and generates expressive character
sprites with dynamic body language, rich facial cues, and transparent backgrounds via ComfyUI.

Usage:
    python generate_expression_sprites.py [OPTIONS]

Examples:
    # 1990s retro anime cel-shaded sprites (default preset):
    python generate_expression_sprites.py \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --style 90s

    # Modern high-res digital art sprites:
    python generate_expression_sprites.py \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --style modern

    # Quick test generating only the first 2 expressions:
    python generate_expression_sprites.py \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --limit 2

    # Generate a single expression by name:
    python generate_expression_sprites.py \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --expression amusement

Arguments:
    --character, -c STR
        Character description and franchise keywords (e.g. "Rei Ayanami from
        Neon Genesis Evangelion, white plugsuit"). Prepend to prompt to activate
        the text encoder's franchise and character priors. Default: "".

    --expression, -e STR
        Generate a single expression by name (e.g. "amusement", "joy", "anger").
        Overrides the --expressions file and saves as sprite_<expression>.png.
        Default: None.

    --style, -s STR
        Style preset name or custom prompt text. Available presets:
          * 90s (aliases: retro, vintage, classic) [default]
              1990s cel-shaded animation, retro palette, soft shading.
              Recommended strength: 0.55, negative: "".
          * modern
              Crisp modern digital anime illustration with dynamic lighting.
              Recommended strength: 0.72, negative: standard quality tags.
          * visual-novel (alias: vn)
              Polished dialogue sprite with sharp clean linework.
              Recommended strength: 0.65.
          * cinematic
              Dramatic lighting and atmospheric key visual aesthetic.
              Recommended strength: 0.60.
          * raw (alias: none)
              No added style tokens; relies purely on character prompt and IP-Adapter.
              Recommended strength: 0.70.
        Any string not matching a preset will be used as custom style prompt text.

    --extra-prompt STR
        Additional prompt keywords to append after character, style, and expression cues.
        Default: "".

    --negative, -n STR
        Custom negative prompt text. If omitted, uses the selected style preset's
        built-in negative prompt.

    --expressions PATH
        Path to file containing expressions (one per line).
        Default: "expressions.txt" in the script's directory.

    --ref-image NAME
        Reference character image filename located in ComfyUI's input/ directory.
        Default: "ComfyUI_00044_.png".

    --output-dir PATH
        Directory where generated transparent PNG sprites will be saved.
        Default: "ComfyUI/output/expression_sprites".

    --server HOST:PORT
        ComfyUI server address. Default: "127.0.0.1:8188".

    --seed INT
        Base random seed for generation to maintain character consistency.
        Default: 42096.

    --steps INT
        Sampling step count for Anima Turbo. Default: 8.

    --cfg FLOAT
        CFG scale for diffusion sampling. Default: 1.0.

    --strength FLOAT
        IP-Adapter conditioning weight. If omitted, automatically uses the
        recommended strength defined by the chosen style preset (e.g. 0.55 for 90s,
        0.72 for modern).

    --width INT
        Canvas width in pixels. Default: 832.

    --height INT
        Canvas height in pixels. Default: 1216.

    --limit INT
        Limit number of expressions to generate from the list (useful for testing).
        Default: None (generates all).
"""

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

# Rich facial and body language cues tailored to each emotion in expressions.txt
EXPRESSION_DETAILS = {
    "admiration": "starry eyes full of wonder, radiant soft smile, hands clasped together near chin, leaning forward in awe, head tilted slightly, sparkle in pupils",
    "amusement": "amused, playful, one eyebrow raised playfully, hand covering half smile, shoulders shaking with suppressed laughter, teasing glint in eyes",
    "anger": "furious scowl, yelling angrily, bared clenched teeth, intense glaring sharp eyes, furrowed sharp eyebrows, clenched shaking fist raised, tense forward-leaning posture",
    "annoyance": "scowling frown, side-glance eye roll, furrowed brow, arms crossed stubbornly over chest, head turned away irritably, irritated pouting mouth",
    "approval": "confident warm smile, nod of satisfaction, proud upright posture, one hand giving enthusiastic thumbs up, encouraging bright eyes",
    "caring": "gentle tender smile, soft comforting gaze, head tilted warmly, one hand resting gently over heart, leaning forward kindly, soft affectionate eyes",
    "confusion": "puzzled furrowed brow, crooked mouth, one eye slightly squinted, hand scratching back of head, head tilted to the side questioningly, baffled shrug",
    "curiosity": "wide curious sparkling eyes, parted intrigued lips, leaning in close with eager curiosity, one finger placed on chin inquisitively, forward eager posture",
    "desire": "captivated longing gaze, parted lips taking a breath, flushed cheeks, hand reaching outward yearnfully, leaning forward with longing, glistening eyes",
    "disappointment": "downcast sorrowful gaze, drooping lips, deflated slumped shoulders, head hanging low, hand dropping limp at side, heavy dejected sigh",
    "disapproval": "stern unimpressed glare, pursed tight lips, heavy furrowed brow, arms tightly crossed across chest, head shaking disapprovingly, rigid stiff posture",
    "disgust": "uncomfortable, repulsed sneer, wrinkled nose, squinted disgusted eyes",
    "embarrassment": "deep red blushing across face and ears, flustered averted eyes, hands flustered covering hot cheeks, nervous sheepish smile, cowering bashful posture",
    "excitement": "ecstatic beaming smile, sparkling starry eyes, pumping both fists high in victory, energetic bouncing pose, leaning forward eagerly",
    "fear": "terrified wide dilated eyes, trembling open mouth, hands held up defensively in terror, cowering backward defensively, cold sweat bead on forehead, panicked tense posture",
    "gratitude": "tearful touched smile, eyes shining with heartfelt thanks, both hands pressed together against chest, deep respectful bow of head, warm radiant glow",
    "grief": "heartbroken sorrow, tears pouring down cheeks, sobbing mouth open in anguish, hands clutching chest tightly, head hung low in despair, trembling slumped body",
    "joy": "ecstatic joy, laughing cheerfully, huge beaming open-mouth smile, happy closed squinting eyes, celebratory victorious fist pump, head tilted back playfully",
    "love": "adoring sweet gaze, blushing cheeks, dreamy heart-struck expression, hands gently cradling face, dreamy affectionate smile, gentle warm lean",
    "nervousness": "anxious trembling smile, darting worried eyes, sweat drop on temple, fidgeting hands playing with collar, stiff awkward posture, tense neck",
    "optimism": "bright hopeful smile, determined confident eyes looking upward, hand on hip proudly, chest puffed out confidently, hopeful forward-looking stance",
    "pride": "smug triumphant grin, chin tilted up proudly, eyes half-closed smugly, arms crossed confidently across chest, haughty confident stance",
    "realization": "sudden epiphany, mouth rounded in a light 'o', eyes widening in sudden clarity, one hand snapping fingers with index finger pointing up, head perked up",
    "relief": "relaxed peaceful smile, gently closed relaxed eyes, deep exhale of breath, one hand wiping brow in relief, shoulders slumping down in eased tension",
    "remorse": "regretful guilty look, eyes looking down apologetically, sorrowful tight lips, head bowed low, one hand gripping opposite forearm bashfully",
    "sadness": "melancholy downcast eyes, trembling downturned mouth, glassy tearful eyes, head drooping slightly, gentle quiet slumped posture",
    "surprise": "completely shocked gasp, wide circular open mouth, huge dilated pupils, raised arched eyebrows, both hands covering open mouth in disbelief, leaning back in sudden shock",
    "neutral": "calm neutral expression, relaxed steady gaze, gentle resting mouth, natural upright posture, balanced poised stance",
}

# Style presets tailoring prompt framing, negative prompts, and recommended IP-Adapter strengths
STYLE_PRESETS = {
    "90s": {
        "description": "1990s hand-painted cel animation with retro palette and classic shading",
        "prompt": "1990s anime style, retro cel shading, vintage anime aesthetic, classic cel animation",
        "negative": "",
        "default_strength": 0.55,
    },
    "modern": {
        "description": "Modern high-res digital anime with crisp highlights and vibrant lighting",
        "prompt": "masterpiece, best quality, modern anime digital art, vibrant detailed coloring",
        "negative": "low quality, blurry, worst quality, artifacts",
        "default_strength": 0.72,
    },
    "visual-novel": {
        "description": "Clean character sprite suitable for visual novels and dialogue portraits",
        "prompt": "visual novel character sprite, clean sharp linework, polished anime illustration",
        "negative": "low quality, blurry, worst quality, artifacts",
        "default_strength": 0.65,
    },
    "cinematic": {
        "description": "Dramatic cinematic anime key visual with rich atmospheric lighting",
        "prompt": "cinematic anime lighting, atmospheric shading, rich depth, dynamic key visual",
        "negative": "low quality, blurry, worst quality, artifacts",
        "default_strength": 0.60,
    },
    "raw": {
        "description": "No additional style tags; relies purely on character prompt and IP-Adapter",
        "prompt": "",
        "negative": "",
        "default_strength": 0.70,
    },
}

STYLE_ALIASES = {
    "retro": "90s",
    "vintage": "90s",
    "classic": "90s",
    "vn": "visual-novel",
    "none": "raw",
}


def parse_expressions(file_path):
    """Parse expressions from expressions.txt, stripping quotes and commas."""
    if not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(file_path))
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Expressions file not found: {file_path}")
    expressions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().strip(",").strip("'").strip('"').strip()
            if clean:
                expressions.append(clean)
    return expressions


def queue_prompt(server_url, prompt_workflow):
    """Submit a workflow prompt to the ComfyUI server."""
    url = f"http://{server_url}/prompt"
    payload = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_prompt(server_url, prompt_id, timeout=180):
    """Wait for prompt execution to finish and return (filename, subfolder)."""
    url = f"http://{server_url}/history/{prompt_id}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if prompt_id in data:
                    item = data[prompt_id]
                    status = item.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI Error: {status.get('messages')}")
                    outputs = item.get("outputs", {})
                    if "9" in outputs and outputs["9"].get("images"):
                        img = outputs["9"]["images"][0]
                        return img["filename"], img.get("subfolder", "")
                    for node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        for img in images:
                            if img.get("type") == "output":
                                return img["filename"], img.get("subfolder", "")
        except urllib.error.URLError:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def build_workflow(
    ref_image_name,
    expression,
    seed,
    steps,
    cfg,
    strength,
    prefix,
    character="",
    style_text="",
    negative_prompt="",
    extra_prompt="",
    width=832,
    height=1216,
):
    """Build the API prompt dictionary with rich facial and body cues."""
    details = EXPRESSION_DETAILS.get(
        expression.lower(),
        f"expressive {expression} expression, dynamic upper body posture, expressive body language",
    )

    parts = []
    if character:
        parts.append(character.strip().rstrip(","))
    if style_text:
        parts.append(style_text.strip().rstrip(","))
    parts.append(expression)
    parts.append(details)
    parts.append("dynamic body language, expressive posture")
    if extra_prompt:
        parts.append(extra_prompt.strip().rstrip(","))

    prompt_text = ", ".join(parts) + ".\n\nsimple solid dark grey background, flat background, no shadows"

    return {
        "26": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "anima-turbo-v1.1.safetensors",
                "weight_dtype": "default"
            }
        },
        "23": {
            "class_type": "AnimaIPAdapterLoader",
            "inputs": {
                "ip_adapter_name": "ip_adapter-Character_Reference-10.safetensors",
                "auto_download": False
            }
        },
        "22": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_name}
        },
        "24": {
            "class_type": "AnimaIPAdapterApply",
            "inputs": {
                "model": ["26", 0],
                "ip_adapter": ["23", 0],
                "ref_image": ["22", 0],
                "strength": strength,
                "ref_image_size": 512,
                "siglip_layer": -1,
                "ip_cfg_scale": 1.0,
                "ip_cfg_separate": False,
                "gray_null": False,
                "use_lora": False
            }
        },
        "19": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_06b_base.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["19", 0], "text": prompt_text}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["19", 0], "text": negative_prompt}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["24", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["20", 0]}
        },
        "34:14": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": "birefnet.safetensors"}
        },
        "34:13": {
            "class_type": "RemoveBackground",
            "inputs": {
                "bg_removal_model": ["34:14", 0],
                "image": ["8", 0]
            }
        },
        "35": {
            "class_type": "InvertMask",
            "inputs": {"mask": ["34:13", 0]}
        },
        "28": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {
                "image": ["8", 0],
                "alpha": ["35", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["28", 0], "filename_prefix": prefix}
        }
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.dirname(os.path.dirname(script_dir))

    default_expressions = os.path.join(script_dir, "expressions.txt")
    if not os.path.exists(default_expressions):
        default_expressions = "./expressions.txt"

    default_output_dir = os.path.join(comfy_root, "output", "expression_sprites")

    preset_names = ", ".join(STYLE_PRESETS.keys())
    parser = argparse.ArgumentParser(description="Generate expressive character sprites with Anima IP-Adapter")
    parser.add_argument("--character", "-c", default="", help="Character description / franchise keywords (e.g. 'Rei Ayanami from Neon Genesis Evangelion, white plugsuit')")
    parser.add_argument("--style", "-s", default="90s", help=f"Style preset ({preset_names}) or custom style prompt (default: 90s)")
    parser.add_argument("--expression", "-e", default=None, help="Generate a single expression by name (e.g. 'amusement', 'joy')")
    parser.add_argument("--extra-prompt", default="", help="Additional prompt keywords to append")
    parser.add_argument("--negative", "-n", default=None, help="Custom negative prompt (defaults to preset's negative prompt)")
    parser.add_argument("--expressions", default=default_expressions, help="Path to expressions file")
    parser.add_argument("--ref-image", default="ComfyUI_00044_.png", help="Reference image in ComfyUI/input")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save sprites")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--seed", type=int, default=42096, help="Seed for character consistency")
    parser.add_argument("--steps", type=int, default=8, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--strength", type=float, default=None, help="IP-Adapter strength (defaults to preset's recommended strength)")
    parser.add_argument("--width", type=int, default=832, help="Image width")
    parser.add_argument("--height", type=int, default=1216, help="Image height")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of expressions to generate")
    args = parser.parse_args()

    style_key = STYLE_ALIASES.get(args.style.lower(), args.style.lower())
    if style_key in STYLE_PRESETS:
        preset = STYLE_PRESETS[style_key]
        style_text = preset["prompt"]
        neg_prompt = preset["negative"] if args.negative is None else args.negative
        strength = args.strength if args.strength is not None else preset["default_strength"]
        style_name = style_key
    else:
        style_text = args.style
        neg_prompt = "low quality, blurry, worst quality, artifacts" if args.negative is None else args.negative
        strength = args.strength if args.strength is not None else 0.65
        style_name = "custom"

    if args.expression:
        expressions = [args.expression.strip()]
    else:
        expressions = parse_expressions(args.expressions)
        if args.limit:
            expressions = expressions[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    comfy_output_dir = os.path.join(comfy_root, "output")

    print(f"Loaded {len(expressions)} expressions.")
    print(f"Character: {args.character or '(none specified, relying on reference image)'}")
    print(f"Style preset: {style_name} (strength={strength})")
    if style_text:
        print(f"Style prompt: {style_text}")
    print(f"Reference character image: {args.ref_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling: seed={args.seed}, steps={args.steps}, cfg={args.cfg}, size={args.width}x{args.height}\n")

    for i, expr in enumerate(expressions, 1):
        prefix = f"sprites/expr_{expr}"
        print(f"[{i}/{len(expressions)}] Generating expression: '{expr}' ...")
        workflow = build_workflow(
            ref_image_name=args.ref_image,
            expression=expr,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            strength=strength,
            prefix=prefix,
            character=args.character,
            style_text=style_text,
            negative_prompt=neg_prompt,
            extra_prompt=args.extra_prompt,
            width=args.width,
            height=args.height,
        )
        resp = queue_prompt(args.server, workflow)
        prompt_id = resp["prompt_id"]

        filename, subfolder = wait_for_prompt(args.server, prompt_id)
        dest_filename = f"sprite_{expr}.png" if args.expression else f"sprite_{i:02d}_{expr}.png"
        dest_path = os.path.join(args.output_dir, dest_filename)

        candidates = [
            os.path.join(comfy_output_dir, subfolder, filename),
            os.path.join(comfy_output_dir, filename),
            os.path.join(".", "output", subfolder, filename),
            os.path.join(".", "output", filename),
        ]
        saved = False
        for src_path in candidates:
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)
                print(f"  -> Saved {dest_filename}")
                saved = True
                break

        if not saved:
            try:
                query = f"filename={urllib.parse.quote(filename)}&type=output"
                if subfolder:
                    query += f"&subfolder={urllib.parse.quote(subfolder)}"
                view_url = f"http://{args.server}/view?{query}"
                with urllib.request.urlopen(view_url) as r:
                    with open(dest_path, "wb") as f:
                        f.write(r.read())
                print(f"  -> Saved {dest_filename}")
            except Exception as e:
                print(f"  -> Generated: {filename} (could not copy to {dest_filename}: {e})")

    print(f"\nAll {len(expressions)} expression sprites successfully generated in {args.output_dir}!")


if __name__ == "__main__":
    main()
