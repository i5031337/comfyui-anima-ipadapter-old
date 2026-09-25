"""Generate character expression sprites using Anima Turbo + IP-Adapter.

Reads expressions from expressions.txt and generates expressive character
sprites with dynamic body language and rich facial cues via ComfyUI.
"""

import argparse
import json
import os
import shutil
import time
import urllib.request
import urllib.error

# Rich facial and body language cues tailored to each emotion in expressions.txt
EXPRESSION_DETAILS = {
    "admiration": "starry eyes full of wonder, radiant soft smile, hands clasped together near chin, leaning forward in awe, head tilted slightly, sparkle in pupils",
    "amusement": "playful smirking grin, one eyebrow raised playfully, hand covering half smile, shoulders shaking with suppressed laughter, teasing glint in eyes",
    "anger": "furious scowl, yelling angrily, bared clenched teeth, intense glaring sharp eyes, furrowed sharp eyebrows, clenched shaking fist raised, tense forward-leaning posture",
    "annoyance": "scowling frown, side-glance eye roll, furrowed brow, arms crossed stubbornly over chest, head turned away irritably, irritated pouting mouth",
    "approval": "confident warm smile, nod of satisfaction, proud upright posture, one hand giving enthusiastic thumbs up, encouraging bright eyes",
    "caring": "gentle tender smile, soft comforting gaze, head tilted warmly, one hand resting gently over heart, leaning forward kindly, soft affectionate eyes",
    "confusion": "puzzled furrowed brow, crooked mouth, one eye slightly squinted, hand scratching back of head, head tilted to the side questioningly, baffled shrug",
    "curiosity": "wide curious sparkling eyes, parted intrigued lips, leaning in close with eager curiosity, one finger placed on chin inquisitively, forward eager posture",
    "desire": "captivated longing gaze, parted lips taking a breath, flushed cheeks, hand reaching outward yearnfully, leaning forward with longing, glistening eyes",
    "disappointment": "downcast sorrowful gaze, drooping lips, deflated slumped shoulders, head hanging low, hand dropping limp at side, heavy dejected sigh",
    "disapproval": "stern unimpressed glare, pursed tight lips, heavy furrowed brow, arms tightly crossed across chest, head shaking disapprovingly, rigid stiff posture",
    "disgust": "repulsed sneer, wrinkled nose, tongue sticking out slightly in distaste, leaning far backward away, one hand raised to ward off in revulsion, squinted disgusted eyes",
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


def parse_expressions(file_path):
    """Parse expressions from expressions.txt, stripping quotes and commas."""
    expressions = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expressions file not found: {file_path}")
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
    """Wait for prompt execution to finish and return output image filename."""
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
                    for node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        if images:
                            return images[0]["filename"]
        except urllib.error.URLError:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def build_workflow(ref_image_name, expression, seed, steps, cfg, strength, prefix):
    """Build the API prompt dictionary with rich facial and body cues."""
    details = EXPRESSION_DETAILS.get(
        expression.lower(),
        f"expressive {expression} expression, dynamic upper body posture, expressive body language"
    )
    prompt_text = (
        f"masterpiece, best quality, 1girl, solo, anime sprite portrait, "
        f"{expression}, {details}, dynamic body language, expressive posture, clean white background"
    )
    return {
        "21": {
            "class_type": "UnetLoaderGGUFAdvanced",
            "inputs": {
                "unet_name": "anima-turbo-v1.1-Q4_K_M.gguf",
                "dequant_dtype": "float16",
                "patch_dtype": "float16",
                "patch_on_device": False
            }
        },
        "22": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_name}
        },
        "23": {
            "class_type": "AnimaIPAdapterLoader",
            "inputs": {
                "ip_adapter_name": "ip_adapter-Character_Reference-10.safetensors",
                "auto_download": False
            }
        },
        "24": {
            "class_type": "AnimaIPAdapterApply",
            "inputs": {
                "model": ["21", 0],
                "ip_adapter": ["23", 0],
                "ref_image": ["22", 0],
                "strength": strength,
                "ref_image_size": 512,
                "siglip_layer": -1,
                "ip_cfg_scale": 1.0,
                "ip_cfg_separate": False,
                "gray_null": False,
                "use_lora": True
            }
        },
        "19": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "split_files\\text_encoders\\qwen_3_06b_base.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "split_files\\vae\\qwen_image_vae.safetensors"}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["19", 0], "text": prompt_text}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["19", 0], "text": "low quality, blurry, worst quality, neutral face"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1}
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
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": prefix}
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate expressive character sprites with Anima IP-Adapter")
    parser.add_argument("--expressions", default="./expressions.txt", help="Path to expressions file")
    parser.add_argument("--ref-image", default="character_reference.png", help="Reference image in ComfyUI/input")
    parser.add_argument("--output-dir", default="./ComfyUI/output/expression_sprites", help="Directory to save sprites")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--seed", type=int, default=42096, help="Seed for character consistency")
    parser.add_argument("--steps", type=int, default=8, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--strength", type=float, default=0.72, help="IP-Adapter strength (0.70-0.75 recommended for vivid expressions)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of expressions to generate")
    args = parser.parse_args()

    expressions = parse_expressions(args.expressions)
    if args.limit:
        expressions = expressions[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    comfy_output_dir = os.path.abspath("./ComfyUI/output")

    print(f"Loaded {len(expressions)} expressions.")
    print(f"Reference character image: {args.ref_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling: seed={args.seed}, steps={args.steps}, cfg={args.cfg}, strength={args.strength}\n")

    for i, expr in enumerate(expressions, 1):
        prefix = f"expr_gen_{expr}"
        print(f"[{i}/{len(expressions)}] Generating expression: '{expr}' ...")
        workflow = build_workflow(args.ref_image, expr, args.seed, args.steps, args.cfg, args.strength, prefix)
        resp = queue_prompt(args.server, workflow)
        prompt_id = resp["prompt_id"]

        filename = wait_for_prompt(args.server, prompt_id)
        src_path = os.path.join(comfy_output_dir, filename)
        dest_filename = f"sprite_{i:02d}_{expr}.png"
        dest_path = os.path.join(args.output_dir, dest_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"  -> Saved {dest_filename}")
        else:
            print(f"  -> Generated: {filename}")

    print(f"\nAll {len(expressions)} expression sprites successfully generated in {args.output_dir}!")


if __name__ == "__main__":
    main()
