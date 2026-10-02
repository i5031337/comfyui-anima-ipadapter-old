"""Generate character expression sprites using Anima Turbo + IP-Adapter.

Reads expressions from expressions.txt and generates expressive character
sprites with dynamic body language, rich facial cues, and transparent backgrounds via ComfyUI.

Usage:
    python generate_expression_sprites.py [OPTIONS]

Examples:
    # 1990s retro anime cel-shaded sprites (default preset):
    python generate_expression_sprites.py \\
        --ref-image character_reference.png \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --style 90s

    # Modern high-res digital art sprites:
    python generate_expression_sprites.py \\
        --ref-image character_reference.png \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --style modern

    # Quick test generating only the first 2 expressions:
    python generate_expression_sprites.py \\
        --ref-image character_reference.png \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --limit 2

    # Generate a single expression by name:
    python generate_expression_sprites.py \\
        --ref-image character_reference.png \\
        --character "Rei Ayanami from Neon Genesis Evangelion, white plugsuit" \\
        --expression amusement

Arguments:
    --workflow PATH
        ComfyUI API-format workflow JSON. Model choices and graph connections
        come from this file. Default: ./workflows/Expression_Sprites_API.json. Editor-format exports are not supported.

    --list-styles
        List style presets, aliases, and recommended strengths, then exit.

    --list-expressions
        List all expressions from --expressions, then exit. Can be combined
        with --list-styles; ignores --expression and --limit.

    --skip-existing
        Skip generation when the destination PNG already exists.

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
        Default: "./expressions.txt".

    --ref-image NAME
        Local reference image path (relative to the working directory or absolute),
        or a filename already in ComfyUI's input/ directory. Existing local files
        are uploaded to the server once per run.
        Required for generation; help and list commands do not need it.

    --output-dir PATH
        Directory where generated transparent PNG sprites will be saved.
        Default: "../../output/expression_sprites".

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
import mimetypes
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_WORKFLOW = "./workflows/Expression_Sprites_API.json"

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


def upload_reference_image(server_url, file_path):
    """Upload a local file and return its name in ComfyUI's input directory."""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # Keep the multipart header independent of user-supplied filename characters.
    upload_name = "reference" + (mimetypes.guess_extension(content_type) or "")
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{upload_name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    with open(file_path, "rb") as f:
        payload = header + f.read() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"http://{server_url}/upload/image", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request) as response:
        uploaded = json.loads(response.read().decode("utf-8"))
    return "/".join(part for part in (uploaded.get("subfolder", ""), uploaded["name"]) if part)


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
    workflow_path=DEFAULT_WORKFLOW,
):
    """Load an API workflow and update sprite prompts and generation inputs."""
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

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    if not isinstance(workflow, dict) or "nodes" in workflow or not workflow:
        raise ValueError("Use a ComfyUI API-format workflow export, not an editor workflow JSON.")
    if any(not isinstance(node, dict) or "class_type" not in node or "inputs" not in node
           for node in workflow.values()):
        raise ValueError("Workflow must contain API nodes with class_type and inputs.")

    def single_node(class_type):
        matches = [node for node in workflow.values() if node["class_type"] == class_type]
        if len(matches) != 1:
            raise ValueError(f"Workflow must contain exactly one {class_type} node.")
        return matches[0]["inputs"]

    sampler = single_node("KSampler")
    adapter = single_node("AnimaIPAdapterApply")
    save = single_node("SaveImage")
    positive = workflow[str(sampler["positive"][0])]
    negative = workflow[str(sampler["negative"][0])]
    if positive is negative:
        raise ValueError("Positive and negative prompts must use separate CLIPTextEncode nodes.")
    latent = workflow[str(sampler["latent_image"][0])]
    reference = workflow[str(adapter["ref_image"][0])]
    for node, expected in ((positive, "CLIPTextEncode"), (negative, "CLIPTextEncode"),
                           (latent, "EmptyLatentImage"), (reference, "LoadImage")):
        if node["class_type"] != expected:
            raise ValueError(f"Workflow requires a directly connected {expected} node.")

    reference["inputs"]["image"] = ref_image_name
    positive["inputs"]["text"] = prompt_text
    negative["inputs"]["text"] = negative_prompt
    latent["inputs"].update(width=width, height=height)
    sampler.update(seed=seed, steps=steps, cfg=cfg)
    adapter["strength"] = strength
    save["filename_prefix"] = prefix
    return workflow


def main():
    preset_names = ", ".join(STYLE_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description="Generate expressive character sprites with Anima IP-Adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--character", "-c", default="", help="Character description / franchise keywords (e.g. 'Rei Ayanami from Neon Genesis Evangelion, white plugsuit')")
    parser.add_argument("--style", "-s", default="90s", help=f"Style preset ({preset_names}) or custom style prompt")
    parser.add_argument("--expression", "-e", default=None, help="Generate a single expression by name (e.g. 'amusement', 'joy')")
    parser.add_argument("--extra-prompt", default="", help="Additional prompt keywords to append")
    parser.add_argument("--negative", "-n", default=None, help="Custom negative prompt (defaults to preset's negative prompt)")
    parser.add_argument("--expressions", default="./expressions.txt", help="Path to expressions file")
    parser.add_argument("--ref-image", help="Local image path or filename in ComfyUI/input (required for generation)")
    parser.add_argument("--output-dir", default="../../output/expression_sprites", help="Directory to save sprites")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--seed", type=int, default=42096, help="Seed for character consistency")
    parser.add_argument("--steps", type=int, default=8, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--strength", type=float, default=None, help="IP-Adapter strength (defaults to preset's recommended strength)")
    parser.add_argument("--width", type=int, default=832, help="Image width")
    parser.add_argument("--height", type=int, default=1216, help="Image height")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of expressions to generate")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="ComfyUI API-format workflow JSON")
    parser.add_argument("--list-styles", action="store_true", help="List style presets and aliases, then exit")
    parser.add_argument("--list-expressions", action="store_true", help="List expressions from --expressions, then exit")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sprites whose destination PNG already exists")
    args = parser.parse_args()

    if args.list_styles:
        for name, preset in STYLE_PRESETS.items():
            aliases = [alias for alias, target in STYLE_ALIASES.items() if target == name]
            alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
            print(f"{name}{alias_text}: {preset['description']}; strength={preset['default_strength']}")
    if args.list_expressions:
        for expression in parse_expressions(args.expressions):
            print(expression)
    if args.list_styles or args.list_expressions:
        return
    if not args.ref_image or not args.ref_image.strip():
        parser.error("--ref-image is required for generation")

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
    comfy_output_dir = "../../output"

    print(f"Loaded {len(expressions)} expressions.")
    print(f"Character: {args.character or '(none specified, relying on reference image)'}")
    print(f"Style preset: {style_name} (strength={strength})")
    if style_text:
        print(f"Style prompt: {style_text}")
    print(f"Reference character image: {args.ref_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling: seed={args.seed}, steps={args.steps}, cfg={args.cfg}, size={args.width}x{args.height}\n")

    skipped = 0
    saved_count = 0
    ref_image_name = None
    for i, expr in enumerate(expressions, 1):
        dest_filename = f"sprite_{expr}.png" if args.expression else f"sprite_{i:02d}_{expr}.png"
        dest_path = os.path.join(args.output_dir, dest_filename)
        if args.skip_existing and os.path.isfile(dest_path):
            print(f"[{i}/{len(expressions)}] Skipping existing: {dest_filename}")
            skipped += 1
            continue
        if ref_image_name is None:
            local_image = os.path.expanduser(args.ref_image)
            if os.path.isfile(local_image):
                ref_image_name = upload_reference_image(args.server, local_image)
            else:
                ref_image_name = args.ref_image
        prefix = f"sprites/expr_{expr}"
        print(f"[{i}/{len(expressions)}] Generating expression: '{expr}' ...")
        workflow = build_workflow(
            ref_image_name=ref_image_name,
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
            workflow_path=args.workflow,
        )
        resp = queue_prompt(args.server, workflow)
        prompt_id = resp["prompt_id"]

        filename, subfolder = wait_for_prompt(args.server, prompt_id)
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
                saved = True
            except Exception as e:
                print(f"  -> Generated: {filename} (could not copy to {dest_filename}: {e})")

        saved_count += int(saved)

    failed = len(expressions) - skipped - saved_count
    print(f"\nSaved {saved_count}, skipped {skipped}, failed to save {failed}. Output: {args.output_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
