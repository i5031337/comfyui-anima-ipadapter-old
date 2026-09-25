"""Download required models for Anima IP-Adapter.

Downloads the IP-Adapter weights, SigLIP2 vision encoder, and optionally
the base Anima Turbo GGUF model, text encoder, and VAE.
"""

import argparse
import os
import sys

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


def get_default_models_dir():
    # If running from inside custom_nodes/comfyui-anima-ipadapter
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, "..", "..", "models"))
    if os.path.isdir(candidate):
        return candidate
    return os.path.abspath("./models")


def download_ipadapter(models_dir):
    dest = os.path.join(models_dir, "ipadapter")
    os.makedirs(dest, exist_ok=True)
    print(f"\n[1/2] Downloading IP-Adapter weights to {dest} ...")
    path = hf_hub_download(
        repo_id="LuciferTC/Anima-IP-Adapter",
        filename="ip_adapter-Character_Reference-10.safetensors",
        local_dir=dest,
    )
    print(f"  -> IP-Adapter saved: {path}")


def download_siglip2(models_dir):
    dest = os.path.join(models_dir, "siglip2", "siglip2-base-patch16-512")
    os.makedirs(dest, exist_ok=True)
    print(f"\n[2/2] Downloading SigLIP2 vision encoder to {dest} ...")
    path = snapshot_download(
        repo_id="google/siglip2-base-patch16-512",
        local_dir=dest,
    )
    print(f"  -> SigLIP2 saved: {path}")


def download_anima_base(models_dir):
    # Diffusion model (GGUF)
    diff_dir = os.path.join(models_dir, "diffusion_models")
    os.makedirs(diff_dir, exist_ok=True)
    print(f"\nDownloading Anima Turbo Q4_K_M GGUF to {diff_dir} ...")
    hf_hub_download(
        repo_id="city96/Anima-GGUF",
        filename="anima-turbo-v1.1-Q4_K_M.gguf",
        local_dir=diff_dir,
    )

    # Text encoder (Qwen)
    te_dir = os.path.join(models_dir, "text_encoders", "split_files", "text_encoders")
    os.makedirs(te_dir, exist_ok=True)
    print(f"\nDownloading Qwen Text Encoder to {te_dir} ...")
    hf_hub_download(
        repo_id="circlestone-labs/Anima",
        filename="split_files/text_encoders/qwen_3_06b_base.safetensors",
        local_dir=os.path.join(models_dir, "text_encoders"),
    )

    # VAE
    vae_dir = os.path.join(models_dir, "vae", "split_files", "vae")
    os.makedirs(vae_dir, exist_ok=True)
    print(f"\nDownloading Qwen Image VAE to {vae_dir} ...")
    hf_hub_download(
        repo_id="circlestone-labs/Anima",
        filename="split_files/vae/qwen_image_vae.safetensors",
        local_dir=os.path.join(models_dir, "vae"),
    )


def main():
    parser = argparse.ArgumentParser(description="Download Anima IP-Adapter and base models")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Path to ComfyUI/models directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--include-base-models",
        action="store_true",
        help="Also download Anima Turbo GGUF, Qwen Text Encoder, and VAE",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or get_default_models_dir()
    print(f"Target ComfyUI models directory: {models_dir}")

    download_ipadapter(models_dir)
    download_siglip2(models_dir)

    if args.include_base_models:
        download_anima_base(models_dir)

    print("\nAll requested models downloaded successfully!")


if __name__ == "__main__":
    main()
