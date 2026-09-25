import os
import folder_paths

models_dir = folder_paths.models_dir
folder_paths.folder_names_and_paths.setdefault("ipadapter", (
    [os.path.join(models_dir, "ipadapter")],
    {".safetensors"},
))
SIGLIP2_DIR = os.path.join(models_dir, "siglip2", "siglip2-base-patch16-512")
