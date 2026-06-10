import os
from pathlib import Path
import torch
from torch import nn
import transformers


def _resolve_local_clip_path():
    explicit_path = os.environ.get("CLIP_HF_LOCAL_PATH")
    if not explicit_path:
        raise FileNotFoundError(
            "CLIP_HF_LOCAL_PATH is required. Point it to a local openai/clip-vit-base-patch32 "
            "Transformers snapshot containing config.json."
        )
    path = Path(explicit_path).expanduser()
    if path.is_dir() and (path / "config.json").is_file():
        return str(path)
    snapshot_dirs = sorted([candidate for candidate in path.iterdir() if candidate.is_dir()]) if path.is_dir() else []
    for snapshot_dir in snapshot_dirs:
        if (snapshot_dir / "config.json").is_file():
            return str(snapshot_dir)
    raise FileNotFoundError(
        f"CLIP_HF_LOCAL_PATH does not point to a valid local CLIP snapshot: {explicit_path}"
    )


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        model_path = _resolve_local_clip_path()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        model_path = _resolve_local_clip_path()
        self.model = transformers.CLIPTextModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
