def fetch_visual_encoders(model_name):
    if model_name == "clip":
        from .clip import load_clip
        return load_clip()
    if model_name == "siglip2":
        raise NotImplementedError("SigLIP2 is not included in this evaluation-only release. Use --backbone clip.")
    return None, None
