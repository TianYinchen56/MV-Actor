def fetch_visual_encoders(model_name):
    if model_name == "clip":
        from .clip import load_clip
        return load_clip()
    if model_name == "siglip2":
        raise NotImplementedError("Use --backbone clip for the released MV-Actor checkpoint.")
    return None, None
