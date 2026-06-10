def fetch_text_encoders(model_name):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        from .clip import ClipTextEncoder
        return ClipTextEncoder(), 512
    if model_name == 'siglip2':
        raise NotImplementedError('Use --backbone clip for the released MV-Actor checkpoint.')
    return None, None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        from .clip import ClipTokenizer
        return ClipTokenizer()
    if model_name == 'siglip2':
        raise NotImplementedError('Use --backbone clip for the released MV-Actor checkpoint.')
    return None
