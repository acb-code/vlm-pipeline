class MockQwenVLModel:
    """
    Lightweight mock replacement for Qwen3-VL models.
    Useful for local development on CPU/WSL where loading real
    multimodal models would crash the system.

    Behaves like Qwen3VLLoader:
        - has generate_caption(image)
        - returns a deterministic fake caption
    """

    def __init__(self, cfg=None):
        print("[MockQwenVLModel] Using mock VLM (no GPU / no real weights).")

    def generate_caption(self, image):
        # You can make this more elaborate if needed
        return "A mock caption for testing the pipeline."
