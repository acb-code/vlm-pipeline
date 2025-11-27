from typing import Dict, Any, Optional
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class Qwen3VLLoader:
    """
    Load Qwen3-VL-8B-Instruct with flexible device/dtype settings.

    Provides:
      - model
      - processor (for vision + text)
      - generate_caption() for image → text
    """

    def __init__(self, cfg: Dict[str, Any]):
        model_name = cfg["model"]["name"]
        revision = cfg["model"].get("revision", "main")
        cache_dir = cfg["model"].get("cache_dir", None)

        # Decide device
        device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Decide dtype for model
        dtype_str = cfg["model"].get("torch_dtype", "float16")
        if dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_str == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"[Qwen3VLLoader] Loading model {model_name} (dtype={dtype_str}, device={device})")

        # Load processor (vision + text)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device,           # "auto" or explicit device
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.model.eval()  # inference mode
        self.generation_cfg = cfg.get("generation", {})

    def generate_caption(self, image, prompt_override: Optional[str] = None) -> str:
        """
        Generate a caption for a PIL image.

        Args:
          image: PIL.Image or raw image data accepted by processor
          prompt_override: optional override of prompt template

        Returns:
          String caption output.
        """
        prompt = prompt_override or self.generation_cfg.get(
            "prompt_template",
            "<image>\nDescribe this image."
        )

        # Compose the multimodal input
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_cfg.get("max_new_tokens", 64),
            temperature=self.generation_cfg.get("temperature", 0.7),
            top_p=self.generation_cfg.get("top_p", 0.9),
            num_beams=self.generation_cfg.get("num_beams", 1),
        )

        # Decode output
        decoded = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # Since batch size = 1, return first output
        return decoded[0]
