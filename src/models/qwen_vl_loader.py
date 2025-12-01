from typing import Dict, Any, Optional

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


class Qwen3VLLoader:
    """
    Load Qwen3-VL-8B-Instruct with flexible device/dtype settings.

    Optionally load a LoRA adapter from `lora_adapter_dir`.

    Provides:
      - model
      - processor (for vision + text)
      - generate_caption() for image → text
    """

    def __init__(self, cfg: Dict[str, Any], lora_adapter_dir: Optional[str] = None):
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

        # Load base model
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device,           # "auto" or explicit device
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Optionally load LoRA adapter
        if lora_adapter_dir is not None:
            print(f"[Qwen3VLLoader] Loading LoRA adapter from: {lora_adapter_dir}")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
        else:
            self.model = base_model

        self.model.eval()  # inference mode
        self.generation_cfg = cfg.get("generation", {})

    def generate_caption(self, image, prompt_override: Optional[str] = None) -> str:
        """
        Generate a caption for a PIL image.

        Args:
          image: PIL.Image or array-like image data accepted by the processor
          prompt_override: optional override of prompt template

        Returns:
          String caption output.
        """

        # Ensure we have a PIL.Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        user_text = prompt_override or self.generation_cfg.get(
            "prompt_template",
            "Describe this image in one concise, informative sentence.",
        )

        # Qwen3-VL expects a chat-style message with an explicit image placeholder.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        # Turn the chat into a text string with the proper <image> tokens.
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Encode text + image together.
        proc_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # Move tensors to the model device.
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in proc_inputs.items()
        }

        max_new_tokens = self.generation_cfg.get("max_new_tokens", 64)
        temperature = self.generation_cfg.get("temperature", 0.7)
        top_p = self.generation_cfg.get("top_p", 0.9)
        num_beams = self.generation_cfg.get("num_beams", 1)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )

        # Skip the prompt tokens when decoding.
        input_length = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[:, input_length:]

        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return decoded[0].strip()
