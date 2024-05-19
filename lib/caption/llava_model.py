from typing import Any, Dict
from omegaconf import OmegaConf
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from lib.inference.model import Model


class LlavaModel(Model):
    """The LLaVA model class"""

    def __init__(self, config: OmegaConf) -> None:
        """Initialize the LLaVA model"""
        super().__init__(config=config)

    def load_model(self) -> None:
        self.processor = LlavaNextProcessor.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to("cuda:0")

    def predict(self, x: Dict[str, Any]) -> Dict[str, str]:
        image = x["Image"]
        prompt = x["Prompt"]
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=1000)
        output_text: str = self.processor.decode(output[0], skip_special_tokens=True)
        return {"caption": output_text}
