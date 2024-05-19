from typing import Any, Dict
from omegaconf import OmegaConf
from open_clip.tokenizer import HFTokenizer
import torch
from lib.inference.model import Model
import open_clip


class ClipModel(Model):
    def __init__(self, config: OmegaConf) -> None:
        """Initialize the clip model"""
        super().__init__(config=config)

    def load_model(self) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.config.model_name, pretrained=self.config.model_weights
        )
        self.tokenizer: HFTokenizer | open_clip.SimpleTokenizer = (
            open_clip.get_tokenizer(model_name=self.config.model_name)
        )

    def predict(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        image: torch.Tensor = self.preprocess(x["Image"]).unsqueeze(0)
        text: torch.Tensor = self.tokenizer(x["Labels"])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image=image)
            text_features = self.model.encode_text(text=text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return {"image_features": image_features, "text_features": text_features}

    def classify(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_features: torch.Tensor = x["image_features"]
        text_features: torch.Tensor = x["text_features"]
        text_probs: torch.Tensor = (100.0 * image_features @ text_features.T).softmax(
            dim=-1
        )
        return text_probs

    def get_most_similar_text_index(
        self, text_probs: torch.Tensor, threshold: float = 0.8
    ) -> int | float | bool:
        sorted_probs, indices = torch.sort(input=text_probs, descending=True)
        highest_prob: int | float | bool = sorted_probs[0, 0].item()
        second_highest_prob: int | float | bool = sorted_probs[0, 1].item()

        if second_highest_prob > threshold * highest_prob:
            return -1
        return indices[0, 0].item()
