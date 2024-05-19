from ast import Dict
from typing import Optional
from omegaconf import OmegaConf
import torch
from lib.inference.model import Model
import open_clip


class ClipModel(Model):
    def __init__(self, config: OmegaConf) -> None:
        """Initialize the clip model"""
        super().__init__(config=config)
        self.model: Optional[torch.nn.Module] = None

    def load_model(self) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.config.model_name, pretrained=self.config.model_weights
        )
        self.tokenizer = open_clip.get_tokenizer(model_name=self.config.model_name)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = self.preprocess(x["Image"]).unsqueeze(0)
        text: torch.Tensor = self.tokenizer(x["Labels"])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return {"image_features": image_features, "text+features": text_features}

    def classify(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_features: torch.Tensor = x["image_features"]
        text_features: torch.Tensor = x["text_features"]
        text_probs: torch.Tensor = (100.0 * image_features @ text_features.T).softmax(
            dim=-1
        )
        return text_probs
