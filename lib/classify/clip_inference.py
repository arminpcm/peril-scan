from typing import Any, Dict
from omegaconf import OmegaConf
import torch
from lib.classify.clip_model import ClipModel
from lib.inference.inference import Inference
from lib.inference.model import Model


class ClipInference(Inference):
    """The Clip Inference Class"""

    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config=config)

    def setup(self) -> Model:
        return ClipModel(config=self.config.model_config)

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return input_data

    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        probs = self.model.classify(x=prediction)
        class_idx = self.model.get_most_similar_text_index(probs)
        return {"class": class_idx, "probabilities": probs}

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        preprocessed_data: Dict[str, Any] = self.preprocess(input_data=input_data)
        predictions: Dict[str, Any] = self.model.predict(x=preprocessed_data)
        postprocessed_data: Dict[str, Any] = self.postprocess(prediction=predictions)
        return postprocessed_data
