from typing import Any, Dict
from omegaconf import OmegaConf
from lib.caption.llava_model import LlavaModel
from lib.inference.inference import Inference
from lib.inference.model import Model


class LlavaInference(Inference):
    """The LLaVA Inference Class"""

    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config=config)

    def setup(self) -> Model:
        return LlavaModel(config=self.config.model_config)

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return input_data

    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return prediction

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        preprocessed_data: Dict[str, Any] = self.preprocess(input_data=input_data)
        predictions: Dict[str, Any] = self.model.predict(x=preprocessed_data)
        postprocessed_data: Dict[str, Any] = self.postprocess(prediction=predictions)
        return postprocessed_data
