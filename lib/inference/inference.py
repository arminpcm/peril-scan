from abc import abstractmethod
from typing import Any

from omegaconf import OmegaConf

from lib.inference.model import Model


class Inference:
    def __init__(self, config: OmegaConf) -> None:
        """Initialize the inference class"""
        self.config: OmegaConf = config
        self.model: Model = self.setup()

    @abstractmethod
    def setup(self) -> Model:
        """Setup the model, load weights, configure environment, etc."""
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Preprocess the input data before feeding it to the model."""
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any:
        """Postprocess the model output to get the final results."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Predict the output using the model."""
        raise NotImplementedError()
