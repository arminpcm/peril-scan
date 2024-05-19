from abc import abstractmethod
from typing import Dict, Optional
from omegaconf import OmegaConf

import torch


class Model:
    def __init__(self, config: OmegaConf) -> None:
        """Intializes and loads the model"""
        self.config: OmegaConf = config
        self.model: Optional[torch.nn.Module] = None

        self.load_model()

        assert self.model is not None

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError()

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
