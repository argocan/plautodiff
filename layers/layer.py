from typing import Tuple
from abc import ABC, abstractmethod
from pldiffer import Tensor


class Layer(ABC):

    @abstractmethod
    def compute(self, tensor_in: Tensor, train_mode=False) -> Tensor:
        pass

    @abstractmethod
    def shape(self) -> Tuple[int]:
        pass