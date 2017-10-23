from abc import ABC, abstractmethod
from typing import List
import numpy as np
from functools import reduce
import pldiffer.tensor


class Operation(ABC):

    def __init__(self, operands: List[pldiffer.tensor.Tensor]):
        self.operands = operands
        self.diff = reduce(lambda x, y: x or y, map(lambda t: t.diff, operands), False)

    @abstractmethod
    def forward(self) -> pldiffer.tensor.Tensor:
        pass

    @abstractmethod
    def backward(self, g_in: np.ndarray) -> List[np.ndarray]:
        pass
