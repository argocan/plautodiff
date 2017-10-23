import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Exp(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = np.exp(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * self.cache if self.x.diff else None
        return [dx]
