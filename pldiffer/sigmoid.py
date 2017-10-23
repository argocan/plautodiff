from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Sigmoid(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = np.true_divide(1, 1 + np.exp(-1 * self.x.data))
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * (self.cache * (1 - self.cache)) if self.x.diff else None
        return [dx]
