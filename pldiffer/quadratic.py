import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Quadratic(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.power(self.x.data, 2), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = 2 * g_in * self.x.data if self.x.diff else None
        return [dx]
