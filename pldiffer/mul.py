from typing import List
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.operation import Operation


class Mul(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(self.x.data * self.y.data, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * self.y.data if self.x.diff else None
        dy = g_in * self.x.data if self.y.diff else None
        return [dx, dy]
