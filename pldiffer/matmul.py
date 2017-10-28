from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def dot(x, y):
    return np.dot(x, y)


class Matmul(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(dot(self.x.data, self.y.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = dot(self.y.data, g_in.T).T if self.x.diff else None
        dy = dot(self.x.data.T, g_in) if self.y.diff else None
        return [dx, dy]
