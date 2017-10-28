import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def forward(x):
    return np.power(x, 2).astype(np.float32)


@jit(nopython=True)
def grad(x, y):
    return (2.0 * x * y).astype(np.float32)


class Quadratic(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(forward(self.x.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = grad(g_in, self.x.data) if self.x.diff else None
        return [dx]
