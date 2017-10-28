import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def forward(x):
    return np.maximum(x, 0)


@jit(nopython=True)
def grad(x):
    return np.minimum(x, 1)


class Relu(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = forward(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * grad(self.cache) if self.x.diff else None
        return [dx]
