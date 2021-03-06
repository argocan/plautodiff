from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def sigmoid(x):
    return np.true_divide(1, 1 + np.exp(-1 * x))


@jit(nopython=True)
def grad(g, x):
    return g * (x * (1 - x))


class Sigmoid(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = sigmoid(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = grad(g_in, self.cache) if self.x.diff else None
        return [dx]
