import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def log(x):
    return np.log(x)


@jit(nopython=True)
def grad(g, x):
    return g * np.true_divide(np.float32(1.0), x)


class Log(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(log(self.x.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = grad(g_in, self.x.data) if self.x.diff else None
        return [dx]
