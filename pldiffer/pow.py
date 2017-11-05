import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


@jit(nopython=True)
def forward(x, p):
    return np.power(x, p).astype(np.float32)


@jit(nopython=True)
def grad(g, x, p):
    return (p * np.power(x, p - 1) * g).astype(np.float32)


class Pow(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.p = operands[1].data

    def forward(self):
        return Tensor(forward(self.x.data, self.p), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = grad(g_in, self.x.data, self.p) if self.x.diff else None
        return [dx, None]
