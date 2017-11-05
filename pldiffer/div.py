from typing import List
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.operation import Operation
from numba import jit, float32


@jit(nopython=True)
def div(x, y):
    return (x / y).astype(np.float32)

@jit(nopython=True)
def mul(x, y):
    return (x * y).astype(np.float32)

@jit(nopython=True)
def div_grad(x, y):
    return -x / (y ** -2)

class Div(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(div(self.x.data, self.y.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = mul(g_in, div(1.0, self.y.data)) if self.x.diff else None
        dy = mul(g_in, div_grad(self.x.data, self.y.data)) if self.y.diff else None
        return [dx, dy]
