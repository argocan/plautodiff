from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from numba import jit


#@jit(nopython=True)
def sum(x):
    return np.sum(x, keepdims=True)


class Sum(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(sum(self.x.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in if self.x.diff else None
        return [dx]
