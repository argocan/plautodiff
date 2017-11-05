from typing import List
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.operation import Operation
from numba import jit, float32
from utils import broadcast


#@jit(nopython=True)
def add(x, y):
    return x + y


class Add(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(add(self.x.data, self.y.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = broadcast(self.x.data, g_in) if self.x.diff else None
        dy = broadcast(self.y.data, g_in) if self.y.diff else None
        return [dx, dy]

