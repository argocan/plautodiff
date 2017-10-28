from typing import List
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.operation import Operation
from numba import jit, float32


#@jit(nopython=True)
def add(x, y):
    return x + y


#@jit(nopython=True)
def t_sum(x, axis=0):
    return np.sum(x, axis=axis)


def broadcast(t, g_in):
    if len(t.shape) < len(g_in.shape):
        t.resize(g_in.shape)
    if len(g_in.shape) < len(t.shape):
        g_in.resize(t.shape)
    if t.shape[0] == 1 and g_in.shape[0] > 1:
        return t_sum(g_in, axis=0)
    if t.shape[1] == 1 and g_in.shape[1] > 1:
        return t_sum(g_in, axis=1)
    return g_in


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
