from typing import List
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.operation import Operation


class Add(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(self.x.data + self.y.data, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = Add.broadcast(self.x.data, g_in) if self.x.diff else None
        dy = Add.broadcast(self.y.data, g_in) if self.y.diff else None
        return [dx, dy]

    @staticmethod
    def broadcast(t, g_in):
        if len(t.shape) < len(g_in.shape):
            t.resize(g_in.shape)
        if len(g_in.shape) < len(t.shape):
            g_in.resize(t.shape)
        if t.shape[0] == 1 and g_in.shape[0] > 1:
            return np.sum(g_in, axis=0, keepdims=True)
        if t.shape[1] == 1 and g_in.shape[1] > 1:
            return np.sum(g_in, axis=1, keepdims=True)
        return g_in
