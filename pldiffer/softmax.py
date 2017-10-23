from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Softmax(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = Softmax.calc_softmax(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        jacobian = np.array([Softmax.jacobian(row) for row in self.cache])
        dx = np.einsum('ij,ijk->ik', g_in, jacobian)
        return [dx]

    @staticmethod
    def calc_softmax(inp):

        def softmax(v):
            den = np.sum(np.exp(v))
            return np.true_divide(np.exp(v), den)

        maxes = np.max(inp, axis=1, keepdims=True)
        x = inp - maxes
        return np.array([softmax(row) for row in x])

    @staticmethod
    def jacobian(s):
        return np.diag(s) - np.outer(s, s)