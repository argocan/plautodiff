from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from utils import softmax
from numba import jit


@jit(nopython=True)
def grad(a, y):
    return a - y


class SoftmaxCrossEntropy(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        self.cache = softmax(self.x.data)
        return Tensor(-self.y.data * np.log(self.cache + 1e-7))

    def backward(self, g_in: np.ndarray):
        dx = grad(self.cache, self.y.data)
        return [dx, None]
