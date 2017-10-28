from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from utils import softmax, softmax_jacobian, softmax_grad


class Softmax(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = softmax(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = softmax_grad(g_in, softmax_jacobian(self.cache))
        return [dx]

