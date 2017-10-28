from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor
from utils import softmax, log_sofmax_jacobian, softmax_grad


class LogSoftmax(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = softmax(self.x.data)
        return Tensor(np.log(self.cache + 1e-7), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = softmax_grad(g_in, log_sofmax_jacobian(self.cache))
        return [dx]

