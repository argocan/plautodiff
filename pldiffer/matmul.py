from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Matmul(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(np.dot(self.x.data, self.y.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = np.dot(self.y.data, g_in.T).T if self.x.diff else None
        dy = np.dot(self.x.data.T, g_in) if self.y.diff else None
        return [dx, dy]
