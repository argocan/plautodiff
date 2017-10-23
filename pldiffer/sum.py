from typing import List
import numpy as np
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Sum(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.sum(self.x.data, keepdims=True), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in if self.x.diff else None
        return [dx]
