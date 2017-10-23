import numpy as np
from typing import List
from pldiffer.operation import Operation
from pldiffer.tensor import Tensor


class Log(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.log(self.x.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * np.true_divide(1.0, self.x.data) if self.x.diff else None
        return [dx]
