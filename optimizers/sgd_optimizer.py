from optimizers import Optimizer
from pldiffer import Tensor
from typing import List


class SGDOptimizer(Optimizer):

    def __init__(self, parameters: List[Tensor]=None, learning_rate: float=0.01):
        Optimizer.__init__(self, parameters)
        self.learning_rate = learning_rate

    def step(self, loss: Tensor):
        loss.calc_gradients()
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad

