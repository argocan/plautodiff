from optimizers import Optimizer
from utils import zeros_like_list, running_avg, running_avg_bias_correction
from pldiffer import Tensor
from typing import List


class MomentumOptimizer(Optimizer):

    def __init__(self, parameters: List[Tensor]=None, learning_rate: float=0.01, beta: float=0.9, bias_correction=False):
        Optimizer.__init__(self, parameters)
        self.learning_rate = learning_rate
        self.beta = beta
        self.bias_correction = bias_correction
        self.t = 0
        self.m = zeros_like_list(self.parameters)

    def step(self, loss: Tensor):
        loss.calc_gradients()
        self.t = self.t + 1
        for i in range(0, len(self.parameters)):
            self.m[i] = running_avg(self.m[i], self.beta, self.parameters[i].grad)
            m_hat = running_avg_bias_correction(self.m[i], self.beta, self.t) if self.bias_correction else self.m[i]
            self.parameters[i].data -= self.learning_rate * m_hat

    def add_parameters(self, new_parameters):
        Optimizer.add_parameters(self, new_parameters)
        self.m += zeros_like_list(new_parameters)
