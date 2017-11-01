from pldiffer import Tensor
from optimizers import Optimizer
from utils import zeros_like_list, running_avg, running_avg_squared, running_avg_bias_correction, grad_square_delta
from typing import List


class AdamOptimizer(Optimizer):

    def __init__(self, parameters: List[Tensor]=None, learning_rate: float=0.001, beta_1: float=0.9, beta_2: float=0.999,
                 epsilon: float=1e-08, bias_correction=True):
        Optimizer.__init__(self, parameters)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.t = 0
        self.m = zeros_like_list(self.parameters)
        self.v = zeros_like_list(self.parameters)

    def step(self, loss: Tensor):
        loss.calc_gradients()
        self.t = self.t + 1
        for i in range(0, len(self.parameters)):
            g = self.parameters[i].grad
            self.m[i] = running_avg(self.m[i], self.beta_1, g)
            self.v[i] = running_avg_squared(self.v[i], self.beta_2, g)
            m_hat = running_avg_bias_correction(self.m[i], self.beta_1, self.t) if self.bias_correction else self.m[i]
            v_hat = running_avg_bias_correction(self.v[i], self.beta_2, self.t) if self.bias_correction else self.v[i]
            self.parameters[i].data -= grad_square_delta(self.learning_rate, m_hat, v_hat, self.epsilon)

    def add_parameters(self, new_parameters):
        Optimizer.add_parameters(self, new_parameters)
        self.m += zeros_like_list(new_parameters)
        self.v += zeros_like_list(new_parameters)
