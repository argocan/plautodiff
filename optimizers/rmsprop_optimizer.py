import numpy as np
from pldiffer import Tensor
from optimizers import Optimizer
from utils import zeros_like_list, running_avg_squared, running_avg_bias_correction, grad_square_delta
from typing import List


class RmspropOptimizer(Optimizer):

    def __init__(self, parameters: List[np.ndarray]=None, learning_rate: float=0.01, beta: float=0.9,
                 epsilon: float=1e-08, bias_correction=False):
        Optimizer.__init__(self, parameters)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.t = 0
        self.v = zeros_like_list(parameters)

    def step(self, loss: Tensor):
        loss.calc_gradients()
        self.t = self.t + 1
        for i in range(0, len(self.parameters)):
            self.v[i] = running_avg_squared(self.v[i], self.beta, self.parameters[i].grad)
            v_hat = running_avg_bias_correction(self.v[i], self.beta, self.t) if self.bias_correction else self.v[i]
            self.parameters[i].data -= grad_square_delta(self.learning_rate, self.parameters[i].grad, v_hat, self.epsilon)

    def add_parameters(self, new_parameters):
        Optimizer.add_parameters(self, new_parameters)
        self.v += zeros_like_list(new_parameters)
