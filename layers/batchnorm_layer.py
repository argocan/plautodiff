import numpy as np
from typing import Tuple
from optimizers.optimizer import Optimizer
from pldiffer import Tensor
from layers.layer import Layer
from pldiffer import Operations as op
from utils import running_avg
from numba import jit


@jit(nopython=True)
def mean(x: np.ndarray):
    s = x.sum(axis=0)
    return np.true_divide(s, x.shape[0])


@jit(nopython=True)
def var(x: np.ndarray, mean: float):
    s = np.sum(np.power((x - mean), 2), axis=0)
    return np.true_divide(s, x.shape[0])


class BatchNorm(Layer):

    def __init__(self, dim: Tuple[int, ...], optimizer: Optimizer):
        self.dim = dim
        self.gamma = Tensor(np.expand_dims(np.ones(dim, dtype=np.float32), axis=0), diff=True)
        self.beta = Tensor(np.expand_dims(np.zeros(dim, dtype=np.float32), axis=0), diff=True)
        self.ravg_mean = np.zeros(dim, dtype=np.float32)
        self.ravg_var = np.zeros(dim, dtype=np.float32)
        self.epsilon = 1e-7
        optimizer.add_parameters([self.beta])

    def compute(self, tensor_in: Tensor, train_mode=False):
        if train_mode:
            m = Tensor(mean(tensor_in.data), diff=False)
            v = Tensor(var(tensor_in.data, m.data), diff=False)
            self.ravg_mean = running_avg(self.ravg_mean, 0.9, m.data)
            self.ravg_var = running_avg(self.ravg_var, 0.9, v.data)
        else:
            m = Tensor(self.ravg_mean)
            v = Tensor(self.ravg_var)
        x = (tensor_in - m) / op.sqrt(v + self.epsilon)
        y = self.gamma * x + self.beta
        return y

    def shape(self):
        return self.layer_in.shape()
