from typing import Tuple
import numpy as np
from optimizers.optimizer import Optimizer
from pldiffer import Tensor
from pldiffer import Operations as op
from layers.layer import Layer


class DenseLayer(Layer):

    def __init__(self, dim: Tuple[int, int], optimizer: Optimizer = None):
        self.dim = dim
        self.w = Tensor(np.random.normal(0, 0.01, dim).astype(np.float32), diff=True)
        self.b = Tensor(np.zeros((1, dim[1]), dtype=np.float32), diff=True)
        self.parameters = [self.w, self.b]
        if optimizer is not None:
            self.optimizer = optimizer
            self.optimizer.add_parameters(self.parameters)

    def compute(self, tensor_in: Tensor, train_mode=False):
        return op.matmul(tensor_in, self.w) + self.b

    def shape(self):
        return (self.dim[1],)
