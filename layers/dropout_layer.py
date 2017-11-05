import numpy as np
from pldiffer import Tensor
from numba import jit


@jit(nopython=True)
def binomial(prob, size):
    return np.true_divide(np.random.binomial(1, prob, size).astype(np.float32), prob)


def dropout(tensor_in: Tensor, keep_prob: float=0.5, train_mode=False):
    if train_mode:
        mask = Tensor(binomial(keep_prob, tensor_in.shape()), diff=False)
        return tensor_in * mask
    else:
        return tensor_in
