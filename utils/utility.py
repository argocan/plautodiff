from torch.autograd import Variable
import torch
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange, autojit


def numpy_to_variable(t, cuda=True):
    tv = torch.from_numpy(t).float().cuda() if cuda else torch.from_numpy(t).float()
    return Variable(tv)


def zeros_like_list(parameter_list: List[Variable]):
    return [torch.zeros(p.size()).cuda() for p in parameter_list]


def bernoulli(dim: List[int], prob: float=0.5):
    return torch.bernoulli(torch.zeros(dim).add_(prob)).cuda()


def plot_loss(train_loss_values, test_loss_values):
    plt.plot(np.arange(0, len(train_loss_values)), train_loss_values, 'b-',
             np.arange(0, len(train_loss_values)), test_loss_values, 'r-')
    plt.show()


@jit(nopython=True, parallel=True)
def row_max(arr: np.ndarray):
    maxes = np.empty((arr.shape[0]))
    for i in prange(0, arr.shape[0]):
        maxes[i] = np.max(arr[0])
    return maxes


@jit(nopython=True, parallel=True)
def row_substract(arr: np.ndarray, subs: np.ndarray):
    to_ret = arr.copy()
    for i in prange(0, arr.shape[0]):
        to_ret[i] -= subs[i]
    return to_ret


@jit(nopython=True)
def einsum_ij_ijk_ik(x: np.ndarray, y: np.ndarray):
    to_ret = np.empty((x.shape[0], y.shape[2]), dtype=np.float32)
    for i in range(0, x.shape[0]):
        to_ret[i] = np.dot(x[i], y[i])
    return to_ret
