from pldiffer.tensor import Tensor
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange, vectorize, float32


def zeros_like_list(parameter_list: List[Tensor]):
    return [np.zeros_like(p.data, dtype=np.float32) for p in parameter_list]


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


@vectorize([float32(float32, float32, float32)], target='parallel')
def running_avg(acc, beta, newval):
    return beta * acc + (1 - beta) * newval


@vectorize([float32(float32, float32, float32)], target='parallel')
def running_avg_bias_correction(acc, beta, iteration):
    return acc / (1 - beta ** iteration)


@vectorize([float32(float32, float32, float32)], target='parallel')
def running_avg_squared(acc, beta, newval):
    return beta * acc + (1 - beta) * (newval ** 2)


@vectorize([float32(float32, float32, float32, float32)], target='parallel')
def grad_square_delta(lr, g, v_hat, eps):
    return (lr * g) / (v_hat ** 0.5 + eps)
