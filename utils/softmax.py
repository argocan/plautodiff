import numpy as np
from numba import jit, prange
from utils.utility import row_max, row_substract, einsum_ij_ijk_ik


@jit(nopython=True)
def __softmax(v):
    den = np.sum(np.exp(v))
    return np.true_divide(np.exp(v), den)


@jit(nopython=True, parallel=True)
def softmax(inp):
    maxes = row_max(inp)
    x = row_substract(inp, maxes)
    s = np.zeros_like(x)
    for i in prange(0, x.shape[0]):
        s[i] = __softmax(x[i])
    return s


@jit(nopython=True)
def __row_jacobian(s):
    return np.diag(s) - np.outer(s, s)


@jit(nopython=True, parallel=True)
def softmax_jacobian(rows):
    j = np.zeros((rows.shape[0], rows.shape[1], rows.shape[1]), dtype=np.float32)
    for i in prange(0, rows.shape[0]):
        j[i] = __row_jacobian(rows[i])
    return j


def softmax_grad(g_in, j):
    #return np.einsum('ij,ijk->ik', g_in, j)
    return einsum_ij_ijk_ik(g_in, j)


@jit(nopython=True)
def __row_log_jacobian(s):
    return np.identity(s.shape[0]) - s


@jit(nopython=True, parallel=True)
def log_sofmax_jacobian(rows):
    j = np.zeros((rows.shape[0], rows.shape[1], rows.shape[1]), dtype=np.float32)
    for i in prange(0, rows.shape[0]):
        j[i] = __row_log_jacobian(rows[i])
    return j
