import numpy as np


def broadcast(t, g_in):
    if len(t.shape) < len(g_in.shape):
        t = t.copy()
        t.resize(g_in.shape)
    if len(g_in.shape) < len(t.shape):
        g_in = g_in.copy()
        g_in.resize(t.shape)
    if t.shape[0] == 1 and g_in.shape[0] > 1:
        return np.sum(g_in, axis=0)
    if t.shape[1] == 1 and g_in.shape[1] > 1:
        return np.sum(g_in, axis=1)
    return g_in
