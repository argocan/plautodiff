import numpy as np
import pldiffer.autodiff


class Tensor:

    def __init__(self, data: np.ndarray, diff=False):
        self.data = data
        self.diff = diff
        self.grad = None
        self.op = None

    def set_op(self, op: 'Operation'):
        self.op = op

    def __str__(self):
        return self.data.__str__()

    def __mul__(self, other):
        return pldiffer.autodiff.Operations.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return pldiffer.autodiff.Operations.div(self, other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __add__(self, other):
        return pldiffer.autodiff.Operations.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return pldiffer.autodiff.Operations.add(self, -1.0 * other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __pow__(self, power, modulo=None):
        return pldiffer.autodiff.Operations.pow(self, power)

    def __neg__(self):
        return pldiffer.autodiff.Operations.mul(-1.0, self)

    def __backward__(self, g_in: np.ndarray):
        if self.grad is None:
            self.grad = g_in
        else:
            self.grad += g_in
        if self.op is not None:
            gs_out = self.op.backward(g_in)
            for i in range(0, len(self.op.operands)):
                t_par = self.op.operands[i]
                t_par.__backward__(gs_out[i])

    def __reset_grad__(self):
        self.grad = None
        if self.op is not None:
            for i in range(0, len(self.op.operands)):
                self.op.operands[i].__reset_grad__()

    def calc_gradients(self):
        self.__reset_grad__()
        self.__backward__(np.array([1.0], dtype=np.float32))

    def shape(self):
        return self.data.shape
