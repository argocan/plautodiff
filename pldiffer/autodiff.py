import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from functools import reduce


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
        return Operations.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return Operations.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

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
        self.__backward__(np.array([1.0]))


class Operation(ABC):

    def __init__(self, operands: List[Tensor]):
        self.operands = operands
        self.diff = reduce(lambda x, y: x or y, operands, False)

    @abstractmethod
    def forward(self) -> Tensor:
        pass

    @abstractmethod
    def backward(self, g_in: np.ndarray) -> List[np.ndarray]:
        pass


class Add(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(self.x.data + self.y.data, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = Add.broadcast(self.x.data, g_in) if self.x.diff else None
        dy = Add.broadcast(self.y.data, g_in) if self.y.diff else None
        return [dx, dy]

    @staticmethod
    def broadcast(t, g_in):
        if len(t.shape) < len(g_in.shape):
            t.resize(g_in.shape)
        if len(g_in.shape) < len(t.shape):
            g_in.resize(t.shape)
        if t.shape[0] == 1 and g_in.shape[0] > 1:
            return np.sum(g_in, axis=0, keepdims=True)
        if t.shape[1] == 1 and g_in.shape[1] > 1:
            return np.sum(g_in, axis=1, keepdims=True)
        return g_in


class Mul(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(self.x.data * self.y.data, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * self.y.data if self.x.diff else None
        dy = g_in * self.x.data if self.y.diff else None
        return [dx, dy]


class Matmul(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.y = operands[1]

    def forward(self):
        return Tensor(np.dot(self.x.data, self.y.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = np.dot(self.y.data, g_in.T).T if self.x.diff else None
        dy = np.dot(self.x.data.T, g_in) if self.y.diff else None
        return [dx, dy]


class Quadratic(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.power(self.x.data, 2), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = 2 * g_in * self.x.data if self.x.diff else None
        return [dx]


class Exp(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = np.exp(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * self.cache if self.x.diff else None
        return [dx]


class Log(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.log(self.x.data), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * np.true_divide(1.0, self.x.data) if self.x.diff else None
        return [dx]


class Sigmoid(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = np.true_divide(1, 1 + np.exp(-1 * self.x.data))
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * (self.cache * (1 - self.cache)) if self.x.diff else None
        return [dx]


class Softmax(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]
        self.cache = None

    def forward(self):
        self.cache = Softmax.calc_softmax(self.x.data)
        return Tensor(self.cache, diff=self.diff)

    def backward(self, g_in: np.ndarray):
        jacobian = np.array([Softmax.jacobian(row) for row in self.cache])
        dx = np.einsum('ij,ijk->ik', g_in, jacobian)
        return [dx]

    @staticmethod
    def calc_softmax(inp):

        def softmax(v):
            den = np.sum(np.exp(v))
            return np.true_divide(np.exp(v), den)

        maxes = np.max(inp, axis=1, keepdims=True)
        x = inp - maxes
        return np.array([softmax(row) for row in x])

    @staticmethod
    def jacobian(s):
        return np.diag(s) - np.outer(s, s)


class Sum(Operation):

    def __init__(self, operands: List[Tensor]):
        Operation.__init__(self, operands)
        self.x = operands[0]

    def forward(self):
        return Tensor(np.sum(self.x.data, keepdims=True), diff=self.diff)

    def backward(self, g_in: np.ndarray):
        dx = g_in * np.ones(self.x.data.shape) if self.x.diff else None
        return [dx]


class Operations:

    @staticmethod
    def cast_to_tensor(x: Union[float, Tensor]):
        if type(x) is float:
            x = Tensor(x)
        return x

    @staticmethod
    def cast_to_tensors(l: List[Union[float, Tensor]]):
        return [Operations.cast_to_tensor(x) for x in l]

    @staticmethod
    def do_op(op_class, operands: List[Tensor]) -> Tensor:
        op = op_class(operands)
        res = op.forward()
        res.set_op(op)
        return res

    @staticmethod
    def add(x: [Tensor, float], y: [Tensor, float]) -> Tensor:
        return Operations.do_op(Add, Operations.cast_to_tensors([x, y]))

    @staticmethod
    def mul(x: [Tensor, float], y: [Tensor, float]) -> Tensor:
        return Operations.do_op(Mul, Operations.cast_to_tensors([x, y]))

    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        return Operations.do_op(Matmul, [x, y])

    @staticmethod
    def quadratic(x: Tensor) -> Tensor:
        return Operations.do_op(Quadratic, [x])

    @staticmethod
    def exp(x: Tensor) -> Tensor:
        return Operations.do_op(Exp, [x])

    @staticmethod
    def log(x: Tensor) -> Tensor:
        return Operations.do_op(Log, [x])

    @staticmethod
    def sigmoid(x: Tensor) -> Tensor:
        return Operations.do_op(Sigmoid, [x])

    @staticmethod
    def softmax(x: Tensor) -> Tensor:
        return Operations.do_op(Softmax, [x])

    @staticmethod
    def sum(x: Tensor) -> Tensor:
        return Operations.do_op(Sum, [x])

