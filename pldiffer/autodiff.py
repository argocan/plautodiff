from typing import List, Union
import numpy as np
from pldiffer.tensor import Tensor
from pldiffer.add import Add
from pldiffer.mul import Mul
from pldiffer.div import Div
from pldiffer.matmul import Matmul
from pldiffer.quadratic import Quadratic
from pldiffer.pow import Pow
from pldiffer.exp import Exp
from pldiffer.log import Log
from pldiffer.sigmoid import Sigmoid
from pldiffer.relu import Relu
from pldiffer.softmax import Softmax
from pldiffer.log_softmax import LogSoftmax
from pldiffer.softmax_cross_entropy import SoftmaxCrossEntropy
from pldiffer.sum import Sum


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
    def div(x: [Tensor, float], y: [Tensor, float]) -> Tensor:
        return Operations.do_op(Div, Operations.cast_to_tensors([x, y]))

    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        return Operations.do_op(Matmul, [x, y])

    @staticmethod
    def quadratic(x: Tensor) -> Tensor:
        return Operations.do_op(Quadratic, [x])

    @staticmethod
    def pow(x: Tensor, p: Tensor) -> Tensor:
        return Operations.do_op(Pow, Operations.cast_to_tensors([x, p]))

    @staticmethod
    def sqrt(x: Tensor) -> Tensor:
        return Operations.pow(x, 0.5)

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
    def relu(x: Tensor) -> Tensor:
        return Operations.do_op(Relu, [x])

    @staticmethod
    def softmax(x: Tensor) -> Tensor:
        return Operations.do_op(Softmax, [x])

    @staticmethod
    def log_softmax(x: Tensor) -> Tensor:
        return Operations.do_op(LogSoftmax, [x])

    @staticmethod
    def softmax_cross_entropy(x: Tensor, y: Tensor = None) -> Tensor:
        return Operations.do_op(SoftmaxCrossEntropy, [x, y])

    @staticmethod
    def sum(x: Tensor) -> Tensor:
        return Operations.do_op(Sum, [x])
