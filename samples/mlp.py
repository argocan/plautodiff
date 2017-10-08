import numpy as np
from pldiffer import Tensor
from pldiffer import Operations as op
from utils import mnist_util as mnist
from utils import BatchIterator

minp, mout, mtest, mtestout = mnist.load_data_set()

num_hidden_neuron = 100

W1 = Tensor(np.random.normal(0, 0.01, (784, num_hidden_neuron)), diff=True)
b1 = Tensor(np.zeros((1, num_hidden_neuron)), diff=True)
W2 = Tensor(np.random.normal(0, 0.01, (num_hidden_neuron, 10)), diff=True)
b2 = Tensor(np.zeros((1, 10)), diff=True)
eta = 0.01


def calc_model(b_in):
    x = Tensor(b_in)
    z1 = op.matmul(x, W1) + b1
    a1 = op.sigmoid(z1)
    z2 = op.matmul(a1, W2) + b2
    return op.softmax(z2)

for i in range(0, 100):
    bit = BatchIterator(minp, mout, 100)
    iter_loss = 0
    for b_in, b_out in bit:
        y = Tensor(b_out)
        model = calc_model(b_in)
        #print(model.data)
        deltas = (-1.0 * y) * op.log(model)
        loss = op.sum(deltas) + 0.001 * (op.sum(op.quadratic(W1)) + op.sum(op.quadratic(W2)))
        iter_loss += loss.data
        loss.calc_gradients()
        #print(np.sum(W2.grad[0,0]))
        W1.data -= eta * W1.grad
        W2.data -= eta * W2.grad
        b1.data -= eta * b1.grad
        b2.data -= eta * b2.grad
        #print(iter_loss)
        #exit(0)
    actual = calc_model(mtest)
    err_ratio = mnist.score_result(actual.data, mtestout)
    print("Iteration {0} loss: {1} score {2}%".format(i, iter_loss, err_ratio))
