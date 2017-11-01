import numpy as np
from pldiffer import Tensor
from pldiffer import Operations as op
from utils import mnist_util as mnist
from utils import BatchIterator
from time import time
from optimizers import SGDOptimizer, MomentumOptimizer, RmspropOptimizer, AdamOptimizer

minp, mout, mtest, mtestout = mnist.load_data_set()
minp = minp.astype(np.float32)
mout = mout.astype(np.float32)
mtest = mtest.astype(np.float32)
mtestout = mtestout.astype(np.float32)

num_hidden_neuron = 1000

W1 = Tensor(np.random.normal(0, 0.01, (784, num_hidden_neuron)).astype(np.float32), diff=True)
b1 = Tensor(np.zeros((1, num_hidden_neuron), dtype=np.float32), diff=True)
W2 = Tensor(np.random.normal(0, 0.01, (num_hidden_neuron, 10)).astype(np.float32), diff=True)
b2 = Tensor(np.zeros((1, 10), dtype=np.float32), diff=True)
eta = 0.001

optimizer = AdamOptimizer(parameters=[W1, b1, W2, b1], learning_rate=0.001, bias_correction=False)


def calc_model(b_in, b_out=None):
    x = Tensor(b_in)
    z1 = op.matmul(x, W1) + b1
    a1 = op.relu(z1)
    z2 = op.matmul(a1, W2) + b2
    if b_out is None:
        return op.softmax(z2)
    else:
        return op.softmax_cross_entropy(z2, Tensor(b_out))

div1 = np.true_divide(1.0, np.size(W1))
div2 = np.true_divide(1.0, np.size(W2))

for i in range(0, 11):
    if i==1:
        start = time()
    bit = BatchIterator(minp, mout, 100)
    iter_loss = 0
    for b_in, b_out in bit:
        deltas = calc_model(b_in, b_out)
        loss = op.sum(deltas) + 0.001 * (op.sum(op.quadratic(div1 * W1)) + op.sum(op.quadratic(div2 * W2)))
        iter_loss += loss.data
        optimizer.step(loss)
    actual = calc_model(mtest)
    err_ratio = mnist.score_result(actual.data, mtestout)
    print("Iteration {0} loss: {1} score {2}%".format(i, iter_loss, err_ratio))
end = time()
print(end - start)
