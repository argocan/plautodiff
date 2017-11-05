import numpy as np
from pldiffer import Tensor
from pldiffer import Operations as op
from utils import mnist_util as mnist
from utils import BatchIterator
from time import time
from optimizers import SGDOptimizer, MomentumOptimizer, RmspropOptimizer, AdamOptimizer
from layers import DenseLayer, BatchNorm

minp, mout, mtest, mtestout = mnist.load_data_set()
minp = minp.astype(np.float32)
mout = mout.astype(np.float32)
mtest = mtest.astype(np.float32)
mtestout = mtestout.astype(np.float32)

num_hidden_neuron = 1600

optimizer = SGDOptimizer(learning_rate=0.001)

l1 = DenseLayer((784, num_hidden_neuron), optimizer)
bn1 = BatchNorm((num_hidden_neuron), optimizer)
l2 = DenseLayer((num_hidden_neuron, num_hidden_neuron), optimizer)
l3 = DenseLayer((num_hidden_neuron, 10), optimizer)


def calc_model(b_in, b_out=None):
    train_mode = b_out is not None
    x = Tensor(b_in)
    a1 = bn1.compute(op.relu(l1.compute(x)), train_mode=train_mode)
    #a1 = op.relu(l1.compute(x))
    a2 = op.relu(l2.compute(a1))
    z3 = l3.compute(a2)
    if not train_mode:
        return op.softmax(z3)
    else:
        return op.softmax_cross_entropy(z3, Tensor(b_out))


for i in range(0, 11):
    if i == 1:
        start = time()
    bit = BatchIterator(minp, mout, 100)
    iter_loss = 0
    for b_in, b_out in bit:
        deltas = calc_model(b_in, b_out)
        loss = op.sum(deltas)
        iter_loss += loss.data
        optimizer.step(loss)
    actual = calc_model(mtest)
    err_ratio = mnist.score_result(actual.data, mtestout)
    print("Iteration {0} loss: {1} score {2}%".format(i, iter_loss, err_ratio))
end = time()
print(end - start)
