from pldiffer import Operations as op
from utils import mnist_util as mnist
from utils import plot_loss
from nnet import learn
from optimizers import SGDOptimizer, MomentumOptimizer, RmspropOptimizer, AdamOptimizer
from layers import DenseLayer, BatchNorm

mnist_in_data, mnist_out_data, mnist_test_in_data, mnist_test_out_data = mnist.load_data_set()

num_hidden_neuron = 1600

optimizer = AdamOptimizer(learning_rate=0.001)

l1 = DenseLayer((784, num_hidden_neuron), optimizer)
bn1 = BatchNorm((num_hidden_neuron,), optimizer)
l2 = DenseLayer((num_hidden_neuron, num_hidden_neuron), optimizer)
l3 = DenseLayer((num_hidden_neuron, 10), optimizer)


def calc_model(x, y=None, train_mode=False):
    a1 = op.relu(bn1.compute(l1.compute(x), train_mode=train_mode))
    a2 = op.relu(l2.compute(a1))
    z3 = l3.compute(a2)
    if not train_mode:
        return op.softmax(z3)
    else:
        return op.softmax_cross_entropy(z3, y)


def loss_function(y, m, train_mode=False):
    if train_mode:
        return op.sum(m)
    else:
        return op.sum(-y * op.log(m))


train_loss_values, test_loss_values, test_score_values = learn(mnist_in_data, mnist_out_data, mnist_test_in_data,
                                                               mnist_test_out_data, calc_model, loss_function,
                                                               optimizer, score_func=mnist.score_result,
                                                               batch_size=128, epoch_number=10)

plot_loss(train_loss_values, test_loss_values)