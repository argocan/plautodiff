from utils.batch_iterator import BatchIterator
from utils.utility import numpy_to_variable, zeros_like_list, bernoulli, plot_loss, row_max, row_substract, einsum_ij_ijk_ik
import utils.mnist.mnist_util as mnist_util
from utils.softmax import softmax, softmax_grad, softmax_jacobian, log_sofmax_jacobian