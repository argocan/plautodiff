from utils.batch_iterator import BatchIterator
from utils.utility import  plot_loss, row_max, row_substract, zeros_like_list, \
    einsum_ij_ijk_ik, running_avg, running_avg_bias_correction, running_avg_squared, grad_square_delta
import utils.mnist.mnist_util as mnist_util
from utils.softmax import softmax, softmax_grad, softmax_jacobian, log_sofmax_jacobian
from utils.broadcast import broadcast
