from time import time
from utils import BatchIterator
from pldiffer import Tensor
from optimizers import Optimizer
from numpy import ndarray
import numpy as np
from typing import Callable


def learn(in_data: ndarray, out_data: ndarray, test_in_data: ndarray,
          test_out_data: ndarray, model_func: Callable[[Tensor, Tensor, bool], Tensor],
          loss_func: Callable[[Tensor, Tensor, bool], Tensor], optimizer: Optimizer,
          score_func: Callable[[Tensor, ndarray], float]=None,
          batch_size: int=100, epoch_number: int=100):
    input_data = in_data.astype(np.float32)
    output_data = out_data.astype(np.float32)
    test_input_data = test_in_data.astype(np.float32)
    test_output_data = test_out_data.astype(np.float32)
    train_loss_values = []
    test_loss_values = []
    test_score_values = []
    start = time()
    for i in range(0, epoch_number):
        bit = BatchIterator(input_data, output_data, batch_size)
        iter_loss = 0
        for b_in, b_out in bit:
            x = Tensor(b_in)
            y = Tensor(b_out)
            model = model_func(x, y, True)
            loss = loss_func(y, model, True)
            iter_loss += loss.data[0] / input_data.shape[0]
            optimizer.step(loss)
        if score_func is not None:
            test_loss, err_ratio = score_test(test_input_data, test_output_data, model_func, loss_func, score_func)
        else:
            err_ratio = 'N/A'
            test_loss = 'N/A'
        train_loss_values.append(iter_loss)
        test_loss_values.append(test_loss)
        test_score_values.append(err_ratio)   
        print("Iteration {0} train-loss: {1}, test-loss: {2}, score: {3}%".format(i, iter_loss, test_loss, err_ratio))
    end = time()
    print("Execution time: {0}s".format(end - start))
    return train_loss_values, test_loss_values, test_score_values


def score_test(input_data, out_data, model_func, loss_func, score_func):
    err_ratio = 0
    y = Tensor(out_data)
    actual = model_func(Tensor(input_data))
    loss = loss_func(y, actual, False)
    err_ratio += score_func(actual.data, out_data)
    return loss.data[0] / input_data.shape[0], err_ratio
