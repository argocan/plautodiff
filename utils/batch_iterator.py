import numpy as np


class BatchIterator:

    def __init__(self, arr_in, arr_out, batch_size):
        self.arr_in = arr_in
        self.arr_out = arr_out
        self.batch_size = batch_size
        self.current_index = 0
        self.arr_size = arr_in.shape[0]
        self.random_indexes = np.arange(arr_in.shape[0])
        self.stop_iter = False
        np.random.shuffle(self.random_indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iter:
            self.current_index = 0
            self.stop_iter = False
            raise StopIteration()
        start = self.current_index
        if self.current_index + self.batch_size < self.arr_size:
            end = self.current_index + self.batch_size
            self.current_index = end
        else:
            end = self.arr_size
            self.stop_iter = True
        indexes = self.random_indexes[start:end]
        ret_in = self.arr_in[indexes, :]
        ret_out = self.arr_out[indexes, :]
        return ret_in, ret_out
