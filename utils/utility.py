from torch.autograd import Variable
import torch
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def numpy_to_variable(t, cuda=True):
    tv = torch.from_numpy(t).float().cuda() if cuda else torch.from_numpy(t).float()
    return Variable(tv)


def zeros_like_list(parameter_list: List[Variable]):
    return [torch.zeros(p.size()).cuda() for p in parameter_list]


def bernoulli(dim: List[int], prob: float=0.5):
    return torch.bernoulli(torch.zeros(dim).add_(prob)).cuda()


def plot_loss(train_loss_values, test_loss_values):
    plt.plot(np.arange(0, len(train_loss_values)), train_loss_values, 'b-',
             np.arange(0, len(train_loss_values)), test_loss_values, 'r-')
    plt.show()
