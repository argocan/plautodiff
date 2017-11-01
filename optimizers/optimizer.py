from abc import ABC, abstractmethod
from typing import List
from pldiffer import Tensor


class Optimizer(ABC):

    def __init__(self, parameters: List[Tensor]=None):
        self.parameters = [] if parameters is None else parameters

    def add_parameters(self, new_parameters: List[Tensor]):
        self.parameters += new_parameters

    @abstractmethod
    def step(self, loss: Tensor):
        pass


