from abc import ABC, abstractmethod
import torch


class BaseModel(ABC):

    def __init__(self, model):
        # If a GPU is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = model.to(self.device)

    @abstractmethod
    def train(self, loader):
        pass

    @abstractmethod
    def valid(self, loader):
        pass

    @abstractmethod
    def test(self, loader):
        pass
