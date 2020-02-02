from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def train(self, loader):
        pass

    @abstractmethod
    def valid(self, loader):
        pass

    @abstractmethod
    def test(self, loader):
        pass
