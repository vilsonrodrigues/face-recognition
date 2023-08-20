from abc import ABC, abstractmethod

class ModelBaseClass(ABC):

    """
    Abstract class to implement minimum methods
    """

    @abstractmethod
    def _predict(self, input_data):
        raise NotImplementedError('Please Implement this method')

    @abstractmethod
    def __call__(self, input_data):
        raise NotImplementedError('Please Implement this method')