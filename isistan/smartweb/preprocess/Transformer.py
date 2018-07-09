import abc

__author__ = 'ignacio'


class Transformer(object, metaclass=abc.ABCMeta):
    #
    # Transforms data

    @abc.abstractmethod
    def transform(self, data):
        pass
