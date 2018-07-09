import abc

__author__ = 'ignacio'


class Preprocessor(object, metaclass=abc.ABCMeta):
    #
    # Abstract preprocessor

    @abc.abstractmethod
    def process_term(self, term):
        pass
