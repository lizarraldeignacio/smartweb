import abc

__author__ = 'ignacio'


class NamedEntityRecognizer(object, metaclass=abc.ABCMeta):

    #
    # Information source abstract class

    @abc.abstractmethod
    def get_entities(self, text):
        pass