import abc

__author__ = 'ignacio'


class ModelFactory(object, metaclass=abc.ABCMeta):
    #
    # Abstract model factory

    @abc.abstractmethod
    def create(self, corpus, dictionary, number_of_topics):
        pass
